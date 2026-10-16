# Reproduce_Baselines

Train TinyEEGNet on BCI Competition IV 2a, reproduce published accuracy, quantize to int8, export an ExecuTorch `.pte` for the Ethos-U85.
Everything downstream (emulator, board) consumes this repo's release artifact.

## Pipeline

```
MOABB download → bandpass 7–30 Hz → 2 s windows → per-subject split
      → TinyEEGNet (float) → eval → int8 quantize → eval → export .pte → eval on x86 runtime
```

## Input

| | |
|---|---|
| Dataset | BCI Competition IV 2a via `moabb.datasets.BNCI2014_001` (auto-downloads, ~700 MB, no login) |
| Subjects | 9 (A01–A09) |
| Channels | 22 EEG at 250 Hz; 3 EOG channels dropped |
| Classes | left hand, right hand (feet and tongue dropped unless `--classes 4`) |
| Trial | 4 s motor-imagery window per trial, 288 trials per session, 2 sessions per subject |
| Split | Session 1 = train/val (80/20), session 2 = test. Never mix sessions. |

## Output

| Artifact | Shape / format | Used by |
|---|---|---|
| `artifacts/tinyeegnet_float.pt` | PyTorch state dict | – |
| `artifacts/tinyeegnet_int8.pte` | ExecuTorch, Ethos-U85 delegated | `Emulator_Deploy`, `Edge_Deploy` |
| `artifacts/test_vectors.npz` | 20 input windows `(20, 1, 22, 500)` int8 + expected logits | On-target correctness check |
| `artifacts/preprocess_params.json` | Filter coefficients, per-channel mean/std, quantization scale and zero point | `Edge_Deploy` on-board preprocessing (must be bit-identical) |
| `results/results.md` | Accuracy table, float vs int8, per subject | Org README, submission |

Model input is fixed: `(1, 1, 22, 500)` = one window, 22 channels, 2 s at 250 Hz. Output `(1, 2)` logits. No dynamic shapes.

## Expectations

| Metric | Target | Source |
|---|---|---|
| Float accuracy, 2-class, subject 1, session 2 | ≥ 75 % | EEGNet paper reports ~70–80 % cross-session for strong subjects |
| Float accuracy, 2-class, mean over 9 subjects | ≥ 70 % | |
| Int8 accuracy drop | ≤ 2 points | |
| `.pte` size | < 100 KB | |
| `.pte` runs under ExecuTorch x86 runtime, logits match PyTorch within 1 e-2 after dequant | required | |

Below these, do not tag a release. Fix it or document why.

## Steps

```bash
# 0. Environment (Python 3.10–3.12)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # torch, executorch, moabb, mne, numpy, scikit-learn
pip install -r requirements-arm.txt        # executorch Arm backend + TOSA deps (see starter repo)

# 1. Data (downloads once, caches under ~/mne_data)
python -m src.data.download --subjects all

# 2. Train float model
python -m src.train --subject 1 --classes 2 --epochs 200 --early-stop 20

# 3. Evaluate float
python -m src.eval --checkpoint artifacts/tinyeegnet_float.pt --subject 1

# 4. Quantize (post-training, int8, Ethos-U quantizer) and evaluate
python -m src.quantize --checkpoint artifacts/tinyeegnet_float.pt --calib-windows 200
python -m src.eval --checkpoint artifacts/tinyeegnet_int8.pt --subject 1

# 5. Export .pte for Ethos-U85 (256 MACs/cycle, matches the starter repo)
python -m src.export --target ethos-u85-256 --out artifacts/tinyeegnet_int8.pte

# 6. Verify .pte on host and dump test vectors
python -m src.verify_pte artifacts/tinyeegnet_int8.pte --dump artifacts/test_vectors.npz

# 7. All subjects, full table
python -m src.run_all --classes 2 > results/results.md
```

## Layout

```
src/
  data/        download.py, preprocess.py (filter, epoch, normalize), splits.py
  models/      tinyeegnet.py, eegnet.py (reference), shallowconvnet.py (reference)
  train.py  eval.py  quantize.py  export.py  verify_pte.py  run_all.py
configs/       default.yaml (window, filter, lr, seed)
artifacts/     git-ignored except *.json; .pte and .pt attached to GitHub releases
results/       results.md, confusion matrices, 2 success + 2 failure example plots
tests/         test_preprocess.py (filter output matches scipy reference), test_shapes.py
```

## Model

TinyEEGNet, per the proposal:

| Layer | Purpose | Shape out |
|---|---|---|
| Conv2d temporal (1×64 kernel, 8 filters) | Band-pass-like filters in 7–30 Hz | (8, 22, 500) |
| Depthwise Conv2d spatial (22×1, depth 2) | Mixes electrodes over motor cortex | (16, 1, 500) |
| BatchNorm, ELU, AvgPool 1×4, Dropout | | (16, 1, 125) |
| Separable Conv2d (1×16), BatchNorm, ELU, AvgPool 1×8 | | (16, 1, 15) |
| Flatten, Linear | | (2,) |

Constraints for the NPU: no ELU in int8 on Ethos-U85 (replace with ReLU or verify operator support), no dynamic control flow, BatchNorm folded at export. ~2 k parameters.

## Experiments (from proposal)

| Exp | Change | Report |
|---|---|---|
| 1 Structural ablation | Replace spatial conv with Linear | accuracy vs `.pte` size |
| 2 Sensor dropout | Zero k channels at test time, k = 0…22 | accuracy vs k, chance line at 50 % |

Run with `python -m src.experiments.ablation` and `python -m src.experiments.dropout`.

## Reproducibility

- Seed fixed in `configs/default.yaml`; `run_all.py` prints torch, executorch, moabb versions into `results.md`.
- Release tag `v0.1-baseline` carries the `.pte`, test vectors and params as release assets.

## Status

| Step | Done |
|---|---|
| Data loads, shapes verified | |
| Float ≥ 75 % subject 1 | |
| Int8 within 2 pts | |
| `.pte` exported and verified on host | |
| Release `v0.1-baseline` | |
| Experiments 1–2 | |
