# Reproduce_Baselines

Train TinyEEGNet on BCI Competition IV 2a (left vs right hand), leave-one-subject-out with Euclidean Alignment, quantize to int8 and export an ExecuTorch `.pte` for the Ethos-U85.
Everything downstream (emulator, board) consumes this repo's release `v0.1-baseline`.

## Pipeline

```
MOABB BNCI2014_001 → causal band-pass 7–30 Hz (sosfilt, continuous) → 2 s windows (cue +0.5 … +2.5 s)
   → Euclidean Alignment (A = R^-1/2 per subject-session) → ÷ input_scale
   → TinyEEGNet float, LOSO (train on 8 subjects, test on the 9th) → KPIs
   → int8 PTQ (EthosUQuantizer, 200 training windows) → fake-quant KPIs on host
   → Ethos-U85 delegated .pte + test vectors + preprocess params → FVP check in Emulator_Deploy (pending)
```

The delegated `.pte` does **not** run on the ExecuTorch x86 runtime: its body is an Ethos-U command stream that only the NPU (or the Corstone-320 FVP) executes. Host-side int8 numbers come from the quantized (fake-quant) PyTorch graph that is lowered into the `.pte`. The target check (test vectors and cycle counts on the Corstone-320 FVP) is **pending** in `Emulator_Deploy`; no FVP run has been made yet.

## Data and protocol

| | |
|---|---|
| Dataset | BCI Competition IV 2a via `moabb.datasets.BNCI2014_001` (auto-downloads to `~/mne_data`, ~700 MB, no login) |
| Subjects | 9 (A01–A09), 2 sessions each |
| Channels | 22 EEG at 250 Hz; 3 EOG dropped |
| Classes | 0 = left hand (769), 1 = right hand (770); feet and tongue dropped → 144 trials per session |
| Filter | 4th-order Butterworth band-pass 7–30 Hz as SOS, applied **causally** with `scipy.signal.sosfilt` on the continuous recording, so the board can run the same sections sample by sample |
| Window | cue + 0.5 s to cue + 2.5 s = 500 samples |
| Alignment | Euclidean Alignment (He & Wu 2020): `R = mean(X Xᵀ / T)`, `A = R^-1/2`, `X̃ = A X`. Training subjects: one `A` per subject-session. Target subject: `A` from the **first 24 trials of session 2** (unlabeled calibration block) |
| Row B (headline) | LOSO: train on the other 8 subjects (both sessions), early stop on a stratified 10 % validation split of every source subject-session (no subject held out), no target labels |
| Row A (comparison) | Per-subject: train on the target's session 1 (80/20 early-stop split) |
| Test set | Target session 2, trials 25–144 (120 per subject). Never used for early stopping, model selection, normalisation or quantization calibration |
| KPIs | Accuracy (primary), ROC-AUC, macro F1, AUPRC (positive = right hand), Cohen's kappa; 9 subjects × 3 seeds |

## Model

TinyEEGNet: EEGNet-8,2 shape, **ReLU instead of ELU**, no max-norm, BatchNorm folded at export, no Softmax. 1,938 parameters.

| Layer | Shape out (C, H, W) |
|---|---|
| Input (aligned µV ÷ `input_scale`) | (1, 22, 500) |
| Conv2d temporal 1×64, 8 filters, pad 32 + BN | (8, 22, 501) |
| Depthwise Conv2d spatial 22×1, depth 2 + BN + ReLU | (16, 1, 501) |
| AvgPool 1×4, Dropout 0.25 | (16, 1, 125) |
| Separable Conv2d (depthwise 1×16 pad 8 + pointwise 1×1) + BN + ReLU | (16, 1, 126) |
| AvgPool 1×8, Dropout 0.25 | (16, 1, 15) |
| Flatten, Linear | 240 → 2 logits |

`src/models/eegnet.py` is the float-only reference (ELU, max-norm). `src/models/tinyeegnet.py` exposes the starter interface `get_model(input_shape)` / `get_calibration_inputs(input_shape, calibration_samples)`.

## Output (contract)

| Artifact | Content | Used by |
|---|---|---|
| `artifacts/tinyeegnet_int8.pte` | ExecuTorch 1.5.1 program, `ethos-u85-256`. All conv / pool / ReLU / linear inside one Ethos-U delegate; only the input `quantize_per_tensor` and output `dequantize_per_tensor` run on the CPU (see `results/op_placement.md`) | `Emulator_Deploy`, `Edge_Deploy` |
| `artifacts/tinyeegnet_float.pt` | Float state dict of the export checkpoint | `Emulator_Deploy/model/` |
| `artifacts/calibration.npz` | `x` (200, 1, 22, 500) float32: aligned, scaled **training** windows used for PTQ | `Emulator_Deploy/model/` |
| `artifacts/test_vectors.npz` | `x` int8 (20, 1, 22, 500) = export subject session 2 trials 25–44; `logits` float32 (20, 2) of the quantized model; `y` int64 (20,) | On-target correctness check |
| `artifacts/preprocess_params.json` | `sos`, `tmin`, `tmax`, `fs`, `n_samples`, `channel_order`, `ea_matrix` (22×22), `input_scale`, `input_quant_scale`, `input_zero_point`, `class_map` `{"0":"left_hand","1":"right_hand"}` | `Edge_Deploy` on-board preprocessing (must match) |
| `results/results.md` | KPI tables, float vs int8, per subject | Org README, submission |
| `results/op_placement.md`, `results/deployment.md` | Ops on Ethos-U vs CPU; params, MACs, `.pte` size, Vela estimate, int8 accuracy on the test vectors | Submission |

Model input `(1, 1, 22, 500)`, output `(1, 2)` logits, static shapes. On-board input path, per window:

```
x_model = (ea_matrix @ x_uV) / input_scale
x_int8  = clamp(round(x_model / input_quant_scale) + input_zero_point, -128, 127)
```

**Contract changes vs the original README:**
- `preprocess_params.json` carries the 22×22 EA matrix `ea_matrix` and one global `input_scale` instead of per-channel mean/std. The board multiplies each filtered sample vector by `ea_matrix` before scaling. `Edge_Deploy` must be updated to match; that is a separate task.
- The `.pte` keeps a float input and output, like the starter's `create_ai_layer.py` export: the CPU quantizes the input with (`input_quant_scale`, `input_zero_point`) and dequantizes the logits. Feeding `(x_int8 - input_zero_point) * input_quant_scale` reproduces `x_int8` exactly inside the program.

## Results

BCI IV 2a, left vs right hand, leave-one-subject-out (LOSO): each target subject is never seen in training; its session 2 trials 1–24 are an unlabeled Euclidean-Alignment calibration block and trials 25–144 (120 trials) are the test set. Mean ± std across the 9 subjects, 3 seeds each.

| Model | Precision | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa |
|---|---|---|---|---|---|---|
| **TinyEEGNet + EA (LOSO)** | float | **71.5 ± 13.7** | 0.776 ± 0.142 | 0.708 ± 0.141 | 0.780 ± 0.141 | 0.429 ± 0.273 |
| **TinyEEGNet + EA (LOSO)** | int8 | **71.5 ± 13.5** | 0.778 ± 0.141 | 0.708 ± 0.140 | 0.782 ± 0.140 | 0.429 ± 0.270 |

Int8 post-training quantization costs 0.0 points on average. Export subject A03: 92.2 % float, 92.5 % int8.

Context (same split, accuracy): CSP+LDA 62.7 %, CSP+LDA+EA 71.3 %, EEGNet-ELU+EA 73.4 % (reference, not NPU-tuned), TinyEEGNet per-subject 70.6 %. Subjects A02 and A05 are near chance for every method; A09 does much better per-subject (90.6 %) than LOSO (64.4 %).

Full tables (per subject, all methods, EA-calibration variant, deployment numbers, gates, confusion matrices): [`results/results.md`](results/results.md).

## Artifacts

**Google Drive folder (v0.1-baseline):** https://drive.google.com/drive/folders/1Q9bduq4tFBu0q5QRQ6d16NgSJGH6ea-w

Export subject A03, checkpoint `runs/loso_s3_seed2/model.pt` (seed chosen by lowest validation loss).

| File | Bytes | SHA-256 | On Drive |
|---|---:|---|---|
| `tinyeegnet_int8.pte` | 10,624 | `6ec05cc3da08503b20068a01bab22f16cd59140e095a073219d3bd9eb00a1644` | yes |
| `tinyeegnet_float.pt` | 14,254 | `e29858104ee0b2c83c11e12668e681c62d7cd4898fe4d5798d34501bbe0d1186` | yes |
| `preprocess_params.json` | 13,576 | `d0b40a6f7fa2699a57b3ef3c00b49fbe6d027968561612a9d1571690c1b8170b` | yes (also in git) |
| `test_vectors.npz` | 221,054 | `4bc90c56dd67d0748801fde8ef215c7f93f601aa81634e91798e3ad26cec72d3` | no: regenerate (below) or GitHub release |
| `calibration.npz` | 8,800,256 | `19e69787dbb3afebe504623f7edb2862d2ef3c273f5c301c34c69c36e977703a` | no: regenerate (below) or GitHub release |

All five files are rebuilt byte-identically (seeded) by:

```bash
python -m src.export --checkpoint runs/loso_s3_seed2/model.pt --subject 3 --out artifacts/tinyeegnet_int8.pte
```

## Gates

| Metric | Target |
|---|---|
| Float accuracy, export subject, LOSO+EA | ≥ 75 % |
| int8 accuracy drop (fake-quant vs float) | ≤ 2 points |
| `.pte` size | < 100 KB |
| Op placement | every conv / linear inside the Ethos-U delegate |
| Test vectors on the FVP (`Emulator_Deploy`) | 20/20 argmax match (verification pending; FVP not run yet) |
| LOSO+EA mean over 9 subjects | reported as measured (no target) |

Below these, do not tag a release. Fix it or document why.

## Steps

```bash
# 0. Environment (Python 3.12)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# TOSA serializer for the Arm backend, without dependencies (as CMSIS-Executorch/setup_venv.py does;
# letting pip resolve them upgrades flatbuffers and breaks serialization)
pip install --no-deps ml_dtypes==0.5.1 flatbuffers==24.3.25 tosa-tools==2026.5.0

# 1. Data (downloads once to ~/mne_data; filtered windows cached in runs/cache)
python -m src.data.download --subjects all

# 2. One LOSO run (row B), and its int8 (fake-quant) predictions
python -m src.train --mode loso --subject 1 --seed 0 --out runs/loso_s1_seed0
python -m src.quantize --run runs/loso_s1_seed0          # writes runs/loso_s1_seed0/int8_predictions.npz

# 3. Everything (9 subjects × 3 seeds, rows A and B, baselines) → results/results.md
python -m src.run_all --jobs 6

# 4. Export the chosen subject's checkpoint (Vela config overridable: --system-config, --memory-mode, --config-ini)
python -m src.export --checkpoint runs/loso_sN_seed0/model.pt --subject N --out artifacts/tinyeegnet_int8.pte

# 5. Tests
python -m pytest tests/
```

## Integration notes

Rebuilding the program with the starter in `Emulator_Deploy`:

- **Model files.** Copy `src/models/tinyeegnet.py` to `model/model.py` and put `artifacts/tinyeegnet_float.pt` and `artifacts/calibration.npz` next to it.
- **csolution.** In the `mlops: model:` node, `input-shape` must become `1x1x22x500` (the starter has `1x16x16x3`). Set `calibration-samples: 200`.
- **`create_ai_layer.py` fix (verified).** The unmodified `export_model()` fails on TinyEEGNet with `call_module is not supported` in the Arm quantizer's `DecomposeAvgPool2dPass`. With torch 2.14, `torch.export.export(...).module()` inserts a `_guards_fn` call module that the pass cannot retrace. The fix is one line at `create_ai_layer.py:198`: `.module(check_guards=False)`, the same call `src/quantize.py` uses. The patched starter exports a 10,624-byte `.pte`, the same size as `artifacts/tinyeegnet_int8.pte`. The starter's demo `TinyCNN` has no AvgPool, which is why it never hit this.

## Layout

```
src/
  data/        download.py, preprocess.py (causal filter, epoch), align.py (EA), splits.py (LOSO, per-subject)
  models/      tinyeegnet.py, eegnet.py (reference), csp_lda.py (baseline)
  train.py  eval.py  quantize.py  export.py  run_all.py
configs/       default.yaml (window, filter, training, seeds)
artifacts/     git-ignored except *.json; files go to the release / Drive folder
results/       results.md, confusion/, op_placement.md, deployment.md
tests/         acceptance tests (preprocess, align, shapes, metrics, leakage, export, artifacts, repro)
runs/          git-ignored training runs and data cache
```

## Reproducibility

- Seeds 0, 1, 2 in `configs/default.yaml`; `run_all.py` writes library versions and the git commit into `results.md`.
- Versions are pinned to `CMSIS-Executorch/requirements.txt`: torch 2.14.0, executorch 1.5.1, torchao 0.18.0, ethos-u-vela 5.1.0. A `.pte` only loads in the runtime pack of the same ExecuTorch version.
- The export is deterministic: same checkpoint and data give a byte-identical `.pte`.

## Out of scope for v0.1

Fine-tuning on the target subject, sensor-dropout and structural ablations, 4 s windows, cross-dataset (Cho2017).
