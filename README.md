# Reproduce_Baselines

TinyEEGNet (1,938 parameters) decodes left vs right hand motor imagery at **71.5 % (int8)** on BCI Competition IV 2a subjects it never saw in training. The int8 ExecuTorch program is **10.4 KiB**, and all of its compute runs on the Ethos-U85 NPU.

Everything downstream (emulator, board) consumes this repo's release `v0.1-baseline`.

## Results

BCI IV 2a, left vs right hand. Each subject is tested on session 2 trials 25–144 (120 trials). Trials 1–24 of session 2 are an unlabeled calibration block used only to compute that subject's alignment matrix. Mean ± std across the 9 subjects; each subject is the mean of 3 seeds. All rows use the same split.

| Method | Training | Precision | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa |
|---|---|---|---|---|---|---|---|
| CSP+LDA | cross-subject | float | 62.7 ± 15.0 | 0.718 | 0.595 | 0.725 | 0.254 |
| CSP+LDA + EA | cross-subject | float | 71.3 ± 15.4 | 0.782 | 0.706 | 0.775 | 0.426 |
| EEGNet-ELU + EA (reference, not NPU-exported) | cross-subject | float | 73.4 ± 13.1 | 0.779 | 0.726 | 0.788 | 0.469 |
| **TinyEEGNet + EA (headline)** | **cross-subject** | **float** | **71.5 ± 13.7** | **0.776** | **0.708** | **0.780** | **0.429** |
| **TinyEEGNet + EA (headline)** | **cross-subject** | **int8** | **71.5 ± 13.5** | **0.778** | **0.708** | **0.782** | **0.429** |
| TinyEEGNet + EA | per-subject | float | 70.6 ± 17.0 | 0.742 | 0.696 | 0.742 | 0.411 |
| TinyEEGNet + EA | per-subject | int8 | 70.6 ± 17.2 | 0.743 | 0.697 | 0.743 | 0.412 |

Cross-subject = leave-one-subject-out: trained on the other 8 subjects, no labels from the test subject. Per-subject = trained on the test subject's own session 1. EA = Euclidean Alignment (see [How it works](#how-it-works)).

Accuracy per subject, int8 TinyEEGNet + EA (%):

| | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| Cross-subject | 85.0 | 53.9 | **92.5** | 67.2 | 58.1 | 64.4 | 66.7 | 91.1 | 64.2 | 71.5 |
| Per-subject | 73.1 | 53.1 | **95.8** | 57.5 | 54.7 | 56.7 | 59.7 | 94.2 | 90.8 | 70.6 |

- **Int8 costs nothing:** the mean accuracy is the same in float and int8.
- **Alignment matters:** it lifts CSP+LDA from 62.7 % to 71.3 %. TinyEEGNet is about 2 points behind the full EEGNet with far fewer parameters.
- **Subject spread is large:** A02 and A05 are near chance for every method. A09 does much better per-subject (90.8 %) than cross-subject (64.2 %).

Full tables (all KPIs per subject, all methods, alignment-calibration variant, confusion matrices): [`results/results.md`](results/results.md).

## Deployment (subject A03, release model)

The release `.pte` is the cross-subject model for A03, seed 2 (the seed with the lowest validation loss). Numbers from [`results/deployment.md`](results/deployment.md) and [`results/op_placement.md`](results/op_placement.md).

| Item | Value | Gate | Status |
|---|---|---|---|
| Float accuracy, A03 (mean of 3 seeds) | 92.2 % | ≥ 75 % | pass |
| Int8 minus float accuracy, A03 | +0.3 points | drop ≤ 2 points | pass |
| Release checkpoint, 120 test trials | 92.5 % float, 92.5 % int8, 100 % argmax agreement | | |
| `.pte` size | 10,624 B (10.4 KiB) | < 100 KB | pass |
| Op placement | 35 ops on Ethos-U, 3 on CPU (input quantize, output dequantize, getitem) | all conv/linear on Ethos-U | pass |
| Parameters / MACs per window | 1,938 / 5.88 M | | |
| SRAM (Vela estimate) | 183 KiB | | |
| Test vectors on the Corstone-320 FVP | not run yet (`Emulator_Deploy`) | 20/20 argmax match | pending |

The delegated `.pte` does **not** run on the ExecuTorch x86 runtime: its body is an Ethos-U command stream that only the NPU (or the FVP) executes. Host int8 numbers come from the quantized (fake-quant) PyTorch graph that is lowered into the `.pte`.

## Artifacts and trained models

**Google Drive folder (all trained models):** https://drive.google.com/drive/folders/1Q9bduq4tFBu0q5QRQ6d16NgSJGH6ea-w

| Drive path | Content |
|---|---|
| top level | The 5 release files below (A03 cross-subject) |
| `models/tinyeegnet_loso_ea/A0N_seedK/` | 27 cross-subject TinyEEGNet models (9 subjects × 3 seeds) |
| `models/tinyeegnet_per_subject_ea/A0N_seedK/` | 27 per-subject TinyEEGNet models |
| `models/eegnet_elu_ea_reference_float/` | 27 float EEGNet reference checkpoints (not exported to the NPU) |
| `MODELS.md`, `models.csv` | Accuracy per model, recommended seed per subject, `.pte` SHA-256 |

Every TinyEEGNet folder is a complete on-chip bundle with the same 5 files as the release, plus `metrics.json`, `op_placement.md` and `deployment.md`. All 54 `.pte` files are fully delegated (35 ops on Ethos-U, 3 on CPU). To run another subject on the board, use that folder's `.pte` **and** its `preprocess_params.json`: the alignment matrix is per subject. `MODELS.md` marks one recommended seed per subject (lowest validation loss, chosen without test data). The best demo candidate is per-subject A03 (95.8 % int8 mean over seeds).

Release files (A03 cross-subject, checkpoint `runs/loso_s3_seed2/model.pt`):

| File | Bytes | SHA-256 |
|---|---:|---|
| `tinyeegnet_int8.pte` | 10,624 | `6ec05cc3da08503b20068a01bab22f16cd59140e095a073219d3bd9eb00a1644` |
| `tinyeegnet_float.pt` | 14,254 | `e29858104ee0b2c83c11e12668e681c62d7cd4898fe4d5798d34501bbe0d1186` |
| `preprocess_params.json` | 13,576 | `d0b40a6f7fa2699a57b3ef3c00b49fbe6d027968561612a9d1571690c1b8170b` (also in git) |
| `test_vectors.npz` | 221,054 | `4bc90c56dd67d0748801fde8ef215c7f93f601aa81634e91798e3ad26cec72d3` |
| `calibration.npz` | 8,800,256 | `19e69787dbb3afebe504623f7edb2862d2ef3c273f5c301c34c69c36e977703a` |

The same files are attached to the GitHub release `v0.1-baseline`. The export is deterministic; this command rebuilds all five byte-identically:

```bash
python -m src.export --checkpoint runs/loso_s3_seed2/model.pt --subject 3 --out artifacts/tinyeegnet_int8.pte
```

## How it works

```
MOABB BNCI2014_001 → causal band-pass 7–30 Hz (sosfilt, continuous) → 2 s windows (cue +0.5 … +2.5 s)
   → Euclidean Alignment (A = R^-1/2 per subject-session) → ÷ input_scale
   → TinyEEGNet float, trained cross-subject or per-subject → KPIs
   → int8 PTQ (EthosUQuantizer, 200 training windows) → fake-quant KPIs on host
   → Ethos-U85 delegated .pte + test vectors + preprocess params → FVP check in Emulator_Deploy (pending)
```

**Euclidean Alignment.** EEG from different people (and different sessions) has different channel scales and correlations, so a model trained on some people transfers poorly to others. Alignment fixes this with one 22×22 matrix per subject-session: `R` is the average channel covariance of that recording's trials, `A = R^-1/2`, and every window is multiplied by `A`. After alignment every recording has the identity as its average covariance, so all subjects look statistically alike to the model. The matrix needs no labels. For a new subject it is computed from their first 24 trials (about 3 minutes of calibration), then frozen. Both TinyEEGNet models (cross-subject and per-subject) are trained and tested on aligned data. After alignment the data is divided by one global constant, `input_scale` (the standard deviation of the aligned training data), so values sit near ±1 for int8.

## Data and protocol

| | |
|---|---|
| Dataset | BCI Competition IV 2a via `moabb.datasets.BNCI2014_001` (auto-downloads to `~/mne_data`, ~700 MB, no login) |
| Subjects | 9 (A01–A09), 2 sessions each |
| Channels | 22 EEG at 250 Hz; 3 EOG dropped |
| Classes | 0 = left hand (769), 1 = right hand (770); feet and tongue dropped → 144 trials per session |
| Filter | 4th-order Butterworth band-pass 7–30 Hz as SOS, applied **causally** with `scipy.signal.sosfilt` on the continuous recording, so the board can run the same sections sample by sample |
| Window | cue + 0.5 s to cue + 2.5 s = 500 samples |
| Alignment | Euclidean Alignment (He & Wu 2020): `R = mean(X Xᵀ / T)`, `A = R^-1/2`, `X̃ = A X`. Training subjects: one `A` per subject-session. Test subject: `A` from the **first 24 trials of session 2** (unlabeled calibration block) |
| Cross-subject (headline) | Leave-one-subject-out: train on the other 8 subjects (both sessions), early stop on a stratified 10 % validation split of every source subject-session, no test-subject labels |
| Per-subject (comparison) | Train on the test subject's session 1 (aligned with its own `A`), 80/20 early-stop split |
| Test set | Test subject's session 2, trials 25–144 (120 per subject). Never used for early stopping, model selection, normalisation or quantization calibration |
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

# 2. One cross-subject run, and its int8 (fake-quant) predictions
python -m src.train --mode loso --subject 1 --seed 0 --out runs/loso_s1_seed0
python -m src.quantize --run runs/loso_s1_seed0          # writes runs/loso_s1_seed0/int8_predictions.npz

# 3. Everything (9 subjects × 3 seeds, both TinyEEGNet modes, baselines) → results/results.md
python -m src.run_all --jobs 6

# 4. Export a checkpoint (Vela config overridable: --system-config, --memory-mode, --config-ini).
#    --results-dir keeps the report out of results/ when exporting a non-release model.
python -m src.export --checkpoint runs/loso_sN_seedK/model.pt --subject N --out artifacts/tinyeegnet_int8.pte
python -m src.export --checkpoint runs/per_subject_sN_seedK/model.pt --subject N --mode per-subject \
    --out runs/exports/per_subject/A0N_seedK/tinyeegnet_int8.pte --results-dir runs/exports/per_subject/A0N_seedK/report

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
  data/        download.py, preprocess.py (causal filter, epoch), align.py (EA), splits.py (cross-subject, per-subject)
  models/      tinyeegnet.py, eegnet.py (reference), csp_lda.py (baseline)
  train.py  eval.py  quantize.py  export.py  run_all.py
configs/       default.yaml (window, filter, training, seeds)
artifacts/     git-ignored except *.json; files go to the release / Drive folder
results/       results.md, confusion/, op_placement.md, deployment.md
tests/         acceptance tests (preprocess, align, shapes, metrics, leakage, export, artifacts, repro)
runs/          git-ignored training runs, exports and data cache
```

## Reproducibility

- Seeds 0, 1, 2 in `configs/default.yaml`; `run_all.py` writes library versions and the git commit into `results.md`.
- Versions are pinned to `CMSIS-Executorch/requirements.txt`: torch 2.14.0, executorch 1.5.1, torchao 0.18.0, ethos-u-vela 5.1.0. A `.pte` only loads in the runtime pack of the same ExecuTorch version.
- The export is deterministic: same checkpoint and data give a byte-identical `.pte`.

## Out of scope for v0.1

Fine-tuning on the target subject, TinyEEGNet without alignment, sensor-dropout and structural ablations, 4 s windows, cross-dataset (Cho2017).
