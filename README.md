# Reproduce_Baselines

TinyEEGNet (1,938 parameters) decodes left vs right hand motor imagery at **71.5 % (int8)** on BCI Competition IV 2a subjects it never saw in training. The int8 ExecuTorch program is **10.4 KiB**, and all of its compute runs on the Ethos-U85 NPU.

Everything downstream (emulator, board) consumes this repo's release [`v0.1-baseline`](https://github.com/ArmExecuTorchEEG/Reproduce_Baselines/releases/tag/v0.1-baseline).

v2 is done: a 125 Hz TinyEEGNet student (3,426 parameters) distilled from a teacher ensemble trained only on Cho2017 + Lee2019_MI reaches **73.5 % (int8)** on all 9 BCI IV 2a subjects without seeing any 2a training data, see [v2](#v2-cross-dataset-teacher-student-model-125-hz) (planned release `v0.3-distill`).

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
| Test vectors on the Corstone-320 FVP | 20/20, max logit error 0.038, 226,056 NPU cycles (`Emulator_Deploy` release `v0.2-fvp`) | 20/20 argmax match | pass |

The delegated `.pte` does **not** run on the ExecuTorch x86 runtime: its body is an Ethos-U command stream that only the NPU (or the FVP) executes. Host int8 numbers come from the quantized (fake-quant) PyTorch graph that is lowered into the `.pte`.

## Artifacts and trained models

**Google Drive folder (all trained models):** https://drive.google.com/drive/folders/1Q9bduq4tFBu0q5QRQ6d16NgSJGH6ea-w (anyone with the link can view)

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

The same files are attached to the GitHub release [`v0.1-baseline`](https://github.com/ArmExecuTorchEEG/Reproduce_Baselines/releases/tag/v0.1-baseline). The export is deterministic; this command rebuilds all five byte-identically:

```bash
python -m src.export --checkpoint runs/loso_s3_seed2/model.pt --subject 3 --out artifacts/tinyeegnet_int8.pte
```

## How it works

```
MOABB BNCI2014_001 → causal band-pass 7–30 Hz (sosfilt, continuous) → 2 s windows (cue +0.5 … +2.5 s)
   → Euclidean Alignment (A = R^-1/2 per subject-session) → ÷ input_scale
   → TinyEEGNet float, trained cross-subject or per-subject → KPIs
   → int8 PTQ (EthosUQuantizer, 200 training windows) → fake-quant KPIs on host
   → Ethos-U85 delegated .pte + test vectors + preprocess params → FVP check in Emulator_Deploy (20/20, release v0.2-fvp)
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

## v2: cross-dataset teacher-student model (125 Hz)

**Headline:** the wide32 student (3,426 params, 3.18 M MACs) reaches **73.5 % int8 mean accuracy on all 9 BCI IV 2a subjects with no labeled 2a training data (only the 24 unlabeled calibration windows per subject enter distillation with teacher soft labels)**, versus 71.5 % for v0.1, which was trained on 8 of them.

Status: **final** (2026-10-17): teachers and both students trained on Cho2017 + Lee2019_MI for seeds 0, 1 and 2; FVP check of the shipped `.pte`: 20/20 on the Corstone-320 (table below). Everything in v0.1 above is unchanged and `v0.1-baseline` stays the released model until `v0.3-distill` is tagged.

### What changed and why

| | v0.1 | v2 | Why |
|---|---|---|---|
| Training data | 8 of the 9 BCI IV 2a subjects (LOSO) | Cho2017 (52 subjects) + Lee2019_MI (subjects 1–20) | ~70 new subjects instead of 8; BCI IV 2a becomes a pure test set |
| Test data | held-out 2a subject, session 2 trials 25–144 | all 9 2a subjects, session 2 trials 25–144 | same test block, but no 2a trial is ever trained on |
| Model rate / input | 250 Hz, `(1,1,22,500)` | **125 Hz, `(1,1,22,256)`** | 7–30 Hz needs no more than 125 Hz; half the samples |
| Window | cue + 0.5 s to cue + 2.5 s | cue + 0.496 s to cue + 2.544 s (125 Hz samples 62..318) | 256 samples at 125 Hz; 0.5 s = 62.5 samples rounds to 62 |
| Reference | as recorded | common average over the 22 channels | removes each dataset's own reference electrode |
| Training | cross-entropy on labels | distillation from a 3-network conv ensemble | a larger teacher transfers better; the student stays NPU-sized |
| Student | TinyEEGNet F1 8, D 2, F2 16, kern_t 64 (1,938 params) | TinyEEGNet **wide32** F1 16, D 2, F2 32, kern_t 32 (3,426 params) | twice the filters for about half the MACs of v0.1 |

### Data

| Source | Subjects | Raw | Trials | Role |
|---|---|---|---|---|
| Cho2017 (GigaDB) | 52 | 512 Hz, 64 ch | 10,520 (9,468 train / 1,052 val) | train / val |
| Lee2019_MI (GigaDB), sessions 1 and 2 | 1–20 | 1000 Hz, 62 ch, no FCz (interpolated) | 4,000 (3,600 train / 400 val; 40 subject-sessions) | train / val |
| BCI IV 2a session 2, trials 1–24 | 9 | 250 Hz, 22 ch | 216 | unlabeled EA calibration per subject, and KD soft labels (labels never used) |
| BCI IV 2a session 2, trials 25–144 | 9 | 250 Hz, 22 ch | 1,080 | **test** |

Validation = stratified 10 % of every training subject-session (early stopping only). 2a session 1 is unused.

Download: `src/data/external.py` **never downloads**; it raises `DataNotDownloaded` when a file is missing and skips that subject with a warning (`meta["skipped"]`). Fetch the files once with MOABB's `data_path` (~10.5 GB Cho2017, ~24 GB Lee2019_MI subjects 1–20, into `~/mne_data`):

```bash
MOABB_DOWNLOAD_PROVIDER=upstream python -c "
from moabb.datasets import Cho2017, Lee2019_MI
for s in range(1, 53): Cho2017().data_path(s)
for s in range(1, 21): Lee2019_MI().data_path(s)"
```

`MOABB_DOWNLOAD_PROVIDER=upstream` pins MOABB 1.7 to the GigaDB files instead of its NEMAR mirrors; `external.py` sets it for its own process.

### Processing

One pipeline for every recording (`src/data/external.py`, `src/data/cache125.py`, `src/data/splits125.py`), cached unaligned in `runs/cache125/{dataset}_s{NN}.npz`:

| Step | Operation |
|---|---|
| 1 | Load via MOABB, drop non-EEG channels, keep left/right cues (codes mapped to 769/770 by event name; Lee2019_MI has them reversed) |
| 2 | Montage `standard_1005` by name; interpolate any of the 22 names that is missing (spherical splines; Lee2019_MI: FCz) |
| 3 | Pick the 22 channels in 2a order (`src.train.CHANNELS`) |
| 4 | Common average reference: subtract the per-sample mean of the 22 channels |
| 5 | To 125 Hz with a causal 4th-order Butterworth 7–30 Hz (`sosfilt`, continuous). **2a (the board path):** v0.1 SOS at 250 Hz, then keep every second sample (phase 0, cue → `cue // 2`). **Cho2017, Lee2019_MI:** MNE polyphase resample to 125 Hz, then the same filter designed at 125 Hz |
| 6 | Epoch: 2a samples `[62, 318)` after the cue; training sets `[37, 343)` (306 samples) so that `X[..., 25:281]` is exactly the 2a window and augmentation can crop ±25 |
| 7 | Euclidean Alignment per subject-session, ridge `R + ea_eps·tr(R)/22·I`, `ea_eps` 0.001 (CAR makes the covariance rank-deficient by one). 2a: `A` from session 2 trials 1–24 |
| 8 | One global `input_scale` = std of the pooled aligned training windows |

Units: Cho2017 values are divided by 32 (`external.UNIT_SCALE`): MOABB reads them as µV, but they behave like BioSemi ADC counts (31.25 nV). EA makes the model insensitive to the factor; it only matters for amplitude checks.

### Models

Input `(1,1,22,256)`, output `(1,2)` logits. MACs = conv + linear multiply-accumulates for one window; params include BatchNorm (folded at export).

| Network | Role | Params | MACs | Runs on |
|---|---|---:|---:|---|
| EEGNet-ELU 8,2 (kern_t 64) | teacher | 1,714 | 3.02 M | host |
| ShallowConvNet (kernels halved for 125 Hz) | teacher | 38,242 | 11.38 M | host |
| DeepConvNet (kernel 5, pool 3) | teacher | 146,302 | 5.51 M | host |
| Teacher ensemble (mean softmax) | teacher | 186,258 | 19.9 M | host |
| **TinyEEGNet wide32** (F1 16, D 2, F2 32, kern_t 32, kern_s 16) | **student, shipped** | **3,426** | **3.18 M** | Ethos-U85 |
| TinyEEGNet control (F1 8, D 2, F2 16, kern_t 32, kern_s 16) | student, ablation | 1,458 | 1.57 M | Ethos-U85 |
| v0.1 TinyEEGNet (250 Hz, 500 samples) | reference | 1,938 | 5.88 M | Ethos-U85 |

The students use the same `src/models/tinyeegnet.py` as v0.1 (ReLU, conv / depthwise / avgpool / linear only). `get_model` reads the widths from a `student.json` next to it; without one it builds the v0.1 defaults.

Wide32 layers:

| Layer | Shape out (C, H, W) |
|---|---|
| Input (aligned µV ÷ `input_scale`) | (1, 22, 256) |
| Conv2d temporal 1×32, 16 filters + BN | (16, 22, 257) |
| Depthwise Conv2d spatial 22×1, depth 2 + BN + ReLU | (32, 1, 257) |
| AvgPool 1×4, Dropout 0.25 | (32, 1, 64) |
| Separable Conv2d (depthwise 1×16 + pointwise 1×1) + BN + ReLU | (32, 1, 65) |
| AvgPool 1×8, Dropout 0.25 | (32, 1, 8) |
| Flatten, Linear | 256 → 2 logits |

### Distillation recipe

| | |
|---|---|
| Teachers | each trained with the v0.1 `src.train.fit` (Adam, cross-entropy, early stop on validation loss) on the canonical crop |
| Loss | `0.3 · CE(hard label) + 0.7 · T² · KL(p_teacher ‖ p_student)`, T = 4; `p_teacher` = mean of the three softmax(z/T) |
| Augmentation, per batch | random crop of 256 from the 306-sample trial (offset 25 ± 25), Gaussian noise 0.05 × input std, channel dropout p 0.1 (whole channels zeroed). The teacher scores the same augmented batch |
| 2a calibration trials | session 2 trials 1–24 of each subject, mixed into the stream with teacher soft labels only (KD term, no CE, no labels) |
| Optimiser | Adam lr 1e-3, batch 64, ≤ 300 epochs, early stop on validation KL, patience 30, best weights restored |
| Seeds | 0, 1, 2 |

### Results

From [`results/results_v2.md`](results/results_v2.md). Test = BCI IV 2a subjects 1–9, session 2 trials 25–144 (1,080 trials), never trained on. Mean ± std across the 9 subjects; each subject is the mean of seeds 0, 1, 2. Training: 13,068 trials (92 subject-sessions), validation 1,452.

| Model | Precision | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa | Params | MACs |
|---|---|---|---|---|---|---|---|---|
| EEGNet-ELU (teacher) | float | 72.9 ± 15.3 | 0.773 ± 0.174 | 0.719 ± 0.163 | 0.783 ± 0.153 | 0.457 ± 0.306 | 1,714 | 3.02 M |
| ShallowConvNet (teacher) | float | 73.2 ± 14.7 | 0.780 ± 0.158 | 0.727 ± 0.151 | 0.791 ± 0.147 | 0.464 ± 0.295 | 38,242 | 11.38 M |
| DeepConvNet (teacher) | float | 72.8 ± 14.4 | 0.772 ± 0.162 | 0.723 ± 0.149 | 0.783 ± 0.141 | 0.457 ± 0.288 | 146,302 | 5.51 M |
| Teacher ensemble (mean softmax) | float | 74.4 ± 14.9 | 0.786 ± 0.162 | 0.737 ± 0.155 | 0.800 ± 0.141 | 0.487 ± 0.298 | 186,258 | 19.91 M |
| Student wide32 | float | 73.8 ± 15.5 | 0.778 ± 0.167 | 0.730 ± 0.163 | 0.790 ± 0.146 | 0.476 ± 0.310 | 3,426 | 3.18 M |
| **Student wide32 (shipped)** | **int8** | **73.5 ± 15.7** | **0.779 ± 0.165** | **0.726 ± 0.166** | **0.791 ± 0.145** | **0.469 ± 0.314** | **3,426** | **3.18 M** |
| Student control | float | 73.6 ± 15.5 | 0.769 ± 0.177 | 0.728 ± 0.164 | 0.779 ± 0.157 | 0.472 ± 0.310 | 1,458 | 1.57 M |
| Student control | int8 | 73.3 ± 15.5 | 0.769 ± 0.176 | 0.724 ± 0.165 | 0.778 ± 0.157 | 0.465 ± 0.310 | 1,458 | 1.57 M |
| *v0.1 TinyEEGNet + EA, LOSO on 2a, 250 Hz (different protocol)* | float | 71.5 ± 13.7 | 0.776 ± 0.142 | 0.708 ± 0.141 | 0.780 ± 0.141 | 0.429 ± 0.273 | 1,938 | 5.88 M |
| *v0.1 TinyEEGNet + EA, LOSO on 2a, 250 Hz (different protocol)* | int8 | 71.5 ± 13.5 | 0.778 ± 0.141 | 0.708 ± 0.140 | 0.782 ± 0.140 | 0.429 ± 0.270 | 1,938 | 5.88 M |

The v0.1 rows train on the other 8 BCI IV 2a subjects (leave-one-subject-out) at 250 Hz with 500-sample windows; context only. Ensemble params and MACs are the sum of its three members (offline only).

Accuracy per subject, int8 (%):

| Model | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| v2 student wide32 | 89.7 | 62.8 | 99.2 | 74.2 | 66.4 | 55.8 | 62.5 | 93.9 | 56.7 | 73.5 |
| v0.1 TinyEEGNet + EA, LOSO | 85.0 | 53.9 | 92.5 | 67.2 | 58.1 | 64.4 | 66.7 | 91.1 | 64.2 | 71.5 |

- **Int8 costs 0.3 points** on average for both students (wide32: −1.4 to +0.8 per subject).
- **The student keeps almost all of the teacher:** 73.5 % int8 at 3,426 params vs 74.4 % for the 186 k-param ensemble.
- **Wide32 vs control is small:** +0.2 points int8 for 2.3× the parameters and 2× the MACs; wide32 was picked on mean int8 accuracy.
- **Not uniform over v0.1:** v2 gains on A01–A05 and A08 but loses on A06 (−8.6), A07 (−4.2) and A09 (−7.5).

All KPIs per subject, int8 minus float per subject, training wall times and confusion matrices: [`results/results_v2.md`](results/results_v2.md), `results/confusion_v2/`.

### Deployment (shipped model: wide32 seed 0, subject A03)

Seed 0 was chosen without test data (lowest best validation KL to the teacher, 0.0010). Numbers from the export reports `artifacts/v2/deployment.md` and `artifacts/v2/op_placement.md` (local, git-ignored with the bundle).

| Item | Value |
|---|---|
| `.pte` size | 11,456 B (11.2 KiB), `ethos-u85-256` |
| Op placement | 35 ops on Ethos-U, 3 on CPU (input quantize, output dequantize, getitem); 1 delegated subgraph, all conv/linear on Ethos-U |
| Parameters / MACs per window | 3,426 / 3.18 M (3,176,128) |
| Input qparams | scale 0.0695584, zero point −12; `input_scale` 0.969264 |
| 20 test vectors (host, fake-quant) | 100 % int8, 100 % float; max \|logit\| diff 0.275 |
| A03, 120 test trials | 99.2 % float, 99.2 % int8, 100 % argmax agreement; max \|logit\| diff 1.60 |
| SRAM (Vela estimate) | 96.4 KiB (v0.1: 183 KiB); 5.75 KiB on-chip flash |
| Test vectors on the Corstone-320 FVP (`SSE-320-U85`) | 20/20 argmax match, 20/20 vs true label, max abs logit error 0.026 (one output quant step) |
| FVP cycles, latency, memory pools | 128,056 NPU cycles + 28,892 CPU cycles per window (v0.1: 226,056 NPU); 0.39 ms at 400 MHz; method pool 28,812 B + temp pool 98,688 B = 127,500 B (124.5 KiB) |

FVP run 2026-10-17 in `Emulator_Deploy` (branch `v2/fvp-256`, `results/fvp_v2_metrics.md`), AC6 6.24, FVP 11.32.23, `.pte` sha256 `6ce9ca4f…`.

Vela's static cycle estimate puts 49 % of the cycles in the NCHW transposes; its MAC count (1.77 M) counts the temporal conv at half, see `artifacts/v2/deployment.md`.

Bundle files (A03, wide32 seed 0, checkpoint `runs/v2/student_wide32_seed0/model.pt`), in `artifacts/v2/`:

| File | Bytes | SHA-256 |
|---|---:|---|
| `tinyeegnet_int8.pte` | 11,456 | `6ce9ca4f5f0491c2c6841655e0db02972886207b39a6c73cd35840e3574cdb73` |
| `tinyeegnet_float.pt` | 20,398 | `ee36b12148114e9871738a9b40734db67b88dca4d11ce0565c40b7f525be4f91` |
| `student.json` | 64 | `162a7dc32bd4741bd7b2a2fff72d4792445d76a30c0e3b0aeae6ac8af92242f9` |
| `preprocess_params.json` | 121,185 | `87af6459137ffb9954c9c627c5bb7c887bc319db1960aa4e0b06a413afde397e` |
| `test_vectors.npz` | 113,694 | `33dbd4ad3850c6ac28dfcf23291ddc4f783234ccba99e34dd852b4696021ef7d` |
| `calibration.npz` | 4,505,856 | `212c94fe67195c64b1231b52497c5fffac05620e9062ad88935b78c5a3df4f7f` |

`artifacts/` is git-ignored except top-level `*.json`, so none of these is in git (not even `artifacts/v2/preprocess_params.json`); they ship with the release `v0.3-distill` and the Drive folder. Rebuild with step 4 below (`--checkpoint runs/v2/student_wide32_seed0/model.pt --run-dir runs/v2/student_wide32_seed0`).

### Commands

```bash
# 0. Data: download once (see Data above); the loaders never download.

# 1. Teacher ensemble, one seed (Cho2017 + Lee2019_MI, scored on all 9 2a subjects)
python -m src.train_teacher --seed 0 --out runs/v2/teacher_seed0 --device mps

# 2. Distil a student from it, then its int8 (fake-quant) predictions
python -m src.distill --seed 0 --student wide32 --teacher runs/v2/teacher_seed0 --out runs/v2/student_wide32_seed0 --device mps
python -m src.quantize --run runs/v2/student_wide32_seed0

# 3. Everything (teachers, then students; finished runs skipped) → results/results_v2.md
python -m src.run_v2 --jobs 1 --seeds 0 1 2 --students wide32 control
python -m src.run_v2 --skip-train          # rebuild the report only

# 4. Export the chosen student for subject 3 (the v2 path is picked from student.json)
python -m src.export --checkpoint runs/v2/student_wide32_seedK/model.pt --run-dir runs/v2/student_wide32_seedK \
    --subject 3 --out artifacts/v2/tinyeegnet_int8.pte
```

Config: `configs/v2.yaml` (window, rate, filter, `ea_eps`, datasets, KD settings, student widths). `src.data.splits125.v2_data` refuses a config that disagrees with the constants in `src/data/cache125.py`.

Release: tag **`v0.3-distill`** with the v2 bundle (`tinyeegnet_int8.pte`, `tinyeegnet_float.pt`, `student.json`, `calibration.npz`, `test_vectors.npz`, `preprocess_params.json`). `v0.1-baseline` stays as it is.

### Contract changes (v2 vs v0.1)

- Model input **`(1,1,22,256)`** at **125 Hz** (v0.1: `(1,1,22,500)` at 250 Hz). Output still `(1,2)` logits, float I/O with CPU quantize/dequantize.
- The UART stream stays 250 Hz. The board adds a **common average reference** over the 22 channels before the filter, keeps the v0.1 SOS at 250 Hz, then **keeps every second sample** (`decimation {factor 2, phase 0}`).
- Window: 256 samples starting 62 samples (125 Hz) after the cue (cue + 0.496 s .. + 2.544 s).
- `preprocess_params.json` gains `version`, `fs_model`, `n_channels`, `input_shape`, `units`, `reference`, `filter`, `decimation`, `window`, `cue_offset_s`, `ea_matrices` (all 9 subjects), `ea_calibration`, `input_formula`, `output`, `student`. `ea_matrix` is the export subject's matrix, as in v0.1. `tmin` / `tmax` are gone; the window is `window.start_samples_after_cue_125hz` + `n_samples`.
- New bundle file `student.json` (`F1`, `D`, `F2`, `kern_t`, `kern_s`): `Emulator_Deploy/model/model.py` stays identical to `src/models/tinyeegnet.py` and builds the student from it.
- `test_vectors.npz`: `x` int8 (20,1,22,256).
- Hand-off for Emulator_Deploy and Edge_Deploy: [`docs/v2_handoff.md`](docs/v2_handoff.md).

New files: `src/data/external.py`, `src/data/cache125.py`, `src/data/splits125.py`, `src/models/convnets.py`, `src/train_teacher.py`, `src/distill.py`, `src/run_v2.py`, `configs/v2.yaml`, `tests/test_v2_*.py`, `results/results_v2.md`, `results/confusion_v2/`, `docs/v2_handoff.md`.
