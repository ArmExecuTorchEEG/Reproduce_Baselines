# Reproduce_Baselines v3 results: teacher ensemble distilled into TinyEEGNet at 125 Hz

## 1. Setup

Training data: Cho2017 + Lee2019_MI (left vs right hand), 22 channels in the BCI IV 2a order, common average reference over the 22, 125 Hz, causal Butterworth 7–30 Hz, Euclidean Alignment per subject-session, one global `input_scale`. Training windows 306 samples with random 256-sample crops for the students; teachers train on the canonical crop at offset 25 (cue + 0.496 s to cue + 2.544 s). Validation = whole held-out training subjects {"Cho2017": [48, 49, 50, 51, 52], "Lee2019_MI": [19, 20]} (cross-subject; early stopping and checkpoint selection only). **Test: BCI IV 2a (MOABB `BNCI2014_001`) subjects 1–9, session 2 trials 25–144 (120 per subject), never trained on.** Session 2 trials 1–24 of every test subject give its EA matrix and enter the distillation stream with teacher soft labels only (their labels are never used). Seeds [0]; each subject's value is the mean over seeds, ± is the std across subjects. KPIs: accuracy, ROC-AUC, macro F1, AUPRC (positive class = right hand), Cohen's kappa.

Distillation: loss = 0.3 CE + 0.7 T² KL, T = 4.0, noise std 0.05, channel dropout 0.2; the teacher scores each augmented batch on the fly.

Environment: Python 3.12.14, torch 2.14.0, moabb 1.7.2, mne 1.13.2, scikit-learn 1.9.1, scipy 1.18.1, numpy 2.5.4; git commit `01e6471`; generated 2026-10-17 12:25.

## 2. Summary (mean ± std across the 9 2a subjects)

| Model | Precision | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa | Params | MACs | n subj | n seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| EEGNet-ELU | float | 70.7 ± 16.1 | 0.743 ± 0.187 | 0.700 ± 0.168 | 0.752 ± 0.164 | 0.415 ± 0.321 | 1,714 | 3.02 M | 9 | 1 |
| ShallowConvNet | float | 70.2 ± 14.9 | 0.754 ± 0.163 | 0.698 ± 0.152 | 0.779 ± 0.140 | 0.404 ± 0.297 | 38,242 | 11.38 M | 9 | 1 |
| DeepConvNet | float | 70.9 ± 14.0 | 0.762 ± 0.158 | 0.702 ± 0.145 | 0.767 ± 0.146 | 0.419 ± 0.281 | 146,302 | 5.51 M | 9 | 1 |
| Teacher ensemble (mean softmax) | float | 73.2 ± 14.4 | 0.766 ± 0.168 | 0.727 ± 0.149 | 0.775 ± 0.148 | 0.465 ± 0.288 | 186,258 | 19.91 M | 9 | 1 |
| Student wide32 | float | 70.5 ± 16.2 | 0.741 ± 0.182 | 0.697 ± 0.170 | 0.753 ± 0.162 | 0.409 ± 0.324 | 3,426 | 3.18 M | 9 | 1 |
| Student wide32 | int8 | 71.3 ± 15.3 | 0.749 ± 0.178 | 0.705 ± 0.159 | 0.759 ± 0.160 | 0.426 ± 0.305 | 3,426 | 3.18 M | 9 | 1 |
| Student control | float | 70.2 ± 16.3 | 0.742 ± 0.176 | 0.696 ± 0.168 | 0.755 ± 0.158 | 0.404 ± 0.326 | 1,458 | 1.57 M | 9 | 1 |
| Student control | int8 | 70.1 ± 16.2 | 0.742 ± 0.176 | 0.695 ± 0.167 | 0.754 ± 0.158 | 0.402 ± 0.323 | 1,458 | 1.57 M | 9 | 1 |
| *v0.1 TinyEEGNet+EA, LOSO on 2a, 250 Hz (different protocol)* | float | 71.5 ± 13.7 | 0.776 ± 0.142 | 0.708 ± 0.141 | 0.780 ± 0.141 | 0.429 ± 0.273 | 1,938 | 5.88 M | 9 | 3 |
| *v0.1 TinyEEGNet+EA, LOSO on 2a, 250 Hz (different protocol)* | int8 | 71.5 ± 13.5 | 0.778 ± 0.141 | 0.708 ± 0.140 | 0.782 ± 0.140 | 0.429 ± 0.270 | 1,938 | 5.88 M | 9 | 3 |

The v0.1 row trains on the other 8 BCI IV 2a subjects (leave-one-subject-out) at 250 Hz with 500-sample windows; it is shown for context only (`results/results.md`).

MACs = multiply-accumulates of the conv and linear layers for one window; params include BatchNorm. Ensemble params/MACs are the sum of its three members (offline only).

## 3. Per-subject accuracy (%)

| Model | Precision | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EEGNet-ELU | float | 85.8 | 60.8 | 98.3 | 73.3 | 65.8 | 55.0 | 55.8 | 90.0 | 51.7 | 70.7 |
| ShallowConvNet | float | 88.3 | 66.7 | 96.7 | 64.2 | 65.0 | 58.3 | 58.3 | 84.2 | 50.0 | 70.2 |
| DeepConvNet | float | 85.8 | 64.2 | 95.8 | 73.3 | 59.2 | 51.7 | 60.8 | 85.0 | 62.5 | 70.9 |
| Teacher ensemble (mean softmax) | float | 88.3 | 68.3 | 99.2 | 71.7 | 67.5 | 55.8 | 63.3 | 88.3 | 56.7 | 73.2 |
| Student wide32 | float | 89.2 | 57.5 | 98.3 | 73.3 | 65.8 | 55.8 | 55.0 | 86.7 | 52.5 | 70.5 |
| Student wide32 | int8 | 87.5 | 59.2 | 99.2 | 72.5 | 66.7 | 54.2 | 58.3 | 86.7 | 57.5 | 71.3 |
| Student control | float | 88.3 | 59.2 | 99.2 | 72.5 | 62.5 | 48.3 | 60.0 | 85.8 | 55.8 | 70.2 |
| Student control | int8 | 88.3 | 59.2 | 98.3 | 73.3 | 62.5 | 49.2 | 59.2 | 85.8 | 55.0 | 70.1 |
| *v0.1 TinyEEGNet+EA, LOSO on 2a, 250 Hz (different protocol)* | float | 85.3 | 53.1 | 92.2 | 67.8 | 58.1 | 63.9 | 66.9 | 91.4 | 64.4 | 71.5 |
| *v0.1 TinyEEGNet+EA, LOSO on 2a, 250 Hz (different protocol)* | int8 | 85.0 | 53.9 | 92.5 | 67.2 | 58.1 | 64.4 | 66.7 | 91.1 | 64.2 | 71.5 |

v0.1 rows: different protocol (LOSO on BCI IV 2a, 250 Hz, 500 samples), context only.

## 4. Selected student (shipped only if its int8 mean is at least the v2 release's)

Selected: **student wide32** (highest mean int8 accuracy across subjects, tie-break on MACs). Export subject A03, seed 0 (lowest best validation KL to the teacher: 0.0006), checkpoint `runs/v3/student_wide32_seed0/model.pt`.

Per-subject KPIs, float (mean ± std over seeds):

| Subject | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa | best epochs |
|---|---|---|---|---|---|---|
| A01 | 89.2 | 0.950 | 0.892 | 0.942 | 0.783 | 208 |
| A02 | 57.5 | 0.643 | 0.568 | 0.634 | 0.150 | 208 |
| A03 (export) | 98.3 | 1.000 | 0.983 | 1.000 | 0.967 | 208 |
| A04 | 73.3 | 0.771 | 0.730 | 0.772 | 0.467 | 208 |
| A05 | 65.8 | 0.685 | 0.656 | 0.639 | 0.317 | 208 |
| A06 | 55.8 | 0.601 | 0.558 | 0.598 | 0.117 | 208 |
| A07 | 55.0 | 0.616 | 0.547 | 0.651 | 0.100 | 208 |
| A08 | 86.7 | 0.963 | 0.866 | 0.965 | 0.733 | 208 |
| A09 | 52.5 | 0.444 | 0.475 | 0.573 | 0.050 | 208 |
| **mean ± std** | 70.5 ± 16.2 | 0.741 ± 0.182 | 0.697 ± 0.170 | 0.753 ± 0.162 | 0.409 ± 0.324 | |

## 5. Int8 minus float accuracy (points)

| Student | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| wide32 | -1.7 | +1.7 | +0.8 | -0.8 | +0.8 | -1.7 | +3.3 | +0.0 | +5.0 | +0.8 |
| control | +0.0 | +0.0 | -0.8 | +0.8 | +0.0 | +0.8 | -0.8 | +0.0 | -0.8 | -0.1 |

## 6. Training facts

Trials: train 13,120, validation 1,400, 2a calibration 216 (unlabelled), 2a test 1,080. Data meta: `{"n_train_subject_sessions": 92, "sources": {"Cho2017": {"subjects": 52, "subject_sessions": 52, "train": 9520, "val": 1000, "skipped": 0}, "Lee2019_MI": {"subjects": 20, "subject_sessions": 40, "train": 3600, "val": 400, "skipped": 0}}, "skipped": [], "fs": 125, "window": 256, "train_window": 306, "crop_offset": 25, "window_start": 62, "ea_eps": 0.001, "seed": 0, "quality": {"config": {"ptp_uv": 100.0, "potato_z": 2.5, "potato_iterations": 2, "flat_uv": 0.1, "max_flagged_fraction": 0.2}, "dropped": [], "datasets": {"Cho2017": {"sessions": 52, "dropped": 0, "trials": 10520, "counts": {"ptp": 4, "potato": 516, "flat": 0, "any": 516}, "flag_rates": {"ptp": 0.00038022813688212925, "potato": 0.04904942965779468, "flat": 0.0, "any": 0.04904942965779468}}, "Lee2019_MI": {"sessions": 40, "dropped": 0, "trials": 4000, "counts": {"ptp": 8, "potato": 222, "flat": 0, "any": 222}, "flag_rates": {"ptp": 0.002, "potato": 0.0555, "flat": 0.0, "any": 0.0555}}}, "calib_2a": {"1": {"ptp": 0, "potato": 1, "flat": 0, "any": 1}, "2": {"ptp": 0, "potato": 0, "flat": 0, "any": 0}, "3": {"ptp": 0, "potato": 0, "flat": 0, "any": 0}, "4": {"ptp": 0, "potato": 0, "flat": 0, "any": 0}, "5": {"ptp": 0, "potato": 2, "flat": 0, "any": 2}, "6": {"ptp": 0, "potato": 4, "flat": 0, "any": 4}, "7": {"ptp": 0, "potato": 0, "flat": 0, "any": 0}, "8": {"ptp": 0, "potato": 2, "flat": 0, "any": 2}, "9": {"ptp": 0, "potato": 1, "flat": 0, "any": 1}}}, "validation": {"mode": "cross_subject", "val_subjects": {"Cho2017": [48, 49, 50, 51, 52], "Lee2019_MI": [19, 20]}, "val_sessions": ["Cho2017_s48_1", "Cho2017_s49_1", "Cho2017_s50_1", "Cho2017_s51_1", "Cho2017_s52_1", "Lee2019_MI_s19_1", "Lee2019_MI_s19_2", "Lee2019_MI_s20_1", "Lee2019_MI_s20_2"]}}`.

Teachers stop and select on the validation cross-entropy, students on the validation KL to the teacher ensemble (the combined distillation loss at that epoch in brackets).

| Run | Seed | Best epochs (per model) | Best val CE (teachers) / val KL (students) | Wall (s) |
|---|---|---|---|---|
| teacher | 0 | EEGNet-ELU 263, ShallowConvNet 25, DeepConvNet 224 | 0.5108, 0.5179, 0.5055 | 31 |
| student control | 0 | 58 | 0.0010 (0.1700) | 287 |
| student wide32 | 0 | 208 | 0.0006 (0.1604) | 833 |

Confusion matrices per subject (rows = true left/right, columns = predicted, summed over seeds): `results/confusion_v3/A01.md` … `A09.md`.

## 7. Signal-quality gate (training sources)

Per trial: peak |x − mean| > 100.0 µV (`ptp`), whitened-covariance distance z > 2.5 within the subject-session (`potato`, 2 iterations), channel std < 0.1 µV (`flat`). A subject-session is dropped when more than 0.2 of its trials are flagged or a channel is flat over the whole session; kept sessions keep all trials.

| Dataset | Sessions | Dropped | Trials | ptp | potato | flat | any (rate) |
|---|---|---|---|---|---|---|---|
| Cho2017 | 52 | 0 | 10,520 | 4 | 516 | 0 | 516 (4.9 %) |
| Lee2019_MI | 40 | 0 | 4,000 | 8 | 222 | 0 | 222 (5.5 %) |

Dropped sessions: none.

BCI IV 2a calibration trials (session 2 trials 1–24, logged only, never gated), flagged per subject (any): A01 1, A02 0, A03 0, A04 0, A05 2, A06 4, A07 0, A08 2, A09 1.

## 8. Cross-subject validation accuracy

Validation subjects `{"Cho2017": [48, 49, 50, 51, 52], "Lee2019_MI": [19, 20]}` (9 subject-sessions); value at the selected epoch (EMA weights).

| Model | Seed | Val accuracy (%) | Val loss / KL | Best epoch | Epochs run |
|---|---|---|---|---|---|
| EEGNet-ELU | 0 | 75.2 | 0.5108 | 263 | 300 |
| ShallowConvNet | 0 | 74.0 | 0.5179 | 25 | 66 |
| DeepConvNet | 0 | 75.0 | 0.5055 | 224 | 265 |
| Teacher ensemble | 0 | 76.6 | 0.5021 | | |
| Student control | 0 | 75.2 | KL 0.0010 | 58 | 89 |
| Student wide32 | 0 | 76.8 | KL 0.0006 | 208 | 239 |

## 9. Change against `v2` (seeds [0, 1, 2]), accuracy points

Positive = this run is better. Reference = the comparison run's per-subject mean over its seeds.

| Model | Precision | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | Mean | Ref mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EEGNet-ELU | float | -4.7 | -0.8 | -0.3 | +0.0 | -1.7 | -2.5 | -5.8 | +0.0 | -3.3 | -2.1 | 72.9 |
| ShallowConvNet | float | -0.3 | +1.4 | -0.8 | -6.7 | +5.3 | -1.7 | -7.2 | -9.2 | -7.8 | -3.0 | 73.2 |
| DeepConvNet | float | -1.4 | +0.0 | -2.8 | +1.4 | -2.5 | -3.1 | -7.2 | -4.4 | +2.8 | -1.9 | 72.8 |
| Teacher ensemble (mean softmax) | float | -1.9 | +3.1 | +0.0 | -3.3 | +2.2 | +0.6 | -4.4 | -3.9 | -2.2 | -1.1 | 74.4 |
| Student wide32 | float | -0.3 | -5.8 | -0.8 | -1.4 | -1.9 | +0.8 | -8.3 | -7.2 | -5.0 | -3.3 | 73.8 |
| Student wide32 | int8 | -2.2 | -3.6 | +0.0 | -1.7 | +0.3 | -1.7 | -4.2 | -7.2 | +0.8 | -2.2 | 73.5 |
| Student control | float | -0.8 | -2.5 | +0.3 | +0.0 | -6.7 | -9.4 | -2.5 | -8.6 | -0.3 | -3.4 | 73.6 |
| Student control | int8 | -0.6 | -1.9 | -0.6 | +0.8 | -5.0 | -9.2 | -3.1 | -8.3 | -0.8 | -3.2 | 73.3 |
