# Reproduce_Baselines v2 results: teacher ensemble distilled into TinyEEGNet at 125 Hz

## 1. Setup

Training data: Cho2017 + Lee2019_MI (left vs right hand), 22 channels in the BCI IV 2a order, common average reference over the 22, 125 Hz, causal Butterworth 7–30 Hz, Euclidean Alignment per subject-session, one global `input_scale`. Training windows 306 samples with random 256-sample crops for the students; teachers train on the canonical crop at offset 25 (cue + 0.496 s to cue + 2.544 s). Validation = stratified 10 % of every training subject-session (early stopping only). **Test: BCI IV 2a (MOABB `BNCI2014_001`) subjects 1–9, session 2 trials 25–144 (120 per subject), never trained on.** Session 2 trials 1–24 of every test subject give its EA matrix and enter the distillation stream with teacher soft labels only (their labels are never used). Seeds [0, 1, 2]; each subject's value is the mean over seeds, ± is the std across subjects. KPIs: accuracy, ROC-AUC, macro F1, AUPRC (positive class = right hand), Cohen's kappa.

Distillation: loss = 0.3 CE + 0.7 T² KL, T = 4.0, noise std 0.05, channel dropout 0.1; the teacher scores each augmented batch on the fly.

Environment: Python 3.12.14, torch 2.14.0, moabb 1.7.2, mne 1.13.2, scikit-learn 1.9.1, scipy 1.18.1, numpy 2.5.4; git commit `c96b6b6`; generated 2026-10-17 01:13.

## 2. Summary (mean ± std across the 9 2a subjects)

| Model | Precision | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa | Params | MACs | n subj | n seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| EEGNet-ELU | float | 72.9 ± 15.3 | 0.773 ± 0.174 | 0.719 ± 0.163 | 0.783 ± 0.153 | 0.457 ± 0.306 | 1,714 | 3.02 M | 9 | 3 |
| ShallowConvNet | float | 73.2 ± 14.7 | 0.780 ± 0.158 | 0.727 ± 0.151 | 0.791 ± 0.147 | 0.464 ± 0.295 | 38,242 | 11.38 M | 9 | 3 |
| DeepConvNet | float | 72.8 ± 14.4 | 0.772 ± 0.162 | 0.723 ± 0.149 | 0.783 ± 0.141 | 0.457 ± 0.288 | 146,302 | 5.51 M | 9 | 3 |
| Teacher ensemble (mean softmax) | float | 74.4 ± 14.9 | 0.786 ± 0.162 | 0.737 ± 0.155 | 0.800 ± 0.141 | 0.487 ± 0.298 | 186,258 | 19.91 M | 9 | 3 |
| Student wide32 | float | 73.8 ± 15.5 | 0.778 ± 0.167 | 0.730 ± 0.163 | 0.790 ± 0.146 | 0.476 ± 0.310 | 3,426 | 3.18 M | 9 | 3 |
| Student wide32 | int8 | 73.5 ± 15.7 | 0.779 ± 0.165 | 0.726 ± 0.166 | 0.791 ± 0.145 | 0.469 ± 0.314 | 3,426 | 3.18 M | 9 | 3 |
| Student control | float | 73.6 ± 15.5 | 0.769 ± 0.177 | 0.728 ± 0.164 | 0.779 ± 0.157 | 0.472 ± 0.310 | 1,458 | 1.57 M | 9 | 3 |
| Student control | int8 | 73.3 ± 15.5 | 0.769 ± 0.176 | 0.724 ± 0.165 | 0.778 ± 0.157 | 0.465 ± 0.310 | 1,458 | 1.57 M | 9 | 3 |
| *v0.1 TinyEEGNet+EA, LOSO on 2a, 250 Hz (different protocol)* | float | 71.5 ± 13.7 | 0.776 ± 0.142 | 0.708 ± 0.141 | 0.780 ± 0.141 | 0.429 ± 0.273 | 1,938 | 5.88 M | 9 | 3 |
| *v0.1 TinyEEGNet+EA, LOSO on 2a, 250 Hz (different protocol)* | int8 | 71.5 ± 13.5 | 0.778 ± 0.141 | 0.708 ± 0.140 | 0.782 ± 0.140 | 0.429 ± 0.270 | 1,938 | 5.88 M | 9 | 3 |

The v0.1 row trains on the other 8 BCI IV 2a subjects (leave-one-subject-out) at 250 Hz with 500-sample windows; it is shown for context only (`results/results.md`).

MACs = multiply-accumulates of the conv and linear layers for one window; params include BatchNorm. Ensemble params/MACs are the sum of its three members (offline only).

## 3. Per-subject accuracy (%)

| Model | Precision | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EEGNet-ELU | float | 90.6 | 61.7 | 98.6 | 73.3 | 67.5 | 57.5 | 61.7 | 90.0 | 55.0 | 72.9 |
| ShallowConvNet | float | 88.6 | 65.3 | 97.5 | 70.8 | 59.7 | 60.0 | 65.6 | 93.3 | 57.8 | 73.2 |
| DeepConvNet | float | 87.2 | 64.2 | 98.6 | 71.9 | 61.7 | 54.7 | 68.1 | 89.4 | 59.7 | 72.8 |
| Teacher ensemble (mean softmax) | float | 90.3 | 65.3 | 99.2 | 75.0 | 65.3 | 55.3 | 67.8 | 92.2 | 58.9 | 74.4 |
| Student wide32 | float | 89.4 | 63.3 | 99.2 | 74.7 | 67.8 | 55.0 | 63.3 | 93.9 | 57.5 | 73.8 |
| Student wide32 | int8 | 89.7 | 62.8 | 99.2 | 74.2 | 66.4 | 55.8 | 62.5 | 93.9 | 56.7 | 73.5 |
| Student control | float | 89.2 | 61.7 | 98.9 | 72.5 | 69.2 | 57.8 | 62.5 | 94.4 | 56.1 | 73.6 |
| Student control | int8 | 88.9 | 61.1 | 98.9 | 72.5 | 67.5 | 58.3 | 62.2 | 94.2 | 55.8 | 73.3 |
| *v0.1 TinyEEGNet+EA, LOSO on 2a, 250 Hz (different protocol)* | float | 85.3 | 53.1 | 92.2 | 67.8 | 58.1 | 63.9 | 66.9 | 91.4 | 64.4 | 71.5 |
| *v0.1 TinyEEGNet+EA, LOSO on 2a, 250 Hz (different protocol)* | int8 | 85.0 | 53.9 | 92.5 | 67.2 | 58.1 | 64.4 | 66.7 | 91.1 | 64.2 | 71.5 |

v0.1 rows: different protocol (LOSO on BCI IV 2a, 250 Hz, 500 samples), context only.

## 4. Shipped student

Selected: **student wide32** (highest mean int8 accuracy across subjects, tie-break on MACs). Export subject A03, seed 0 (lowest best validation KL to the teacher: 0.0010), checkpoint `runs/v2/student_wide32_seed0/model.pt`.

Per-subject KPIs, float (mean ± std over seeds):

| Subject | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa | best epochs |
|---|---|---|---|---|---|---|
| A01 | 89.4 ± 1.4 | 0.964 ± 0.003 | 0.894 ± 0.015 | 0.963 ± 0.004 | 0.789 ± 0.028 | 58, 75, 56 |
| A02 | 63.3 ± 1.8 | 0.703 ± 0.014 | 0.625 ± 0.020 | 0.679 ± 0.019 | 0.267 ± 0.036 | 58, 75, 56 |
| A03 (export) | 99.2 ± 0.0 | 1.000 ± 0.000 | 0.992 ± 0.000 | 1.000 ± 0.000 | 0.983 ± 0.000 | 58, 75, 56 |
| A04 | 74.7 ± 1.4 | 0.797 ± 0.004 | 0.744 ± 0.015 | 0.799 ± 0.003 | 0.494 ± 0.028 | 58, 75, 56 |
| A05 | 67.8 ± 1.0 | 0.728 ± 0.004 | 0.676 ± 0.011 | 0.704 ± 0.017 | 0.356 ± 0.021 | 58, 75, 56 |
| A06 | 55.0 ± 2.5 | 0.615 ± 0.010 | 0.549 ± 0.024 | 0.634 ± 0.011 | 0.100 ± 0.049 | 58, 75, 56 |
| A07 | 63.3 ± 0.7 | 0.711 ± 0.009 | 0.626 ± 0.007 | 0.730 ± 0.012 | 0.267 ± 0.014 | 58, 75, 56 |
| A08 | 93.9 ± 0.8 | 0.990 ± 0.003 | 0.939 ± 0.008 | 0.990 ± 0.003 | 0.878 ± 0.016 | 58, 75, 56 |
| A09 | 57.5 ± 1.4 | 0.492 ± 0.033 | 0.523 ± 0.016 | 0.614 ± 0.030 | 0.150 ± 0.027 | 58, 75, 56 |
| **mean ± std** | 73.8 ± 15.5 | 0.778 ± 0.167 | 0.730 ± 0.163 | 0.790 ± 0.146 | 0.476 ± 0.310 | |

## 5. Int8 minus float accuracy (points)

| Student | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| wide32 | +0.3 | -0.6 | +0.0 | -0.6 | -1.4 | +0.8 | -0.8 | +0.0 | -0.8 | -0.3 |
| control | -0.3 | -0.6 | +0.0 | +0.0 | -1.7 | +0.6 | -0.3 | -0.3 | -0.3 | -0.3 |

## 6. Training facts

Trials: train 13,068, validation 1,452, 2a calibration 216 (unlabelled), 2a test 1,080. Data meta: `{"n_train_subject_sessions": 92, "sources": {"Cho2017": {"subjects": 52, "subject_sessions": 52, "train": 9468, "val": 1052, "skipped": 0}, "Lee2019_MI": {"subjects": 20, "subject_sessions": 40, "train": 3600, "val": 400, "skipped": 0}}, "skipped": [], "fs": 125, "window": 256, "train_window": 306, "crop_offset": 25, "window_start": 62, "ea_eps": 0.001, "seed": 0}`.

Teachers stop and select on the validation cross-entropy, students on the validation KL to the teacher ensemble (the combined distillation loss at that epoch in brackets).

| Run | Seed | Best epochs (per model) | Best val CE (teachers) / val KL (students) | Wall (s) |
|---|---|---|---|---|
| teacher | 0 | EEGNet-ELU 57, ShallowConvNet 1, DeepConvNet 46 | 0.5157, 0.5415, 0.5286 | 499 |
| teacher | 1 | EEGNet-ELU 75, ShallowConvNet 2, DeepConvNet 38 | 0.5103, 0.5329, 0.5179 | 528 |
| teacher | 2 | EEGNet-ELU 35, ShallowConvNet 2, DeepConvNet 76 | 0.5193, 0.5434, 0.5294 | 502 |
| student control | 0 | 99 | 0.0012 (0.1665) | 543 |
| student control | 1 | 63 | 0.0016 (0.1706) | 392 |
| student control | 2 | 34 | 0.0017 (0.1762) | 271 |
| student wide32 | 0 | 58 | 0.0010 (0.1631) | 366 |
| student wide32 | 1 | 75 | 0.0013 (0.1659) | 433 |
| student wide32 | 2 | 56 | 0.0013 (0.1692) | 357 |

Confusion matrices per subject (rows = true left/right, columns = predicted, summed over seeds): `results/confusion_v2/A01.md` … `A09.md`.
