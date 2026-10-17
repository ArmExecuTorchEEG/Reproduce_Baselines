# Reproduce_Baselines v4 results: teacher ensemble distilled into TinyEEGNet at 125 Hz

## 1. Setup

Training data: Cho2017 + Lee2019_MI (left vs right hand), 22 channels in the BCI IV 2a order, common average reference over the 22, 125 Hz, causal Butterworth 7–30 Hz, Euclidean Alignment per subject-session, one global `input_scale`. Training windows 306 samples with random 256-sample crops for the students; teachers train on the canonical crop at offset 25 (cue + 0.496 s to cue + 2.544 s). Validation = stratified 10 % of every training subject-session (early stopping only). **Test: BCI IV 2a (MOABB `BNCI2014_001`) subjects 1–9, session 2 trials 25–144 (120 per subject), never trained on.** Session 2 trials 1–24 of every test subject give its EA matrix and enter the distillation stream with teacher soft labels only (their labels are never used). Seeds [0]; each subject's value is the mean over seeds, ± is the std across subjects. KPIs: accuracy, ROC-AUC, macro F1, AUPRC (positive class = right hand), Cohen's kappa.

Distillation: loss = 0.3 CE + 0.7 T² KL, T = 4.0, noise std 0.05, channel dropout 0.1; the teacher scores each augmented batch on the fly.

Environment: Python 3.12.14, torch 2.14.0, moabb 1.7.2, mne 1.13.2, scikit-learn 1.9.1, scipy 1.18.1, numpy 2.5.4; git commit `01e6471`; generated 2026-10-17 13:30.

## 2. Summary (mean ± std across the 9 2a subjects)

| Model | Precision | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa | Params | MACs | n subj | n seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| EEGNet-ELU | float | 74.4 ± 15.3 | 0.775 ± 0.175 | 0.737 ± 0.160 | 0.779 ± 0.159 | 0.489 ± 0.306 | 1,714 | 3.02 M | 9 | 1 |
| ShallowConvNet | float | 74.3 ± 13.7 | 0.777 ± 0.162 | 0.738 ± 0.140 | 0.789 ± 0.147 | 0.485 ± 0.274 | 38,242 | 11.38 M | 9 | 1 |
| DeepConvNet | float | 73.5 ± 14.1 | 0.772 ± 0.176 | 0.726 ± 0.151 | 0.785 ± 0.151 | 0.470 ± 0.282 | 146,302 | 5.51 M | 9 | 1 |
| Riemann TS + LR | float | 73.1 ± 16.0 | 0.793 ± 0.154 | 0.722 ± 0.169 | 0.786 ± 0.164 | 0.463 ± 0.321 | 254 | 0.00 M | 9 | 1 |
| Teacher ensemble (mean softmax) | float | 74.7 ± 15.0 | 0.794 ± 0.160 | 0.739 ± 0.158 | 0.796 ± 0.150 | 0.494 ± 0.300 | 186,512 | 19.91 M | 9 | 1 |
| Student wide32 | float | 73.9 ± 15.4 | 0.777 ± 0.173 | 0.731 ± 0.161 | 0.785 ± 0.154 | 0.478 ± 0.307 | 3,426 | 3.18 M | 9 | 1 |
| Student wide32 | int8 | 73.8 ± 15.4 | 0.776 ± 0.173 | 0.730 ± 0.162 | 0.784 ± 0.155 | 0.476 ± 0.308 | 3,426 | 3.18 M | 9 | 1 |
| Student control | float | 73.5 ± 15.8 | 0.780 ± 0.171 | 0.727 ± 0.166 | 0.786 ± 0.154 | 0.470 ± 0.315 | 1,458 | 1.57 M | 9 | 1 |
| Student control | int8 | 73.5 ± 15.6 | 0.781 ± 0.169 | 0.727 ± 0.164 | 0.786 ± 0.152 | 0.470 ± 0.311 | 1,458 | 1.57 M | 9 | 1 |
| Student wide32_r0 | float | 73.6 ± 15.0 | 0.780 ± 0.167 | 0.729 ± 0.157 | 0.788 ± 0.149 | 0.472 ± 0.301 | 3,426 | 3.18 M | 9 | 1 |
| Student wide32_r0 | int8 | 73.6 ± 14.9 | 0.780 ± 0.166 | 0.730 ± 0.155 | 0.787 ± 0.149 | 0.472 ± 0.298 | 3,426 | 3.18 M | 9 | 1 |
| *v0.1 TinyEEGNet+EA, LOSO on 2a, 250 Hz (different protocol)* | float | 71.5 ± 13.7 | 0.776 ± 0.142 | 0.708 ± 0.141 | 0.780 ± 0.141 | 0.429 ± 0.273 | 1,938 | 5.88 M | 9 | 3 |
| *v0.1 TinyEEGNet+EA, LOSO on 2a, 250 Hz (different protocol)* | int8 | 71.5 ± 13.5 | 0.778 ± 0.141 | 0.708 ± 0.140 | 0.782 ± 0.140 | 0.429 ± 0.270 | 1,938 | 5.88 M | 9 | 3 |

The v0.1 row trains on the other 8 BCI IV 2a subjects (leave-one-subject-out) at 250 Hz with 500-sample windows; it is shown for context only (`results/results.md`).

MACs = multiply-accumulates of the conv and linear layers for one window; params include BatchNorm. Ensemble params/MACs are the sum of its 4 members (offline only); the Riemannian member's params are its logistic-regression coefficients and its MACs are not counted (covariance + matrix logarithm).

## 3. Per-subject accuracy (%)

| Model | Precision | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EEGNet-ELU | float | 91.7 | 60.0 | 99.2 | 77.5 | 69.2 | 58.3 | 61.7 | 92.5 | 60.0 | 74.4 |
| ShallowConvNet | float | 86.7 | 66.7 | 100.0 | 73.3 | 63.3 | 60.8 | 63.3 | 90.8 | 63.3 | 74.3 |
| DeepConvNet | float | 87.5 | 65.0 | 99.2 | 71.7 | 65.8 | 55.0 | 70.8 | 88.3 | 58.3 | 73.5 |
| Riemann TS + LR | float | 81.7 | 55.0 | 99.2 | 78.3 | 61.7 | 55.8 | 71.7 | 96.7 | 58.3 | 73.1 |
| Teacher ensemble (mean softmax) | float | 89.2 | 62.5 | 99.2 | 77.5 | 68.3 | 56.7 | 65.0 | 94.2 | 60.0 | 74.7 |
| Student wide32 | float | 88.3 | 60.8 | 99.2 | 75.0 | 70.0 | 55.8 | 63.3 | 94.2 | 58.3 | 73.9 |
| Student wide32 | int8 | 88.3 | 60.8 | 99.2 | 74.2 | 68.3 | 57.5 | 62.5 | 95.0 | 58.3 | 73.8 |
| Student control | float | 89.2 | 60.8 | 97.5 | 77.5 | 67.5 | 54.2 | 65.8 | 94.2 | 55.0 | 73.5 |
| Student control | int8 | 89.2 | 60.8 | 97.5 | 76.7 | 66.7 | 55.8 | 65.8 | 94.2 | 55.0 | 73.5 |
| Student wide32_r0 | float | 89.2 | 60.8 | 98.3 | 73.3 | 68.3 | 57.5 | 63.3 | 93.3 | 58.3 | 73.6 |
| Student wide32_r0 | int8 | 88.3 | 60.0 | 98.3 | 73.3 | 68.3 | 57.5 | 65.0 | 93.3 | 58.3 | 73.6 |
| *v0.1 TinyEEGNet+EA, LOSO on 2a, 250 Hz (different protocol)* | float | 85.3 | 53.1 | 92.2 | 67.8 | 58.1 | 63.9 | 66.9 | 91.4 | 64.4 | 71.5 |
| *v0.1 TinyEEGNet+EA, LOSO on 2a, 250 Hz (different protocol)* | int8 | 85.0 | 53.9 | 92.5 | 67.2 | 58.1 | 64.4 | 66.7 | 91.1 | 64.2 | 71.5 |

v0.1 rows: different protocol (LOSO on BCI IV 2a, 250 Hz, 500 samples), context only.

## 4. Variant comparison on the test block (exploratory)

Variant comparison on the test block (exploratory; the shipped variant was chosen by the team from this table, which makes its test estimate optimistic by selection among 5 candidates); the seed within the variant is chosen by validation KL. Highest mean int8 test accuracy in this run: **student wide32**; its lowest-validation-KL seed is 0 (0.0007), checkpoint `runs/v4/student_wide32_seed0/model.pt`, export subject A03. See the selection note.

Per-subject KPIs, float (mean ± std over seeds):

| Subject | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa | best epochs |
|---|---|---|---|---|---|---|
| A01 | 88.3 | 0.965 | 0.883 | 0.964 | 0.767 | 28 |
| A02 | 60.8 | 0.669 | 0.596 | 0.642 | 0.217 | 28 |
| A03 (export) | 99.2 | 1.000 | 0.992 | 1.000 | 0.983 | 28 |
| A04 | 75.0 | 0.821 | 0.747 | 0.819 | 0.500 | 28 |
| A05 | 70.0 | 0.744 | 0.699 | 0.695 | 0.400 | 28 |
| A06 | 55.8 | 0.620 | 0.556 | 0.621 | 0.117 | 28 |
| A07 | 63.3 | 0.717 | 0.628 | 0.736 | 0.267 | 28 |
| A08 | 94.2 | 0.992 | 0.942 | 0.992 | 0.883 | 28 |
| A09 | 58.3 | 0.468 | 0.537 | 0.599 | 0.167 | 28 |
| **mean ± std** | 73.9 ± 15.4 | 0.777 ± 0.173 | 0.731 ± 0.161 | 0.785 ± 0.154 | 0.478 ± 0.307 | |

## 5. Int8 minus float accuracy (points)

| Student | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| wide32 | +0.0 | +0.0 | +0.0 | -0.8 | -1.7 | +1.7 | -0.8 | +0.8 | +0.0 | -0.1 |
| control | +0.0 | +0.0 | +0.0 | -0.8 | -0.8 | +1.7 | +0.0 | +0.0 | +0.0 | +0.0 |
| wide32_r0 | -0.8 | -0.8 | +0.0 | +0.0 | +0.0 | +0.0 | +1.7 | +0.0 | +0.0 | +0.0 |

## 6. Training facts

Trials: train 13,068, validation 1,452, 2a calibration 216 (unlabelled), 2a test 1,080. Data meta: `{"n_train_subject_sessions": 92, "sources": {"Cho2017": {"subjects": 52, "subject_sessions": 52, "train": 9468, "val": 1052, "skipped": 0}, "Lee2019_MI": {"subjects": 20, "subject_sessions": 40, "train": 3600, "val": 400, "skipped": 0}}, "skipped": [], "fs": 125, "window": 256, "train_window": 306, "crop_offset": 25, "window_start": 62, "ea_eps": 0.001, "seed": 0}`.

Teachers stop and select on the validation cross-entropy, students on the validation KL to the teacher ensemble (the combined distillation loss at that epoch in brackets).

| Run | Seed | Best epochs (per model) | Best val CE (teachers) / val KL (students) | Wall (s) |
|---|---|---|---|---|
| teacher | 0 | EEGNet-ELU 31, ShallowConvNet 2, DeepConvNet 30, Riemann TS + LR 0 | 0.5217, 0.5331, 0.5262, 0.5774 | 39 |
| student control | 0 | 57 | 0.0009 (0.1676) | 452 |
| student wide32 | 0 | 28 | 0.0007 (0.1635) | 262 |
| student wide32_r0 | 0 | 29 | 0.0008 (0.1638) | 311 |

Confusion matrices per subject (rows = true left/right, columns = predicted, summed over seeds): `results/confusion_v4/A01.md` … `A09.md`.

## 7. Self-training on the unlabelled target session

Round 0: the 4-member ensemble (EEGNet-ELU, ShallowConvNet, DeepConvNet, Riemann TS + LR) is trained on the training split only and scores BNCI2014_001 session 1 (left/right trials of every test subject, EA from that session's own trials; its labels are never read for training). Windows with ensemble max probability ≥ 0.8 get the argmax as a hard label (weight 1) and round 1 retrains every member from scratch on train + pseudo-labelled windows (same recipe and validation split). Agreement columns compare with the true session-1 labels and are computed afterwards for this report only. The session-2 test block (trials 25–144) is not involved.

Seed 0:

| Subject | Trials | Kept | Kept (%) | Pseudo left / right | Mean conf | Agreement of kept (%) | Round-0 ensemble on all trials (%) |
|---|---|---|---|---|---|---|---|
| A01 | 144 | 65 | 45.1 | 35 / 30 | 0.921 | 100.0 | 91.7 |
| A02 | 144 | 19 | 13.2 | 13 / 6 | 0.851 | 84.2 | 61.8 |
| A03 | 144 | 114 | 79.2 | 65 / 49 | 0.937 | 100.0 | 97.2 |
| A04 | 144 | 41 | 28.5 | 14 / 27 | 0.876 | 78.0 | 73.6 |
| A05 | 144 | 13 | 9.0 | 8 / 5 | 0.844 | 84.6 | 68.1 |
| A06 | 144 | 48 | 33.3 | 22 / 26 | 0.882 | 85.4 | 66.7 |
| A07 | 144 | 11 | 7.6 | 5 / 6 | 0.851 | 100.0 | 70.8 |
| A08 | 144 | 97 | 67.4 | 51 / 46 | 0.901 | 96.9 | 89.6 |
| A09 | 144 | 72 | 50.0 | 37 / 35 | 0.903 | 94.4 | 80.6 |
| **all** | 1296 | 480 | 37.0 | | | 94.2 | 77.8 |

Round-1 training set: 13,548 windows (13,068 labelled + 480 pseudo-labelled).

## 8. Teachers before and after self-training (2a test block, accuracy %)

| Model | Round | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | Mean | ROC-AUC | Kappa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EEGNet-ELU | 0 (before) | 89.2 | 64.2 | 98.3 | 73.3 | 70.0 | 57.5 | 59.2 | 93.3 | 52.5 | 73.1 | 0.766 | 0.461 |
| EEGNet-ELU | 1 (after) | 91.7 | 60.0 | 99.2 | 77.5 | 69.2 | 58.3 | 61.7 | 92.5 | 60.0 | 74.4 | 0.775 | 0.489 |
| ShallowConvNet | 0 (before) | 85.8 | 65.8 | 100.0 | 71.7 | 58.3 | 60.0 | 67.5 | 89.2 | 55.8 | 72.7 | 0.781 | 0.454 |
| ShallowConvNet | 1 (after) | 86.7 | 66.7 | 100.0 | 73.3 | 63.3 | 60.8 | 63.3 | 90.8 | 63.3 | 74.3 | 0.777 | 0.485 |
| DeepConvNet | 0 (before) | 86.7 | 64.2 | 99.2 | 70.0 | 62.5 | 55.8 | 66.7 | 89.2 | 60.0 | 72.7 | 0.774 | 0.454 |
| DeepConvNet | 1 (after) | 87.5 | 65.0 | 99.2 | 71.7 | 65.8 | 55.0 | 70.8 | 88.3 | 58.3 | 73.5 | 0.772 | 0.470 |
| Riemann TS + LR | 0 (before) | 81.7 | 55.0 | 98.3 | 77.5 | 64.2 | 55.8 | 69.2 | 96.7 | 58.3 | 73.0 | 0.792 | 0.459 |
| Riemann TS + LR | 1 (after) | 81.7 | 55.0 | 99.2 | 78.3 | 61.7 | 55.8 | 71.7 | 96.7 | 58.3 | 73.1 | 0.793 | 0.463 |
| Teacher ensemble | 0 (before) | 87.5 | 66.7 | 99.2 | 77.5 | 69.2 | 57.5 | 68.3 | 93.3 | 58.3 | 75.3 | 0.792 | 0.506 |
| Teacher ensemble | 1 (after) | 89.2 | 62.5 | 99.2 | 77.5 | 68.3 | 56.7 | 65.0 | 94.2 | 60.0 | 74.7 | 0.794 | 0.494 |

## 9. Per-member KPIs per subject (final teachers, float)

**EEGNet-ELU**

| Subject | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa |
|---|---|---|---|---|---|
| A01 | 91.7 | 0.968 | 0.916 | 0.966 | 0.833 |
| A02 | 60.0 | 0.666 | 0.586 | 0.612 | 0.200 |
| A03 | 99.2 | 1.000 | 0.992 | 1.000 | 0.983 |
| A04 | 77.5 | 0.814 | 0.773 | 0.817 | 0.550 |
| A05 | 69.2 | 0.741 | 0.691 | 0.697 | 0.383 |
| A06 | 58.3 | 0.628 | 0.579 | 0.636 | 0.167 |
| A07 | 61.7 | 0.713 | 0.610 | 0.708 | 0.233 |
| A08 | 92.5 | 0.989 | 0.925 | 0.989 | 0.850 |
| A09 | 60.0 | 0.456 | 0.560 | 0.586 | 0.200 |
| **mean** | 74.4 | 0.775 | 0.737 | 0.779 | 0.489 |

**ShallowConvNet**

| Subject | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa |
|---|---|---|---|---|---|
| A01 | 86.7 | 0.949 | 0.867 | 0.952 | 0.733 |
| A02 | 66.7 | 0.709 | 0.662 | 0.701 | 0.333 |
| A03 | 100.0 | 1.000 | 1.000 | 1.000 | 1.000 |
| A04 | 73.3 | 0.837 | 0.730 | 0.851 | 0.467 |
| A05 | 63.3 | 0.657 | 0.633 | 0.636 | 0.267 |
| A06 | 60.8 | 0.645 | 0.604 | 0.650 | 0.217 |
| A07 | 63.3 | 0.709 | 0.632 | 0.692 | 0.267 |
| A08 | 90.8 | 0.982 | 0.908 | 0.983 | 0.817 |
| A09 | 63.3 | 0.511 | 0.609 | 0.640 | 0.267 |
| **mean** | 74.3 | 0.777 | 0.738 | 0.789 | 0.485 |

**DeepConvNet**

| Subject | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa |
|---|---|---|---|---|---|
| A01 | 87.5 | 0.960 | 0.875 | 0.959 | 0.750 |
| A02 | 65.0 | 0.700 | 0.640 | 0.673 | 0.300 |
| A03 | 99.2 | 1.000 | 0.992 | 1.000 | 0.983 |
| A04 | 71.7 | 0.796 | 0.714 | 0.800 | 0.433 |
| A05 | 65.8 | 0.675 | 0.657 | 0.674 | 0.317 |
| A06 | 55.0 | 0.602 | 0.550 | 0.615 | 0.100 |
| A07 | 70.8 | 0.781 | 0.705 | 0.782 | 0.417 |
| A08 | 88.3 | 0.980 | 0.883 | 0.977 | 0.767 |
| A09 | 58.3 | 0.450 | 0.519 | 0.589 | 0.167 |
| **mean** | 73.5 | 0.772 | 0.726 | 0.785 | 0.470 |

**Riemann TS + LR**

| Subject | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa |
|---|---|---|---|---|---|
| A01 | 81.7 | 0.957 | 0.810 | 0.960 | 0.633 |
| A02 | 55.0 | 0.599 | 0.540 | 0.550 | 0.100 |
| A03 | 99.2 | 0.998 | 0.992 | 0.998 | 0.983 |
| A04 | 78.3 | 0.859 | 0.783 | 0.858 | 0.567 |
| A05 | 61.7 | 0.674 | 0.617 | 0.644 | 0.233 |
| A06 | 55.8 | 0.586 | 0.556 | 0.584 | 0.117 |
| A07 | 71.7 | 0.754 | 0.717 | 0.737 | 0.433 |
| A08 | 96.7 | 0.991 | 0.967 | 0.991 | 0.933 |
| A09 | 58.3 | 0.722 | 0.519 | 0.747 | 0.167 |
| **mean** | 73.1 | 0.793 | 0.722 | 0.786 | 0.463 |

**Teacher ensemble**

| Subject | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa |
|---|---|---|---|---|---|
| A01 | 89.2 | 0.973 | 0.891 | 0.973 | 0.783 |
| A02 | 62.5 | 0.693 | 0.615 | 0.651 | 0.250 |
| A03 | 99.2 | 1.000 | 0.992 | 1.000 | 0.983 |
| A04 | 77.5 | 0.845 | 0.774 | 0.845 | 0.550 |
| A05 | 68.3 | 0.722 | 0.683 | 0.670 | 0.367 |
| A06 | 56.7 | 0.621 | 0.565 | 0.637 | 0.133 |
| A07 | 65.0 | 0.769 | 0.648 | 0.758 | 0.300 |
| A08 | 94.2 | 0.992 | 0.942 | 0.992 | 0.883 |
| A09 | 60.0 | 0.535 | 0.544 | 0.637 | 0.200 |
| **mean** | 74.7 | 0.794 | 0.739 | 0.796 | 0.494 |

## 10. Per-member learning rate and validation

| Seed | Member | Round | lr | Best epoch | Epochs run | Best val loss | Val acc (%) | lr choice |
|---|---|---|---|---|---|---|---|---|
| 0 | EEGNet-ELU | 0 | 0.001 | 57 | 88 | 0.5157 | 72.2 |  |
| 0 | ShallowConvNet | 0 | 0.0003 | 2 | 33 | 0.5362 | 72.2 | tried 0.0003 (val 0.5362) vs reference 0.5415 → 0.0003 |
| 0 | DeepConvNet | 0 | 0.001 | 46 | 77 | 0.5286 | 72.5 |  |
| 0 | Riemann TS + LR | 0 | — | 0 | 1 | 0.5772 | 69.5 |  |
| 0 | EEGNet-ELU | 1 | 0.001 | 31 | 62 | 0.5217 | 73.0 |  |
| 0 | ShallowConvNet | 1 | 0.0003 | 2 | 33 | 0.5331 | 72.0 |  |
| 0 | DeepConvNet | 1 | 0.001 | 30 | 61 | 0.5262 | 73.3 |  |
| 0 | Riemann TS + LR | 1 | — | 0 | 1 | 0.5774 | 69.1 |  |

## 11. Change against `v2` (seeds [0, 1, 2]), accuracy points

Positive = this run is better. Reference = the comparison run's per-subject mean over its seeds.

| Model | Precision | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | Mean | Ref mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EEGNet-ELU | float | +1.1 | -1.7 | +0.6 | +4.2 | +1.7 | +0.8 | +0.0 | +2.5 | +5.0 | +1.6 | 72.9 |
| ShallowConvNet | float | -1.9 | +1.4 | +2.5 | +2.5 | +3.6 | +0.8 | -2.2 | -2.5 | +5.6 | +1.1 | 73.2 |
| DeepConvNet | float | +0.3 | +0.8 | +0.6 | -0.3 | +4.2 | +0.3 | +2.8 | -1.1 | -1.4 | +0.7 | 72.8 |
| Teacher ensemble (mean softmax) | float | -1.1 | -2.8 | +0.0 | +2.5 | +3.1 | +1.4 | -2.8 | +1.9 | +1.1 | +0.4 | 74.4 |
| Student wide32 | float | -1.1 | -2.5 | +0.0 | +0.3 | +2.2 | +0.8 | +0.0 | +0.3 | +0.8 | +0.1 | 73.8 |
| Student wide32 | int8 | -1.4 | -1.9 | +0.0 | +0.0 | +1.9 | +1.7 | +0.0 | +1.1 | +1.7 | +0.3 | 73.5 |
| Student control | float | +0.0 | -0.8 | -1.4 | +5.0 | -1.7 | -3.6 | +3.3 | -0.3 | -1.1 | -0.1 | 73.6 |
| Student control | int8 | +0.3 | -0.3 | -1.4 | +4.2 | -0.8 | -2.5 | +3.6 | -0.0 | -0.8 | +0.2 | 73.3 |

## 12. Change against `v2` (seeds [0]), accuracy points

Positive = this run is better. Reference = the comparison run's per-subject mean over its seeds.

| Model | Precision | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | Mean | Ref mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EEGNet-ELU | float | +2.5 | -4.2 | +0.8 | +4.2 | -0.8 | +0.8 | +2.5 | -0.8 | +7.5 | +1.4 | 73.1 |
| ShallowConvNet | float | +0.8 | +4.2 | +1.7 | +0.0 | +6.7 | +0.8 | +0.0 | -2.5 | +4.2 | +1.8 | 72.5 |
| DeepConvNet | float | +0.8 | +0.8 | +0.0 | +1.7 | +3.3 | -0.8 | +4.2 | -0.8 | -1.7 | +0.8 | 72.7 |
| Teacher ensemble (mean softmax) | float | +0.0 | -5.8 | -0.8 | +3.3 | +3.3 | +0.8 | -0.8 | +0.8 | +0.8 | +0.2 | 74.5 |
| Student wide32 | float | +0.8 | -0.8 | +0.0 | +1.7 | +2.5 | -1.7 | +0.0 | -0.8 | +0.8 | +0.3 | 73.6 |
| Student wide32 | int8 | +0.8 | -0.8 | +0.0 | +1.7 | +0.8 | -0.8 | +0.0 | +0.8 | +1.7 | +0.5 | 73.3 |
| Student control | float | +0.8 | -0.8 | -0.8 | +4.2 | -2.5 | -1.7 | +5.0 | -0.8 | +0.8 | +0.5 | 73.1 |
| Student control | int8 | +0.8 | -0.8 | -0.8 | +3.3 | -0.8 | -0.8 | +5.0 | -0.8 | +0.8 | +0.6 | 72.9 |

## 14. Selection note

Variant comparison on the test block (exploratory; the shipped variant was chosen by the team from this table, which makes its test estimate optimistic by selection among 5 candidates); the seed within the variant is chosen by validation KL.

Candidates: (1) v2 wide32 (release v0.3-distill); (2) v4 joint (hard pseudo-labels); (3) v4 round-0 teacher student (wide32_r0); (4) v4soft joint (soft targets); (5) v4soft student-ft (v2 wide32 + KD-only fine-tune on 2a session 1 + calibration).

Paired per seed (fine-tuned student vs the v2 student it starts from), int8 accuracy (%) over the 9 subjects:

| Seed | `v4soft/student_wide32_ft` | `v2/student_wide32` | Δ |
|---|---|---|---|
| 0 | 74.4 | 73.3 | +1.1 |
| 1 | 75.5 | 73.9 | +1.6 |
| 2 | 73.8 | 73.1 | +0.6 |
| **mean ± std (seeds)** | 74.6 ± 0.7 | 73.5 ± 0.3 | +1.1 (3/3 seeds better) |
