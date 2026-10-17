# Reproduce_Baselines v4soft results: teacher ensemble distilled into TinyEEGNet at 125 Hz

## 1. Setup

Training data: Cho2017 + Lee2019_MI (left vs right hand), 22 channels in the BCI IV 2a order, common average reference over the 22, 125 Hz, causal Butterworth 7–30 Hz, Euclidean Alignment per subject-session, one global `input_scale`. Training windows 306 samples with random 256-sample crops for the students; teachers train on the canonical crop at offset 25 (cue + 0.496 s to cue + 2.544 s). Validation = stratified 10 % of every training subject-session (early stopping only). **Test: BCI IV 2a (MOABB `BNCI2014_001`) subjects 1–9, session 2 trials 25–144 (120 per subject), never trained on.** Session 2 trials 1–24 of every test subject give its EA matrix and enter the distillation stream with teacher soft labels only (their labels are never used). Seeds [0, 1, 2]; each subject's value is the mean over seeds, ± is the std across subjects. KPIs: accuracy, ROC-AUC, macro F1, AUPRC (positive class = right hand), Cohen's kappa.

Distillation: loss = 0.3 CE + 0.7 T² KL, T = 4.0, noise std 0.05, channel dropout 0.1; the teacher scores each augmented batch on the fly.

Environment: Python 3.12.14, torch 2.14.0, moabb 1.7.2, mne 1.13.2, scikit-learn 1.9.1, scipy 1.18.1, numpy 2.5.4; git commit `01e6471`; generated 2026-10-17 13:30.

## 2. Summary (mean ± std across the 9 2a subjects)

| Model | Precision | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa | Params | MACs | n subj | n seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| EEGNet-ELU | float | 73.8 ± 15.2 | 0.774 ± 0.172 | 0.729 ± 0.161 | 0.783 ± 0.152 | 0.476 ± 0.303 | 1,714 | 3.02 M | 9 | 1 |
| ShallowConvNet | float | 72.9 ± 14.3 | 0.779 ± 0.162 | 0.724 ± 0.148 | 0.792 ± 0.144 | 0.457 ± 0.286 | 38,242 | 11.38 M | 9 | 1 |
| DeepConvNet | float | 72.9 ± 14.0 | 0.774 ± 0.157 | 0.723 ± 0.144 | 0.780 ± 0.142 | 0.457 ± 0.280 | 146,302 | 5.51 M | 9 | 1 |
| Riemann TS + LR | float | 73.1 ± 15.9 | 0.795 ± 0.153 | 0.721 ± 0.168 | 0.788 ± 0.163 | 0.461 ± 0.318 | 254 | 0.00 M | 9 | 1 |
| Teacher ensemble (mean softmax) | float | 74.9 ± 14.3 | 0.795 ± 0.154 | 0.742 ± 0.151 | 0.797 ± 0.145 | 0.498 ± 0.286 | 186,512 | 19.91 M | 9 | 1 |
| Student wide32 | float | 73.8 ± 14.9 | 0.780 ± 0.165 | 0.731 ± 0.155 | 0.788 ± 0.149 | 0.476 ± 0.298 | 3,426 | 3.18 M | 9 | 1 |
| Student wide32 | int8 | 73.6 ± 15.0 | 0.780 ± 0.164 | 0.729 ± 0.156 | 0.787 ± 0.149 | 0.472 ± 0.300 | 3,426 | 3.18 M | 9 | 1 |
| Student control | float | 73.2 ± 16.0 | 0.780 ± 0.169 | 0.724 ± 0.168 | 0.786 ± 0.151 | 0.465 ± 0.319 | 1,458 | 1.57 M | 9 | 1 |
| Student control | int8 | 73.2 ± 15.8 | 0.781 ± 0.168 | 0.724 ± 0.167 | 0.789 ± 0.150 | 0.465 ± 0.316 | 1,458 | 1.57 M | 9 | 1 |
| Student wide32_ft | float | 74.7 ± 14.9 | 0.786 ± 0.164 | 0.741 ± 0.154 | 0.797 ± 0.147 | 0.493 ± 0.297 | 3,426 | 3.18 M | 9 | 3 |
| Student wide32_ft | int8 | 74.6 ± 14.8 | 0.787 ± 0.163 | 0.739 ± 0.155 | 0.796 ± 0.147 | 0.491 ± 0.297 | 3,426 | 3.18 M | 9 | 3 |
| *v0.1 TinyEEGNet+EA, LOSO on 2a, 250 Hz (different protocol)* | float | 71.5 ± 13.7 | 0.776 ± 0.142 | 0.708 ± 0.141 | 0.780 ± 0.141 | 0.429 ± 0.273 | 1,938 | 5.88 M | 9 | 3 |
| *v0.1 TinyEEGNet+EA, LOSO on 2a, 250 Hz (different protocol)* | int8 | 71.5 ± 13.5 | 0.778 ± 0.141 | 0.708 ± 0.140 | 0.782 ± 0.140 | 0.429 ± 0.270 | 1,938 | 5.88 M | 9 | 3 |

The v0.1 row trains on the other 8 BCI IV 2a subjects (leave-one-subject-out) at 250 Hz with 500-sample windows; it is shown for context only (`results/results.md`).

MACs = multiply-accumulates of the conv and linear layers for one window; params include BatchNorm. Ensemble params/MACs are the sum of its 4 members (offline only); the Riemannian member's params are its logistic-regression coefficients and its MACs are not counted (covariance + matrix logarithm).

## 3. Per-subject accuracy (%)

| Model | Precision | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EEGNet-ELU | float | 89.2 | 62.5 | 98.3 | 75.0 | 70.0 | 59.2 | 60.8 | 93.3 | 55.8 | 73.8 |
| ShallowConvNet | float | 86.7 | 64.2 | 99.2 | 70.0 | 62.5 | 60.0 | 69.2 | 89.2 | 55.0 | 72.9 |
| DeepConvNet | float | 86.7 | 64.2 | 99.2 | 70.8 | 63.3 | 56.7 | 66.7 | 88.3 | 60.0 | 72.9 |
| Riemann TS + LR | float | 81.7 | 54.2 | 98.3 | 78.3 | 62.5 | 55.8 | 71.7 | 96.7 | 58.3 | 73.1 |
| Teacher ensemble (mean softmax) | float | 87.5 | 65.8 | 99.2 | 76.7 | 68.3 | 56.7 | 66.7 | 93.3 | 60.0 | 74.9 |
| Student wide32 | float | 89.2 | 61.7 | 98.3 | 74.2 | 66.7 | 56.7 | 65.0 | 93.3 | 59.2 | 73.8 |
| Student wide32 | int8 | 89.2 | 60.8 | 98.3 | 74.2 | 65.8 | 57.5 | 64.2 | 93.3 | 59.2 | 73.6 |
| Student control | float | 89.2 | 60.0 | 97.5 | 77.5 | 68.3 | 54.2 | 63.3 | 94.2 | 55.0 | 73.2 |
| Student control | int8 | 90.0 | 60.0 | 97.5 | 76.7 | 67.5 | 55.8 | 64.2 | 93.3 | 54.2 | 73.2 |
| Student wide32_ft | float | 88.3 | 64.7 | 98.6 | 74.4 | 71.1 | 55.6 | 65.6 | 95.0 | 58.6 | 74.7 |
| Student wide32_ft | int8 | 88.6 | 63.9 | 98.6 | 74.4 | 70.0 | 56.4 | 66.1 | 94.7 | 58.3 | 74.6 |
| *v0.1 TinyEEGNet+EA, LOSO on 2a, 250 Hz (different protocol)* | float | 85.3 | 53.1 | 92.2 | 67.8 | 58.1 | 63.9 | 66.9 | 91.4 | 64.4 | 71.5 |
| *v0.1 TinyEEGNet+EA, LOSO on 2a, 250 Hz (different protocol)* | int8 | 85.0 | 53.9 | 92.5 | 67.2 | 58.1 | 64.4 | 66.7 | 91.1 | 64.2 | 71.5 |

v0.1 rows: different protocol (LOSO on BCI IV 2a, 250 Hz, 500 samples), context only.

## 4. Variant comparison on the test block (exploratory)

Variant comparison on the test block (exploratory; the shipped variant was chosen by the team from this table, which makes its test estimate optimistic by selection among 5 candidates); the seed within the variant is chosen by validation KL. Highest mean int8 test accuracy in this run: **student wide32_ft**; its lowest-validation-KL seed is 0 (0.0009), checkpoint `runs/v4soft/student_wide32_ft_seed0/model.pt`, export subject A03. See the selection note.

Per-subject KPIs, float (mean ± std over seeds):

| Subject | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa | best epochs |
|---|---|---|---|---|---|---|
| A01 | 88.3 ± 1.2 | 0.971 ± 0.002 | 0.882 ± 0.012 | 0.971 ± 0.001 | 0.767 ± 0.024 | 29, 29, 29 |
| A02 | 64.7 ± 3.9 | 0.705 ± 0.015 | 0.641 ± 0.041 | 0.676 ± 0.021 | 0.294 ± 0.077 | 29, 29, 29 |
| A03 (export) | 98.6 ± 0.4 | 1.000 ± 0.000 | 0.986 ± 0.004 | 1.000 ± 0.000 | 0.972 ± 0.008 | 29, 29, 29 |
| A04 | 74.4 ± 0.8 | 0.803 ± 0.003 | 0.740 ± 0.007 | 0.806 ± 0.003 | 0.489 ± 0.016 | 29, 29, 29 |
| A05 | 71.1 ± 1.4 | 0.749 ± 0.004 | 0.711 ± 0.014 | 0.730 ± 0.006 | 0.422 ± 0.028 | 29, 29, 29 |
| A06 | 55.6 ± 1.4 | 0.611 ± 0.011 | 0.555 ± 0.015 | 0.623 ± 0.007 | 0.111 ± 0.028 | 29, 29, 29 |
| A07 | 65.6 ± 0.4 | 0.737 ± 0.007 | 0.653 ± 0.004 | 0.757 ± 0.007 | 0.311 ± 0.008 | 29, 29, 29 |
| A08 | 95.0 ± 0.7 | 0.993 ± 0.002 | 0.950 ± 0.007 | 0.992 ± 0.002 | 0.900 ± 0.014 | 29, 29, 29 |
| A09 | 58.6 ± 0.4 | 0.505 ± 0.003 | 0.548 ± 0.001 | 0.615 ± 0.003 | 0.172 ± 0.008 | 29, 29, 29 |
| **mean ± std** | 74.7 ± 14.9 | 0.786 ± 0.164 | 0.741 ± 0.154 | 0.797 ± 0.147 | 0.493 ± 0.297 | |

## 5. Int8 minus float accuracy (points)

| Student | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| wide32 | +0.0 | -0.8 | +0.0 | +0.0 | -0.8 | +0.8 | -0.8 | +0.0 | +0.0 | -0.2 |
| control | +0.8 | +0.0 | +0.0 | -0.8 | -0.8 | +1.7 | +0.8 | -0.8 | -0.8 | +0.0 |
| wide32_ft | +0.3 | -0.8 | +0.0 | +0.0 | -1.1 | +0.8 | +0.6 | -0.3 | -0.3 | -0.1 |

## 6. Training facts

Trials: train 13,068, validation 1,452, 2a calibration 216 (unlabelled), 2a test 1,080. Data meta: `{"n_train_subject_sessions": 92, "sources": {"Cho2017": {"subjects": 52, "subject_sessions": 52, "train": 9468, "val": 1052, "skipped": 0}, "Lee2019_MI": {"subjects": 20, "subject_sessions": 40, "train": 3600, "val": 400, "skipped": 0}}, "skipped": [], "fs": 125, "window": 256, "train_window": 306, "crop_offset": 25, "window_start": 62, "ea_eps": 0.001, "seed": 0}`.

Teachers stop and select on the validation cross-entropy, students on the validation KL to the teacher ensemble (the combined distillation loss at that epoch in brackets).

| Run | Seed | Best epochs (per model) | Best val CE (teachers) / val KL (students) | Wall (s) |
|---|---|---|---|---|
| teacher | 0 | EEGNet-ELU 0, ShallowConvNet 0, DeepConvNet 3, Riemann TS + LR 0 | 0.5155, 0.5312, 0.5294, 0.5777 | 42 |
| student control | 0 | 57 | 0.0011 (0.1682) | 384 |
| student wide32 | 0 | 29 | 0.0009 (0.1641) | 310 |
| student wide32_ft | 0 | 29 | 0.0009 (0.0148) | 26 |
| student wide32_ft | 1 | 29 | 0.0015 (0.0233) | 16 |
| student wide32_ft | 2 | 29 | 0.0013 (0.0200) | 16 |

Confusion matrices per subject (rows = true left/right, columns = predicted, summed over seeds): `results/confusion_v4soft/A01.md` … `A09.md`.

## 7. Self-training on the unlabelled target session

Round 0: the 4-member ensemble (EEGNet-ELU, ShallowConvNet, DeepConvNet, Riemann TS + LR) is trained on the training split only and scores BNCI2014_001 session 1 (left/right trials of every test subject, EA from that session's own trials; its labels are never read for training). Windows with ensemble max probability ≥ 0.8 get the argmax as a hard label (weight 1) and round 1 retrains every member from scratch on train + pseudo-labelled windows (same recipe and validation split). Agreement columns compare with the true session-1 labels and are computed afterwards for this report only. The session-2 test block (trials 25–144) is not involved.

Seed 0:

Soft mode: round 1 uses **all** 1,296 target windows with the round-0 ensemble probability as target, KL weighted by w = 2·|p(right) − 0.5| (Riemannian member: argmax label, sample weight w); total weight 625.3. The 'Kept' columns show how many windows pass max prob ≥ 0.8 (the hard variant's set), for comparison.

| Subject | Trials | Mean w | Weight sum | ≥ threshold | Argmax left / right (kept) | Weighted agreement, all (%) | Agreement of ≥ threshold (%) | Round-0 ensemble on all trials (%) |
|---|---|---|---|---|---|---|---|---|
| A01 | 144 | 0.541 | 77.9 | 65 | 35 / 30 | 97.2 | 100.0 | 91.7 |
| A02 | 144 | 0.321 | 46.2 | 19 | 13 / 6 | 73.9 | 84.2 | 61.8 |
| A03 | 144 | 0.768 | 110.6 | 114 | 65 / 49 | 99.4 | 100.0 | 97.2 |
| A04 | 144 | 0.445 | 64.0 | 41 | 14 / 27 | 78.0 | 78.0 | 73.6 |
| A05 | 144 | 0.284 | 40.9 | 13 | 8 / 5 | 75.8 | 84.6 | 68.1 |
| A06 | 144 | 0.474 | 68.3 | 48 | 22 / 26 | 76.3 | 85.4 | 66.7 |
| A07 | 144 | 0.296 | 42.6 | 11 | 5 / 6 | 77.5 | 100.0 | 70.8 |
| A08 | 144 | 0.654 | 94.2 | 97 | 51 / 46 | 94.4 | 96.9 | 89.6 |
| A09 | 144 | 0.559 | 80.4 | 72 | 37 / 35 | 90.4 | 94.4 | 80.6 |
| **all** | 1296 | | 625.3 | 480 | | 87.6 | 94.2 | 77.8 |

Round-1 training set: 14,364 windows (13,068 labelled + 1,296 soft-labelled).

## 8. Teachers before and after self-training (2a test block, accuracy %)

| Model | Round | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | Mean | ROC-AUC | Kappa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EEGNet-ELU | 0 (before) | 89.2 | 64.2 | 98.3 | 73.3 | 70.0 | 57.5 | 59.2 | 93.3 | 52.5 | 73.1 | 0.766 | 0.461 |
| EEGNet-ELU | 1 (after) | 89.2 | 62.5 | 98.3 | 75.0 | 70.0 | 59.2 | 60.8 | 93.3 | 55.8 | 73.8 | 0.774 | 0.476 |
| ShallowConvNet | 0 (before) | 85.8 | 65.8 | 100.0 | 71.7 | 58.3 | 60.0 | 67.5 | 89.2 | 55.8 | 72.7 | 0.781 | 0.454 |
| ShallowConvNet | 1 (after) | 86.7 | 64.2 | 99.2 | 70.0 | 62.5 | 60.0 | 69.2 | 89.2 | 55.0 | 72.9 | 0.779 | 0.457 |
| DeepConvNet | 0 (before) | 86.7 | 64.2 | 99.2 | 70.0 | 62.5 | 55.8 | 66.7 | 89.2 | 60.0 | 72.7 | 0.774 | 0.454 |
| DeepConvNet | 1 (after) | 86.7 | 64.2 | 99.2 | 70.8 | 63.3 | 56.7 | 66.7 | 88.3 | 60.0 | 72.9 | 0.774 | 0.457 |
| Riemann TS + LR | 0 (before) | 81.7 | 55.0 | 98.3 | 77.5 | 64.2 | 55.8 | 69.2 | 96.7 | 58.3 | 73.0 | 0.792 | 0.459 |
| Riemann TS + LR | 1 (after) | 81.7 | 54.2 | 98.3 | 78.3 | 62.5 | 55.8 | 71.7 | 96.7 | 58.3 | 73.1 | 0.795 | 0.461 |
| Teacher ensemble | 0 (before) | 87.5 | 66.7 | 99.2 | 77.5 | 69.2 | 57.5 | 68.3 | 93.3 | 58.3 | 75.3 | 0.792 | 0.506 |
| Teacher ensemble | 1 (after) | 87.5 | 65.8 | 99.2 | 76.7 | 68.3 | 56.7 | 66.7 | 93.3 | 60.0 | 74.9 | 0.795 | 0.498 |

## 9. Per-member KPIs per subject (final teachers, float)

**EEGNet-ELU**

| Subject | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa |
|---|---|---|---|---|---|
| A01 | 89.2 | 0.967 | 0.891 | 0.965 | 0.783 |
| A02 | 62.5 | 0.706 | 0.617 | 0.673 | 0.250 |
| A03 | 98.3 | 1.000 | 0.983 | 1.000 | 0.967 |
| A04 | 75.0 | 0.803 | 0.744 | 0.796 | 0.500 |
| A05 | 70.0 | 0.714 | 0.697 | 0.703 | 0.400 |
| A06 | 59.2 | 0.644 | 0.591 | 0.658 | 0.183 |
| A07 | 60.8 | 0.677 | 0.598 | 0.677 | 0.217 |
| A08 | 93.3 | 0.990 | 0.933 | 0.990 | 0.867 |
| A09 | 55.8 | 0.466 | 0.506 | 0.588 | 0.117 |
| **mean** | 73.8 | 0.774 | 0.729 | 0.783 | 0.476 |

**ShallowConvNet**

| Subject | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa |
|---|---|---|---|---|---|
| A01 | 86.7 | 0.949 | 0.866 | 0.946 | 0.733 |
| A02 | 64.2 | 0.692 | 0.640 | 0.670 | 0.283 |
| A03 | 99.2 | 1.000 | 0.992 | 1.000 | 0.983 |
| A04 | 70.0 | 0.841 | 0.690 | 0.852 | 0.400 |
| A05 | 62.5 | 0.646 | 0.625 | 0.644 | 0.250 |
| A06 | 60.0 | 0.655 | 0.599 | 0.665 | 0.200 |
| A07 | 69.2 | 0.749 | 0.692 | 0.753 | 0.383 |
| A08 | 89.2 | 0.977 | 0.891 | 0.977 | 0.783 |
| A09 | 55.0 | 0.506 | 0.520 | 0.625 | 0.100 |
| **mean** | 72.9 | 0.779 | 0.724 | 0.792 | 0.457 |

**DeepConvNet**

| Subject | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa |
|---|---|---|---|---|---|
| A01 | 86.7 | 0.946 | 0.866 | 0.943 | 0.733 |
| A02 | 64.2 | 0.704 | 0.637 | 0.647 | 0.283 |
| A03 | 99.2 | 1.000 | 0.992 | 1.000 | 0.983 |
| A04 | 70.8 | 0.789 | 0.699 | 0.789 | 0.417 |
| A05 | 63.3 | 0.676 | 0.631 | 0.674 | 0.267 |
| A06 | 56.7 | 0.598 | 0.561 | 0.602 | 0.133 |
| A07 | 66.7 | 0.767 | 0.666 | 0.759 | 0.333 |
| A08 | 88.3 | 0.958 | 0.883 | 0.953 | 0.767 |
| A09 | 60.0 | 0.528 | 0.569 | 0.657 | 0.200 |
| **mean** | 72.9 | 0.774 | 0.723 | 0.780 | 0.457 |

**Riemann TS + LR**

| Subject | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa |
|---|---|---|---|---|---|
| A01 | 81.7 | 0.956 | 0.810 | 0.959 | 0.633 |
| A02 | 54.2 | 0.601 | 0.532 | 0.551 | 0.083 |
| A03 | 98.3 | 0.999 | 0.983 | 0.999 | 0.967 |
| A04 | 78.3 | 0.859 | 0.783 | 0.858 | 0.567 |
| A05 | 62.5 | 0.676 | 0.625 | 0.654 | 0.250 |
| A06 | 55.8 | 0.587 | 0.556 | 0.585 | 0.117 |
| A07 | 71.7 | 0.758 | 0.717 | 0.740 | 0.433 |
| A08 | 96.7 | 0.991 | 0.967 | 0.991 | 0.933 |
| A09 | 58.3 | 0.732 | 0.519 | 0.755 | 0.167 |
| **mean** | 73.1 | 0.795 | 0.721 | 0.788 | 0.461 |

**Teacher ensemble**

| Subject | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa |
|---|---|---|---|---|---|
| A01 | 87.5 | 0.967 | 0.874 | 0.965 | 0.750 |
| A02 | 65.8 | 0.699 | 0.654 | 0.648 | 0.317 |
| A03 | 99.2 | 1.000 | 0.992 | 1.000 | 0.983 |
| A04 | 76.7 | 0.841 | 0.764 | 0.842 | 0.533 |
| A05 | 68.3 | 0.714 | 0.683 | 0.673 | 0.367 |
| A06 | 56.7 | 0.621 | 0.565 | 0.639 | 0.133 |
| A07 | 66.7 | 0.770 | 0.665 | 0.760 | 0.333 |
| A08 | 93.3 | 0.987 | 0.933 | 0.987 | 0.867 |
| A09 | 60.0 | 0.556 | 0.544 | 0.661 | 0.200 |
| **mean** | 74.9 | 0.795 | 0.742 | 0.797 | 0.498 |

## 10. Per-member learning rate and validation

| Seed | Member | Round | lr | Best epoch | Epochs run | Best val loss | Val acc (%) | lr choice |
|---|---|---|---|---|---|---|---|---|
| 0 | EEGNet-ELU | 0 | 0.001 | 57 | 88 | 0.5157 | 72.2 |  |
| 0 | ShallowConvNet | 0 | 0.0003 | 2 | 33 | 0.5362 | 72.2 | tried 0.0003 (val 0.5362) vs reference 0.5415 → 0.0003 |
| 0 | DeepConvNet | 0 | 0.001 | 46 | 77 | 0.5286 | 72.5 |  |
| 0 | Riemann TS + LR | 0 | — | 0 | 1 | 0.5772 | 69.5 |  |
| 0 | EEGNet-ELU | 1 | 0.0001 | 0 | 16 | 0.5155 | 72.4 |  |
| 0 | ShallowConvNet | 1 | 0.0001 | 0 | 16 | 0.5312 | 71.9 |  |
| 0 | DeepConvNet | 1 | 0.0001 | 3 | 19 | 0.5294 | 72.6 |  |
| 0 | Riemann TS + LR | 1 | — | 0 | 1 | 0.5777 | 69.1 |  |

## 11. Change against `v2` (seeds [0, 1, 2]), accuracy points

Positive = this run is better. Reference = the comparison run's per-subject mean over its seeds.

| Model | Precision | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | Mean | Ref mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EEGNet-ELU | float | -1.4 | +0.8 | -0.3 | +1.7 | +2.5 | +1.7 | -0.8 | +3.3 | +0.8 | +0.9 | 72.9 |
| ShallowConvNet | float | -1.9 | -1.1 | +1.7 | -0.8 | +2.8 | +0.0 | +3.6 | -4.2 | -2.8 | -0.3 | 73.2 |
| DeepConvNet | float | -0.6 | +0.0 | +0.6 | -1.1 | +1.7 | +1.9 | -1.4 | -1.1 | +0.3 | +0.0 | 72.8 |
| Teacher ensemble (mean softmax) | float | -2.8 | +0.6 | +0.0 | +1.7 | +3.1 | +1.4 | -1.1 | +1.1 | +1.1 | +0.6 | 74.4 |
| Student wide32 | float | -0.3 | -1.7 | -0.8 | -0.6 | -1.1 | +1.7 | +1.7 | -0.6 | +1.7 | +0.0 | 73.8 |
| Student wide32 | int8 | -0.6 | -1.9 | -0.8 | +0.0 | -0.6 | +1.7 | +1.7 | -0.6 | +2.5 | +0.2 | 73.5 |
| Student control | float | +0.0 | -1.7 | -1.4 | +5.0 | -0.8 | -3.6 | +0.8 | -0.3 | -1.1 | -0.3 | 73.6 |
| Student control | int8 | +1.1 | -1.1 | -1.4 | +4.2 | +0.0 | -2.5 | +1.9 | -0.8 | -1.7 | -0.0 | 73.3 |

## 12. Change against `v2` (seeds [0]), accuracy points

Positive = this run is better. Reference = the comparison run's per-subject mean over its seeds.

| Model | Precision | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | Mean | Ref mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EEGNet-ELU | float | +0.0 | -1.7 | +0.0 | +1.7 | +0.0 | +1.7 | +1.7 | +0.0 | +3.3 | +0.7 | 73.1 |
| ShallowConvNet | float | +0.8 | +1.7 | +0.8 | -3.3 | +5.8 | +0.0 | +5.8 | -4.2 | -4.2 | +0.4 | 72.5 |
| DeepConvNet | float | +0.0 | +0.0 | +0.0 | +0.8 | +0.8 | +0.8 | +0.0 | -0.8 | +0.0 | +0.2 | 72.7 |
| Teacher ensemble (mean softmax) | float | -1.7 | -2.5 | -0.8 | +2.5 | +3.3 | +0.8 | +0.8 | +0.0 | +0.8 | +0.4 | 74.5 |
| Student wide32 | float | +1.7 | +0.0 | -0.8 | +0.8 | -0.8 | -0.8 | +1.7 | -1.7 | +1.7 | +0.2 | 73.6 |
| Student wide32 | int8 | +1.7 | -0.8 | -0.8 | +1.7 | -1.7 | -0.8 | +1.7 | -0.8 | +2.5 | +0.3 | 73.3 |
| Student control | float | +0.8 | -1.7 | -0.8 | +4.2 | -1.7 | -1.7 | +2.5 | -0.8 | +0.8 | +0.2 | 73.1 |
| Student control | int8 | +1.7 | -1.7 | -0.8 | +3.3 | +0.0 | -0.8 | +3.3 | -1.7 | +0.0 | +0.4 | 72.9 |

## 13. Round 0 vs hard pseudo-labels vs soft targets (2a test block, accuracy %)

| Model | Precision | A01 | A02 | A03 | A04 | A05 | A06 | A07 | A08 | A09 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Teacher ensemble, round 0 (before self-training) | float | 87.5 | 66.7 | 99.2 | 77.5 | 69.2 | 57.5 | 68.3 | 93.3 | 58.3 | 75.3 |
| Teacher ensemble, hard pseudo-labels (`v4`) | float | 89.2 | 62.5 | 99.2 | 77.5 | 68.3 | 56.7 | 65.0 | 94.2 | 60.0 | 74.7 |
| Teacher ensemble, soft targets (this run) | float | 87.5 | 65.8 | 99.2 | 76.7 | 68.3 | 56.7 | 66.7 | 93.3 | 60.0 | 74.9 |
| Student control, hard (`v4`) | float | 89.2 | 60.8 | 97.5 | 77.5 | 67.5 | 54.2 | 65.8 | 94.2 | 55.0 | 73.5 |
| Student control, hard (`v4`) | int8 | 89.2 | 60.8 | 97.5 | 76.7 | 66.7 | 55.8 | 65.8 | 94.2 | 55.0 | 73.5 |
| Student wide32, hard (`v4`) | float | 88.3 | 60.8 | 99.2 | 75.0 | 70.0 | 55.8 | 63.3 | 94.2 | 58.3 | 73.9 |
| Student wide32, hard (`v4`) | int8 | 88.3 | 60.8 | 99.2 | 74.2 | 68.3 | 57.5 | 62.5 | 95.0 | 58.3 | 73.8 |
| Student wide32_r0, round-0 teacher (`v4`) | float | 89.2 | 60.8 | 98.3 | 73.3 | 68.3 | 57.5 | 63.3 | 93.3 | 58.3 | 73.6 |
| Student wide32_r0, round-0 teacher (`v4`) | int8 | 88.3 | 60.0 | 98.3 | 73.3 | 68.3 | 57.5 | 65.0 | 93.3 | 58.3 | 73.6 |
| Student wide32, soft (this run) | float | 89.2 | 61.7 | 98.3 | 74.2 | 66.7 | 56.7 | 65.0 | 93.3 | 59.2 | 73.8 |
| Student wide32, soft (this run) | int8 | 89.2 | 60.8 | 98.3 | 74.2 | 65.8 | 57.5 | 64.2 | 93.3 | 59.2 | 73.6 |
| Student control, soft (this run) | float | 89.2 | 60.0 | 97.5 | 77.5 | 68.3 | 54.2 | 63.3 | 94.2 | 55.0 | 73.2 |
| Student control, soft (this run) | int8 | 90.0 | 60.0 | 97.5 | 76.7 | 67.5 | 55.8 | 64.2 | 93.3 | 54.2 | 73.2 |
| Student wide32_ft, fine-tuned v2 student (KD only) | float | 88.3 | 64.7 | 98.6 | 74.4 | 71.1 | 55.6 | 65.6 | 95.0 | 58.6 | 74.7 |
| Student wide32_ft, fine-tuned v2 student (KD only) | int8 | 88.6 | 63.9 | 98.6 | 74.4 | 70.0 | 56.4 | 66.1 | 94.7 | 58.3 | 74.6 |

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
