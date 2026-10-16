# Reproduce_Baselines results

## 1. Setup

BCI Competition IV 2a (MOABB `BNCI2014_001`), left vs right hand, 22 EEG channels, 250 Hz, causal Butterworth 7–30 Hz (order 4, `sosfilt`), window cue+0.5 s to cue+2.5 s (500 samples). Test set for every method: target subject's session 2 trials 25–144 (120 trials/subject); trials 1–24 of session 2 are the unlabeled Euclidean-Alignment calibration block. LOSO rows train on the other 8 subjects (both sessions, EA per subject-session), early stopping on a stratified, seed-dependent 10 % of every source subject-session's trials; per-subject rows train on the target's session 1 (80/20 stratified early-stop split). Seeds [0, 1, 2] per subject; each subject's value is the mean over seeds, the ± in the summary is the std across the 9 subjects. KPIs: accuracy, ROC-AUC, macro F1, AUPRC (positive class = right hand), Cohen's kappa; chance = 50 % / 0.5 / 0.5 / 0.5 / 0.

Environment: Python 3.12.14, torch 2.14.0, executorch 1.5.1, torchao 0.18.0, moabb 1.7.2, mne 1.13.2, scikit-learn 1.9.1, scipy 1.18.1, numpy 2.5.4; git commit `23b5d4b`; generated 2026-10-16 15:27.

## 2. Summary (mean ± std across 9 subjects)

| Method | Training | Precision | Accuracy (%) | ROC-AUC | F1 | AUPRC | Kappa | n subj |
|---|---|---|---|---|---|---|---|---|
| CSP+LDA | LOSO | float | 62.7 ± 15.0 | 0.718 ± 0.162 | 0.595 ± 0.171 | 0.725 ± 0.166 | 0.254 ± 0.300 | 9 |
| CSP+LDA+EA | LOSO | float | 71.3 ± 15.4 | 0.782 ± 0.151 | 0.706 ± 0.162 | 0.775 ± 0.152 | 0.426 ± 0.308 | 9 |
| EEGNet-ELU+EA | LOSO | float | 73.4 ± 13.1 | 0.779 ± 0.159 | 0.726 ± 0.138 | 0.788 ± 0.148 | 0.469 ± 0.263 | 9 |
| **TinyEEGNet+EA, row B** | LOSO | float | 71.5 ± 13.7 | 0.776 ± 0.142 | 0.708 ± 0.141 | 0.780 ± 0.141 | 0.429 ± 0.273 | 9 |
| **TinyEEGNet+EA, row B** | LOSO | int8 | 71.5 ± 13.5 | 0.778 ± 0.141 | 0.708 ± 0.140 | 0.782 ± 0.140 | 0.429 ± 0.270 | 9 |
| TinyEEGNet, row A | per-subject | float | 70.6 ± 17.0 | 0.742 ± 0.179 | 0.696 ± 0.178 | 0.742 ± 0.177 | 0.411 ± 0.340 | 9 |
| TinyEEGNet, row A | per-subject | int8 | 70.6 ± 17.2 | 0.743 ± 0.180 | 0.697 ± 0.179 | 0.743 ± 0.179 | 0.412 ± 0.343 | 9 |

Published references (not comparable: different class count or protocol): BCI Competition IV 2a winner kappa 0.57 (4-class, within-subject; Tangermann et al. 2012, Review of the BCI Competition IV, Front. Neurosci., https://doi.org/10.3389/fnins.2012.00055); EEGNet 73.13 % on 2a (4-class, within-session; Yang et al. 2025, Sci. Data, https://doi.org/10.1038/s41597-025-04826-y, Table 4).

**Headline rows are the two bold TinyEEGNet+EA row-B rows (float and int8).**

## 3. Per subject, row B (TinyEEGNet+EA, LOSO)

Values are mean ± std over seeds; Δ int8 = int8 − float accuracy in points.

| Subject | Acc float (%) | Acc int8 (%) | Δ int8 | ROC-AUC | F1 | AUPRC | Kappa | best epoch | wall/fold (s) |
|---|---|---|---|---|---|---|---|---|---|
| A01 | 85.3 ± 0.8 | 85.0 ± 0.7 | -0.3 | 0.942 ± 0.011 | 0.853 ± 0.008 | 0.945 ± 0.009 | 0.706 ± 0.016 | 17, 29, 40 | 427 |
| A02 | 53.1 ± 2.7 | 53.9 ± 3.7 | +0.8 | 0.636 ± 0.028 | 0.524 ± 0.031 | 0.608 ± 0.040 | 0.061 ± 0.055 | 24, 21, 67 | 563 |
| A03 (export) | 92.2 ± 0.4 | 92.5 ± 0.7 | +0.3 | 0.992 ± 0.002 | 0.922 ± 0.004 | 0.993 ± 0.002 | 0.844 ± 0.008 | 22, 48, 18 | 511 |
| A04 | 67.8 ± 1.7 | 67.2 ± 2.1 | -0.6 | 0.742 ± 0.033 | 0.677 ± 0.017 | 0.746 ± 0.013 | 0.356 ± 0.034 | 22, 37, 18 | 445 |
| A05 | 58.1 ± 2.6 | 58.1 ± 2.6 | +0.0 | 0.643 ± 0.064 | 0.566 ± 0.018 | 0.640 ± 0.073 | 0.161 ± 0.052 | 9, 15, 12 | 358 |
| A06 | 63.9 ± 1.7 | 64.4 ± 1.4 | +0.6 | 0.689 ± 0.008 | 0.639 ± 0.017 | 0.678 ± 0.020 | 0.278 ± 0.034 | 64, 17, 13 | 501 |
| A07 | 66.9 ± 2.2 | 66.7 ± 0.7 | -0.3 | 0.727 ± 0.024 | 0.667 ± 0.024 | 0.720 ± 0.033 | 0.339 ± 0.044 | 22, 17, 8 | 401 |
| A08 | 91.4 ± 0.8 | 91.1 ± 0.8 | -0.3 | 0.975 ± 0.005 | 0.914 ± 0.008 | 0.977 ± 0.003 | 0.828 ± 0.016 | 18, 23, 18 | 336 |
| A09 | 64.4 ± 2.2 | 64.2 ± 2.5 | -0.3 | 0.637 ± 0.034 | 0.615 ± 0.032 | 0.715 ± 0.027 | 0.289 ± 0.044 | 51, 20, 21 | 238 |
| **mean ± std** | 71.5 ± 13.7 | 71.5 ± 13.5 | +0.0 | 0.776 ± 0.142 | 0.708 ± 0.141 | 0.780 ± 0.141 | 0.429 ± 0.273 |  |  |

## 4. Per-subject accuracy (%) across methods

| Subject | CSP+LDA (LOSO, float) | CSP+LDA+EA (LOSO, float) | EEGNet-ELU+EA (LOSO, float) | TinyEEGNet+EA, row B (LOSO, float) | TinyEEGNet+EA, row B (LOSO, int8) | TinyEEGNet, row A (per-subject, float) | TinyEEGNet, row A (per-subject, int8) |
|---|---|---|---|---|---|---|---|
| A01 | 55.8 | 85.0 | 85.6 | 85.3 | 85.0 | 72.8 | 73.1 |
| A02 | 48.3 | 49.2 | 57.5 | 53.1 | 53.9 | 53.1 | 53.1 |
| A03 | 86.7 | 90.8 | 92.8 | 92.2 | 92.5 | 95.6 | 95.8 |
| A04 | 58.3 | 78.3 | 70.3 | 67.8 | 67.2 | 58.9 | 57.5 |
| A05 | 53.3 | 55.8 | 62.2 | 58.1 | 58.1 | 54.4 | 54.7 |
| A06 | 52.5 | 63.3 | 70.3 | 63.9 | 64.4 | 56.4 | 56.7 |
| A07 | 58.3 | 70.0 | 67.5 | 66.9 | 66.7 | 59.4 | 59.7 |
| A08 | 93.3 | 93.3 | 94.2 | 91.4 | 91.1 | 93.9 | 94.2 |
| A09 | 57.5 | 55.8 | 60.6 | 64.4 | 64.2 | 90.6 | 90.8 |
| **mean** | 62.7 | 71.3 | 73.4 | 71.5 | 71.5 | 70.6 | 70.6 |

## 5. EA calibration variant (row B, float)

Same trained row-B models re-scored on the same 120 test trials; only the target's EA matrix changes: default `A` from session 2 trials 1–24 vs `A` from all 144 trials of session 1 (labels unused in both; `predictions_calib_s1.npz` in each run directory).

| Subject | Acc (%) A from session 2 calib | Acc (%) A from session 1 | Kappa (s2 calib) | Kappa (s1) |
|---|---|---|---|---|
| A01 | 85.3 | 83.1 | 0.706 | 0.661 |
| A02 | 53.1 | 59.7 | 0.061 | 0.194 |
| A03 | 92.2 | 81.7 | 0.844 | 0.633 |
| A04 | 67.8 | 62.2 | 0.356 | 0.244 |
| A05 | 58.1 | 58.1 | 0.161 | 0.161 |
| A06 | 63.9 | 59.2 | 0.278 | 0.183 |
| A07 | 66.9 | 66.1 | 0.339 | 0.322 |
| A08 | 91.4 | 89.2 | 0.828 | 0.783 |
| A09 | 64.4 | 75.0 | 0.289 | 0.500 |
| **mean ± std** | 71.5 ± 13.7 | 70.5 ± 11.2 | 0.429 ± 0.273 | 0.409 ± 0.225 |

## 6. Deployment (export subject)

Export subject: **A03**, checkpoint `runs/loso_s3_seed2/model.pt` (seed with the lowest best validation loss: 0.3522, float test accuracy 92.5 %). Mean over seeds: float 92.2 %, int8 92.5 %. TinyEEGNet: 1938 parameters.

Key deployment numbers (host fake-quant checks; from [deployment.md](deployment.md), operator detail in [op_placement.md](op_placement.md), both written by `src.export`):

| Item | Value |
|---|---|
| Checkpoint | `runs/loso_s3_seed2/model.pt` |
| Parameters | 1,938 |
| MACs per window | 5,884,608 |
| `.pte` size | 10,624 B (10.4 KiB) |
| Ops on Ethos-U / CPU | 35 / 3 (1 delegated subgraph(s)); see [op_placement.md](op_placement.md) |
| Vela SRAM estimate | 182.97 KiB (Vela estimate, not an FVP measurement) |
| int8 accuracy, 20 test vectors | 90.0% (float 90.0%) |
| Max \|logit\| diff int8 vs float, 20 vectors | 1.5643 |

FVP latency and arena size are added from Emulator_Deploy.

## 7. Confusion matrices and gates

| Gate | Value | Status |
|---|---|---|
| Export subject float accuracy ≥ 75 % | 92.2 % | pass |
| Int8 accuracy drop ≤ 2 points (export subject) | +0.3 | pass |
| `.pte` < 100 KB | 10.4 KiB | pass |
| All conv/linear ops on Ethos-U | [op_placement.md](op_placement.md): yes | pass |

Confusion matrices per subject and method (rows = true left/right, columns = predicted, summed over seeds): `results/confusion/A01.md` … `A09.md`. Row B, export subject, summed over seeds:

| A03 row B | pred left | pred right |
|---|---|---|
| true left | 155 | 25 |
| true right | 3 | 177 |
