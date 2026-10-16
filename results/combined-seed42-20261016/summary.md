# Combined-participant TinyEEGNet baseline

Run: `combined-seed42-20261016`; branch: `feature/tinyeegnet-training`.
One CPU model, seed 42, 1,938 parameters. Adam lr=0.001, batch 64, dropout 0.5.
Training stopped at epoch 67; epoch 47 selected by lowest Subject-8 validation loss.
Duration including preprocessing/export: 9.1 minutes.

| Group | Shape | Left/right | Source-flagged trials |
|---|---|---|---:|
| train | `(1728, 1, 22, 500)` | [864, 864] | 142 |
| validation | `(288, 1, 22, 500)` | [144, 144] | 22 |
| demonstration | `(288, 1, 22, 500)` | [144, 144] | 40 |
| test | `(288, 1, 22, 500)` | [144, 144] | 42 |

Training: subjects 1–5 both sessions plus 6–7 session 1. Validation: subject 8 both sessions.
Demonstration: subjects 6–7 session 2. Test: subject 9 both sessions.
No shared recordings or trial identities. All trials retained; normalization fitted only on training windows.
22 EEG channels; 250 Hz; causal 7–30 Hz Butterworth SOS; cue-relative [1,3) seconds.
Per-trial zero-state filtering with 3 seconds warm-up. Float32 X; int64 labels (left=0, right=1).

| Evaluation | Accuracy | Balanced accuracy | Confusion matrix |
|---|---:|---:|---|
| validation | 93.06% | 93.06% | `[[136, 8], [12, 132]]` |
| demonstration | 62.15% | 62.15% | `[[102, 42], [67, 77]]` |
| test | 59.03% | 59.03% | `[[70, 74], [44, 100]]` |

Confusion matrices: rows=true, columns=predicted; label order left, right.
Demonstration and test groups were evaluated only after selecting the checkpoint; no subsequent tuning.
Held-out generalization is modest despite high validation accuracy. This is a working float baseline,
not evidence of reliable cursor control or of meeting the original release thresholds.

Verification: 13 offline tests pass, including the original preprocessing tests, participant/session
allocation, train-only normalization, seeded training-step reproducibility, forward shapes, and strict export.
Saved data audit: 2,592 unique trials, disjoint groups, expected class counts/dtypes, finite tensors,
zero training channel means and unit population standard deviations. Checkpoint reload is exact;
saved export matches real validation logits; saved held-out metrics reproduce from the reloaded checkpoint.
A full second training run was not performed; cross-platform bitwise reproducibility is not promised.

Artifacts are under `artifacts/combined-seed42-20261016/`: `best_checkpoint.pt`,
`tinyeegnet_float.pt`, `tinyeegnet_float.pt2`, `preprocess_params.json`, `config.json`,
`label_mapping.json`, `split_manifest.json`, `source_recordings.json` (18 SHA256 fingerprints),
`prepared_tensors.pt`, `history.json`, and `metrics.json`. All remain Git-ignored.
MOABB emitted upstream montage-deprecation and download TLS-verification warnings; loading succeeded.
No quantization, ExecuTorch/Arm lowering, deployment, experiments, commits, or pushes were performed.

Commands:
```bash
.venv/bin/python -m src.data.download --subjects 2 3 4 5 6 7 8 9
.venv/bin/python -m src.train --run-name combined-seed42-20261016
.venv/bin/python -m unittest discover -s tests -v
```
