# Model 1 — Multi-subject TinyEEGNet

Deployment candidate; physical Alif E8 execution has not been validated.

- Training: S1–5 both sessions plus S6–7 Session 1, 1,728 left/right trials.
- Validation/early stopping: S8 both sessions, 288 trials, accuracy 93.06%.
- Reserved demonstration: S6–7 Session 2, 288 trials, accuracy 62.15%.
- Cross-subject test: S9 both sessions, 288 trials, accuracy 59.03%.
- Balanced accuracy equals accuracy on these balanced evaluation groups.
- Original dropout-0.5 baseline, 1,938 parameters, seed 42. Training-only channel
  normalization; retain artifact flags. No checkpoint or weights were changed.

PyTorch input: float32 `(N,1,22,500)`. Compiled input: signed int8
`(1,1,22,500)` after THIS model's saved EEG preprocessing/normalization and
quantization (scale 0.07146575301885605, zero point 6). Output: float32 `(1,2)`
logits; left=0/right=1. EEG: 22 channels at 250 Hz, causal 7–30 Hz Butterworth,
cue-relative [1,3) seconds. Follow `preprocess_params.json` exactly.

The 13,056-byte PTE targets ethos-u85-256, Ethos_U85_SRAM_MRAM, Shared_Sram.
ExecuTorch exporter/runtime version 1.5.1. All model compute delegates to U85;
output dequantization is CPU. On four S8 validation inputs, FVP and host Q/DQ
predictions agree 4/4; maximum logit error 0.026093 exceeds 0.01 (one output
quantization step). This is execution success, not a complete numerical pass.
Full S8 host Q/DQ accuracy remains 93.06%; the FVP check is not a new full-dataset
accuracy result. Float-versus-Q/DQ logit discrepancies remain documented.

Original locations (all preserved):
- [Training checkpoint/config/normalization](../../artifacts/combined-seed42-20261016/)
- [Compiled PTE and INT8 parameters](../../artifacts/ethosu85-export-seed42-20261016-alif-e8/)
- [Training results](../../results/combined-seed42-20261016/summary.md)
- [FVP results and reproduction](../../results/fvp-execution-20261016/README.md)
- [Generated FVP harness](../../artifacts/fvp-multisubject-20261016/project/)

See [provenance.json](provenance.json) for per-file SHA-256 and exact source paths.
Use `preprocess_params.json` alongside this PTE, never Model 2's parameters.
Checkpoint and PTE are tracked on this branch. See the [package guide](../README.md)
for download and verification; only optional NPZ vectors remain local.
