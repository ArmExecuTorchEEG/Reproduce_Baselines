# Model 2 — Participant 1 TinyEEGNet

Preferred candidate for a Participant 1 recording-based demonstration; physical
Alif E8 execution and live EEG/cursor control are not yet validated.

- Training: Participant 1 Session 1, 115 trials, accuracy 94.78% (balanced 94.74%).
- Validation: same session, 29 distinct stratified trials, accuracy 75.86%
  (balanced 76.19%); minimum validation loss selected epoch 110.
- Held-out test: Participant 1 Session 2, 144 trials, accuracy/balanced accuracy
  79.86%; confusion matrix `[[48,24],[5,67]]` (true rows, left/right columns).
- Seed 42, fresh weights, original 1,938-parameter architecture and dropout 0.5.
  Only the 115 training windows fit normalization; no Session 2 calibration/tuning.
  All artifact flags retained. Different evaluation participants prevent a fair
  superiority claim over Model 1.

PyTorch input: float32 `(N,1,22,500)`. Compiled input: signed int8
`(1,1,22,500)` after THIS model's saved EEG preprocessing/normalization and
quantization (scale 0.04422667250037193, zero point -10). Output: float32 `(1,2)`
logits; left=0/right=1. EEG: 22 channels at 250 Hz, causal 7–30 Hz Butterworth,
cue-relative [1,3) seconds. Follow `preprocess_params.json` exactly.

The 13,072-byte PTE targets ethos-u85-256, Ethos_U85_SRAM_MRAM, Shared_Sram.
ExecuTorch exporter/runtime 1.5.1. All model compute delegates to U85; output
dequantization is CPU. FVP versus host Q/DQ on four Session 1 validation windows:
4/4 prediction agreement, maximum printed-logit difference 2.28e-9, within 0.01.
This is a four-input smoke test, not full-dataset or streaming validation.
The host Q/DQ full-validation accuracy is 75.86%; float-versus-Q/DQ logit
error was 0.27734, so FVP agreement does not establish all release criteria.

Original locations (all preserved):
- [Training checkpoint/config/normalization](../../artifacts/subject1-seed42-20261016/)
- [Compiled PTE and INT8 parameters](../../artifacts/subject1-ethosu85-seed42-20261016/)
- [Training results and loss curves](../../results/subject1-seed42-20261016/summary.md)
- [FVP results and reproduction](../../results/fvp-execution-20261016/README.md)
- [Generated FVP harness](../../artifacts/fvp-subject1-20261016-ready/project/)

See [provenance.json](provenance.json) for per-file SHA-256 and exact source paths.
Use `preprocess_params.json` alongside this PTE, never Model 1's parameters.
Checkpoint and PTE are tracked on this branch. See the [package guide](../README.md)
for download and verification; only optional NPZ vectors remain local.
