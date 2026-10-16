# Participant 1 TinyEEGNet baseline

A second independent model trained successfully with fresh seed-42 weights,
using the unchanged 1,938-parameter TinyEEGNet and dropout 0.5. No additional
hyperparameter experiments were performed. Original models/results are preserved.

## Protocol and results

| Split | Source | Shape | Left/right | Flagged | Accuracy | Balanced accuracy |
|---|---|---|---|---|---|---|
| Training | Participant 1 Session 1 | (115,1,22,500) | 57/58 | 4 | 94.78% | 94.74% |
| Validation | Participant 1 Session 1 | (29,1,22,500) | 15/14 | 2 | 75.86% | 76.19% |
| Test | Participant 1 Session 2 | (144,1,22,500) | 72/72 | 3 | 79.86% | 79.86% |

Stratified 80/20 Session 1 split, seed 42, no duplicated identities or overlapping
window/filter histories. Session 2 was loaded only after selecting the checkpoint.
New channel mean/population std fitted solely on the 115 training trials, not
copied from the multi-subject normalization. Reused the original filter/extractor:
22 ordered EEG channels, causal 7–30 Hz Butterworth, 250 Hz, cue-relative [1,3)
windows, float32 input/int64 labels, left=0/right=1, retain all artifact flags.

Same Adam/CrossEntropyLoss schedule: lr 0.001, batch 64, maximum 200 epochs,
patience 20, CPU 4 threads, deterministic algorithms. Minimum validation loss
selected epoch 110; training stopped at epoch 130, taking 81.49 seconds
(85.72 seconds through final evaluation/export). Loss curves: `loss_curves.png`;
full history is in the artifact directory. Training loss includes dropout-active
minibatches; final reported training accuracy uses evaluation mode. Diverging
training/validation losses indicate some overfitting, limited by early stopping.

Confusion matrices (rows=true, columns=predicted; left then right):
- Training: `[[51,6],[0,58]]`
- Validation: `[[10,5],[2,12]]`
- Held-out Session 2: `[[48,24],[5,67]]`

Artifact subgroup accuracy (balanced accuracy):
- Training clean 111: 94.59% (94.55%); flagged 4: 100% (100%).
- Validation clean 27: 77.78% (78.30%); flagged 2: 50% (50%).
- Test clean 141: 80.14% (80.24%); flagged 3: 66.67% (75%).
Flagged subgroups are too small for reliable conclusions. Test predictions were
computed once, with subgroup metrics derived from those same predictions.

## Interpretation

79.86% on an unseen session is promising for a Participant 1 recording-based
cursor demonstration, with an important right-hand bias: left recall 66.67%,
right recall 93.06%. This is offline classification, not validation of live EEG
or end-to-end cursor control. The 29-trial validation set is small.

Previously reported multi-subject results: S8 validation 93.06%, S6–7 Session 2
demonstration 62.15%, S9 test 59.03%. Those involve different participants and
cannot establish that this subject-specific model is superior. Participant 1
Session 2 was included in the multi-subject model's training, so evaluating that
model here would not supply an unseen-data comparison. No old-model inference
or reserved S6–7/S9 evaluation was performed in this task.

## Verification and deployment candidate

Checkpoint reload matched exactly; (1,1,22,500) produces (1,2) logits.
Strict torch.export, saved-graph reload and output comparison passed.
Checkpoint SHA256:
`63e3479baa0b00cadce5ee7906f25eee873ba561f9040cd07505f5d32b7a51d9`.

Separate INT8 export: `artifacts/subject1-ethosu85-seed42-20261016/`.
Used the proven official Arm environment (ExecuTorch 1.5.1, torch 2.14.0,
torchao 0.18.0, Vela 5.1.0) and saved Alif E8 SRAM/MRAM profile. All 115 training
windows (57 left/58 right) calibrated observers; only 29 validation windows were
used for the quantization comparison. No Session 2 quantized evaluation.
Host Q/DQ validation accuracy/balanced accuracy remain 75.86%/76.19%, with 100%
prediction agreement. Maximum absolute logit difference 0.27734 exceeds the
README's 1e-2 numerical target; unchanged labels do not prove numerical equivalence.

Compiled int8-input PTE: **13,072 bytes**, one EthosUBackend delegate with all
four convolutions and classifier; delegate payload 9,936 bytes. Serialized input
is signed int8 (1,1,22,500), output float32 (1,2); only CPU kernel is output
dequantization. Saved preprocessing includes this model's input scale
0.04422667250037193 and zero point -10, applied AFTER its own normalization.
All training test-vector expected logits are host Q/DQ references, not NPU results.

FVP and physical E8 execution remain unvalidated because Docker/FVP,
CMSIS-Toolbox and Arm compiler are unavailable. Next: run training-derived test
vectors through matching runtime/FVP, integrate E8 RTSS-HP BSP and U85 driver,
link SRAM/MRAM buffers correctly, implement exact EEG preprocessing/int8 rounding,
and verify on-board outputs, memory use and latency before live cursor control.
No root canonical release files were overwritten or release published.

## Files and reproduction

Configuration: `configs/subject1/baseline.yaml`.
Training artifacts: `artifacts/subject1-seed42-20261016/` includes checkpoint,
float state/export, configuration, labels, preprocessing, seed, trial manifest,
source hashes, prepared tensors, predictions, metrics and history.
Export report: `results/subject1-ethosu85-seed42-20261016/export_report.json`.

Commands (new output directory required for each run):
```
.venv/bin/python -m src.train_subject --config configs/subject1/baseline.yaml --run-name <new-run>
.venv/bin/python -m src.data.subject_export_data --run artifacts/<new-run> --out artifacts/<new-export>
# Copy the proven Alif ensemble_vela.ini into the new export directory.
/private/tmp/tinyeegnet-cmsis-executorch/.venv/bin/python -m src.export --workdir artifacts/<new-export> --checkpoint-dir artifacts/<new-run> --config-ini artifacts/<new-export>/ensemble_vela.ini --system-config Ethos_U85_SRAM_MRAM
.venv/bin/python -m unittest discover -s tests -v
```

19 unit tests passed, including training-only normalization and reproducible
subject split integrity; git diff --check passed. Existing artifacts/results were
SHA256-verified unchanged. Branch feature/tinyeegnet-training; no commit or push.
