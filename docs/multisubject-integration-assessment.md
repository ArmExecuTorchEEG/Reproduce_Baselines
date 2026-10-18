# Multi-subject TinyEEGNet integration assessment

Analysis date: 2026-10-18. Analysis and branch preparation only; no model, frontend,
firmware, serial-port, debugger, or board changes. No inference was rerun for this assessment.

## Branches and preserved state

- Frontend worktree is now on `feature/multisubject-eeg-trials`, created directly from
  `f9c40dd0fcfcec633f18786e63c1f6c0ea74a2a2`.
- Local and remote `feature/model2-random-eeg-trials` both remain at that exact commit.
  All tracked working implementation files were committed and pushed. The pre-existing
  untracked `.vscode/` remains local; ignored datasets/binaries remain local too.
- Local and remote `demo/edgebci-model2-4trial` remain at
  `c85ab70f6030243283431f3a12b1a7ea87aa8b72`.
- Inspected, without checking out or merging, `origin/main` at
  `ad61c79de17243f5befce069a1e9ca458057f43e` (Moamen Soliman).
- Original firmware worktree remains on `codex/e8-validation-20261017`; its existing
  modified AGENTS.md and untracked settings/old result directory were not touched.
- Current streaming source, HEX, AXF and sustained capture match
  `tools/model2_random_trials/validation/physical-streaming-build.json`.
- Four-trial rollback archive still hashes to
  `90434792bdc757e225c78136e1c5261dd56acb6e3d5a51a1b8fc192d73c36015`.
  Location: `artifacts/streaming-rollback/validated-four-trial.tar.gz` in the frontend worktree.
- This report is the only new worktree file. No commit, push, or integration implementation
  was performed. The new branch is local; its existing base commit is already on GitHub.

## Which model is Moamen's current work?

Do not confuse our old `models_ready_for_alif_e8/01_multi_subject_tinyeegnet` with
Moamen's independently developed models on main.

| Version | Architecture | Reported accuracy and scope |
|---|---|---|
| v0.1-baseline | 1,938 parameters, 22 x 500 at 250 Hz, separate LOSO checkpoints | 71.5% INT8 mean over subjects/seeds |
| v0.3-distill / v2 | wide32, 3,426 parameters, 22 x 256 at 125 Hz | 73.5% INT8 mean over subjects/seeds |
| v0.5-distill / v4soft wide32_ft | Same wide32 architecture; KD fine-tune of v2 | 74.6% INT8 three-seed mean; shipped seed 0 is 74.4444% |
| v0.5-fused | Nine subject-specific programs derived from v4, EA/scaling folded into weights | Exploratory host mean 74.54%; different input quantization per subject |

**Recommendation: integrate the original, unfused v4soft wide32_ft seed-0 PTE first.**
It is one model for all nine subjects. Host-side preprocessing already performs EA;
the fused version saves no board EA work in our architecture and introduces nine
model images and subject-dependent quantizers. Source docs still call v4/fused
releases planned, but remote tags `v0.5-distill` and `v0.5-fused` now exist. A tag alone
does not verify release attachments; those attachments were not inspected.

Sources: [v4 handoff][v4], [fused design][fused], [architecture][arch].

## Architecture and tensor contract

| Property | Working Model 2 | Recommended Moamen v4 |
|---|---|---|
| Parameters | 1,938 | 3,426 |
| F1 / depth multiplier / F2 | 8 / 2 / 16 | 16 / 2 / 32 |
| Temporal / separable kernel | 64 / 16 | 32 / 16 |
| Temporal padding | Explicit asymmetric padding | Symmetric Conv2d padding |
| Sample rate and tensor shape | 250 Hz, `(1,1,22,500)` | 125 Hz, `(1,1,22,256)` |
| Actual PTE input dtype | INT8 | FLOAT32, internally quantized by a CPU operator |
| Output | FLOAT32 `(1,2)` logits | FLOAT32 `(1,2)` logits |
| INT8 transport elements | 11,000 | 5,632 |
| Normalization | Saved training channel mean/std | Saved subject EA matrix and one global scale |
| Input quantizer scale / zero point | 0.04422667250037193 / -10 | 0.06955837458372116 / -12 |

Both are TinyEEGNet-family temporal convolution, depthwise spatial convolution,
separable convolution, BatchNorm, ReLU, average pooling and a two-logit classifier.
They are not identical architectures or interchangeable tensor contracts. Moamen's
wide32 parameter count includes BN; inference dropout is inactive. Classes stay
0=left, 1=right. All model compute delegates to U85; the v4 PTE includes CPU input
quantize and output dequantize. Runtime target is ExecuTorch 1.5.1 / Ethos-U85-256.

## Artifact inventory and provenance

The training checkpoint is `runs/v4soft/student_wide32_ft_seed0/model.pt`.
The documented exported float state dict is `artifacts/v4/tinyeegnet_float.pt`
(20,398 bytes; SHA256 `987231fefed1815de280d0d11437716b6e7e59abb5cb0dd66f915c1d1432c9bd`).
These are identified but were not found in our local training workspace or the
tracked main tree. Main tracks only the older top-level preprocessing JSON; do not
use it with v4.

The exact deployment assets ARE recoverable from pinned downstream source:

| Asset | Verified location / identity |
|---|---|
| v4 PTE | `Emulator_Deploy/ai_layer/model_pte.c` and `Edge_Deploy/firmware/ai_layer/model_pte.c` both decode to 11,456 bytes |
| PTE SHA256 | `6caa8a9876d3cd0c36b39b3d4474a0fe7dbdbfb32a4d6f80ed4776623a19e02e` |
| v4 parameters | `Edge_Deploy/firmware/model/params.json`, 121,188 bytes |
| Params SHA256 | `3d34f6326c67441e9e6b636454436c63951e81f6e221da977723700f1c8152f7` |
| Architecture config | Downstream `model/student.json`; F1=16, D=2, F2=32, kern_t=32, kern_s=16 |
| Known inputs | Emulator `src/test_vectors.h` contains 20 exported test inputs; Edge generated headers contain preprocessing/self-test fixtures |
| Full saved desktop references | Edge `host/reference/offline_int8_predictions.json`, all 1,080 trials, full-precision logits, labels and L/R trial ordinals |
| Desktop-reference SHA256 | `d41cf8bca4149071b737cbbf8ad73f49e1f1223a2c6ef3b3a88108ec7aa66a70` |

Pinned downstream revisions:
- Emulator_Deploy `v4/fvp`: `b882ddc5ad2fbd28b6cc18da289cde1830052a4c`.
- Edge_Deploy `claude/edge-v2`: `c616ea07e42926c6490d43a220f4bc0915b8f54f`.

Temporary read-only inspection clones are under
`/private/tmp/edgebci-multisubject-analysis-yfqx2y5y/`; they are not part of the project
and should not be treated as a permanent backup. No deployment assets were installed.

Still needed for complete desktop/export reproduction: exact float checkpoint,
original `calibration.npz` (200 external training windows), full bundle test-vector
NPZ, and saved run metadata. Their published hashes are in the v4 handoff. They are
not necessary to retrain anything: we should reuse the verified PTE. Do not invoke
an exporter with missing weights/calibration: its starter helpers have random-data
fallbacks. Fail closed instead.

## Training, normalization, and the ~75% figure

- Supervised sources: Cho2017 (52 subjects, 9,468 train + 1,052 validation windows)
  and Lee2019_MI (20 subjects, both sessions, 3,600 train + 400 validation windows).
  Total: 13,068 training and 1,452 validation windows, stratified 10% per session.
- Base students learn from teacher probabilities as well as labeled external EEG.
  The teachers include EEGNet-ELU, ShallowConvNet, DeepConvNet and, for v4, a
  Riemannian covariance/tangent-space/logistic-regression member.
- v4soft also uses all 1,296 BCI IV 2a Session-1 L/R windows without their true labels.
  Shipped student fine-tuning uses these plus the 216 Session-2 calibration windows,
  with teacher soft targets only. Fixed 30 epochs, last epoch retained.
- Target calibration: first 24 L/R trials in Session 2 of each subject compute its
  EA matrix and participate in KD. Unlabeled use still means those inputs influenced
  training; they are not eligible for training-unseen replay.
- PTQ calibration: separate 200 external training windows. This is distinct from
  target EA/KD calibration. Source-session EA uses each complete source session
  before its validation split; global `input_scale` uses the training subset.
- Test block: Session-2 L/R trials 25-144 of each subject, 120 each / 1,080 total.
- Headline 74.6 +/- 0.7% is the mean/spread over three seeds (74.4/75.5/73.8%).
  The report's +/-14.8% instead describes variation across subject means.
- The shipped seed-0 saved desktop logits give **804/1,080 = 74.4444%**. This was
  independently recomputed during this inspection, with every label checked against
  the original local MAT recording. It is a saved desktop Q/DQ evaluation, not a new
  physical-board accuracy measurement.
- The team selected the variant after comparing five candidates on this test block.
  Accordingly these are training-unseen recordings, not an untouched final test
  estimate. Do not claim fully unseen participants: their unlabeled Session-1 and
  calibration recordings influenced adaptation.

Sources: [v4 results][results], `src/data/splits125.py`, `src/finetune_student.py`,
`src/quantize.py` at the pinned main revision.

## Genuine recordings available locally

All A01E.mat through A09E.mat already exist in the original worktree at:
`mne_data/MNE-bnci-data/~bci/database/001-2014/`.
No EEG download is required for this pool.

| Subject | Eligible L/R trials | Left / right | Artifact flagged | Saved Q/DQ correct |
|---|---:|---:|---:|---:|
| A01 | 120 | 60 / 60 | 1 | 105 |
| A02 | 120 | 60 / 60 | 1 | 75 |
| A03 | 120 | 60 / 60 | 5 | 119 |
| A04 | 120 | 60 / 60 | 24 | 89 |
| A05 | 120 | 60 / 60 | 5 | 85 |
| A06 | 120 | 60 / 60 | 33 | 68 |
| A07 | 120 | 60 / 60 | 4 | 79 |
| A08 | 120 | 60 / 60 | 9 | 113 |
| A09 | 120 | 60 / 60 | 12 | 71 |
| Total | 1,080 | 540 / 540 | 94 | 804 |

Trial numbers 25-144 are ordinals within the L/R subset, not original four-class
trial numbers. A preparation manifest must retain original subject/session/run,
MAT trial index, cue sample, source-file hash, subset ordinal, label and artifact
flag, and map each to a unique transport index. All 94 flagged windows stay in the
pool. Selection must not depend on correctness.

Moamen's cache saves X/y/session/version; it does not preserve full original trial
IDs/artifact metadata. The desktop reference includes subject and subset ordinal;
we verified its label sequence against the raw files, but have not yet reproduced
or hash-verified all 1,080 input tensors. That is a required next step.

## Exact preprocessing to reuse

For BCI IV 2a, `download.load_raw` concatenates the six runs in numeric order.
Use Moamen's implementation and saved parameters, not Model 2's processed inputs:

1. Retain/reorder 22 channels; convert MNE volts to microvolts.
2. Common-average reference over the 22 channels at 250 Hz.
3. Causal fourth-order Butterworth 7-30 Hz SOS filtering over the continuous
   concatenated session, with the same initial state. Do not independently reset
   the filter for each selected window/run.
4. Keep samples 0,2,4,... (phase 0) to obtain 125 Hz; cue index becomes `cue_250 // 2`.
5. Take `[cue_125+62, cue_125+318)`: 256 samples / 2.048 seconds, nominally
   cue +0.496 to +2.544 seconds. Cue is the trial marker plus 2 seconds.
6. Multiply by the saved participant Session-2 22x22 EA matrix. EA uses covariance
   inverse square root with ridge `0.001 * trace(R) / 22`.
7. Divide by saved global `input_scale = 0.9692638516426086`.
8. Quantize, round half to even, clip INT8, using scale
   `0.06955837458372116`, zero point `-12`.

Channel order (same physical montage as our UI):
`Fz FC3 FC1 FCz FC2 FC4 C5 C3 C1 Cz C2 C4 C6 CP3 CP1 CPz CP2 CP4 P1 Pz P2 POz`.

Save CAR-filtered-decimated **pre-EA microvolt** windows for waveform/scalp display.
EA spatially mixes channels, so aligned model tensors must not be displayed as
physical electrode amplitudes. Use relative RMS and retain the prerecorded label.
Our existing 22-electrode/three-column renderer accepts 256 points; sample-rate,
duration, participant and preprocessing captions need accurate profile metadata.
No CSS, illustration, positions, colors, cursor or arm animation redesign is needed.

Pin supplied EA matrices/global scale. Existing upstream quantization code merely
warns on a regenerated EA mismatch; future integration should stop on mismatch.

## Existing FVP and host evidence: verified scope

Directly inspected [Emulator FVP metrics][fvp] and its raw log:
- 20/20 predictions match desktop argmax and ground truth (A03 exported subset).
- 11/20 vectors have nonzero logit differences; max 0.024769, approximately one
  reported output step 0.024768. PASS checks argmax, not our 0.01 logit diagnostic.
- 128,056 NPU cycles; mean 28,892 CPU cycles; method/temp peaks 28,812/98,688 bytes.
- 0.392 ms at 400 MHz is an estimate, not measured physical E8 latency.
- Raw log has an ELF zero-fill load warning before the successful execution.
  Our future harness should use the proven loading method and inspect startup.

[Edge self-test][selftest] exercises parser, preprocessing and real FVP NPU inference
on five A03 windows. All 28,160 input elements match, all five argmaxes agree, no
frame errors. It uses float64 preprocessing to match its reference.

[Edge replay report][replay] is explicitly host-side: simulated replay looks up
saved logits by construction; its 'Board target' column is a host quantized model.
`expected_board_predictions.json` is therefore not a physical-board capture; its
logits are rounded. Use full-precision `offline_int8_predictions.json` as the Q/DQ
reference. Neither report establishes v4 physical E8 operation.

The observed 0.024769 discrepancy exceeds our preserved 0.01 Q/DQ diagnostic.
Characterize v4 independently with TOSA and FVP before deployment; do not transfer
Model 2's rounding diagnosis or silently relax tolerances. Generate full-pool
backend-specific references; the saved 1,080 desktop references alone are insufficient.

## Minimal integration plan, for review only

1. Pin/copy the verified non-fused PTE, params and model metadata into a separate
   versioned model profile. Obtain missing float checkpoint/calibration assets for
   desktop reproduction. Preserve both current fallback images and recipes.
2. Prepare a separate 1,080-trial manifest, INT8 payloads and pre-EA visualization
   windows from the existing raw files. Verify original trial identity, labels,
   artifact flags and exact preprocessing against known exported vectors. Keep
   datasets/generated binaries ignored; check licensing before distributing them.
3. Create a NEW isolated firmware project/image. Reuse our UART4/startup fixes and
   persistent ExecuTorch module. Preserve Moamen's PTE bytes. Receive 5,632 INT8
   bytes, reconstruct float input `(q-zp)*scale` and feed `(1,1,22,256)` FLOAT32 to
   `forward`, as his FVP runner does. Verify internal requantization reproduces q.
   Add the required quantize runtime operator alongside dequantize. Float input
   storage is 22,528 bytes; validate memory pools/alignment and actual E8 compile
   memory configuration. U85 compatibility alone is not a deployment proof.
4. Retain COBS/CRC, session/request/trial identifiers, ACK/result matching, one
   in-flight request, bounded timeouts and no automatic retry. Parameterize size,
   hash, qparams, READY dimensions and allowed trial IDs. Existing protocol fields
   can carry the new payload; wire dtype remains INT8 even though PTE input is float.
   One active firmware model at a time avoids runtime model switching complexity.
5. Parameterize the frontend dataset/handshake profile and accurate captions using
   existing controls. Retain Cindy's visual design, public APIs, 1.8-second delayed
   visualization, real-result-only cursor, independent ground truth, retained
   captures and Play/Pause/Resume/Reset. The sequence engine already supports an
   arbitrary pool: shuffle all 1,080 without replacement, then reshuffle; no fixed
   trial cap. Track total attempts, unique recordings and per-participant accuracy.
6. Run existing Model 2/four-trial regressions plus new profile/parser/shape/identity,
   malformed-frame, disconnect, in-flight Pause and reshuffle tests. Run known 20
   vectors, then full-pool TOSA/FVP validation; retain original Q/DQ diagnostic.
7. Only after review and later physical-test authorization, deploy the separate
   application image, verify READY's identity and compare a bounded initial set of
   actual board results before continuous use. Preserve rollback throughout.

115200 baud, 8N1 transfer estimate: `(5632 + 36 + COBS overhead)*10/115200` is about
0.49 seconds per input, before ACK/result/inference. This fits the existing visual
pacing. Sending float32 bytes would take about 1.96 seconds and need larger bounds;
INT8 wire payloads avoid that change.

Primary future touchpoints (none modified now):
- `firmware/streaming/protocol.hpp`: fixed 11,000 samples, READY identity/shape,
  `trial < 144` check. New isolated profile must replace these safely.
- `firmware/streaming/app_main.cpp`: Char tensor, fixed 22x500 input.
- `frontend/streaming/{protocol,dataset,bridge}.js`: Model 2 hash, shape, S1 IDs,
  fixed counts/captions. `controller.js` selection and state machine are reusable.
- `frontend/gui.js`: sample-rate/preprocessing captions; retain rendering/design.
- Preparation/FVP fixtures: add separate model-specific paths; never overwrite
  `tools/model2_random_trials`, its generated project or existing datasets.

## Checks performed and remaining limits

Performed: Git fetch/status/ref comparison and branch creation; source/report
inspection at pinned commits; C-array PTE and params SHA verification; saved Q/DQ
logit argmax recomputation; all 1,080 labels cross-checked against local raw MAT
files; class/artifact counts; existing streaming image/archive checksum checks.
No regression/build/FVP/hardware test was run because no implementation changed.

Remaining prerequisites: prepare/hash-check all v4 input tensors and original
trial identities; obtain missing desktop reproduction bundle; independently
validate full-pool backend numerics; verify isolated E8 runtime/memory compatibility.
The current frontend, model binaries, original worktree and physical board remain unchanged.

[v4]: https://github.com/ArmExecuTorchEEG/Reproduce_Baselines/blob/ad61c79de17243f5befce069a1e9ca458057f43e/docs/v4_handoff.md
[fused]: https://github.com/ArmExecuTorchEEG/Reproduce_Baselines/blob/ad61c79de17243f5befce069a1e9ca458057f43e/docs/fused_export.md
[arch]: https://github.com/ArmExecuTorchEEG/Reproduce_Baselines/blob/ad61c79de17243f5befce069a1e9ca458057f43e/src/models/tinyeegnet.py
[results]: https://github.com/ArmExecuTorchEEG/Reproduce_Baselines/blob/ad61c79de17243f5befce069a1e9ca458057f43e/results/results_v4soft.md
[fvp]: https://github.com/ArmExecuTorchEEG/Emulator_Deploy/blob/b882ddc5ad2fbd28b6cc18da289cde1830052a4c/results/fvp_v4_metrics.md
[selftest]: https://github.com/ArmExecuTorchEEG/Edge_Deploy/blob/c616ea07e42926c6490d43a220f4bc0915b8f54f/results/fvp_selftest.md
[replay]: https://github.com/ArmExecuTorchEEG/Edge_Deploy/blob/c616ea07e42926c6490d43a220f4bc0915b8f54f/results/replay_accuracy.md

## Implementation follow-up — 2026-10-18

The release bundle and full genuine pool have since been recovered and verified.
The isolated streaming variant now passes full-pool FVP backend validation; the
DevKit-E8 application compiles. See `tools/multisubject_trials/IMPLEMENTATION_STATUS.md`
and `NUMERICAL_VALIDATION_REPORT.md` there for current status, including two
changed desktop tie decisions. No physical deployment has been performed.
