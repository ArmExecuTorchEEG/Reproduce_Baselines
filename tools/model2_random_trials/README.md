# Model 2 held-out Session 2 preparation — Phase 1

Branch: `feature/model2-random-eeg-trials`. Validated demo base remains
`c85ab70f6030243283431f3a12b1a7ea87aa8b72` on `demo/edgebci-model2-4trial`.
No frontend, model, firmware, serial or debugger changes are part of Phase 1.

## Eligibility and scope

144 real Participant 1 Session 2 (`A01E.mat`, session `1test`) left/right windows:
72 left (0), 72 right (1). All 3 artifact-flagged windows remain eligible under the
original retain-all policy. Model 2 used 115 Session 1 training windows and 29
Session 1 checkpoint-selection windows. Its pipeline loads Session 2 only after
checkpoint selection. Session 2 has no trial-identity overlap with either split
or the training-only quantization calibration set. These are **weight-training
unseen**, but were already used for the original final evaluation; they are not
an untouched test set or evidence from a new independent evaluation.

Source MAT SHA-256:
`53d415f39c3d7b0c88b894d7b08d99bcdfe855ede63831d3691af1a45607fb62`.
Model PTE SHA-256:
`644bf8865ed515ed41c5418070a0b147008c4326b9da34d3232884459d9b3298`.
Checkpoint SHA-256:
`63e3479baa0b00cadce5ee7906f25eee873ba561f9040cd07505f5d32b7a51d9`.

The preparation script reads the original worktree, imports its `extract_session`
implementation, and uses the same MOABB MAT converter and annotation mapping.
It never calls dataset downloading, training, export, calibration or hardware APIs.
Run Python with `-B` so imports cannot write bytecode into the original worktree.
Matplotlib cache is directed to `/private/tmp/edgebci-phase1/matplotlib`.

## Exact processing contract

250 Hz; cue-relative [1,3) seconds; 22 channels × 500 samples. Source microvolts
are converted to volts. Causal Butterworth N=4 prototype bandpass (8th-order),
7–30 Hz, saved SOS; zero filter state at cue−2 s, three seconds warm-up before
retained window. Saved Session 1 training-only means/population standard
deviations are reused unchanged. Normalized float32 arrays match the original
saved Session 2 tensors bit for bit; labels and complete trial metadata also match.

Channel order:
`Fz FC3 FC1 FCz FC2 FC4 C5 C3 C1 Cz C2 C4 C6 CP3 CP1 CPz CP2 CP4 P1 Pz P2 POz`.

Signed INT8 quantization uses original torch float32 ties-to-even rounding:
`clip(round(normalized_float / 0.04422667250037193) - 10, -128, 127)`.
One trial is `(1,1,22,500)`, contiguous channel-major C order, 11,000 bytes.
23 of 1,584,000 samples exceed the representable range before clipping.
No normalization or scale is fitted on Session 2.

Display buffers are genuine **filtered microvolts**, float32, before channel
normalization or INT8 quantization. Normalized float32 buffers are also retained.
Future scalp rendering can use per-channel `sqrt(mean(filtered_uv**2))` on the
same trial; relative RMS is a display statistic, not cortical localization or a
clinical measurement. These assets are prerecorded and must stay labeled so.

## Outputs (local, Git-ignored)

`artifacts/model2-session2-phase1/`:

- `windows.npz` — 12,956,004 bytes: `inputs_int8`, `normalized_float32`,
  `filtered_uv_float32`, `labels`, `float_logits`. Trial dimension is 144;
  use `inputs_int8[i:i+1]` for the required single-model-input shape.
- `manifest.json` — 81,203 bytes: original file hashes/sizes, preprocessing,
  versions, split checks, artifact audit, trial identities/labels, each input
  SHA-256, float logits and predictions, dataset checksum.
- `qdq-reference.npz` — 1,393 bytes: expected desktop quantized logits,
  predictions, labels.
- `qdq-validation.json` — 23,873 bytes: per-trial quantized references, metrics,
  environment versions and reference checksum.
- `selection-seed42-300.jsonl` — 74,222 bytes: reproducible example of more than
  two complete cycles, including cycle, ordinal, source ID, index, truth and hash.

Full source recordings and prepared datasets remain local. Redistribution/license
review is pending before publishing EEG assets; nothing is committed or pushed.
Do not force-add these ignored files. Tools and this documentation contain no EEG
sample arrays. The validated four-window frontend remains unchanged.

## Validation results

- Float: **115/144, 79.8611%**, confusion matrix `[[48,24],[5,67]]`.
  All predictions match saved final evaluation; maximum float logit difference
  2.3841858e-6 from single-window versus original batched evaluation.
- Saved host PT2E Q/DQ: **114/144, 79.1667%**, matrix `[[48,24],[6,66]]`.
- Float/QDQ classification agreement: **143/144**. Only difference:
  `S1/1test/run2/cue71202`, truth 1, float predicts 1, Q/DQ predicts 0.
- Dequantizing prepared INT8 inputs using saved scale/zero-point and evaluating
  the same Q/DQ model gives **bit-exact logits** to the normalized-input path.
- Saved Q/DQ development references rechecked to tolerance 1e-6.
- Independent NumPy quantization, all 144 input checksums, display consistency,
  trial identities and labels, and three full selection cycles pass.
- Selector unit tests cover 593 selections across cycles, exact state resume,
  reset, dataset identity mismatch and invalid input.
- Existing frontend integration, recorded-EEG and Cindy design tests pass.

These are desktop model references, **not new FVP or physical Alif E8 results**.
No hardware logit tolerance claim is made for these 144 inputs yet.

## Reproduce

From the frontend development worktree, set the original worktree path:

```sh
BASELINE_REPO='/path/to/original/Reproduce_Baselines'
# Choose a NEW output directory: preparation refuses overwrites.
"$BASELINE_REPO/.venv/bin/python" -B tools/model2_random_trials/prepare_session2.py prepare \
  --baseline-repo "$BASELINE_REPO" --out-dir artifacts/model2-session2-phase1
# Existing official Arm environment; do not install its packages into baseline venv.
/private/tmp/tinyeegnet-cmsis-executorch/.venv/bin/python -B \
  tools/model2_random_trials/prepare_session2.py validate-qdq \
  --baseline-repo "$BASELINE_REPO" --out-dir artifacts/model2-session2-phase1
"$BASELINE_REPO/.venv/bin/python" -B tools/model2_random_trials/verify_prepared.py \
  artifacts/model2-session2-phase1
python3 -B tools/model2_random_trials/test_select_trials.py
python3 -B tools/model2_random_trials/select_trials.py \
  --manifest artifacts/model2-session2-phase1/manifest.json --seed 42 --count 300
```

Dependencies are available locally: baseline Python environment supplies torch,
NumPy, SciPy, MNE, MOABB and scikit-learn. Q/DQ uses the existing official Arm
environment (torch 2.14.0, torchao 0.18.0, ExecuTorch 1.5.1); its paths and source
artifacts must be restored separately on another machine. The provenance manifest
records exact preparation versions. No dependency installation was required.

## Selection and future execution semantics

`TrialSequence` orders verified source IDs by SHA-256(seed, cycle, ID), with ID
as a collision tie-breaker. This is deterministic pseudorandom ordering, has no
fixed trial limit, includes every ID once per 144-selection cycle, and reshuffles
on the next cycle. A new explicit seed creates a new sequence. Algorithm version,
seed, dataset-ID digest and consumed position are serializable; resume rejects a
different dataset/order. This is a data-selection utility, not a streaming UI.
It reads no serial port and generates no EEG signals or predictions.

The future controller must consume selections only when ready, keep one request
in flight, wait for its real hardware result, and match responses by trial and
request IDs. Pause must stop new requests while safely finishing an in-flight
request; resume must retain sequence position. Reset must record a new seed and
clear session state. Cycle/occurrence counters distinguish repeated source trials.
A completed-result ledger should separately report unique source trials evaluated
and all inference attempts/repeated trials; this selector intentionally computes
no accuracy from merely selected or sent trials. Neither that controller nor a
Random Trial button or firmware protocol is implemented in Phase 1. Hardware
transport investigation is deferred until review.
