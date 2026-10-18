# Physical Model 2 sustained streaming validation

The isolated streaming variant completed 160 actual E8 requests, not FVP or host
inference: all144 held-out Session2recordings once, then16 after reshuffling.
Selection: sha256-sort-v1, seed20261017. One request at a time, no retransmission,
reset or reflash during this successful run. Session2026101702.

All320 returned float32 logits match FVP exactly (maximum difference0); all160
predictions agree. Request/session/trial identifiers and input CRCs are checked.
Truth accuracy:114/144unique recordings (79.1667%),129/160total attempts (80.625%).
The16 repeat predictions agree with their first results. The difference in attempt
accuracy reflects which recordings repeat, not an improved model. All three
artifact-flagged recordings remain in the pool. No selection on correctness.

Raw READY and response frames, per-trial IDs/logits/labels and counter readings:
`artifacts/streaming-firmware/physical-sustained-160.json` (local ignored evidence).
File/image fingerprints: `artifacts/streaming-firmware/physical-sustained-manifest.json`.
The exact output quantization grid and original Q/DQ0.01diagnostic remain as
approved in NUMERICAL_VALIDATION_REPORT.md; this run does not alter that threshold.

## UART correction and limits of attribution

Old adapter counted ARM_USART_EVENT_RX_TIMEOUT as a fatal error, aborting receive
and invalidating the frame. Installed Alif Driver_USART.c/uart.c explicitly uses
that event for FIFO-empty/character gaps while reception remains busy. The
streaming-only adapter now counts idle events without aborting. Genuine UART
errors and ring overflow still invalidate; protocol deadlines remain unchanged.
Additional bounded counters distinguish idle/errors/rearm/overflow.

The first sustained attempt completed11inferences then timed out on request12,
trial76. Receiver inspection showed dropping=true,rejected=7,timeouts=0 while
app_main remained responsive. Earlier Chrome capture completed7inferences and
failed request8. Old firmware lacked reason counters, so the exact old fault
source cannot be conclusively assigned retrospectively to RX_TIMEOUT. The change
corrects an identified handling bug; a160request pass establishes observed
reliability, not a universal proof of the earlier failure mechanism.
After the successful run all four UART counters, receiver.rejected and
receiver.timeouts were0. No positive idle-event count occurred in this run.

## Deployment verification

An earlier CMSIS load task reported success while target connection failed; direct
flash inspection later showed old event instructions. Successful task exit alone
is insufficient. After power recovery, d-23load_and_debug was followed by a read
of0x8021de00:64bytes matched the corrected HEX event code. Continue explicitly,
confirm app_main receive loop, then require a model/shape/quantization-valid READY.
The decoder harness initially reported malformed COBS while collecting the first
handshake; a subsequent complete107byte READY was CRC/model validated before any
EEG request. No model or tensor preprocessing changes were made.

Native parser600request soak and JS failure tests passed; DevKit-E8 build b-20
passed. `python3 tests/streaming/uart-event-tests.py` compiles the actual UART C
adapter against a mocked driver under ASan/UBSan and checks idle, byte reception,
framing-error invalidation and ring overflow. This adapter is hardware-only; FVP
cannot exercise its physical UART behavior. The existing FVP references are reused.

Board left running; CMSIS serial closed. Chrome continuous playback and its actual
visual synchronization must be assessed separately from this CMSIS-serial test.
Cindy design files and the trained model are unchanged. Original fallback archive
and all136original files were reverified unchanged. No commit or push performed.

## Chrome end-to-end follow-up

The second exported Chrome capture contains12archived results and9current results,
all21unique across sessions. All21actual predictions and42logits match FVP exactly;
21INFER/ACCEPTED/RESULT triples pass source-ID, session, request, inputSHA/CRC and
response checks. Both seeded sequences reproduce exactly. No protocol ERROR or
parser failure. Current session is ready and paused, no pending request/error,
with8/9correct; across captured attempts19/21correct. OriginalQ/DQ0.01diagnostic
fails3/21and is retained; no tolerance changed. Reset preserved the12archived
results and started a new sequence. User separately confirmed the EEG/person/cursor
visuals work well; wire capture alone does not verify animation rendering.
Local preserved capture/comparison: artifacts/streaming-firmware/chrome-verified-21/.
No new physical actions, UART access, commit or push occurred during this review.

## Source preservation

Compact physical output evidence is also retained in tracked
`validation/physical-sustained-160.json` and `validation/chrome-physical-validation.json`.
These contain result metadata and returned outputs, not input EEG recordings.
`validation/physical-streaming-build.json` fingerprints the tested image and UART
source; it does not embed generated binaries. Raw captures remain local-only.
The successful corrected image was deployed with d-23 after the earlier failed
connection attempt; prior manifests remain historical and must not identify the
new image. Use the physical-streaming manifest for this tested firmware.

## Preservation regression run

Before the physical-validation commit, all23repository unittest tests passed,
along with3trial-selection tests, fresh ASan/UBSan C++ parser600request soak,
160request JS/C++ simulated transport test, UART callback regression, connection
status regression, failure handling, saved UART playback, genuine recorded EEG
and Cindy-design fidelity checks. `pytest` could not run because the selected
local Python environment has no pytest module; no dependency was installed.
The documented unittest discovery command completed successfully. These checks
were offline; no board/debugger/serial tool was used for source preservation.
All136original fallback files, archive, supplemental rollback recipe, tested
streaming image/source and installed UART dependency fingerprints were reverified.
