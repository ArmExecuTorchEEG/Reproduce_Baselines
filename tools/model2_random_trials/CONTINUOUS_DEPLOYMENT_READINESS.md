# Continuous physical streaming deployment preparation

This plan supersedes the optional five-trial console guard in PREDEPLOYMENT_REVIEW.md.
No firmware/frontend code changes, console guard, fixed sequence of five, or fixed
trial limit are required. Existing seed-42 reproducible randomized selection runs
through all 144 real trial IDs without replacement, then reshuffles each cycle.
References are separate validation evidence, never browser prediction substitutes.

Verified: preserved archive and 136 original-source checksums; streaming image/source
manifest; 144 trial identities and both full binary asset hashes; local HTTP page,
controller, bridge and metadata; offline 160-request test across reshuffle and
pause/in-flight/reset/error tests. Existing FVP/TOSA validation remains applicable.
Preserved Git commit: a7efe5046bb0f34cdd64ad09612de547647dd08f.

CMSIS physical inspection: M55_HP JLink attach is DAP-responsive, stopped at SIGTRAP,
with the original worktree's DevKit-E8 application and CMSIS Run holding the probe.
SEGGER serial device detected: /dev/tty.usbmodem0012191658361.
No halt/resume/reset, detach, port open, load, flash or provisioning was performed.
The existing stopped state was preserved. Serial detection is not a successful
bidirectional physical protocol test.

Prepared image: artifacts/streaming-firmware/project/out/cmsis-executorch/DevKit-E8/Debug/cmsis-executorch.hex
SHA256: 84ca339645e211bffdeed17d60d6f2fee7e448829b3e44506e4da81394a13afa
Target DevKit-E8 / M55_HP, J-Link SWD 4MHz, application vector base 0x80200000.
Model PTE SHA256: 644bf8865ed515ed41c5418070a0b147008c4326b9da34d3232884459d9b3298.
UART4: 115200/8N1/no flow control. No exporter, rebuild or provisioning is needed.

After explicit deployment authorization, recheck session/image, release the old
attach/CMSIS Run ownership through CMSIS, select the isolated DevKit-E8 solution,
and perform one application-only load/start. The existing generated recipe contains
system-reset behavior; authorization must cover the required load/start reset.
No chip erase, provisioning, SE/debug-stub changes or automatic retry/recovery.
Stop if actual tooling requires a different programming path or reset scope.

Chrome exclusively owns UART: use http://127.0.0.1:8000/?stream=1 in a fresh tab,
release any other tab/Serial Monitor reader first, Connect UART selects SEGGER,
then verify READY model SHA, tensor shape and quantization before Play.
Do not install the old console guard. Play sends one actual input and waits for
ACCEPTED/RESULT before visualization and another input. Pause stops new requests;
an in-flight result is retained. Resume continues the current sequence. Reset
archives results and starts a new sequence; it is not a physical board reset.
Ground truths remain independent; prerecorded EEG is identified as such.

Observe the first few continuous results, then press Pause for review (no automatic
count cap). Export capture before closing/reloading. Compare each by full source ID,
request/session IDs and input CRC/hash against saved TOSA/FVP raw and float outputs.
Require finite grid-consistent logits, 0.01 absolute float agreement, matching class,
and retain original Q/DQ 0.01 diagnostic separately. Report unique and total attempts.
Only resume prolonged operation after those physical results have been reviewed.
No response means stop, preserve evidence, and investigate; never invent results.

Rollback archive and original image/configuration are verified as documented in
PREDEPLOYMENT_REVIEW.md and validation/rollback-verification.json. An actual rollback
is a separate authorized board operation. Continuous UART throughput, physical NPU
results and sustained memory behavior remain unverified until deployment/test.

Status: preparation complete; physical deployment NOT authorized.

## Physical startup correction and verified results

The earlier load_and_run result was insufficient: the generated CMSIS Load task
only performs Connect/LoadFile/Exit; CMSIS Run starts a server with -nohalt, not an
explicit application reset-and-continue. An existing halted/exit state can persist.
The active Run task subsequently had no listening server on port3333.

Corrected and physically verified sequence: CMSIS stop_run, load_and_debug on the
isolated DevKit-E8 context, wait for main, then clear diagnostic breakpoints and
continue_execution. Debugger inspection confirms app_main's receive loop, running
millisecond tick and no UART fault. No model/firmware arithmetic change was needed;
no firmware rebuild was needed because the image itself was unchanged. Do not call
a successful load or active task alone proof of startup; require HELLO/READY.

With Chrome explicitly disconnected, CMSIS serial tools sent HELLO and received
a CRC-valid READY with the expected model/shape/quantization. Three real INT8
windows (seed42 indices71,47,12) produced matching ACCEPTED/RESULT IDs and input
CRCs. Actual logits were float32-exact to the saved FVP references (max error0),
predictions0,0,1. Raw wire evidence and comparisons are in the ignored local file
artifacts/streaming-firmware/physical-verified-results.json. This was actual E8
inference, not mock/FVP execution. The serial port was closed afterward.

Frontend status correction: no connected badge before READY; connection errors
survive Play; progress is visible. On successful READY reinitialize the fresh
visualization session so pre-handshake disconnected status cannot mark it complete.
Continuous Chrome playback remains subject to verification from its own capture.
