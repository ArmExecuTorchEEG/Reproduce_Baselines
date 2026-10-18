# Preservation and initial hardware test review

No commit, push, port access or physical action is authorized by this document.
Branch: `feature/model2-random-eeg-trials`. Fallback frontend branch:
`demo/edgebci-model2-4trial`, commit `c85ab70f6030243283431f3a12b1a7ea87aa8b72`.

## Proposed commit

`EdgeBCI: preserve Model 2 random EEG streaming and validation`

Body: Preserve the isolated firmware, host controller and backend validation before controlled physical testing; keep the validated four-trial fallback unchanged.

The exact selection is in COMMIT_FILES.txt. No staging, commit or push has occurred.
Cindy's original design references and styles remain unchanged. The three existing
frontend files contain guarded streaming hooks. All other proposed files are new.
Excluded: .vscode, datasets, NPZ/intermediate tensors, streaming data assets,
virtual environments, build outputs, HEX/AXF, rollback archive and local caches.
Small per-trial output/reference metadata is included, not EEG recordings.
The ignored local assets and licensed Arm tools are required to reproduce this
working installation; this source commit alone is not a self-contained firmware binary backup.

## Verified fallback and rollback

`validation/rollback-verification.json` records 136 archive-member, original-source
and extracted-file checks. Archive SHA256:
`90434792bdc757e225c78136e1c5261dd56acb6e3d5a51a1b8fc192d73c36015`.
The archive remains at `artifacts/streaming-rollback/validated-four-trial.tar.gz`.
Verified extraction: `artifacts/rollback-four-trial-verified/`.
Original HEX SHA256: `fa5367b73db9c28be9b2ebffb337ec0aedd6d0b7513af2ba21bcc160bfc61b07`.
Original AXF SHA256: `7058d5360f04b441cfd550fe0b1bfa0733437188ce6322b54f560c0437c8bf56`.
The separately preserved `validation/validated-four-trial.cbuild-run.yml` is copied
into the extraction's original `project/out/` directory. Its relative AXF/HEX paths
resolve to those verified originals. Source, board configuration, model and original
UART validation evidence are included in the archive. Original worktree is untouched.

If streaming fails, stop and export capture; do not automatically reset/recover.
After separate rollback authorization:
1. Disconnect Chrome's streaming reader, retaining/exporting results.
2. Recheck archive/image hashes; use the verified extraction, never overwrite the original worktree.
3. Select its `artifacts/fvp-subject1-20261016-ready/project/cmsis-executorch.csolution.yml`
   with CMSIS Developer Assistant, target DevKit-E8 / M55_HP. Preserve the copied
   cbuild-run recipe. Do not rebuild or invoke provisioning/SE/debug-stub tasks.
4. Open the legacy frontend `http://127.0.0.1:8000/?v=eeg-1`, select the verified
   four-trial metadata and Connect UART before the authorized application load/start.
5. Use CMSIS to load the original HEX (image) and AXF (symbols), with startup/reset
   scope explicitly approved in that rollback authorization. No chip erase.
6. Verify four predictions `[0,1,1,1]`, saved logits within 0.01, and original
   `Execution_result: PASS (...)`. Stop on failure; no automatic second attempt.

Image/configuration restoration is prepared and hash-verified. Actual debugger
connection, load and recovery have NOT been tested in this phase. The binary
archive is local-only; GitHub will preserve source/evidence, not this ignored archive.
Keep an independent copy of the archive and supplemental recipe for machine-loss protection.

## Proposed controlled deployment (requires separate explicit approval)

Use only the isolated streaming DevKit-E8 image in validation/streaming-image-manifest.json.
HEX SHA256: `84ca339645e211bffdeed17d60d6f2fee7e448829b3e44506e4da81394a13afa`.
Application vector base: `0x80200000`. CMSIS application-only load/start; no erase,
provisioning or SE/debug-stub changes. Inspect the debugger state first. If loading
requires a different reset scope or recovery, stop for review instead of retrying.
The existing recipe contains a system reset sequence: approval must explicitly
cover the actual load/start reset behavior; it must not be described as reset-free.

Chrome is the sole serial owner. Disconnect any old frontend tab through its UI;
close Serial Monitor without terminating unrelated processes. Keep SW4 on UART4.
Open `http://127.0.0.1:8000/?stream=1`. Connect UART selects SEGGER at
115200 baud, 8N1, no flow control. HELLO/READY must verify the Model 2 SHA,
shape, scale and zero point before Play. No prediction is sent/computed by the laptop.
If the handshake fails, stop and export evidence; do not reopen/reset automatically.

## Five-trial test

`validation/initial-hardware-test.json` records full source IDs, ground truths,
input hashes, TOSA integers and FVP logits for the first five seed-42 trials.
No results-based selection; no production trial-count limit is introduced.
On a fresh connected paused streaming page, before pressing Play, install this
reviewed DevTools guard (do not run it against the old firmware):

```js
const c = window.EdgeStreamingController;
if (!c?.ready || !c.paused || c.pending || c.records.length || c.seed !== 42 || c.sequence.position !== 0)
  throw new Error('Require a fresh, verified seed-42 streaming session');
const previous = c.onRecord;
c.onRecord = function(record) {
  previous(record);
  if (c.records.length >= 5) c.pause();
};
```

Press Play once. The guard pauses immediately on result five, preventing request
six; all five actual results remain captured. The fifth visualization may remain
paused in the queue. Do NOT press Play again during this bounded test: that would
resume continuous requests. Review its waveform/label and all five returned results
from retained capture; full uninterrupted continuous visualization remains a later
hardware test. Pause at any earlier point is safe; an in-flight inference may finish.

Export capture before reload/disconnection. Check exactly five unique IDs and five
ACCEPTED/RESULT pairs with matching request/session/trial and input CRCs. Compare
finite returned floats with the corresponding TOSA/FVP reference at absolute 0.01,
verify output quantization grid and predicted class separately. Retain the original
Q/DQ 0.01 diagnostic even when it fails. No mock/replay is permitted in this test.
Stop on any timeout, corruption, disconnect, model mismatch or unexpected request.
The browser carries no expected logits; validation references are separate review
metadata. Actual board responses, not laptop evaluation, populate the UI.
