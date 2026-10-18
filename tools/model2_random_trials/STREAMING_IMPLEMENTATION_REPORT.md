# Streaming implementation and validation report

Working branch: `feature/model2-random-eeg-trials`. No commit/push, physical UART
open, debugger attach/detach, reset, flash, erase or provisioning was performed.
All CMSIS actions were explicitly routed to frontend VS Code window92823; original
window90388 and its J-Link session were left untouched.

## Status

Local streaming implementation, native/simulated tests, browser integration,
FVP inference and DevKit compile are complete. **Deployment is blocked by the
unchanged0.01logit tolerance.** Do not present this as a fully numerically validated
physical streaming system or deploy it without resolving/reviewing that failure.

| Evidence | Result |
|---|---|
| Original rollback-source integrity | 136/136files unchanged |
| Native C++ protocol, ASan/UBSan | Pass; malformed/oversized/CRC/session/timeout/duplicate cases +600request reuse |
| JS↔C++ simulated serial | Pass;160real EEG payloads through explicit mock inference,144unique, reshuffle |
| JS/Python selection identity | Same300seed42selections |
| Host failure tests | Pass; wrong model, checksum, NaN/class mismatch, timeout, disconnect, duplicate, reset-in-flight |
| Browser fake serial | Pass; six trials, actual prerecorded EEG/cues, pause/resume, reset/archive, disconnect |
| Four-trial saved-replay/browser regression | Pass; L/R/R/R, cues L/L/R/R,75%,22EEG channels |
| Cindy geometry/style comparison | Original desktop geometry/fonts/borders/backgrounds identical; responsive checks pass |
| FVP smoke | 11/11inference responses; logits bit-exact to saved references;2invalid frames rejected |
| FVP full pool | 151/151responses; IDs/CRCs correct; all predictions match saved Q/DQ |
| Unique held-out FVP accuracy | 114/144=79.1667%; includes all3artifact flags |
| Full-pool0.01logit tolerance | FAIL:40/144unique trials exceed threshold; max0.05260968208312988 |
| Reordered/repeated diagnostic | Same failing trial produces identical FVP logits as first input and repeated input |
| DevKit-E8/M55_HP firmware build | Pass, compile/link only; no load/run on hardware |

Full FVP run contains original4 + all144held-out in seed42order + first3repeated.
Repeated first3results match their earlier occurrences exactly. No trial was chosen
based on correctness. All151classifications agree with the saved quantized desktop
reference. The desktopfloat baseline remains115/144; quantized desktop and FVP are
114/144. Protocol/parser smoke success is distinct from numerical validation.

## Numerical discrepancy and next investigation

Output quantization scale in the ORIGINAL saved `quantized_graph.txt` is
0.052609603852033615, zero-point−5. All observed deviations are approximately one
output integer step; maximumfloat difference is0.05260968208312988. Forty unique
held-out trials are outside0.01; there are zero prediction mismatches. Input IDs,
lengths and CRCs agree, model bytes/quantization were unchanged, and a targeted
fresh-order test reproduced the same values. This is consistent with fixed-point
NPU/requantization behavior differing from hostfloat Q/DQ, but the exact operator
causing it has NOT been established. Do not claim a confirmed rounding root cause.

Example `S1/1test/run0/cue6623` (dataset index3):

- Saved host Q/DQ: `[3.0513570308685303, -2.6304802894592285]`, class0.
- FVP: `[3.10396671295166, -2.6830897331237793]`, class0.
- FVP values identical in the full sequence, as the first inference in a new
  simulation, and on an immediate repeated request with a new requestID.

The comparison tool intentionally returns nonzero when tolerance fails and writes
all per-trial evidence. No tolerance, labels, logits, model bytes, quantization or
reference file has been adjusted to make it pass. Recommended next step: inspect
compiled integer-vs-Q/DQ operator rounding using this minimal failing input and
existing compiler artefacts, with Model2kept immutable. Any future change to the
acceptance criterion requires explicit review; no physical approval is requested now.

## Preserved fallback and local evidence

`artifacts/streaming-rollback/validated-four-trial.tar.gz`:1,867,882bytes,
SHA256 `90434792bdc757e225c78136e1c5261dd56acb6e3d5a51a1b8fc192d73c36015`.
`manifest.json` lists all136member hashes; archive contents were read back and
verified and source hashes were rechecked after implementation. Includes validated
source/board/model packages, original AXF/HEX, deployment configuration, pinned
versions and physical four-trial UART evidence. Original worktree unchanged.

`artifacts/streaming-firmware/` (ignored/local):

- `project/`: isolated model/board/runtime copy plus new streaming sources.
- `fvp-11-trial.log`, `fvp-11-comparison.json`: smoke evidence.
- `fvp-151-trial.log`, `expected-151.json`, `fvp-151-comparison.json`: full evidence.
- `fvp-diagnostic-trial3.log`, `fvp-diagnostic-comparison.json`: stable mismatch.
- `deployment-manifest.json`: final DevKit binary/source checksums and closed gate.
- `parser-tests`: native sanitized test executable.

Current `project/fvp-stream.log` and `expected.json` describe the last targeted
2request diagnostic. Aggregate full-pool comparison is separately preserved; do
not compare that targeted log against the full151expected manifest. Regenerate
matching fixtures before a new comparison run.

## Implementation files

- `firmware/streaming/{protocol.hpp,app_main.cpp,uart_stream.c,fvp-capture.sh,README.md}`.
- `frontend/streaming/{protocol.js,controller.js,dataset.js,bridge.js,data/.gitignore}`.
- Guarded extension points in `frontend/{index.html,gui.js,playback.js}` only;
  no stylesheet, human/scalp SVG, ground-truth mapping or old transport edits.
- `tests/streaming/{native_tests.cpp,host-tests.cjs,failure-tests.cjs,browser-stream.mjs,compare-fvp.cjs}`.
- `tools/model2_random_trials/{create_streaming_project.py,prepare_streaming_assets.py}`
  plus existing Phase1tools/design/report files.
- `.vscode/` contains generated local CMSIS configuration and proven tool PATH.
  Treat as local environment material, not part of an automatic future commit.

Frontend EEG assets are7,920,000binary bytes plus metadata, all ignored under
`frontend/streaming/data/`. They contain no expected predictions or model evaluator.
No large EEG data was staged/committed/pushed; redistribution review is outstanding.

## Future deployment (not authorized)

Full procedure is in `firmware/streaming/README.md`. Necessary preconditions:
resolve/review numerical gate, verify rollback, final image checksums, ensure CMSIS
connection/load scope is reliable, and obtain explicit authorization for the exact
application load/start operation. Do not touch original debugger/provisioning.

Current isolated M55_HP HEX416012bytes SHA256
`84ca339645e211bffdeed17d60d6f2fee7e448829b3e44506e4da81394a13afa`;
AXF3919212bytes SHA256
`41f6b09e18ef63bb9913302aed2d5d3d1c1aec6af8b0c023dc61bcd727a80efb`.
CMSIS section inspection places application vectors at0x80200000 and ROM at
0x802007c0. Image file content151248bytes; use generated application-only load
recipe. Do NOT treat the ELF memsz3,509,992bytes as an MRAM erase/write extent:
it also includes runtime memory. CMSIS memory check: inference pools3MiB unchanged;
other static RW_RAM49,944bytes; main stack32KiB, runtime heap96KiB+app heap32KiB.

Physical test would use Chrome alone at115200/8N1: HELLO/READY while paused, then
one small seeded sequence; compare real results using a separate reference path.
Stop on failure; never automatically reset/reflash/retry. FVP transport injects
encoded data and exercises production parsing/NPU dispatch, but does not validate
physical UART IRQ latency, ring overflow margins, USB bandwidth or cache behavior
on the real E8. Long unattended operation and RAM capture growth are not qualified.

Legacy URL: `http://127.0.0.1:8000/?v=eeg-1`.
Streaming preview: `http://127.0.0.1:8000/?stream=1` (do not connect yet).
The demo branch remains exactly `c85ab70f6030243283431f3a12b1a7ea87aa8b72`.

## Numerical investigation addendum

The subsequent investigation is in [NUMERICAL_VALIDATION_REPORT.md](NUMERICAL_VALIDATION_REPORT.md).
It establishes exact TOSA/FVP integer agreement for all288output values and
identifies the two average-pool rounding rules as sufficient to explain all
40Q/DQdifferences. Fresh12case FVP order/repeat tests pass. The original0.01
Q/DQdiagnostic remains failed as reported above. A stricter backend-specific
integer-reference criterion is proposed for explicit user review; no production
model, saved reference, acceptance policy, hardware or firmware was changed.
