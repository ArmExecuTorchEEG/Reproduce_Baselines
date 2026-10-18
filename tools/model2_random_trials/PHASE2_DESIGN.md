# Phase 2 technical design — review before implementation

Status: design only. No firmware/frontend code changes, builds, serial opens,
resets, provisioning or board writes. Development branch remains
`feature/model2-random-eeg-trials`; validated UI fallback is
`demo/edgebci-model2-4trial` at `c85ab70f6030243283431f3a12b1a7ea87aa8b72`.

## Findings from the validated source

Original read-only firmware project:
`Reproduce_Baselines/artifacts/fvp-subject1-20261016-ready/project/`.
The original workspace's launch configuration selects this project's DevKit-E8
Debug AXF. The earlier physical validation report associates it with Model 2;
no new board-memory inspection was performed for this design.

- `src/eeg_vectors.h`: four constant `int8_t eeg_inputs[4][11000]`, alignment16.
- `src/app_main.cpp:77–102`: one static aligned writable 11,000-byte input,
  `TensorImpl(ScalarType::Char)` shape `[1,1,22,500]`, dimension order `[0,1,2,3]`.
  Each embedded vector is copied into the same buffer, then `module.forward(input)`
  runs inference. Output is checked for one `(1,2)` float tensor and finite logits.
  Prediction is `logit1 > logit0 ? 1 : 0` (ties choose0). It emits four EEG_RESULT
  lines, PASS/EOT, returns0 and reaches runtime exit. There is no input command loop.
- `src/arm_embedded_module.cpp`: program/method are loaded lazily and cached;
  forward sets inputs then executes the loaded method. Keep this instance alive
  for all trials. Copy output logits before the next call reuses runtime storage.
- DevKit board layer configures method/temp pools as1MiB/2MiB, alignment16 in
  `.bss.ai_pool`; these are configured capacities, not measured memory usage.
  Do not resize them or change model/tensor memory placement without evidence.
- DevKit `retarget_stdio.c` enables both RX and TX at115200,8N1,no flow control;
  generated RTE defines custom STDIN. Pins select UART4_RX_B P12_1 and UART4_TX_B
  P12_2, alternate function2. `RTE_UART4_BLOCKING_MODE_ENABLE=1`, DMA disabled.
- Pack USART driver has Receive/GetRxCount/abort/status paths, and stdin has a
  one-byte Receive wrapper. Current polling Receive waits for the full requested
  count with no application deadline. A half-sent tensor must never use that path.
- Frontend `serial-transport.js` opens115200/8N1 and takes only a readable stream.
  It sends no commands, accepts only trials0–3, and declares completion at four
  plus PASS. `playback.js` also completes at four; `recorded-eeg.js` intentionally
  accepts only the verified four-window profile. Streaming needs a separate mode,
  not removal of the existing demo protections.

The current firmware cannot accept arbitrary new tensors. A separately built,
loaded streaming firmware can reuse the SAME INT8 input buffer and inference call
without any reflash between trials. It still requires one future authorized
firmware deployment to replace the current four-vector application loop.

Board documentation confirms PRG USB is a USB-to-UART adapter switched to UART4
by SW4: `web/alifsemi-com/bd109175fb08`, AUGD0023 pp4,7 (indexed under Ensemble
pack2.2.1; no revision printed). SEGGER UM08001,2021 §7.17 p228 and §18.1.3 p357
(`web/updates-iar-com/569a5e9504c1`) documents VCOM TX and RX. Alongside the pin
configuration, this supports bidirectional feasibility. Generic external J-Link
connector pin numbers are NOT a DevKit wiring instruction. Existing physical
validation proves board-to-host TX; host-to-board RX has not yet been exercised.
Leave SW4 and J-Link settings unchanged. No adapter/probe firmware update proposed.

## Recommended minimal protocol (v1)

One outstanding inference, binary request/response frames, no pipelining, no model
or ground-truth transfer. Chrome remains the sole serial owner. Use Web Serial's
writable stream through one writer, with backpressure; never equate write completion
with board receipt. Keep binary bytes out of TextDecoder and the legacy line parser.

Each frame is `COBS(header + payload + CRC32) + 0x00`. COBS reserves zero as an
unambiguous delimiter even when INT8 payload includes every possible byte. Receiver
bounds both encoded and decoded sizes and drops malformed/oversized frames until
the next delimiter. A delimiter is also sent before a new session handshake.
CRC-32/ISO-HDLC covers the entire decoded header and payload, not its own4bytes:
reflected polynomial0xEDB88320, init0xFFFFFFFF, final xor0xFFFFFFFF; reference check
`123456789 -> 0xCBF43926`. CRC field is little-endian. This is error detection,
not authentication.

Fixed32-byte header, explicitly serialized little-endian (no packed-struct casts):

| Bytes | Field |
|---|---|
| 0–3 | Magic ASCII `EEG2` |
| 4 | Protocol version1 |
| 5 | Message type |
| 6–7 | Flags, must0 |
| 8–15 | Host session nonce, uint64 |
| 16–19 | Request sequence, uint32 |
| 20–23 | Dataset trial key, uint32 (0–143; control frames use0xFFFFFFFF) |
| 24–27 | Reserved, must0 |
| 28–29 | Payload length, uint16 |
| 30–31 | Reserved, must0 |

Trial key maps to the complete `S1/1test/runN/cueSAMPLE` identity through a hashed
canonical dataset index. Store/verify that mapping in the host session manifest.
Never use arrival position or predicted class as the identity. Before sending,
host checks the selected tensor's SHA-256 against its Phase1 trial record. Wire
INT8 is unchanged two's-complement bytes, channel-majorC order; no ASCII numbers,
base64, new normalization or re-quantization.

Messages:

1. HELLO: nonce + protocol negotiation; response READY echoes nonce, model PTE
   SHA-256, streaming firmware build ID, shape/dtype/quantization fingerprint,
   max payload11000, current boot/session state. Refuse legacy/unrecognized
   firmware or model/shape mismatch; never send tensor bytes to the four-trial image.
2. INFER: unique request ID, trial key, length exactly11000, complete tensor.
   Do not use another trial's bytes under the same key. Validate framing, length,
   CRC, session, monotonic ID, UART status and input shape before inference.
3. ACCEPTED: echoes session/request/trial and CRC32 of the11000 payload bytes.
   Emit only after full validated receipt. Duplicate result requests never count
   as another inference. Host still waits for RESULT before advancing.
4. RESULT: same IDs, payloadCRC32, two IEEE754 float32 little-endian logits,
   uint8 predicted class and3zero reserved bytes (16-byte payload). Sender and
   host both enforce finite logits, class0/1 and the existing tie rule.
   This is the streaming equivalent of EEG_RESULT; decoder forwards the same
   public prediction/logit interface after validation. Preserve legacy text
   EEG_RESULT parsing in four-trial mode. Save decoded records and original frames.
5. ERROR: same IDs when header is trustworthy; fixed numeric error code for
   bad length/CRC/version/session/busy/UART/timeout/inference/output failures.
   Corrupt headers do not justify echoing arbitrary IDs or invoking inference.
6. STATUS: same request ID, no input payload; queries in-progress/result/error.
   Cache the most recent completed response. Repeated same-ID/same-input frames
   return cached state/result, not duplicate execution; conflicting reuse rejects.
   Older IDs reject. Host does not automatically resend INFER on uncertainty.
7. LOG: optional bounded framed diagnostics; never insert raw printf text inside
   binary frames. Ignore nonprotocol boot noise until a valid READY handshake.

Unknown message types reject. Control-frame size is bounded separately (e.g256).
No generic memory writes, executable payloads, flash/reset/provision commands,
filesystem uploads or tensor shape changes exist in this protocol.

## Receive/execute lifecycle and deadlines

State machine: `WAIT_HELLO -> IDLE -> RECEIVING -> VALIDATING -> RUNNING -> IDLE`.
Keep exactly one aligned input and one model instance. Never execute partial data;
never overwrite the input during forward. Reuse output/response storage only after
copying logits into the cached response. Input valid flag clears on every error.

In the separate streaming board layer, replace UART4 blocking receive with an
interrupt-driven bounded receive adapter (CMSIS nonblocking receive + partial-byte
handling/rearming and small RX ring). Set UART4 blocking flag0 ONLY in that copy.
The existing retarget_config polling-only guard then cannot be reused unchanged:
replace streaming STDIN/STDOUT/STDERR adapters with a single UART owner that
initializes once, installs the callback and serializes protocol/log TX. Do not
edit cached vendor packs or original board config to bypass this guard globally.
Do not mix getchar/scanf with binary reception or reinitialize USART mid-session.
RX callback only buffers bytes/flags errors; CRC, parsing and inference run outside
interrupt context. RX must already be armed when READY is emitted, and partial
control frames must be visible without waiting for a full arbitrary receive count.

Suggested starting deadlines, tuned by measured behavior later:
- Partial receive:1s inter-byte inactivity and5s total frame duration; local monotonic
  timer aborts receive, invalidates buffer, and resynchronizes at delimiter.
- Host after writing INFER:5s for ACCEPTED. writer.write has its own bounded wait;
  timeout/disconnection stops scheduling and preserves pending IDs/bytes.
- After ACCEPTED:10s for RESULT initially; actual inference duration must be measured.
  A host timeout does NOT cancel a running NPU or prove inference failed. Stop
  new requests and query STATUS if link is still healthy; otherwise require user
  reconnect/review. Never reset/reflash automatically.
- Firmware RX timeout does not solve an NPU hang: module.forward is synchronous.
  Do not bolt on an automatic watchdog reset. Diagnose a stuck inference separately
  with CMSIS after authorization. Fatal runtime errors enter an error/idle state,
  not the bare-metal runtime exit trap. Transport faults do not create predictions.

For bounded COBS accumulation, allow an encoded frame buffer11200bytes plus a
small RX ring (e.g1KiB) and fixed metadata/cache, alongside existing11000input.
After receiving a full bounded frame, decode into the existing input only after
validating header bounds; no inference until CRC passes. No extra EEG tensor or
per-request heap allocation is needed in protocol code. Existing ExecuTorch result
vectors may still allocate; test steady-state usage over many cycles rather than
claim allocation-free runtime. Confirm linker/memory capacities with CMSIS build
artefact tools after building; do not infer headroom from pool constants.

## Bandwidth and demonstration cadence

115200 baud,8N1 is11520bytes/s. Payload alone takes11000×10/115200 =0.95486s.
With32header+4CRC, worst-case COBS overhead44bytes+delimiter, request <=11081bytes:
about0.962s. READY/ACCEPTED/RESULT add milliseconds of wire time plus USB/browser
scheduling. Budget roughly1.0–1.2s per upload, PLUS measured inference latency.
No inference latency is assumed from historical four-result bursts.

Keep one request in flight and, initially, finish the existing1.8s visualization
before sending the next. Total cycle roughly2.8–3.0s plus inference/USB overhead.
This is a plausible interactive prerecorded-trial demo at115200, not real-time
headset streaming. No baud-rate change is recommended before bidirectional tests.
Full-window framing is simpler than chunk ACKs at this size; if sustained RX tests
show overruns, stop and assess chunked flow control before board deployment.

## Frontend behavior, preserving Cindy's design and the fallback

Add a separately selected streaming mode behind existing controls; keep legacy
four-result/replay paths and public ArmEEGDemo/ArmEEGHardware APIs intact.
Required modules: bounded binary codec, one reader/writer transport, session
controller, verified144-trial asset provider and completed-result ledger. Existing
four-trial profile, scripts and replay remain available; do not merely remove the
0–3 guard or complete-at-four check and assume this provides streaming.
No layout, colors, figures or animation timing changes are needed. Minimal source/
status text can distinguish `Physical E8 inference · prerecorded Session2 EEG`,
`Saved UART replay`, `Sending`, `Waiting`, `Paused with request in flight`, errors.
Full EEG datasets remain local pending redistribution review, separate from Git.

- Play: handshake if needed; pull next verified trial from the recorded seeded
  sequence; save immutable pending metadata (sourceID/key,label,artifact,inputhash,
  cycle,attemptID,session); send one INFER, wait for real matching RESULT. Display
  that trial's genuine EEG and task cue paired with the returned hardware logits,
  prediction and cursor direction. Only then complete its1.8s visual playback and
  schedule the next. No desktop reference is used to supply a prediction.
- Pause: immediately block scheduling and freeze visualization. Let an already
  started upload/inference finish; do not truncate it or cancel the port reader.
  Retain matching result, defer its visual playback until Resume. Pause is local
  and does not claim to stop an executing NPU.
- Resume: retain seed, cycle and position; finish retained/in-flight work before
  selecting another trial. Exclude failed sends from evaluated counts.
- Reset: block scheduling; if in flight, defer reset until response/error or expose
  unresolved timeout. New local seed/session starts only after the old request is
  settled; old responses must not score in a new session. Clear session playback
  and counters, keep prior captured records exportable. Reset never resets board.
- After144selections, reshuffle deterministically using next cycle index. No fixed
  four/twelve limit. Keep all three artifact flags; never choose based on success.
- Total predictions = valid completed inference attempts, including repeats;
  unique evaluated = distinct sourceIDs with valid results. Show attempts accuracy
  separately from unique-first-result accuracy; duplicate responses add neither.
  If repeated predictions differ, retain/report disagreement instead of replacing
  an earlier score silently. A send, ACK, timeout or desktop result is not accuracy.

## Firmware changes to propose for a separate copy

1. Freeze current app/source/board/model/build metadata and known firmware binaries.
2. New isolated streaming firmware project/branch; keep embedded model bytes,
   model settings, NPU setup, allocation sections, tensor contract and forward path.
3. Add bounded frame/CRC codec and UART transport backend; change only copied
   board UART ownership/config as described above, retaining115200/8N1 routing.
4. Replace four-vector memcpy loop with valid-frame input reception and forward.
   Build/load method once; return to receive after response, not `_sys_exit`.
5. Add READY/ACCEPTED/RESULT/ERROR/STATUS and last-response caching; no unsolicited
   EOT or four-trial PASS in continuous mode. Finite FVP test harness can emit
   its own test completion status without changing streaming hardware behavior.
6. Validate output type/shape/finite values as today. Never change compiled PTE,
   weights, quantization, operators or frontend predictions to hide a failure.

These are proposed changes only; none has been implemented.

## Preservation and restoration gate before any deployment

The GitHub demo branch preserves frontend+four prerecorded display windows; it
DOES NOT by itself preserve the complete firmware toolchain/generated project.
Before deployment, create a checksum-verified local rollback archive outside the
original worktree containing source, board layer/config/linker, AI layer/model data,
pack/tool version locks, build descriptors, current HEX+AXF, reference inputs,
UART validation log and existing load recipe (no secrets/provisioning credentials).
Record exact core/image address scope from CMSIS build artefacts. Do not invoke
model exporter or provisioning to create it. Verify copied model/input hashes match
the earlier physical record; this design does not re-read current board memory.

Existing files inventoried read-only:
- DevKit Debug AXF3931576bytes SHA256
  `7058d5360f04b441cfd550fe0b1bfa0733437188ce6322b54f560c0437c8bf56`.
- DevKit Debug HEX529752bytes SHA256
  `fa5367b73db9c28be9b2ebffb337ec0aedd6d0b7513af2ba21bcc160bfc61b07`.
- `src/app_main.cpp` SHA256
  `2ba86940c7f121a40d5969b46926fb1a6bd55cdc6137bffa115b22991b51a7de`.
- Embedded model PTE13072bytes SHA256
  `644bf8865ed515ed41c5418070a0b147008c4326b9da34d3232884459d9b3298`.
- Four embedded input bytes SHA256
  `038c1f93ea0818fb2c2b21da6521506e38d222280a1178b09ec5ce3e28d9c8fb`.

Restoration is a separate user-authorized CMSIS load of the preserved four-trial
M55_HP application image using its existing address scope; no chip-wide erase,
SE update, provisioning, debugger stub change or new export. Verify image and
model/input hashes, reconnect capture, and reproduce all four known outputs at
0.01tolerance before calling rollback successful. Debugger reliability/load scope
must be established before first deployment; if not, defer deployment. Do not
promise rollback solely because a source branch exists. No archive/load is
performed as part of this design-only phase.

## Test plan and evidence of actual hardware inference

1. Host-only simulated serial: exact codec golden vectors/CRC, all256bytevalues,
   INT8 extremes/zeroes, frame chunk splits at every boundary, random fragmentation,
   merged frames, partial writes, backpressure, bootnoise, malformed COBS, overlength,
   bad version/CRC/IDs, missing delimiter, timeout, UART error, wrong model handshake,
   duplicate/stale responses, cache/status recovery, disconnect during send/result,
   pause during send/inference/animation, resume/reset with late replies. No failed
   case emits a prediction. Assert <=1outstanding request, payload checksums,
   metadata identity, no-repeat144cycles, separately correct unique/attempt scores.
   Use fake browser serial streams only; no physical port is needed or permitted.
2. Native C/C++ parser tests using same golden frames as JS; fuzz/truncation tests
   prove no forward invocation before complete valid CRC and bounds. Mock model
   invocation count validates duplicates/errors safely. Repeated cycles check heap,
   buffers, input reuse and output lifetimes. Test injection is explicitly simulated.
3. FVP in separate firmware copy, through CMSIS only after implementation approval:
   first reproduce original four reference outputs, then use the first6Session2
   IDs from fixed seed42 (no correctness-based selection), plus one artifact-flagged
   input if absent. Feed actual encoded frames through a test transport adapter
   sharing production decoder/dispatch; existing Corstone backend uses semihosting,
   not the physical E8 UART. Compare trialID, inputCRC, prediction and logits with
   Phase1 host Q/DQ references at absolute0.01tolerance. Validate corrupt/truncated
   input does not call forward. Extend to144and multi-cycle soak only after smoke
   success. FVP validates NPU+protocol application behavior, not electrical UART4
   reception, USB scheduling or real interrupt/FIFO timing. Record this limitation.
4. Only after separate deployment approval and rollback archive verification:
   deploy streaming application once via CMSIS; Chrome alone opens UART4. Verify
   model fingerprint/nonce handshake and short control exchange before sending
   the first11000bytes. No reset per trial. Run a few fixed new trials, then seeded
   random trials; verify accepted byte counts, returned inputCRC and matching IDs.
   If no response or CRC/UART/NPU error, retain logs and stop; no automated recovery.
5. Physical proof: archive timestamped outbound tensor hashes, nonce/session and
   request/trial IDs, board ACCEPTED/RESULT frames and response logits. The normal
   frontend must not load Q/DQ predictions as a fallback. Compare references in a
   separate validation path. As an optional separately authorized debug check,
   CMSIS breakpoint at forward can inspect the selected input hash and then resume;
   never halt in the middle of a UART transfer. Model/build hashes and fresh
   responses establish useful provenance, not cryptographic device attestation.

Approval requested next is implementation/testing in a separate firmware copy,
not board flashing. Hardware deployment/reset remains a later explicit gate.
