# TinyEEGNet FVP integration — pending user license activation

Actual embedded inference has NOT yet run for either model. No firmware has been
built. There is no actual FVP prediction agreement or numerical-error result.

## Official resources and compatibility

Reviewed all requested resources:
- https://arm-executorch.devpost.com/resources recommends the hackathon branch.
- https://github.com/Arm-Examples/CMSIS-Executorch/tree/hackathon
  pinned snapshot `a6364946165f21b27c1f5cf6998a701ded8ed73e`.
- https://github.com/alifsemi/alif_ml-embedded-evaluation-kit and
  `docs/deploying_model_executorch_export.md` (its 1.0 example must not replace
  the matching 1.5.1 exporter/runtime).

The hackathon branch's documentation/example.md explicitly shares one
Ensemble SRAM/MRAM-profile PTE between E8 and FVP: same U85, 256 MACs,
Ethos_U85_SRAM_MRAM, Shared_Sram. Existing Alif PTEs should therefore be usable
without re-export, provided the matching runtime, driver and memory integration
are built. This is documented compatibility, not execution evidence.
An FVP loads compiled Cortex-M firmware (.axf), not a PTE file directly.

Both models were exported with ExecuTorch 1.5.1, PyTorch 2.14.0, torchao 0.18.0,
Vela 5.1.0, TOSA tools 2026.5.0. Use `PyTorch::ExecuTorch@1.5.1` for the runtime.
The official project pins CMSIS-Toolbox 2.15.0, Arm Compiler 6.24.0,
FVP 11.32.23, CMake 4.2.1, Ninja 1.13.2, and offers GNU 14.3.1/LLVM 22.1.0.
The supplied solution selects AC6. It requests ARM::SSE_320_BSP@1.0.0,
ARM::ethos-u-core-driver@^1.25.2 and AlifSemiconductor::Ensemble@^2.2.0-0.
Resolved versions of currently missing packs cannot yet be reported.

## Host and installation

Host: Apple Silicon arm64, macOS 15.6.1 (24G90). No native macOS FVP is
provided by Arm; the official wrapper uses a Linux arm64 Docker container.
Initial PATH had clang but no Docker/cbuild/Arm cross-compiler/FVP. An old Docker
app initially appeared, its CLI failed, and the application subsequently was
absent; no existing Docker installation was intentionally deleted by this task.

After explicit user authorization to install, leaving licenses to the user:
- Installed Keil Studio Pack 1.20.3, Arm Environment Manager 1.26.0,
  CMSIS Solution 1.72.0, CMSIS Debugger 1.8.0 and extension dependencies.
- Installed CMSIS-Toolbox 2.15.0 under
  `artifacts/fvp-tools-20261016/cmsis-toolbox-darwin-arm64/`; cbuild --version works.
  Download SHA256 matches official release checksum:
  `f43f929bb661ef02de0e2d436d03a0de899c3e5f5a2e7988f7e4ca705c56b5bd`.
- Downloaded/signature-verified and installed Docker Desktop 4.95.0 in
  `/Applications/Docker.app`, without sudo or --accept-license; launched it.
- Opened the staged project in VS Code for user-controlled tool activation.
- Did NOT accept Docker, MDK or FVP licenses or activate an Arm account.
- Did NOT build the Docker FVP image: official fvp.sh treats first build/use as
  EULA acceptance, so it awaits the user's explicit confirmation.

First cbuild setup without --packs failed because these packs are not installed:
ARM::CMSIS, CMSIS-Compiler, CMSIS-NN, Cortex_DFP, SSE_320_BSP,
ethos-u-core-driver, AlifSemiconductor::Ensemble, PyTorch::ExecuTorch.
It also reported no registered compiler and could not resolve the SSE-320 board.
These are environment blockers, not discovered model/operator failures.

## Prepared Model 2 harness

Location: `artifacts/fvp-subject1-20261016-ready/project/` (ignored generated
workspace). The official board support, driver, EmbeddedModule runtime wrapper,
linker configuration and Docker FVP shim remain reused. Only the staged example
application, AI layer and solution input/debug fields were adapted.

- Original subject-specific PTE copied byte-for-byte; source SHA256 recorded in
  `reference.json`. Original model artifacts/checkpoints/parameters unchanged.
- Four development-validation windows, two from each class, exclusively
  Participant 1 Session 1. No Session 2, S6–7 reserved sessions or S9 used.
- Input signed int8 NCHW (1,1,22,500), 11,000 bytes/window, contiguous.
  Existing saved normalization and input quantization scale/zero point apply.
- Host Q/DQ reference evaluates the reconstructed quantized input bytes, so
  firmware and host compare exactly the same input values.
- Output float32 (1,2); argmax 0=left_hand, 1=right_hand; ties choose 0.
- Firmware prints `EEG_RESULT <trial> <left_logit> <right_logit> <class>` and
  rejects wrong shapes/types or non-finite outputs. Its execution-success banner
  is separate from numerical validation, which the Python comparator performs.
- All four convolutions, padding, activations, pools and classifier are already
  compiled in one Ethos-U delegate; CPU output dequantization is linked. Nothing
  has yet been observed running on simulated NPU hardware.

Official FVP memory configuration retained: 256-MAC U85, secure vector table
0x12000000; method/temp pools 4 MiB each in board-mapped NPU-accessible DDR,
heap 0x18000 and stack 0x1000. Four test inputs live in read-only model/application
storage and are copied to one aligned writable 11,000-byte tensor. This generous
FVP allocation does not establish E8 memory suitability. E8 pools/linker placement
must use its BSP settings and be measured with the actual runtime map.

The first staging attempt failed to register torchao custom Q/DQ operators before
loading the saved graph. Adding the registration import fixed staging; the failed
workspace remains preserved separately. No model change/re-export was needed.

## Continue after licenses/tools are ready

1. In Docker Desktop personally review/accept its agreement and complete setup.
2. In the opened VS Code project, accept tool-install prompts, sign in with your
   Arm account and activate the free MDK Community license. Install/approve the
   requested packs. No need to regenerate the AI layer or run Python setup again.
3. Personally review and confirm the FVP EULA:
   https://artifacts.tools.arm.com/avh/11.32.23/license_agreement.txt
4. Run from a Keil/Arm-tools-activated terminal in this repository (ensure Docker
   is on PATH, typically `/Applications/Docker.app/Contents/Resources/bin`):

```sh
export FVP_EULA_ACCEPTED=11.32.23  # ONLY after personally accepting the EULA
bash src/fvp/run.sh artifacts/fvp-subject1-20261016-ready/project results/fvp-integration-20261016/model2-console.log
.venv/bin/python -m src.fvp.compare \
  --reference artifacts/fvp-subject1-20261016-ready/reference.json \
  --log results/fvp-integration-20261016/model2-console.log \
  --out results/fvp-integration-20261016/model2-comparison.json
```

The runner invokes setup/build for SSE-320-U85 and official .vscode/fvp.sh with
`--simlimit 60 -f board/Corstone-320/fvp_config.txt -a out/cmsis-executorch/SSE-320-U85/Debug/cmsis-executorch.axf`.
No create_ai_layer.py call: that would replace the staged trained model with the
example TinyCNN. Comparison requires every trial, reports prediction agreement,
maximum/mean absolute logit differences and checks the 0.01 tolerance. Missing,
duplicate, non-finite or inconsistent records fail rather than claiming success.

Only AFTER Model 2 succeeds, use the same preparation script with Model 1's export
run and a new output directory, then the same run/comparison commands. Its existing
export_data.pt contains Subject 8 validation windows, quantized with its own
normalization and input qparams. Model 1 has not been staged/evaluated in this task.

```sh
/private/tmp/tinyeegnet-cmsis-executorch/.venv/bin/python -m src.fvp.prepare \
  --official /private/tmp/tinyeegnet-fvp-hackathon \
  --export-run artifacts/ethosu85-export-seed42-20261016-alif-e8 \
  --out artifacts/fvp-multisubject-<new-run>
```

Temporary official checkout/venv paths must be recreated at the recorded commits
if macOS removes /private/tmp; no tool availability is assumed permanently.

## Physical E8 follow-up

After FVP execution and numerical checks pass, switch to DevKit-E8 or AppKit-E8,
use the correct M55_HP/U85 BSP and 1.5.1 runtime, inspect full firmware memory map
and configure NPU-accessible SRAM/MRAM/alignment/cache handling. Match the actual
pack Vela profile to the recorded export profile (do not silently re-export).
Use Alif SETOOLS >=1.110.000 and J-Link >=8.42; board preparation and debug stubs
must match the exact DevKit/AppKit part. The official guide covers SW4 SEUART for
SETOOLS, then UART4 for the 115200-baud application console. Run the same test
vectors, compare logits, then measure actual memory use and latency. Only then
integrate real streaming EEG preprocessing, with exact trial warm-up/filter state,
channel order, normalization and int8 rounding. FVP is functional verification,
not an E8 latency/power measurement or a live EEG demonstration.
