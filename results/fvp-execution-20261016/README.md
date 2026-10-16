# Actual TinyEEGNet Corstone-320 / Ethos-U85 execution

Both existing Alif-targeted PTEs executed on the simulated 256-MAC Ethos-U85.
Model 2 passes the host-Q/DQ numerical comparison. Model 1 agrees on predictions
but exceeds the existing 0.01 logit tolerance; it is not a full numerical pass.
No retraining, architecture changes, new quantization, or re-export was performed.
Original checkpoints, preprocessing settings, PTEs and prior results are unchanged.

## Results

| Model | Development inputs | PTE bytes | Prediction agreement | Maximum absolute logit difference | Mean absolute difference | 0.01 tolerance |
|---|---|---:|---:|---:|---:|---|
| Model 2, Participant 1 | Four Session 1 validation windows, 2/class | 13,072 | 4/4 (100%) | 2.277832e-9 | 5.444813e-10 | PASS |
| Model 1, multi-subject | Four Subject 8 validation windows, 2/class | 13,056 | 4/4 (100%) | 0.0260930047 | 0.0032616272 | FAIL |

The comparator exits nonzero for Model 1's numerical mismatch. One of its eight
logits differs by approximately one output quantization step: output scale is
0.026093021035194397. The source of that integer-arithmetic difference is not
established; no tolerance was relaxed or model changed to obtain a pass. Model 2's
remaining decimal differences are within the precision of the printed logits.
These are four-input execution smoke tests, not estimates of full-dataset
classification accuracy. No reserved demonstration or Subject 9 data was used.
Participant 1 Session 2 was not accessed for these tests.

References are saved host PT2E Q/DQ models evaluated using the exact reconstructed
int8 inputs provided to firmware. They are NOT float PyTorch numerical references.
Previously reported float-vs-Q/DQ discrepancies remain; FVP agreement does not
remove them or establish all release criteria.

Console evidence, per-trial logits, numerical comparisons and build logs are in
this directory. Both runs print U85 Arch v2.0.0, 256 MACs/cc and command stream v1.
All four convolutions (temporal, depthwise spatial, depthwise temporal, pointwise),
padding, ReLU, average pooling, reshape and classifier run in the serialized
EthosUBackend delegate. The only CPU tensor kernel is output dequantization.
Successful inference uses the compiled delegate and official Ethos-U driver;
there is no convolution CPU fallback in these PTEs.

## Workspace and interfaces

- Model 2: `artifacts/fvp-subject1-20261016-ready/project/`
- Model 1: `artifacts/fvp-multisubject-20261016/project/`
- Each workspace includes the official board/runtime wrapper, staged identical
  PTE, model C array, four int8 EEG windows, firmware, HEX, AXF and linker map.
- `reference.json` and `validation_vectors.npz` in each workspace's parent record
  source PTE SHA256, model-specific quantization, development trial identities and
  host-reference outputs. Pre-existing training-only test_vectors files unchanged.
- One reusable Python preparation script and one common application template
  handle both models. Separate generated build directories embed each model.
- Input: contiguous signed int8 NCHW `(1,1,22,500)`; 11,000 bytes/window.
- Output: float32 `(1,2)` logits, argmax left=0/right=1 (ties=left).
- Model 2 scale/zero-point: 0.04422667250037193 / -10.
- Model 1 scale/zero-point: 0.07146575301885605 / 6.
- Input quantization follows each model's own unchanged EEG normalization:
  `clip(round(normalized_input / scale) + zero_point, -128, 127)`, ties to even.
  Channel order, causal 7–30 Hz filtering, cue [1,3) window and 250 Hz are unchanged.

## Verified environment

- Host: macOS 15.6.1, Apple Silicon arm64.
- Docker Desktop 4.95.0; Engine 29.8.2, Linux arm64 containers.
- Official hackathon snapshot: `a6364946165f21b27c1f5cf6998a701ded8ed73e`.
- FVP: Fast Models 11.32+23 (Aug 9 2026), image `cmsis-fvp:11.32.23`.
- Image ID: `sha256:4fe0f21919254d02ef4f6dfdf1c655fcbf17ec4c2b5e3f80ad4a66eab161d5fb`.
- CMSIS-Toolbox 2.15.0, Arm Compiler 6.24.0, CMake 4.2.1, Ninja 1.13.2.
- Resolved runtime/BSP packs: PyTorch::ExecuTorch@1.5.1,
  ARM::ethos-u-core-driver@1.26.5, ARM::SSE_320_BSP@1.0.0,
  ARM::CMSIS@6.4.0, ARM::CMSIS-Compiler@2.3.0.
- Original exporter: ExecuTorch 1.5.1, torch 2.14.0, torchao 0.18.0, Vela 5.1.0,
  TOSA tools 2026.5.0. No replacement exporter/runtime version introduced.
- User explicitly accepted the FVP EULA before container download/run. Arm compiler
  activation became available on this resumed run; build succeeded without any
  agent license bypass. Docker helper repair is recorded separately in
  `results/docker-helper-repair-20261016/`.

## Memory and integration issues resolved

The hackathon branch explicitly supports sharing the Ensemble E8 SRAM/MRAM PTE
with Corstone-320: U85-256, Ethos_U85_SRAM_MRAM, Shared_Sram. No FVP-specific
recompile of either model was necessary. Actual firmware compilation is required:
the simulator loads an embedded executable, not a standalone PTE.

Official FVP board memory layout retained: code/vector table in FPGA SRAM at
0x12000000, writable memory in NPU-accessible DDR at 0x70000000, 4 MiB method pool
and 4 MiB temporary pool, heap 0x18000, stack 0x1000. Both firmware images report
Code 118,632 bytes, RW-data 393,376 bytes, ZI-data 8,527,156 bytes; RO-data is
67,700 bytes for Model 2 and 67,684 for Model 1. These include the runtime and test
harness, not just the 13 KiB PTE. These generous debug allocations are not E8 RAM
requirements or measured latency. Inspect the linker map when changing targets.

Two integration issues were addressed without changing models:
1. Official macOS Docker shim fails under Bash 3.2 `set -u` when NAME/PORTS arrays
   are empty. The generated integration copy now conditionally expands those
   arrays; the preparation script applies the same narrow compatibility fix.
2. The AXF's first ELF LOAD segment has memsz 0x8af5b4 but only 0x2e460 file bytes.
   The FVP tries to zero-fill beyond the 2 MiB FPGA SRAM load region, prints an
   image-loading warning, but still runs the application. The initial log is
   preserved. Loading the build's HEX instead avoids that loader zero-fill;
   firmware startup initializes the real writable DDR execution region. Final
   HEX runs complete without the warning and give identical Model 2 logits.
   `src/fvp/run.sh` now selects the generated HEX image.

The missing debugger config warning is not needed for these headless runs and
was left unchanged. No unrelated RTE or board configuration edits were made.

## Reproduce

Use a Keil tool-activated terminal with the user-activated Arm license, CMSIS
packs installed, Docker running, and tools on PATH. This run used:

```
AC6_TOOLCHAIN_6_24_0=/Users/philippinechauvet/.vcpkg/artifacts/2139c4c6/compilers.arm.armclang/6.24.0/bin
CMSIS_PACK_ROOT=/Users/philippinechauvet/.cache/arm/packs
```

CMSIS-Toolbox, CMake/Ninja and Docker CLI must also be on PATH; the exact local
vcpkg paths are in build logs and may differ on another computer. No compiler
license files or tokens are included in this report.

From the repository root, choose NEW log/report filenames to preserve results:

```sh
export FVP_EULA_ACCEPTED=11.32.23  # user has personally accepted this version
bash src/fvp/run.sh artifacts/fvp-subject1-20261016-ready/project /tmp/model2-new.log
.venv/bin/python -m src.fvp.compare \
  --reference artifacts/fvp-subject1-20261016-ready/reference.json \
  --log /tmp/model2-new.log --out /tmp/model2-new-comparison.json

# Repeat with artifacts/fvp-multisubject-20261016/project and its reference.json.
# Model 1 comparison currently exits 1 because its maximum error exceeds 0.01.
```

The runner executes cbuild setup/build for SSE-320-U85, then:

```sh
bash .vscode/fvp.sh -f board/Corstone-320/fvp_config.txt --simlimit 60 \
  -a out/cmsis-executorch/SSE-320-U85/Debug/cmsis-executorch.hex
```

Do NOT invoke the template's create_ai_layer.py: the trained unchanged EEG PTE
is already embedded; regenerating would replace it with the example TinyCNN.
The FVP shim mounts the host home directory and reuses the user-activated Arm
license as designed by the official project. No training or data download needed.

## Physical Alif E8 next steps

1. Select the exact DevKit-E8/AppKit-E8 target and matching M55_HP/U85 BSP. The
   same NPU model passed FVP; Cortex-M55 board bring-up is still unverified.
2. Retain the official E8 linker/pool configuration: it places `.bss.ai_pool` in
   NPU-accessible bulk SRAM, with 1 MiB method and 2 MiB temporary pools instead
   of FVP's two 4 MiB pools. Build and inspect the complete E8 memory map.
3. Use SETOOLS >=1.110.000 and J-Link >=8.42; follow the official guide's exact
   board-part/debug-stub and UART-switch procedure. No flashing performed here.
4. Run the same development test vectors on the physical board, verify Model 2
   outputs first, then check Model 1's one-output-step discrepancy. Record actual
   latency/memory. FVP is not a physical timing/power measurement.
5. Adapt the input transport for recorded/live EEG while preserving every model's
   filter state/warm-up, channel order, normalization, int8 rounding and labels.
   These four preprocessed vectors do not test streaming preprocessing or cursor
   control. No new model accuracy claim is made.

Validation: 23 unit tests passed; Bash syntax and git diff --check passed. Original
training/export files are SHA256-verified unchanged. Branch feature/tinyeegnet-training;
no commits, pushes, retraining, model modifications or reserved-data evaluation.
