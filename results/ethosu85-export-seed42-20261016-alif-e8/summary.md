# TinyEEGNet: investigative Alif E8 / Ethos-U85 export

Compiled successfully; FVP and physical E8 execution NOT validated. No retraining,
architecture changes, reserved demonstration/S9 evaluation, commits, or pushes.

## Provenance and workflow

Original checkpoint: `artifacts/combined-seed42-20261016/best_checkpoint.pt`.
SHA256: `1dd02a4d5b61999d49238da87a492fe4b103d29d71b17622010f4890d5e34ec9`.
Reload passed in both environments; original float logits agreed within 1e-5.
Model: 1,938 parameters, dropout 0.5, unchanged weights and architecture.

Official workflow: https://github.com/Arm-Examples/CMSIS-Executorch
commit `0a243f1ee054c64e3cc698f5a8c42c63ba3710c7`.
Separate export environment created with its setup_venv.py: torch 2.14.0,
ExecuTorch 1.5.1, torchao 0.18.0, Vela 5.1.0, TOSA tools 2026.5.0.
Original training environment retains torch 2.14.1.
The official no-deps TOSA installation leaves pip-check complaints: missing
jsonschema/semver, flatbuffers 24.3.25 vs required 25.2.10, numpy 2.5.4 vs <2.2.
These were not repaired by mixing dependency versions; export succeeded.

Alif export guidance:
https://github.com/alifsemi/alif_ml-embedded-evaluation-kit/blob/main/docs/deploying_model_executorch_export.md
Alif profile source at commit `d7c39929fe482c1ce17b7be03edfab1db03acf06`:
https://github.com/alifsemi/alif_ml-embedded-evaluation-kit/blob/d7c39929fe482c1ce17b7be03edfab1db03acf06/scripts/vela/ensemble_vela.ini
Target ethos-u85-256, system Ethos_U85_SRAM_MRAM, Shared_Sram.
The guide's older-version examples must not be mixed with this 1.5.1 exporter/runtime.

Data preparation reused existing causal 7–30 Hz filtering, cue-relative [1,3)
windows at 250 Hz, all 22 EEG channels, training-only per-channel normalization,
left=0/right=1 and the retain-and-track artifact policy. Recomputed training
statistics match the original saved parameters. Calibration: 200 seeded windows,
100/class, stratified over all 12 allowed training recordings (S1–5 both sessions,
S6–7 session 1). Development: S8 both sessions only, 288 trials. Trial provenance
is recorded in data_manifest.json. No reserved source files were loaded.

## Results

| Item | Result |
|---|---|
| Fixed input | signed int8 (1,1,22,500) |
| Output | float32 logits (1,2) |
| PTE size | 13,056 bytes (12.75 KiB) |
| Compiled delegate payload | 9,920 bytes |
| Float-I/O comparison PTE | 12,128 bytes |
| EthosUBackend partitions | 1 |
| Vela estimated SRAM | 185.40625 KiB |
| Vela estimated MRAM constants, modeled as Off-chip Flash | 4.46875 KiB |
| Vela estimated inference time | 0.5573 ms; NOT measured hardware latency |
| Float S8 accuracy / balanced accuracy | 93.0556% / 93.0556% |
| Host quantized Q/DQ S8 accuracy / balanced accuracy | 93.0556% / 93.0556% |
| Prediction agreement | 284/288 (98.6111%) |
| Maximum absolute logit difference | 1.5674579; exceeds README 1e-2 target |

Float confusion matrix (rows true, columns predicted; left/right):
`[[136,8],[12,132]]`; host Q/DQ: `[[138,6],[14,130]]`.
These are PT2E host quantization-reference results, NOT compiled PTE/FVP/NPU
accuracy. Unchanged accuracy does not establish numerical equivalence.

The temporal convolution, depthwise spatial convolution (depth multiplier 2),
separable depthwise/pointwise convolutions, padding, ReLUs, average pools,
reshape and linear classifier all compile inside one delegate. Batch norms fold;
evaluation-mode dropout disappears. No unsupported model operator or CPU
convolution fallback was found. Serialized final PTE inspection confirms its
only CPU kernel is output `quantized_decomposed::dequantize_per_tensor`.

Input quantization is applied AFTER saved EEG preprocessing/normalization:
`clip(round(x / 0.07146575301885605) + 6, -128, 127)`; round ties to even.
The run's preprocess_params.json contains the full filter/statistics/quantization
configuration. Training-derived test_vectors.npz uses host Q/DQ expected logits,
explicitly labeled as such; it is not a hardware validation certificate.

## Artifacts and checks

Run artifacts: `artifacts/ethosu85-export-seed42-20261016-alif-e8/`:
PTE files, Q/DQ exported graph, preprocessing parameters, training test vectors,
development predictions, data manifest, pinned environment freeze, Alif profile
and source provenance, graphs, Vela debug output/CSV, export report and serialized
verification. JSON report is also copied into this results directory.
SHA256 comparison verified 113 pre-existing artifact/result files unchanged.
18 unit tests passed via `.venv/bin/python -m unittest discover -s tests -v`.

An initial export attempt compiled float-I/O successfully but int8-boundary
conversion failed because serializing had mutated the Edge graph. Serializing a
deep copy before input conversion resolved it without model changes. Initial
attempt and separate Corstone-profile retry remain preserved in their own run
folders. This final run uses Alif's published memory profile.

## Remaining deployment steps

No Docker, Corstone-320 FVP, CMSIS-Toolbox or Arm firmware compiler was found;
Arm's template requires Docker for FVP on macOS. No simulator/license/toolchain
installation or hardware-specific firmware edits were attempted.

1. Provide a supported Linux/Docker FVP environment, CMSIS-Toolbox and Arm
   compiler. Build the official template with a runtime matching ExecuTorch 1.5.1,
   the Ethos-U backend and the output-dequantization kernel. Run the training-derived
   vectors and compare compiled outputs with the saved host reference.
2. Adapt the template's example image input/output harness to 22-channel EEG,
   500 samples and two logits. Confirm preprocessing, int8 rounding/saturation,
   labels, tensor layout, alignment and cache handling end to end.
3. Integrate the Alif E8 RTSS-HP BSP, Ethos-U85 driver and SRAM/MRAM linker/memory
   placement. Vela scratch estimates and the 13 KiB PTE exclude firmware/runtime,
   driver, stacks and preprocessing buffers; check the full firmware memory map.
4. Flash/run on E8 and validate output agreement, memory use and measured latency
   before publishing canonical contract artifacts or declaring deployment complete.

No architecture/operator correction is currently indicated by compilation.
Numerical FVP/board validation is the next gate, rather than further model tuning.
