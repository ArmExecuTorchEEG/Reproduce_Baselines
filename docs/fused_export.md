# Fused export: EA + input scale (and CAR) inside the `.pte`, one program per subject

Status 2026-10-17: exploratory, not released. Built from `runs/v4soft/student_wide32_ft_seed0` (the shipped v4 wide32-ft student) by `src/export_fused.py`; files in `artifacts/v4_fused/` (git-ignored). Same graph, input `(1,1,22,256)` and output `(1,2)` as v4. The params schema changes (`version` `"v2-fused"`), so this is a contract change if it ships.

```
python -m src.export_fused --run-dir runs/v4soft/student_wide32_ft_seed0 --subjects 1..9 --out artifacts/v4_fused/
#   defaults: --input car (the board keeps CAR) --calib session1
```

## The fold

1. Shipped v4: `logits = model(A_s C X / input_scale)`. `X` is the band-passed, decimated 22 × 256 µV window, `C = I − 11ᵀ/22` is the CAR and `A_s` the subject's EA matrix. CAR is linear and identical on every sample, so applying it at 250 Hz before the filter is the same as applying it after the filter and decimation.
2. `conv_temporal` uses one kernel for every channel and has no bias, so `conv_temporal(M X) = M · conv_temporal(X)` for any 22 × 22 `M`. Zero padding commutes with `M`.
3. `bn1` is per filter: `bn1(z) = a_f z + c_f`, the same on all 22 channels.
4. Spatial filter `k` (group `f`) sees `w_kᵀ(a_f M Z + c_f 1) = a_f (Mᵀw_k)ᵀ Z + c_f w_kᵀ1`.
5. Folding `w_k' = Mᵀ w_k` into `conv_spatial.weight`, with `M_s = A_s C / input_scale`, gives `a_f (Mᵀw_k)ᵀ Z + c_f w_k'ᵀ1`.
6. `M_s 1 = 0` (CAR), so `w_k'ᵀ1 = 0`. The two outputs differ by the constant `δ_k = c_f (w_kᵀ1 − w_k'ᵀ1)`.
7. `bn2.running_mean −= δ_k` absorbs it exactly. No other layer changes, and the model file stays identical.
8. Without step 7 the identity fails: max |logit diff| 0.025 on A03. With it, float32 gives ≤ 9.5e-7 on 100 test windows of every subject (gate 1e-4).
9. `C` is idempotent (`C C = C`), so in float the folded model gives the same logits whether or not the window is referenced.
10. In int8 the two inputs are not equivalent. The int8 input quantizer sits in front of the folded matrix, so its range is set by the window the board feeds (next section).

## What the board does

| Step (v2_handoff numbering) | v4 shipped | Fused |
|---|---|---|
| decode int16 → µV | board | board |
| 1 CAR (250 Hz) | board | **board** (also inside the `.pte`, where it does nothing on referenced input) |
| 2 band-pass SOS, 3 decimate, 4–5 ring buffer + window | board | board |
| 6 EA `ea_matrices[s] @ X` (22×22×256 MACs) | board | **inside the `.pte`** |
| 7 `/ input_scale` | board | **inside the `.pte`** (`input_scale` 1.0) |
| 8 quantize | board, one qparam pair | board, **per-subject** `input_quant_scale[s]`, `input_zero_point[s]` |
| 9 run `.pte`, 10 argmax | one `.pte` | **the subject's `.pte`** |

**Why CAR stays on the board.** The brief planned to drop CAR as well. With un-referenced windows, the int8 accuracy fails the gate:

- The common mode has std 5.7 µV against 2.7 µV for the referenced signal (A03). It widens the int8 input range by about 2×.
- `M_s` then throws the common mode away, so that extra range carries no information.

| Board input / PTQ calibration | per-subject int8 % (A01..A09) | mean | Δ vs shipped | worst Δ | gate |
|---|---|---:|---:|---:|---|
| shipped v4 (one `.pte`) | 87.5 62.5 99.2 74.2 70.8 56.7 65.8 94.2 59.2 | 74.44 | | | |
| **CAR'd, session 1 (200 windows) — exported** | 87.5 63.3 99.2 74.2 70.8 56.7 65.8 95.0 58.3 | **74.54** | **+0.09** | **−0.83** (A09) | **PASS** |
| CAR'd, session 2 trials 1–24 | 85.8 62.5 99.2 73.3 73.3 55.8 65.8 95.0 59.2 | 74.44 | +0.00 | −1.67 (A01) | PASS |
| un-referenced, session 1 | 85.0 58.3 98.3 72.5 70.8 55.0 66.7 95.0 59.2 | 73.43 | −1.02 | −4.17 (A02) | FAIL |
| un-referenced, session 2 trials 1–24 | 86.7 60.0 99.2 74.2 71.7 55.0 65.8 95.8 59.2 | 74.17 | −0.28 | −2.50 (A02) | FAIL |

How these numbers were made:

- Host fake-quant int8 (`src.quantize`) on session 2 trials 25–144 (120 per subject).
- Gate: mean within 0.5 points of 74.4, and no subject worse by more than 2 points.
- PTQ on 2a session 1: 200 seeded windows from all four MI classes. Cue event codes are read only to locate the trials; no class label enters calibration. The test block (session 2 trials 25–144) is used for the final metrics and, inputs only, for the float fold-identity check (no fitting, no selection).
- The v2 `calibration.npz` windows cannot be mapped back to the raw domain: CAR is singular, and their source sessions are not recorded.
- Float accuracy of the folded model equals the shipped float on every subject (74.8 % mean).
- Raw windows come from `download.load_raw`, cut like `cache125.windows_2a` without the CAR step (`runs/cache125_noref/`). CAR applied to them equals the `cache125` windows to within 1e-5 µV.

## Per-subject programs (`artifacts/v4_fused/`, `--input car --calib session1`)

Every program has 35 ops on Ethos-U and 3 on the CPU (input quantize, output dequantize, getitem). Vela: 96.38 KiB SRAM, 5.77 KiB flash, the same as v4.

| Subject | `.pte` bytes | SHA-256 | int8 % (120) | qparams (scale / zp) |
|---|---:|---|---:|---|
| A01 | 11,472 | `439c91db38b7b6a32d0a039150ed3e32121ca03e1ff2ac0f13432b8eea3056b4` | 87.5 | 0.126893 / 2 |
| A02 | 11,456 | `957037824c0aa66aa372cd3f052478e4b2eb9688a3dd950b6d8d9f5037fea12c` | 63.3 | 0.106858 / −2 |
| A03 | 11,472 | `634692e84321c203be189d3dd8421cf3f82771419a7f113e479c9364e9020b21` | 99.2 | 0.139791 / −3 |
| A04 | 11,440 | `70995a3aec3d7ac24a84b41eb32edec05f275fc89c267a956575dd0b46447b14` | 74.2 | 0.146785 / −6 |
| A05 | 11,488 | `d3be3f44aeb2350b94f6c391e00f1b45e3d55de8c75f7d1e669d34559cab4a92` | 70.8 | 0.086498 / 2 |
| A06 | 11,440 | `d9efcc19a471c2d51a5f4111be371c5d80ddb794937ab573b1c0cb3fae000263` | 56.7 | 0.189932 / 3 |
| A07 | 11,456 | `6d8307dc29ab7f603ea50048a2800ef48949212886d16d109d4a4301e53931a6` | 65.8 | 0.110712 / 1 |
| A08 | 11,456 | `8e6930486171af98054fe96f5eae54cc34d1ff4f0bd19493ca69029ac47de133` | 95.0 | 0.257964 / −6 |
| A09 | 11,456 | `96875d1d2fd60a442bb5f7b8eea7d3cef98222989eb078aefe32e654c2ffa38f` | 58.3 | 0.294062 / 6 |

Other files:

- `tinyeegnet_fused_sN_float.pt`: the folded state dict.
- `test_vectors_sN.npz`: `x` int8 (20,1,22,256) holds session 2 trials 25–44, CAR'd, filtered, decimated and quantized with that subject's qparams; also `logits` and `y`. On A03 the argmax matches the shipped v4 vectors 20/20.
- `preprocess_params_fused.json`: `version` `"v2-fused"`, `reference` `"common_average_22"` (`reference_folded_in_pte` true), `ea_matrix` / `ea_matrices` `"folded"`, `input_scale` 1.0 (`input_scale_folded` 0.969264), and per-subject `input_quant_scale` / `input_zero_point` dicts keyed `"1"`..`"9"`. It also carries `pte` (file, bytes, sha256 per subject) and `fused` (what is inside the `.pte` and what runs on the board). Filter, decimation and window fields are equal to v4.
- `fused_metrics.json` and `fused_metrics_*_nopte.json` (the four variants above), plus `vela_sN.log`.

## FVP (Corstone-320, Emulator_Deploy branch `v4/fused`, not committed)

`model/` holds the A03 program, vectors and fused params. `tools/npz_to_c.py ... --subject 3` reads the per-subject qparams.

Result for the A03 `.pte`:

- 20/20 argmax match, 20/20 correct against the true label.
- Max abs logit error 0.035, which is 1 output quantization step.
- 128,056 NPU cycles, the same as v4, plus 27,564 CPU cycles: 0.39 ms at 400 MHz.
- Pools: method 28,812 B + temp 98,688 B = 124.5 KiB.

Files: `results/fvp_v4_fused_metrics.md` and `results/fvp_v4_fused_release.log`.

## Edge_Deploy firmware change (if adopted)

1. **Select the `.pte` by `SET_SUBJECT`.** Store all 9 programs as const arrays: 9 × ~11.5 KB ≈ 103 KB of MRAM. This replaces the 9 EA matrices (≈ 17 KiB). Each program is loaded into the ExecuTorch runtime on a subject switch, or every one is kept loaded if the method pool allows it (28.8 KB each).
2. **`infer_window`:** drop the EA matmul and the `/ input_scale` step. Quantize with `input_quant_scale[s]` and `input_zero_point[s]`.
3. **Keep CAR** in `preprocess.cpp`, at 22 subtractions per 250 Hz sample. Dropping it costs 1–4 points (table above).
4. **`params.h`:** per-subject qparams instead of one pair. Firmware must accept `version` `"v2-fused"`. `Edge_Deploy/firmware/model/params.json` must equal `preprocess_params_fused.json`.
5. **Gain:** about 124 k MACs per window on the Cortex-M55 (22 × 22 × 256), plus the float scale.
6. **Cost:** about 86 KB more flash, and 9 programs to keep in sync. The NPU graph and cycle count do not change, because the fold changes weights only.
