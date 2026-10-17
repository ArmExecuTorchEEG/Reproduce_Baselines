# v2 hand-off: Emulator_Deploy and Edge_Deploy

Model: TinyEEGNet **wide32** student (F1 16, D 2, F2 32, kern_t 32, kern_s 16; 3,426 params, 3.18 M MACs), distilled from a conv ensemble trained on Cho2017 + Lee2019_MI. Input `(1,1,22,256)` at **125 Hz**, output `(1,2)` logits. Release tag **`v0.3-distill`** (planned). `v0.1-baseline` stays valid for the `(1,1,22,500)` model.

Status 2026-10-17: final bundle in `artifacts/v2/`, exported from `runs/v2/student_wide32_seed0/model.pt` (wide32, seed 0, export subject A03; trained on Cho2017 + Lee2019_MI). Accuracy 73.5 % int8 mean over the 9 2a subjects (`results/results_v2.md`). FVP: 20/20 on the Corstone-320 (see step 6).

## Bundle (`v0.3-distill`)

| File | Content |
|---|---|
| `tinyeegnet_int8.pte` | Ethos-U85 (`ethos-u85-256`) delegated program, 11,456 B. 35 ops on Ethos-U, 3 on CPU (input quantize, output dequantize, getitem). Float I/O |
| `tinyeegnet_float.pt` | float state dict of the student |
| `student.json` | `{"F1": 16, "D": 2, "F2": 32, "kern_t": 32, "kern_s": 16}` |
| `calibration.npz` | `x` (200,1,22,256) float32, aligned + scaled training windows (PTQ) |
| `test_vectors.npz` | `x` int8 (20,1,22,256) = export subject (A03) session 2 trials 25–44; `logits` float32 (20,2) of the quantized model; `y` int64 (20,) |
| `preprocess_params.json` | board pipeline, below |

Rebuild locally: `python -m src.export --checkpoint runs/v2/student_wide32_seedK/model.pt --run-dir runs/v2/student_wide32_seedK --subject 3 --out artifacts/v2/tinyeegnet_int8.pte` with `K` = 0 (`artifacts/` is git-ignored except top-level `*.json`, so `artifacts/v2/` is not in git; take the files from the release or the Drive folder).

SHA-256 of the bundle:

| File | Bytes | SHA-256 |
|---|---:|---|
| `tinyeegnet_int8.pte` | 11,456 | `6ce9ca4f5f0491c2c6841655e0db02972886207b39a6c73cd35840e3574cdb73` |
| `tinyeegnet_float.pt` | 20,398 | `ee36b12148114e9871738a9b40734db67b88dca4d11ce0565c40b7f525be4f91` |
| `student.json` | 64 | `162a7dc32bd4741bd7b2a2fff72d4792445d76a30c0e3b0aeae6ac8af92242f9` |
| `preprocess_params.json` | 121,185 | `87af6459137ffb9954c9c627c5bb7c887bc319db1960aa4e0b06a413afde397e` |
| `test_vectors.npz` | 113,694 | `33dbd4ad3850c6ac28dfcf23291ddc4f783234ccba99e34dd852b4696021ef7d` |
| `calibration.npz` | 4,505,856 | `212c94fe67195c64b1231b52497c5fffac05620e9062ad88935b78c5a3df4f7f` |

## Emulator_Deploy

1. **Model files.** Copy `src/models/tinyeegnet.py` to `model/model.py` unchanged (it must stay identical). Put `student.json`, `tinyeegnet_float.pt` and `calibration.npz` next to it. `get_model(input_shape)` reads `student.json` and builds wide32; without it you get the v0.1 widths and the checkpoint will not load.
2. **csolution.** `mlops: model:` `input-shape: 1x1x22x256` (v0.1: `1x1x22x500`), `calibration-samples: 200`.
3. **`create_ai_layer.py:198`** still needs `.module(check_guards=False)` (same fix as v0.1).
4. **Prebuilt `.pte` path.** To skip the starter export, use the release `tinyeegnet_int8.pte` (locally `Reproduce_Baselines/artifacts/v2/tinyeegnet_int8.pte`, 11,456 B). The starter re-export must give the same size.
5. **Check.** Run the 20 `test_vectors.npz` inputs: argmax must match `logits` argmax 20/20. Feed the CPU-side input as `(x_int8 - input_zero_point) * input_quant_scale` to reproduce `x_int8` inside the program.
6. **FVP result (Corstone-320, `SSE-320-U85`, 2026-10-17): 20/20 argmax match**, max abs logit error 0.026 (one output quant step), 128,056 NPU cycles + 28,892 CPU cycles per window, 0.39 ms at 400 MHz, method pool 28,812 B + temp pool 98,688 B (127,500 B, 124.5 KiB). Log and table: `Emulator_Deploy/results/fvp_v2_release.log`, `results/fvp_v2_metrics.md` (branch `v2/fvp-256`). Host reference: 20/20 test vectors 100 % int8 (fake-quant), Vela SRAM estimate 96.4 KiB.


Layout note: the export is NCHW `(1,1,22,256)`; Vela inserts transposes for it (the largest cycle share in the Vela estimate, see the export's `deployment.md`).

## Edge_Deploy firmware: per-sample pipeline

UART input is unchanged: frame `0xAA55, uint8 n_ch, uint16 seq, int16[n_ch]` at **250 Hz**, 22 channels in `channel_order`, in µV. Per received sample `x` (22 values):

```
1. CAR (250 Hz):        x[c] -= mean(x[0..21])                         # reference = common_average_22
2. Band-pass (250 Hz):  y = sosfilt(sos, x), 4 sections, DF2-T state per channel, never reset
3. Decimate:            keep y only when n_250 is even (phase 0)  ->  125 Hz sample
4. Ring buffer:         push the 125 Hz sample into a 22 x 256 buffer
5. Window (replay):     the window is 125 Hz samples [cue_125 + 62, cue_125 + 318), cue_125 = cue_250 // 2
6. EA:                  x_ea = ea_matrices[subject] @ x_window           # 22x22 @ 22x256
7. Scale:               x_model = x_ea / input_scale
8. Quantize:            x_int8 = clamp(round(x_model / input_quant_scale) + input_zero_point, -128, 127)
9. Model:               logits = run(.pte, (x_int8 - input_zero_point) * input_quant_scale)   # float I/O
10. Output:             cls = argmax(logits); 0 = left_hand, 1 = right_hand
```

- Steps 1–3 run every sample; 6–10 once per window. EA is linear, so it can also run per 125 Hz sample before the ring buffer (22 × 22 MACs per sample) instead of once per window.
- Phase and filter state: the reference pipeline filters the **continuous** recording from its first sample and keeps samples 0, 2, 4, … of it. To reproduce `test_vectors.npz` bit-for-bit, the replay must stream the session from its start (or a long lead-in) and the firmware must count `n_250` from that first sample.
- Replay alignment: the 2a marker sits at fixation; the cue is `cue_offset_s` = 2.0 s after it. At 250 Hz the window starts 124 samples (0.496 s) after the cue and ends 636 samples (2.544 s) after it. The bridge must tell the firmware the cue sample (or the trial index) of each replayed trial; that signalling is Edge_Deploy's choice.
- `ea_calibration`: each matrix comes from that subject's session 2 trials 1–24 (unlabeled), regularised with `eps` 0.001. Only trials 25–144 are test trials; demo on those.

## `preprocess_params.json` fields (v2)

| Field | Value (A03) | Use |
|---|---|---|
| `version` | `"v2"` | refuse anything else |
| `subject` | 3 | export subject |
| `fs` | 250.0 | UART rate |
| `fs_model` | 125.0 | model rate |
| `n_channels` | 22 | |
| `n_samples` | 256 | window length at 125 Hz |
| `input_shape` | `[1, 1, 22, 256]` | |
| `channel_order` | Fz FC3 FC1 FCz FC2 FC4 C5 C3 C1 Cz C2 C4 C6 CP3 CP1 CPz CP2 CP4 P1 Pz P2 POz | UART channel order |
| `units` | `"uV"` | |
| `reference`, `reference_applied` | `"common_average_22"`, subtract the per-sample mean of the 22 at 250 Hz, before the filter | step 1 |
| `sos` | 4 × 6 (v0.1 SOS at 250 Hz) | step 2 |
| `filter` | Butterworth band-pass 7–30 Hz, order 4, `fs_hz` 250, causal, DF2-T | step 2 description |
| `decimation` | `{factor 2, phase 0, applied_after "sos_filter_250hz"}` | step 3 |
| `window` | `{start_samples_after_cue_125hz 62, length 256, start_s_after_cue 0.496, end_s_after_cue 2.544}` | step 5 |
| `cue_offset_s` | 2.0 | 2a cue = marker + 2 s |
| `ea_matrix` | 22 × 22 | EA of the export subject |
| `ea_matrices` | `{"1": 22×22, …, "9": 22×22}` | EA per 2a subject, step 6 |
| `ea_calibration` | `{subject, session 2, trials "1-24", labels_used false, window, eps 0.001}` | provenance |
| `input_scale` | 0.969264 | step 7 |
| `input_quant_scale`, `input_zero_point` | 0.0695584, −12 | step 8 |
| `input_formula` | text of steps 6–8 | |
| `class_map` | `{"0": "left_hand", "1": "right_hand"}` | step 10 |
| `output` | logits (1, 2), argmax, no softmax | |
| `student` | `{name "wide32", F1 16, D 2, F2 32, kern_t 32, kern_s 16}` | must match `student.json` |

v0.1 keys `tmin` / `tmax` are gone; use `window`. `Edge_Deploy/firmware/model/params.json` must equal this file.

## Per-subject EA matrices

`ea_matrices` carries one 22 × 22 matrix per 2a subject (keys `"1"`..`"9"`, A01..A09); `ea_matrix` equals `ea_matrices[str(subject)]`. The model, `input_scale` and int8 qparams are shared by all 9 subjects (no 2a subject was trained on), so **one `.pte` serves every replay subject**: switch subjects by switching the matrix only. This differs from v0.1, where each subject had its own model and bundle.

Firmware: store all 9 as constant float arrays (9 × 484 floats ≈ 17 KiB), select by the replayed subject, apply at step 6. Do not recompute them on the board; they are computed offline from session 2 trials 1–24 of the 125 Hz CAR-filtered windows.
