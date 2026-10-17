# v4 hand-off: Emulator_Deploy and Edge_Deploy

What changed against [`v2_handoff.md`](v2_handoff.md): **only the numbers.** Same TinyEEGNet wide32 student
(F1 16, D 2, F2 32, kern_t 32, kern_s 16; 3,426 params, 3.18 M MACs), same input `(1,1,22,256)` at 125 Hz,
same output `(1,2)` logits, same file names, same `preprocess_params.json` schema (`version` `"v2"`: SOS,
CAR, decimation, window and `ea_matrices` from session 2 trials 1–24 as before). The firmware pipeline
(`firmware/src/preprocess.cpp`) does not change; regenerate `params.h` and the test vectors from the new bundle.

Status 2026-10-17: bundle in `artifacts/v4/`, exported from `runs/v4soft/student_wide32_ft_seed0/model.pt`
(export subject A03). Release tag **`v0.5-distill` (planned)**; `v0.3-distill` (v2) stays until it is published.

## What is new in training (offline only, nothing new runs on the board)

1. **Riemannian teacher member:** OAS covariance → tangent space (Riemannian mean) → logistic regression,
   joining EEGNet-ELU, ShallowConvNet and DeepConvNet (`src/models/riemann.py`). Alone it scores 73.0 % on
   the 2a test block, on par with the CNNs; the 4-member ensemble scores 75.3 % (v2 three-member: 74.4 %).
2. **Self-training on BCI IV 2a session 1, unlabeled:** the round-0 ensemble scores session 1 (1,296 windows,
   X only; its labels are never read for training). Hard variant (`runs/v4`): 480 windows with confidence
   ≥ 0.8 added with pseudo-labels, members retrained (ensemble 74.7 %). Soft variant (`runs/v4soft`): all
   1,296 windows with the round-0 probabilities as targets, KL weighted by 2·|p(right) − 0.5|, members
   fine-tuned from round 0 (ensemble 74.9 %).
3. **Shipped student: the v2 wide32 student fine-tuned with KD only** (`src/finetune_student.py`): starts
   from `runs/v2/student_wide32_seedK/model.pt`, 30 epochs Adam 1e-4 on the 1,296 session-1 windows + the
   216 calibration windows, soft labels of the v4soft ensemble, last epoch kept. No labeled 2a data.

**Headline:** int8 **74.6 ± 0.7 %** (mean over 3 seeds ± std over seeds; mean over the 9 subjects) vs 73.5 %
for the v2 students these runs start from. Paired per seed: 74.4 / 75.5 / 73.8 vs 73.3 / 73.9 / 73.1 (3/3
seeds better, +1.1 points mean). The shipped seed 0 has 74.4 % int8 (74.8 % float).

## Selection note

Variant comparison on the test block (exploratory; the shipped variant was chosen by the team from this
table, which makes its test estimate optimistic by selection among 5 candidates); the seed within the
variant is chosen by validation KL. Candidates (wide32 int8, seed 0 unless stated): (1) v2 wide32 73.5
(3 seeds), (2) v4 joint 73.8, (3) v4 round-0 teacher student 73.6, (4) v4soft joint 73.6, (5) v4soft
student-ft 74.4 (3 seeds 74.6). Full tables: `results/results_v4.md`, `results/results_v4soft.md`.

## Bundle (`v0.5-distill`, planned)

| File | Bytes | SHA-256 |
|---|---:|---|
| `tinyeegnet_int8.pte` | 11,456 | `6caa8a9876d3cd0c36b39b3d4474a0fe7dbdbfb32a4d6f80ed4776623a19e02e` |
| `tinyeegnet_float.pt` | 20,398 | `987231fefed1815de280d0d11437716b6e7e59abb5cb0dd66f915c1d1432c9bd` |
| `calibration.npz` | 4,505,856 | `212c94fe67195c64b1231b52497c5fffac05620e9062ad88935b78c5a3df4f7f` |
| `student.json` | 64 | `162a7dc32bd4741bd7b2a2fff72d4792445d76a30c0e3b0aeae6ac8af92242f9` |
| `test_vectors.npz` | 113,694 | `250241cb0967b99e04c355c86c3f8ae4671ed42e71a1f0d65f82fea5affb98c3` |
| `preprocess_params.json` | 121,188 | `3d34f6326c67441e9e6b636454436c63951e81f6e221da977723700f1c8152f7` |

`student.json` and `calibration.npz` are byte-identical to v2. New numbers in `preprocess_params.json`:
`input_quant_scale` 0.0695584, `input_zero_point` −12 (`input_scale` 0.969264 and `ea_matrices` as in v2);
`student.name` is `wide32_ft` (widths unchanged). Ethos-U ops 35 / CPU 3 (input quantize, output
dequantize, getitem), Vela 96.38 KiB SRAM, 5.75 KiB flash. Host check on A03: 20/20 test vectors,
int8 = float on all 120 test trials (99.2 %). Corstone-320 FVP (Emulator_Deploy branch `v4/fvp`, `results/fvp_v4_metrics.md`): 20/20, max abs logit error 0.025, 128,056 NPU + 28,892 CPU cycles, 0.39 ms at 400 MHz, pools 124.5 KiB (identical to v2). Edge_Deploy (branch `claude/edge-v2`): firmware self-test PASS, 0 of 28,160 int8 elements differ, replay accuracy 74.4 % with 1080/1080 agreement.

Rebuild: `python -m src.export --checkpoint runs/v4soft/student_wide32_ft_seed0/model.pt --run-dir
runs/v4soft/student_wide32_ft_seed0 --subject 3 --out artifacts/v4/tinyeegnet_int8.pte`.
