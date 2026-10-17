# AGENTS.md — Reproduce_Baselines

Rules for people and coding agents (Claude Code, Codex, Cursor, Copilot) working in this repo. `CLAUDE.md` imports this file.
Project overview and repo map: https://github.com/ArmExecuTorchEEG

## Ownership
| Path | Owner | Others may |
|---|---|---|
| `src/data/` | TBD | PR |
| `src/models/`, `src/train.py`, `src/eval.py`, `src/run_all.py` | TBD | PR |
| `src/quantize.py`, `src/export.py`, `artifacts/*.json` | TBD | PR |
| `src/experiments/` | TBD | PR |
| `configs/default.yaml` | TBD | PR; never change `seed`, `window`, `fs` silently |
| `src/data/external.py`, `src/data/cache125.py`, `src/data/splits125.py` (v2 data) | TBD | PR |
| `src/models/convnets.py`, `src/train_teacher.py`, `src/distill.py`, `src/run_v2.py` (v2 training) | TBD | PR |
| `configs/v2.yaml` | TBD | PR; never change `fs_model`, `window`, `window_start`, `crop_offset`, `filter`, `ea_eps`, `seeds` silently (mirrored in `src/data/cache125.py`) |
| `docs/v2_handoff.md` | TBD | PR |
| `results/` | author of the run | add only; never overwrite another run's file |

## Contracts
Model input `(1,1,22,500)` int8, output `(1,2)` logits. Release tag `v0.1-baseline`. Consumed by Emulator_Deploy and Edge_Deploy.
- `artifacts/tinyeegnet_int8.pte`: Ethos-U85 (`ethos-u85-256`) delegated; float I/O with CPU quantize/dequantize at the boundary, as the starter's `create_ai_layer.py` exports it.
- `artifacts/test_vectors.npz`: `x` int8 (20,1,22,500), `logits` float32 (20,2), `y` int64 (20,).
- `artifacts/preprocess_params.json`: `sos`, `tmin`, `tmax`, `fs`, `n_samples`, `channel_order`, `ea_matrix` (22×22 Euclidean Alignment matrix of the export subject), `input_scale`, `input_quant_scale`, `input_zero_point`, `class_map`. `ea_matrix` replaced per-channel mean/std; Edge_Deploy must apply it.
- `artifacts/tinyeegnet_float.pt`, `artifacts/calibration.npz` (200 training windows): inputs for the starter's `create_ai_layer.py`.
- Filter is causal (`sosfilt` on the continuous signal); window cue+0.5 s to cue+2.5 s; `src/models/tinyeegnet.py` must stay identical to `Emulator_Deploy/model/model.py`.
- Test data (target subject session 2 trials 25–144) is never used for early stopping, model selection, normalisation or calibration.
- The delegated `.pte` does not run on the x86 runtime. Host int8 check = fake-quant model (`src/quantize.py`); target check = FVP in Emulator_Deploy (v0.1: 20/20, release `v0.2-fvp`; v2: 20/20 on branch `v2/fvp-256`).

v2 bundle (release tag `v0.3-distill`, planned; `v0.1-baseline` stays). Model input `(1,1,22,256)` int8 at 125 Hz, output `(1,2)` logits. Hand-off: `docs/v2_handoff.md`.
- `artifacts/v2/` holds the bundle (shipped: wide32 seed 0, export subject A03). `.gitignore` keeps everything in `artifacts/` except top-level `*.json` out of git, so all of `artifacts/v2/` (including its `preprocess_params.json`) ships via the release and the Drive folder, not git.
- Same file names as v0.1 plus `student.json` (`F1`, `D`, `F2`, `kern_t`, `kern_s`; shipped wide32 = 16, 2, 32, 32, 16). `src/models/tinyeegnet.get_model` reads it; without it the v0.1 defaults apply. `src/models/tinyeegnet.py` must still stay identical to `Emulator_Deploy/model/model.py`.
- `test_vectors.npz`: `x` int8 (20,1,22,256) = export subject session 2 trials 25–44, `logits` float32 (20,2), `y` int64 (20,). `calibration.npz`: `x` (200,1,22,256).
- `preprocess_params.json` (`version` `"v2"`): `fs` 250 (UART), `fs_model` 125, `n_samples` 256, `input_shape`, `channel_order`, `reference` `common_average_22` (per-sample mean of the 22 subtracted at 250 Hz, before the filter), `sos` (v0.1 SOS at 250 Hz), `filter`, `decimation` `{factor 2, phase 0, applied_after sos_filter_250hz}`, `window` (start 62 samples after the cue at 125 Hz, length 256), `cue_offset_s`, `ea_matrix` (export subject), `ea_matrices` (all 9 2a subjects, keys `"1"`..`"9"`), `ea_calibration` (session 2 trials 1–24, no labels, `eps` 0.001), `input_scale`, `input_quant_scale`, `input_zero_point`, `input_formula`, `class_map`, `output`, `student`. No `tmin` / `tmax` (v0.1 keys); use `window`.
- BCI IV 2a is test only in v2: no labeled 2a trial is trained on; session 2 trials 1–24 enter distillation as unlabeled calibration (EA + teacher soft labels), trials 25–144 are the test block.
- `src/data/external.py` never downloads; Cho2017 and Lee2019_MI must already be in `~/mne_data` (MOABB `data_path`, provider `upstream`).

## Commands
```
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
pip install --no-deps ml_dtypes==0.5.1 flatbuffers==24.3.25 tosa-tools==2026.5.0   # TOSA serializer; --no-deps is load-bearing
python -m pytest tests/                 # must pass before any PR
python -m src.train --mode loso --subject 1 --seed 0 --out runs/loso_s1_seed0
python -m src.quantize --run runs/loso_s1_seed0          # int8 (fake-quant) predictions for the run
python -m src.run_all --jobs 6                           # all runs + results/results.md
python -m src.export --checkpoint runs/loso_sN_seed0/model.pt --subject N --out artifacts/tinyeegnet_int8.pte
# v2 (configs/v2.yaml; Cho2017 + Lee2019_MI must be downloaded first)
python -m src.train_teacher --seed 0 --out runs/v2/teacher_seed0 --device mps
python -m src.distill --seed 0 --student wide32 --teacher runs/v2/teacher_seed0 --out runs/v2/student_wide32_seed0 --device mps
python -m src.quantize --run runs/v2/student_wide32_seed0             # int8 (fake-quant) predictions
python -m src.run_v2 --jobs 1 --seeds 0 1 2 --students wide32 control  # all v2 runs + results/results_v2.md
python -m src.export --checkpoint runs/v2/student_wide32_seedK/model.pt --run-dir runs/v2/student_wide32_seedK --subject 3 --out artifacts/v2/tinyeegnet_int8.pte
```
Large files (`.pt`, `.pte`, `.npz`, `~/mne_data`) are git-ignored; attach them to the GitHub release.

## Shared rules (identical in every ArmExecuTorchEEG repo)

### 1. Before touching anything
- Read `README.md` and this file. Then `git fetch && git status`.
- Work on a branch named `<yourname>/<topic>`. Never commit on `main`.
- One task per branch. One branch per agent session. Two sessions on one repo → two worktrees.

### 2. Ownership
See the table at the top of this file. Touching a path you don't own: open a PR, tag the owner, don't merge it yourself.
Root config, lockfiles, `requirements*.txt`, `.gitignore`, `README.md`, this file: PR only, never direct.

### 3. Contracts
Files other repos or people depend on are listed at the top of this file (shapes, file names, UART frame, params, release tags).
Changing one: update the README in the same PR and post in the team chat before merging. Never rename or delete a contract file.

### 4. Commits and pushes
- Commit only files you changed for this task. No `git add .` across unrelated files.
- Never commit: `.env`, tokens, datasets, `.pte`, checkpoints, generated `ai_layer/`, worktree dirs. If one slips in, say so; don't rewrite history.
- Message: `<area>: <what>` + one body line of why. Agents append their session link.
- Push your own branch freely. Never force-push a branch someone else has pulled. `--force-with-lease` only on your own branch.

### 5. Merging
- PR into `main`. Tests pass. Self-merge only when the diff touches only your own paths; otherwise one approval.
- Several PRs open: merge contracts → data → model → export → firmware → docs, one at a time.
- Before merge: `git rebase main` on your branch, re-run tests.

### 6. Conflicts
- Never resolve by discarding the other side. Same lines changed by both: stop, post the file and both diffs, let the two humans decide.
- `main` is never rewritten. Fix forward with a new commit.

### 7. Agents specifically
- Don't push, open PRs, merge, or delete branches unless the human in the session asked.
- Don't modify files outside this task's listed paths, even to "clean up". Don't reformat files you didn't change.
- Don't run `rm -rf`, `git reset --hard`, `git clean`, `git checkout -- .` without an explicit instruction naming the target.
- Two failed fix attempts → stop and report; don't widen the change.
- Final report: files changed, commands run, test output, branch name, open doubts.

### 8. Keep this file true
Changed a command, path, or contract? Update this file in the same PR. Stale rules are worse than none.
