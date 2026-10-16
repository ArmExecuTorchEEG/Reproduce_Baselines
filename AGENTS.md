# AGENTS.md — Reproduce_Baselines

Rules for people and coding agents (Claude Code, Codex, Cursor, Copilot) working in this repo. `CLAUDE.md` imports this file.
Project overview and repo map: https://github.com/ArmExecuTorchEEG

## Ownership
| Path | Owner | Others may |
|---|---|---|
| `src/data/` | TBD | PR |
| `src/models/`, `src/train.py`, `src/eval.py` | TBD | PR |
| `src/quantize.py`, `src/export.py`, `src/verify_pte.py` | TBD | PR |
| `src/experiments/` | TBD | PR |
| `configs/default.yaml` | TBD | PR; never change `seed`, `window`, `fs` silently |
| `results/` | author of the run | add only; never overwrite another run's file |

## Combined float baseline
Train one model: subjects 1–5 both sessions plus 6–7 session 1 (1,728 trials).
Validate/select on subject 8 both sessions (288). Reserve 6–7 session 2 for
demonstration (288), and subject 9 both sessions for final testing (288).
Normalize with training-only channel statistics; retain and track artifact flags.
Defaults: seed 42, 250 Hz, cue-relative [1,3) seconds, causal 7–30 Hz filtering.
Run outputs live in ignored `artifacts/<run>/`; metrics in new `results/<run>/`.
`python -m src.data.download --subjects all` uses ignored `mne_data/`.
The original subject-1 diagnostic remains `python -m src.data.preprocess --seed 42`.
Offline tests: `python -m unittest discover -s tests -v`.
Investigative export is implemented; see README for the isolated official Arm environment.
Four-window FVP execution checks passed for both models; Model 1 exceeds the
0.01 logit tolerance. Physical E8 execution remains unvalidated; no verified release.

## Controlled dropout CV
`python -m src.experiments.dropout_cv --run-name <unique-run-name>` compares
activation dropout 0.5/0.25 using held-out groups {1,4}, {2,5}, {3,6,7}.
Use original training recordings only; fit normalization separately per fold.
Never load/evaluate S6–7 session 2 or S9 in this experiment. Mean participant
balanced accuracy selects the configuration under the predeclared rule in README.
A conditional final refit uses S8 only for early stopping after fixing dropout.
Keep the original baseline files unchanged; new runs must have distinct paths.

## Participant 1 run
`python -m src.train_subject --config configs/subject1/baseline.yaml --run-name <new-run>`
reuses the original architecture/training settings with fresh weights. Session 1
is stratified seed-42 into 115 training / 29 validation trials; fit new training-only
normalization. Load Session 2 only after checkpoint selection; evaluate once.
Keep all multi-subject and subject-specific artifacts in separate run directories.
`src.export --checkpoint-dir <subject-run>` can export the separate checkpoint;
calibrate on its training split only and compare Q/DQ on its validation split.

## FVP integration
`src.fvp.prepare` creates a separate official-hackathon project with unchanged
PTE and development-only EEG vectors. `src/fvp/run.sh` builds/runs it only after
user-controlled Arm/Docker/FVP license setup; `src.fvp.compare` verifies outputs.
Current results/commands: `results/fvp-execution-20261016/README.md`. Model 2 first;
Model 1 only after actual Model 2 success. Do not run the example model exporter
over the staged AI layer. The runner uses the generated HEX image to avoid an
ELF zero-fill loading warning; the local Docker shim supports macOS Bash 3.2.

## Navigation and retained assets
`models_ready_for_alif_e8/01_multi_subject_tinyeegnet/` and
`02_subject_01_tinyeegnet/` contain selected byte-identical convenience copies.
`python -m src.package_models --verify-only` checks their provenance manifests;
without that flag it creates missing copies and refuses changed-file overwrites.
Original artifact/run paths remain authoritative; never interchange model params.
`datasets/bci_competition_iv_2a/recordings` is a relative navigation link into the
existing ignored `mne_data/` cache; loaders continue using mne_data unchanged.
`results/model_training/`, `accuracy_experiments/`, `hardware_validation/` are
indexes: never move or overwrite the historical results they link to. See
`docs/reproduce_workflows.md` for fresh-clone setup and retained path dependencies.
By explicit user authorization, only best_checkpoint.pt and tinyeegnet_int8.pte
in the two named models_ready_for_alif_e8 packages are tracked in Git. Other
PT/PTE/NPZ binaries and datasets remain Git-ignored; small metadata,
provenance manifests and guides are intended for Git. No release is published.

## Contracts
Model input `(1,1,22,500)` int8, output `(1,2)`. `artifacts/tinyeegnet_int8.pte`, `artifacts/test_vectors.npz`, `artifacts/preprocess_params.json`. Release tag `v0.1-baseline`. Consumed by Emulator_Deploy and Edge_Deploy.

## Commands
```
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m pytest tests/                 # must pass before any PR
python -m src.train --config configs/default.yaml --run-name <unique-run-name>
python -m src.data.export_data --out artifacts/<new-export-run>
# In the official Arm template environment, with its checkout and Alif profile:
python -m src.export --official-repo /path/to/CMSIS-Executorch --workdir artifacts/<new-export-run> --config-ini artifacts/<new-export-run>/ensemble_vela.ini --system-config Ethos_U85_SRAM_MRAM
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
- Never commit `.env`, tokens, datasets, generated `ai_layer/`, worktree dirs or arbitrary checkpoints/PTEs. Exception explicitly authorized by the user: the two final best_checkpoint.pt and tinyeegnet_int8.pte pairs in models_ready_for_alif_e8/01_multi_subject_tinyeegnet and 02_subject_01_tinyeegnet. All other binaries remain excluded.
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
