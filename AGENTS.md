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

## Contracts
Model input `(1,1,22,500)` int8, output `(1,2)`. `artifacts/tinyeegnet_int8.pte`, `artifacts/test_vectors.npz`, `artifacts/preprocess_params.json`. Release tag `v0.1-baseline`. Consumed by Emulator_Deploy and Edge_Deploy.

## Commands
```
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m pytest tests/                 # must pass before any PR
python -m src.train --subject 1 --classes 2
python -m src.export --target ethos-u85-256 --out artifacts/tinyeegnet_int8.pte
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
