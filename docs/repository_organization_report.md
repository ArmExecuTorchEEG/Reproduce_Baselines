# Repository organization verification

Branch: feature/tinyeegnet-training. No commits or pushes.

## Changes

- Added two descriptive finished-model directories under models_ready_for_alif_e8.
  Each has 10 byte-identical copies: checkpoint, E8 PTE, deployment parameters,
  float normalization parameters, model config, labels, training metrics, export
  report, training test vectors and Vela profile. Provenance records original
  paths, SHA256 and byte counts. READMEs distinguish training allocations,
  evaluation datasets, FVP numerical limits and pending physical deployment.
- Added src/package_models.py for safe repeatable packaging and --verify-only.
  It preflights destinations and refuses changed-copy/provenance overwrites.
- Added datasets/bci_competition_iv_2a/README.md and a relative recordings symlink
  into the existing cache. No dataset files were copied, renamed or moved.
- Added results/README.md and model_training/, accuracy_experiments/,
  hardware_validation/ indexes. Every original run result directory remains
  in place; the hardware index distinguishes final reports and historical attempts.
- Replaced the main README's planned-layout inventory with the actual directory
  guide, model status and practical clone workflow. Detailed commands live in
  docs/reproduce_workflows.md. Updated AGENTS.md to reflect the navigation paths.
- Updated .gitignore to allow only the dataset guide/link under datasets, ignore
  recording contents, and explicitly ignore finished-package NPZ/PT2 assets.
  Existing global PT/PTE ignores and ignored artifacts/mne_data paths remain.

## Verification

- Before changes: 23 complete offline unit tests passed; both original checkpoints
  loaded and accepted (1,1,22,500) -> (1,2).
- After changes: 23 complete offline unit tests passed; both packaged checkpoints
  loaded with the same forward shape. No dataset/test inference was repeated.
- SHA256 verified all 1,000 original files protected before organization:
  artifacts, results, mne_data, src, configs and tests (excluding pycache).
- Package verification confirmed all 20 copied files match their manifests and
  original sources, with float/deployment normalization/channel/label consistency.
- Relative dataset link resolves to exactly all 18 A01–A09 T/E MAT files.
- Both existing FVP wrappers, configurations, HEX images and reference files exist
  at their documented paths. FVP was not rerun because no inference code, model,
  firmware or original path changed.
- New Markdown navigation links resolve locally. Broken links on a fresh clone
  to ignored binary assets are expected until local assets/data are restored.
- git check-ignore confirms package PT/PTE/NPZ, original EEG and accidental data
  copies under datasets are excluded; guide/link, provenance and JSON metadata
  are eligible for Git. git diff --check passed.

Commands used: git fetch/status/branch; path-reference rg audit; complete
`python -m unittest discover -s tests -v` before and after; checkpoint reload and
forward checks; `python -m src.package_models` and `--verify-only`; SHA256,
symlink/data inventory, FVP-path and Markdown-link checks; git check-ignore and
untracked-file listing. Details: repository_organization_verification.json.
Local full hash audit/test logs: ignored artifacts/repository-organization-20261016/.

## Git contents and remaining limitations

Intended Git files: source/config/tests from the prior work, the new packaging
utility, README/AGENTS/.gitignore edits, model package READMEs/provenance/small JSON
metadata/Vela profiles, dataset README and relative symlink, result indexes,
existing reports/plots/logs and reproduction/verification docs.

Excluded: checkpoints, PTEs, NPZ tensors/vectors, local original artifact runs,
generated firmware/FVP workspaces, downloaded tool binaries, raw EEG, venv/cache.
A clone does not contain a runnable model binary or the actual dataset. No release
has been published: restore matching assets from the team or reproduce separate
runs; verify their hashes before deployment. Retrained weights may differ and
must not overwrite the retained package manifest.

Timestamped historical directories remain intentionally unchanged. Some helpers
still use the exact retained combined-run path, as documented in the reproduction
guide. The navigation symlink assumes the current MOABB upstream cache layout;
Windows may require Git symlink support. No source refactor was done to hide
these compatibility constraints. Export reports retain their export-time FVP
status; the actual later FVP report is the current execution evidence.

Model 1 still exceeds the 0.01 FVP-versus-Q/DQ logit tolerance (0.026093); Model 2
passes on four validation inputs. Neither is validated on physical E8. No
training results, model weights or quantization settings were changed.

The pre-existing working tree already contained uncommitted/untracked project
code, configs, tests and results. Those remain intact; nothing was staged.
