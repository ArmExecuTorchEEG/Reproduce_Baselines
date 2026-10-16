# Finished TinyEEGNet model packages

| Package | Intended use | Current deployment status |
|---|---|---|
| [01 Multi-subject](01_multi_subject_tinyeegnet/README.md) | One model trained across participants | Compiled for Alif E8/U85; FVP execution passed, but one logit differs by 0.026093 (above 0.01 tolerance) |
| [02 Participant 1](02_subject_01_tinyeegnet/README.md) | Participant 1 recording-based demonstration | Compiled for Alif E8/U85; four-window FVP numerical check passed |

Both are deployment candidates. Neither has been validated on physical E8 hardware.
`src/models/` contains Python architecture code; this directory contains selected
trained assets and their provenance. Never interchange their preprocessing files.

Each package contains byte-identical copies of:
- `best_checkpoint.pt`: original PyTorch checkpoint.
- `tinyeegnet_int8.pte`: compiled Alif E8 U85-256 model.
- `preprocess_params.json`: matching EEG normalization AND INT8 quantization.
- `normalization_params.json`: original float preprocessing; quantization is null.
- `model_config.json`, `label_mapping.json`, `training_metrics.json`, `export_report.json`.
- `ensemble_vela.ini`: the exact export memory profile.
- `test_vectors.npz`: original TRAINING-derived vectors with host Q/DQ expected
  logits; these are not the four development-validation vectors used for FVP.

`export_report.json` is an unchanged export-time report; its FVP status predates
the later successful tests. Follow each package README's actual FVP result link
for current execution status.

`provenance.json` lists each file's original location, byte count and SHA-256.
Original `artifacts/<run>/` directories remain authoritative and unchanged.

From the repository root:
```sh
python -m src.package_models                # create missing copies from original local runs
python -m src.package_models --verify-only  # verify copies and available original sources
```
The copier refuses to overwrite a changed file or provenance manifest. It never
trains, quantizes, or reads EEG. The exact `best_checkpoint.pt` and `tinyeegnet_int8.pte` pair in each package
is explicitly approved for Git and included on `feature/tinyeegnet-training`,
alongside documentation, provenance, JSON metadata and the Vela profile. Clone
this branch to obtain both models directly; use GitHub Raw/Download for individual
files. Optional `test_vectors.npz` remains Git-ignored and verification skips it
when absent. Original artifact directories remain local. No release/tag is
published; these are branch-hosted deployment candidates. Freshly retrained
weights may differ across hosts and must get a separate package/manifest.
