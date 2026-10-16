# Reproduce training, export and FVP checks

Run commands from the repository root. Commands create local data/binaries; never
force-add them to Git. Existing run names are immutable: use a fresh clone or a
new output directory. The completed models are tracked on feature/tinyeegnet-training and can be used
without retraining after hash verification. Only optional NPZ vectors and full
original export workspaces must be restored separately for FVP regeneration.

## Training environment and data

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.data.download --subjects all
python -m src.data.inspect_dataset
python -m unittest discover -s tests -v
```

Data remains under ignored `mne_data/`. The descriptive
`datasets/bci_competition_iv_2a/recordings` link resolves after download.
Preprocessing: ordered 22 EEG channels, 250 Hz, causal fourth-order-prototype
Butterworth 7–30 Hz (realized order 8), zero filter state at trial start with
3-second history, cue-relative [1,3) seconds/500 samples; left=0/right=1. Retain
and track artifact flags. Fit population channel mean/std on training only.

## Training

```sh
python -m src.train --config configs/default.yaml --run-name my-new-combined-run
python -m src.train_subject --config configs/subject1/baseline.yaml --run-name my-new-subject1-run
```

Both use fresh seed-42 weights, 1,938 parameters, ReLU, dropout 0.5, Adam,
lr 0.001, cross-entropy, batch 64, maximum 200 epochs, patience 20, CPU/4 threads.
Minimum validation loss selects the checkpoint. Multi-subject train: S1–5 both
sessions, S6–7 Session 1; validation S8; reserved demo S6–7 Session 2 and test S9.
Subject-specific: S1 Session 1 stratified 115/29, S1 Session 2 test only after
selection; new training-only normalization.

Compatibility detail: `src.train_subject` reads filter/metadata from the retained
`artifacts/combined-seed42-20261016/preprocess_params.json`, then replaces mean/std
with its own training statistics. The combined export-data helper and dropout CV
also reference that original named run. Those paths were intentionally preserved.
On a blank clone without original artifacts, first reproduce the combined run
using `--run-name combined-seed42-20261016`, then the subject-specific run using
`--run-name subject1-seed42-20261016` if you want the retained helper paths to work.
Do not use these existing names on this working machine. This path dependency is
documented rather than refactored during organization. Retrained checkpoints may
not match the original package hashes and must not replace the retained packages.

## Isolated official Arm exporter

Keep the training environment unchanged. Create a separate checkout/environment:

```sh
git clone https://github.com/Arm-Examples/CMSIS-Executorch.git /path/to/CMSIS-Executorch
# In that checkout:
git checkout 0a243f1ee054c64e3cc698f5a8c42c63ba3710c7
python3.12 setup_venv.py
```

This official setup pins torch 2.14.0, ExecuTorch 1.5.1, torchao 0.18.0 and Vela
5.1.0. Its intentional no-deps TOSA overrides cause documented pip-check conflicts;
use the existing proven environment rather than mixing it into `.venv`.

Prepare combined-model calibration/development data with the training environment:

```sh
.venv/bin/python -m src.data.export_data --out artifacts/my-new-combined-export
```

This helper targets the retained original combined run, not `my-new-combined-run`.
For a new Participant 1 training run:

```sh
.venv/bin/python -m src.data.subject_export_data \
  --run artifacts/my-new-subject1-run --out artifacts/my-new-subject1-export
```

Copy the matching proven Alif profile into each NEW export directory (or obtain
the same profile from the pinned Alif source recorded in the original report):

```sh
cp models_ready_for_alif_e8/02_subject_01_tinyeegnet/ensemble_vela.ini artifacts/my-new-subject1-export/ensemble_vela.ini
/path/to/CMSIS-Executorch/.venv/bin/python -m src.export \
  --official-repo /path/to/CMSIS-Executorch \
  --workdir artifacts/my-new-subject1-export \
  --checkpoint-dir artifacts/my-new-subject1-run \
  --config-ini artifacts/my-new-subject1-export/ensemble_vela.ini \
  --system-config Ethos_U85_SRAM_MRAM
```

For the combined original, substitute its new export directory and use
`--checkpoint-dir artifacts/combined-seed42-20261016`. Calibration is training-only;
Q/DQ comparison is development-only. Never evaluate reserved recordings for export
selection. Output is int8 `(1,1,22,500)` to float32 `(1,2)`, U85-256/Shared_Sram.

## FVP

[Actual FVP report](../results/fvp-execution-20261016/README.md) records full tool
versions, memory map, build/run/comparison commands and the one-step Model 1
numerical discrepancy. Original local generated project paths remain valid.
A fresh clone excludes those generated workspaces; recreate one from the official
hackathon checkout and the full matching local export run:

```sh
git clone --branch hackathon https://github.com/Arm-Examples/CMSIS-Executorch.git /path/to/CMSIS-Executorch-hackathon
# In that checkout, select a6364946165f21b27c1f5cf6998a701ded8ed73e.
/path/to/CMSIS-Executorch/.venv/bin/python -m src.fvp.prepare \
  --official /path/to/CMSIS-Executorch-hackathon \
  --export-run artifacts/my-new-subject1-export --out artifacts/my-new-subject1-fvp
```

The full export run needs export_data.pt, quantized_qdq.pt2, data_manifest.json,
export_report.json and PTE; the small finished-model package alone is not a full
FVP regeneration bundle. Install/activate official Keil tools, the matching
ExecuTorch 1.5.1 pack and Arm license. On macOS run Docker Desktop and personally
review/accept the FVP EULA. Existing machine setup is already complete.

```sh
export FVP_EULA_ACCEPTED=11.32.23  # only after personal acceptance
bash src/fvp/run.sh artifacts/my-new-subject1-fvp/project /tmp/my-new-model2.log
.venv/bin/python -m src.fvp.compare \
  --reference artifacts/my-new-subject1-fvp/reference.json \
  --log /tmp/my-new-model2.log --out /tmp/my-new-model2-comparison.json
```

Model 2 first; run Model 1 with its own development inputs/parameters afterward.
The runner loads HEX to avoid an ELF zero-fill warning. Never invoke the template
TinyCNN exporter on a staged EEG AI layer. FVP checks do not replace physical E8
bring-up, memory/latency verification or validation of streaming EEG preprocessing.
