# Reproduce_Baselines — EdgeBCI

Train TinyEEGNet to classify left/right motor imagery from BCI Competition IV
Dataset 2a, then compile INT8 inference for the Alif Ensemble E8's Ethos-U85.
Two independently trained models are available locally. Both have executed in
Arm's Corstone-320 FVP; physical E8 execution is still pending.

## Find what you need

| Location | Purpose |
|---|---|
| [EEG recordings](datasets/bci_competition_iv_2a/README.md) | Clear dataset entry point; relative link into the existing ignored `mne_data/` cache |
| [Model 1: multi-subject](models_ready_for_alif_e8/01_multi_subject_tinyeegnet/README.md) | Selected checkpoint, Alif `.pte`, matching parameters and provenance |
| [Model 2: Participant 1](models_ready_for_alif_e8/02_subject_01_tinyeegnet/README.md) | Separate checkpoint, Alif `.pte`, matching parameters and provenance |
| [Training results](results/model_training/README.md) | Both models' metrics, protocols and loss curves |
| [Accuracy experiments](results/accuracy_experiments/README.md) | Participant-held-out dropout comparison; retained dropout 0.5 |
| [Hardware validation](results/hardware_validation/README.md) | Compilation, actual FVP outputs, numerical comparisons and setup history |
| [Reproduction guide](docs/reproduce_workflows.md) | Training, isolated Arm export and FVP commands, including fresh-clone requirements |

```text
Reproduce_Baselines/
├── models_ready_for_alif_e8/          # finished trained model packages
│   ├── 01_multi_subject_tinyeegnet/
│   └── 02_subject_01_tinyeegnet/
├── datasets/bci_competition_iv_2a/
│   ├── README.md
│   └── recordings -> ../../mne_data/MNE-bnci-data/~bci/database/001-2014
├── src/
│   ├── data/                        # download, inspect, preprocess, split/calibrate
│   ├── models/tinyeegnet.py          # Python architecture, not trained assets
│   ├── experiments/dropout_cv.py     # completed regularization comparison
│   ├── fvp/                         # reusable preparation, run and comparison harness
│   ├── train.py                     # combined-participant training
│   ├── train_subject.py             # Participant 1 training
│   ├── eval.py  export.py
│   └── package_models.py             # copy/verify selected assets without overwriting
├── configs/                         # original defaults and subject1/baseline.yaml
├── tests/                           # offline automated suite
├── results/
│   ├── model_training/              # descriptive index
│   ├── accuracy_experiments/        # descriptive index
│   ├── hardware_validation/         # descriptive index
│   └── <original-run-directories>/  # preserved reports, metrics and logs
├── artifacts/<original-runs>/       # ignored originals and generated FVP workspaces
├── mne_data/                        # ignored actual EEG files; loader paths unchanged
└── docs/reproduce_workflows.md
```

Original timestamped run directories remain in place because code and provenance
reference them. The descriptive results folders are indexes; no historical results
were moved. Finished-model files are hash-verified copies, not renamed originals.

## Model status

| | Model 1: multi-subject | Model 2: Participant 1 |
|---|---|---|
| Training | S1–5 both sessions; S6–7 Session 1 (1,728 trials) | S1 Session 1, stratified 115 train / 29 validation |
| Validation accuracy | S8: 93.06% | S1 Session 1: 75.86% |
| Held-out accuracy | S6–7 Session 2: 62.15%; S9: 59.03% | S1 Session 2: 79.86% |
| Compiled E8 PTE | 13,056 bytes | 13,072 bytes |
| Actual FVP vs host Q/DQ, four validation inputs | 100% prediction agreement; max logit error 0.026093 **exceeds 0.01** | 100% agreement; max error 2.28e-9 **passes 0.01** |
| Physical E8 | Pending | Pending; promising Participant 1 demo candidate |

The evaluations use different people/session allocations, so these numbers do not
establish which model is superior. FVP results are small execution smoke tests,
not full-dataset accuracy. Float-versus-Q/DQ discrepancies remain documented;
no verified release or `v0.1-baseline` tag has been published.

Both use the same 1,938-parameter TinyEEGNet: temporal convolution, depthwise
spatial convolution, separable convolution, ReLU, average pooling, dropout 0.5
and two logits. BatchNorm folds at export. All model compute delegates to U85;
output dequantization runs on CPU. Experiments 1 (structural ablation) and 2
(sensor dropout) remain unimplemented.

EEG preprocessing: ordered 22 channels, 250 Hz, causal 7–30 Hz Butterworth,
cue-relative [1,3) seconds, training-only channel mean/std, retain artifact flags.
PyTorch input is float32 `(N,1,22,500)`; deployment input is signed int8
`(1,1,22,500)`, output float32 `(1,2)`, left=0/right=1. **Use each model's own
`preprocess_params.json`; never interchange normalization or quantization.**

## Start from a fresh clone

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.data.download --subjects all
python -m src.data.inspect_dataset
```

The dataset link becomes usable after download. Git includes code, guides,
provenance, small JSON metadata and the two explicitly selected checkpoint/PTE
pairs under `models_ready_for_alif_e8/`. Cindy and Moamen can clone this branch
or download either package directly from GitHub. Other checkpoints/PTEs, NPZ
vectors, raw EEG and FVP build workspaces stay local. Run
`python -m src.package_models --verify-only` after cloning. Optional training
test vectors remain local; no release tag is required to download the tracked models. If the original local run folders are available,
`python -m src.package_models` creates missing convenience copies and refuses to
overwrite changed files. Every copy's original path and SHA-256 are recorded in
its package's `provenance.json`.

For new training runs, use unused names:

```sh
python -m src.train --config configs/default.yaml --run-name my-new-combined-run
python -m src.train_subject --config configs/subject1/baseline.yaml --run-name my-new-subject1-run
```

See [the reproduction guide](docs/reproduce_workflows.md) first: retained helpers
reference the original combined run's metadata, so a blank clone must restore or
reproduce that named baseline before the subject-specific workflow. The Arm
exporter uses a **separate** official environment (ExecuTorch/runtime 1.5.1,
torch 2.14.0, torchao 0.18.0, Vela 5.1.0). Do not mix it into the pinned training
environment. FVP additionally requires activated Arm tools/licenses and Docker
on macOS; exact commands and memory configuration are in the linked FVP report.

The downstream release contract remains `artifacts/tinyeegnet_int8.pte`,
`artifacts/test_vectors.npz`, `artifacts/preprocess_params.json`, input/output
shapes above, tag `v0.1-baseline`. Those canonical release paths were not replaced
by these convenience packages. Physical E8 bring-up, on-board numerical checks,
memory/latency verification and streaming EEG integration are the next steps.
