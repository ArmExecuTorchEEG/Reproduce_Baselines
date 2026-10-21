
# EEG-BCI Pipeline Development

**EEG Signal Processing | Neural Decoding | Embedded AI | ExecuTorch | Hardware Integration**

Development and experimentation repository for an EEG-based brain-computer interface (BCI), covering neural signal preprocessing, machine learning, model optimization, embedded inference, firmware integration, and interactive visualization.

This repository preserves the technical development behind **[CorteX ARM](https://github.com/ArmExecuTorchEEG/CorteX-ARM)**, a finalist project at the **Arm + ExecuTorch Edge AI Challenge (October 2026)**.

The project explores how neural signals associated with imagined left- and right-hand movements can be decoded by compact neural networks running on resource-constrained embedded hardware.

The final demonstration used genuine prerecorded EEG signals from publicly available datasets, with predictions generated on the **Alif Ensemble E8** and displayed through an interactive browser interface.

## Project Overview

The development process covered the end-to-end engineering workflow required to translate EEG signals into embedded AI predictions.

Key workstreams included:

- **EEG processing:** Channel selection, temporal filtering, referencing, normalization, and motor-imagery trial extraction.
- **Neural decoding:** Training and evaluating compact convolutional neural networks for left-versus-right motor imagery.
- **Model optimization:** Cross-subject generalization, Euclidean Alignment, knowledge distillation, and INT8 quantization.
- **Embedded deployment:** Exporting PyTorch models through ExecuTorch and compiling inference operations for the Arm Ethos-U85 NPU.
- **Hardware integration:** Firmware development, UART communication, physical-board testing, and numerical validation.
- **Frontend development:** Displaying recorded EEG, task cues, predictions, cursor visualization, and session statistics.

The repository documents both the initial development implementations and the more advanced neural decoder incorporated into the final project.

## System Architecture

The final system uses 22-channel EEG recordings to classify imagined hand movements.

### Neural decoding pipeline

```text
            EEG Recordings
      BCI Competition IV Dataset 2a
                  |
                  v
           EEG Preprocessing
       - Common Average Reference
       - 7–30 Hz Band-pass Filtering
       - Decimation to 125 Hz
       - Trial Window Extraction
       - Euclidean Alignment
       - Scaling and Quantization
                  |
                  v
         TinyEEGNet-wide Student
          (3,426 parameters)
                  |
                  v
         ExecuTorch / Ethos-U85
                  |
                  v
        Left / Right Classification
                  |
                  v
          Interactive BCI Demo
       EEG Display / Cursor / Metrics
```

### Embedded execution paths

Two related hardware-software implementations were developed and validated at different stages.

| Component | Physical Browser Demonstration | Embedded Preprocessing Implementation |
|---|---|---|
| EEG source | Prerecorded EEG | Prerecorded raw EEG samples |
| Laptop | Preprocessing, prepared-window streaming, visualization | Raw sample replay |
| Embedded CPU | UART communication, tensor preparation, ExecuTorch runtime | Referencing, filtering, decimation, windowing, alignment, quantization |
| Neural inference | Ethos-U85 NPU | Ethos-U85 NPU |
| Validation | Physical Alif E8 demonstrations | Corstone-320 FVP self-test |

**Physical demonstration:** The laptop prepares quantized EEG windows and transmits them to the Alif E8. The embedded application performs neural-network inference and returns predictions to the browser.

**Embedded preprocessing implementation:** A separate firmware implementation processes raw EEG samples on the embedded CPU before NPU inference. This complete processing path was validated using the Arm Corstone-320 FVP; physical execution of the complete raw-signal path was not established in the recorded results.

The demonstrated browser cursor represents the classifier's predicted direction. It does not control the operating system's mouse pointer.

For the complete submitted system architecture, see [CorteX ARM Architecture](https://github.com/ArmExecuTorchEEG/CorteX-ARM/blob/main/docs/architecture.md).

## Neural Decoding Models

### Development Prototypes

The initial development used compact TinyEEGNet models trained on BCI Competition IV Dataset 2a.

These included combined-participant and subject-specific classifiers operating on 22 EEG channels sampled at 250 Hz.

Their primary role in the project was to establish and validate the workflow for:

- EEG preprocessing and model training.
- Classification and generalization experiments.
- Model checkpoint packaging and reproducibility.
- ExecuTorch export and quantization.
- Fixed Virtual Platform execution.
- Firmware integration and physical hardware bring-up.
- Continuous EEG streaming and frontend interaction.

These models provided the functional foundation for the embedded system before the more advanced generic decoder was integrated.

[Original Training and Model Development](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/tree/feature/tinyeegnet-training)

### Final Neural Decoder

The final prototype uses **TinyEEGNet-wide `wide32_ft`**, a knowledge-distilled student network optimized for cross-participant EEG decoding and embedded deployment.

Its training incorporates:

1. Supervised training on the Cho2017 and Lee2019_MI motor-imagery datasets.
2. An offline teacher ensemble containing EEGNet, ShallowConvNet, DeepConvNet, and a Riemannian classifier.
3. Knowledge distillation into a compact student network.
4. Unlabeled target-domain adaptation using BCI Competition IV 2a recordings.
5. Student fine-tuning and INT8 post-training quantization.
6. Export through ExecuTorch for Arm Ethos-U85 inference.

Only the compact student network is deployed to the embedded device. Teacher models are used offline during training.

| Property | Final Model |
|---|---|
| Architecture | TinyEEGNet-wide, `wide32_ft` |
| Parameters | 3,426 |
| Input | 22 EEG channels × 256 samples |
| Sampling frequency | 125 Hz |
| Input tensor | `(1, 1, 22, 256)` |
| Output | Left/right classification logits |
| ExecuTorch program size | 11,456 bytes |
| Final model version | `v0.5-distill`, seed 0 |
| Target hardware | Arm Ethos-U85 on Alif Ensemble E8 |

The model uses subject-specific Euclidean Alignment parameters to adapt recordings from different participants without requiring an independently trained neural network for each individual.

The final demonstration uses the **unfused distilled student**. The later `v0.5-fused` model export is a separate experiment and was not the deployed model used in the submitted demonstration.

**Technical documentation:**
- [Model Architecture](src/models/tinyeegnet.py)
- [Knowledge Distillation](src/distill.py)
- [Student Fine-Tuning](src/finetune_student.py)
- [Final Model Results](results/results_v4soft.md)
- [Model Deployment Handoff](docs/v4_handoff.md)

## Embedded AI and Hardware Integration

A major engineering objective was to move beyond desktop-based EEG classification and execute neural decoding models on embedded hardware.

### Model Deployment

The deployment workflow included:

- Preparing trained model checkpoints and preprocessing parameters.
- Applying post-training INT8 quantization.
- Exporting model graphs through ExecuTorch.
- Compiling supported neural-network operations for the Arm Ethos-U85.
- Validating inference through the Arm Corstone-320 FVP.
- Comparing embedded predictions against reference implementations.
- Preparing the model and firmware for execution on the Alif Ensemble E8.

The deployed neural-network operations execute on the NPU. Protocol handling, runtime coordination, and tensor conversion remain CPU responsibilities.

For the final model, the UART protocol transmits quantized INT8 windows, while the ExecuTorch program accepts float32 input tensors and performs internal quantization before neural execution.

### Firmware and Physical-Board Validation

Development included firmware integration and communication between the embedded hardware and host computer.

Key engineering areas included:

- UART communication and binary message handling.
- Model and input-tensor verification.
- Physical E8 inference testing.
- Prediction and logit comparison against reference outputs.
- Communication debugging and error handling.
- Continuous prerecorded EEG-trial streaming.
- Synchronization between hardware responses and the frontend.

Earlier subject-specific firmware completed **160 physical inference requests**, including 144 unique EEG trials and 16 repeated trials.

All 160 returned predictions matched the corresponding FVP references, with the complete returned logits also matching in the recorded validation.

This established a functioning physical inference and streaming workflow that supported subsequent integration of the final generic model.

[Physical E8 Streaming Validation](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/blob/feature/model2-random-eeg-trials/tools/model2_random_trials/PHYSICAL_STREAMING_VALIDATION.md)

### Advanced Model Integration

The final distilled neural decoder was subsequently integrated with the embedded inference and browser demonstration workflow.

This development included:

- Matching the final model's tensor and quantization requirements.
- Preparing EEG windows and model-specific alignment parameters.
- Implementing and testing model-aware UART communication.
- Validating predictions against saved reference outputs.
- Supporting trials across all nine BCI Competition IV 2a participants.
- Connecting hardware inference results to the frontend.
- Preparing the final demonstration interface and session summaries.

The complete prepared evaluation pool contains **1,080 recordings** from nine participants.

[Multi-Subject Integration Branch](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/tree/feature/multisubject-eeg-trials)

## Results and Validation

The repository preserves different validation stages, including model training, desktop inference, FVP execution, and physical hardware demonstrations.

These results should be interpreted according to their respective environments and evaluation protocols.

### Selected Results

| Experiment | Result | Validation Environment |
|---|---|---|
| Final student, nine participants × three seeds | **74.6% INT8 mean accuracy** | Desktop quantized reference |
| Deployed model, seed 0 | 74.44% (804/1,080) | Saved desktop reference |
| Deployed model, seed 0 | 74.26% (802/1,080) | Arm Corstone-320 FVP |
| Final generic-model demonstration | **77.1% (54/70)** | Physical Alif E8 |
| Earlier subject-specific model | 79.17% (114/144 unique trials) | Physical Alif E8 |

The reported 74.6% value is a mean over three training seeds. The final shipped seed achieved approximately 74.44% in the saved desktop reference evaluation.

The final model was selected following exploratory comparison of multiple candidate variants using the evaluation block. The reported accuracy may therefore be optimistic due to model selection.

Unlabeled BCI Competition IV 2a recordings were used during adaptation; target labels were not used as training targets.

### Numerical Validation

The advanced generic-model FVP evaluation covered all 1,080 prepared windows.

- TOSA and FVP integer outputs agreed across the full evaluation pool.
- Desktop and FVP classifications agreed on 1,078 of 1,080 trials.
- Two classification differences and additional floating-point logit discrepancies were documented.

The recorded tests therefore support successful embedded execution and backend verification, without claiming identical desktop and FVP outputs in every representation.

[Generic-Model Numerical Validation](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/blob/feature/final-ui-design/tools/multisubject_trials/NUMERICAL_VALIDATION_REPORT.md)

[Final Physical-Board Metrics](https://github.com/ArmExecuTorchEEG/CorteX-ARM/blob/main/results/board_metrics.md)

[FVP Self-Test](https://github.com/ArmExecuTorchEEG/CorteX-ARM/blob/main/results/fvp_selftest.md)

Physical end-to-end latency, power consumption, and complete on-device raw EEG preprocessing remain outside the confirmed physical-board measurements.

## Frontend and Visualization

The demonstration interface connects the EEG classification pipeline with interactive visual feedback.

The final frontend includes:

- Prerecorded EEG waveform visualization.
- Electrode and scalp activity displays.
- Motor imagery task cues and ground-truth labels.
- Actual classification predictions returned from the hardware.
- In-page cursor movement corresponding to predicted direction.
- Running accuracy and session summaries.
- Session controls and recorded result handling.

The interface distinguishes genuine hardware responses from simulated or replayed results.

Prediction identities, model configuration, and request/response information are validated during hardware communication.

Visual animation timing is separate from hardware inference latency.

The final interface development is preserved in:

**[feature/final-ui-design](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/tree/feature/final-ui-design)**

## Development History and Branches

The repository preserves separate development and research branches documenting the progression from early experiments to the final integrated system.

The `main` branch focuses on model research and optimization, while other branches retain substantial firmware, hardware, streaming, and frontend implementation work.

| Branch | Development Focus |
|---|---|
| [`main`](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/tree/main) | Cross-subject baselines, model architectures, knowledge distillation, evaluation, and optimization |
| [`feature/tinyeegnet-training`](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/tree/feature/tinyeegnet-training) | Preliminary model development, training, packaging, and export workflows |
| [`demo/edgebci-model2-4trial`](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/tree/demo/edgebci-model2-4trial) | Early physical inference and frontend demonstration |
| [`feature/model2-random-eeg-trials`](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/tree/feature/model2-random-eeg-trials) | Continuous EEG-trial streaming, UART development, and physical hardware validation |
| [`feature/multisubject-eeg-trials`](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/tree/feature/multisubject-eeg-trials) | Final-model integration and full-pool FVP validation |
| [`demo/edgebci-multisubject-working`](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/tree/demo/edgebci-multisubject-working) | Working multi-subject integration and demo checkpoint |
| [`feature/session-summary`](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/tree/feature/session-summary) | Frontend session-summary components |
| [`feature/final-ui-design`](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/tree/feature/final-ui-design) | Final interface integration, presentation, and session summaries |

These branches are preserved as development milestones rather than interchangeable releases. Different branches may use distinct model architectures, preprocessing configurations, and validation procedures.

## Contributors

The development and research were conducted collaboratively across separate workstreams.

| Contributor | Documented Contributions |
|---|---|
| **[Philippine Chauvet](https://github.com/philippinech3025)** | Preliminary model development and reproducibility, embedded deployment preparation, firmware/UART integration, physical hardware validation, numerical investigation, EEG streaming, browser/hardware integration, and final interface integration |
| **[Moamen Soliman](https://github.com/msoliman6)** | Cross-subject baseline research, advanced neural decoding models, training, knowledge distillation, model optimization and export, embedded preprocessing firmware, and final repository integration |
| **[Cindy Ku](https://github.com/jihyungku)** | Frontend UI/UX design, layout, EEG visualization, session-summary development, and presentation components |

Contribution attribution reflects Git history, implementation records, and documented source provenance. Imported or merged files should not be interpreted as exclusively authored by the person who performed the import or merge.

### Development Provenance

Selected implementation commits associated with Philippine Chauvet:

- [Training models and reproducible workflows](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/commit/b833fb7)
- [Physical hardware and frontend integration](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/commit/c85ab70)
- [Continuous physical EEG streaming validation](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/commit/f9c40dd)
- [Generic-model streaming integration](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/commit/7f8699a)
- [Working multi-subject demo checkpoint](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/commit/19dc889)
- [Final interface and session-summary integration](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/commit/08c93bc)

The [CorteX ARM board-demo documentation](https://github.com/ArmExecuTorchEEG/CorteX-ARM/blob/main/board_demo/README.md) identifies the corresponding development sources incorporated into the final submission.

## Reproducibility

The repository retains model implementations, training configurations, evaluation reports, deployment documentation, and integration workflows.

Because the development branches contain different implementations, reproduction should begin with the documentation for the specific model or experiment.

### Advanced Model Research

- [Training Configuration](configs/v4soft.yaml)
- [Training and Distillation](src/run_v2.py)
- [Student Fine-Tuning](src/finetune_student.py)
- [Quantization](src/quantize.py)
- [Model Export](src/export.py)
- [Model Results](results/results_v4soft.md)

### Embedded and Frontend Development

- [Original Reproduction Guide](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/blob/feature/final-ui-design/docs/reproduce_workflows.md)
- [Physical Streaming Validation](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/blob/feature/model2-random-eeg-trials/tools/model2_random_trials/PHYSICAL_STREAMING_VALIDATION.md)
- [Multi-Subject Validation](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/blob/feature/final-ui-design/tools/multisubject_trials/NUMERICAL_VALIDATION_REPORT.md)
- [Final UI Development](https://github.com/ArmExecuTorchEEG/EEG-BCI-Pipeline-Development/tree/feature/final-ui-design/frontend)

Some raw EEG recordings, generated model artifacts, and firmware build assets are intentionally excluded from Git. Reproducing physical demonstrations may require additional generated files, matching model parameters, Arm development tools, and hardware-specific dependencies.

## Final Project — CorteX ARM

The work documented here contributed to **CorteX ARM**, selected as one of four finalist teams at the Arm + ExecuTorch Edge AI Challenge.

The final prototype demonstrates embedded EEG neural decoding using the Alif Ensemble E8, with inference predictions connected to interactive cursor visualization.

The final submission repository contains the integrated model, firmware, demonstration code, technical architecture, validation results, and demonstration media.

**[View CorteX ARM — Final Project](https://github.com/ArmExecuTorchEEG/CorteX-ARM)**

**[View Hackathon Submission Snapshot](https://github.com/ArmExecuTorchEEG/CorteX-ARM/tree/v1.0-submission)**

---

*Research and engineering prototype. The demonstrated system uses prerecorded EEG signals and is not a clinically validated medical device.*
