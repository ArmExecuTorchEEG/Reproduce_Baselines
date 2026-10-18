# Local assets required beyond Git

These paths are relative to the frontend worktree unless stated otherwise. None
of the datasets, binaries, archives, environments or machine-specific settings
listed here are included in the physical-validation commit. A fresh clone alone
does not recreate the current installation. Back up these assets separately.

- Original four-trial rollback: `artifacts/streaming-rollback/validated-four-trial.tar.gz`
  and adjacent `manifest.json`. Archive SHA256:
  `90434792bdc757e225c78136e1c5261dd56acb6e3d5a51a1b8fc192d73c36015`.
  Supplemental recipe is tracked at `validation/validated-four-trial.cbuild-run.yml`
  relative to this directory. Verified extraction: `artifacts/rollback-four-trial-verified/`.
- Tested streaming image: `artifacts/streaming-firmware/project/out/cmsis-executorch/DevKit-E8/Debug/cmsis-executorch.hex`
  and `cmsis-executorch.axf` in the same folder. Hashes: `validation/physical-streaming-build.json`.
- Complete generated streaming source/configuration workspace:
  `artifacts/streaming-firmware/project/`, including `board/`, `src/`, `ai_layer/`,
  solution/project/layer files, version locks, `.cmsis/` tool environment and
  `out/cmsis-executorch+DevKit-E8.cbuild-run.yml`. This includes original board
  support and unchanged ExecuTorch/model files that the preparer imports locally.
- Source inputs for regeneration: sibling original `Reproduce_Baselines` worktree,
  especially `artifacts/fvp-subject1-20261016-ready/`, original selected Model 2
  training/export directories, training normalization parameters and source EEG cache.
  Keep it read-only; do not interchange Model 1 and Model 2 assets.
- Prepared 144-window dataset and reference files: `artifacts/model2-session2-phase1/`.
  Exact browser assets: `frontend/streaming/data/{inputs.bin,eeg.f32,metadata.json}`.
  Tensor bytes and display EEG are real recordings; review licensing before distribution.
- Numerical evidence: `artifacts/model2-numerical-investigation/` and
  `artifacts/model2-numerical-fvp/`, including backend reference NPZs and saved TOSA.
- Physical captures: `artifacts/streaming-firmware/physical-sustained-160.json`,
  `physical-sustained-manifest.json`, `chrome-verified-21/`, earlier failure records
  and `uart-idle-fix-review.json`. Full Chrome captures include input EEG bytes.
- Installed AlifSemiconductor Ensemble 2.2.1 pack under the local Arm pack cache.
  The successful installation includes existing driver changes; do not assume an
  arbitrary pristine pack is identical. Preserve the exact files identified by
  `validation/streaming-image-manifest.json`'s `uart_dependency_fingerprints`:
  Driver_USART.c, Driver_USART_Private.h, uart.c, uart.h and vectors.c. These are
  external installed dependencies, not edited by the current source commit.
- Installed/licensed AC6 6.24, CMSIS-Toolbox, CMake, Ninja and Arm FVP/Docker setup;
  original Python training environment and isolated official ExecuTorch environment.
  Local VS Code launch/tasks/tool-PATH settings are excluded. Reconfigure them for
  the target machine through CMSIS; never copy credentials into Git.

Reproduction: create a NEW isolated project with `create_streaming_project.py`,
then apply the checked-in source/configuration changes and create the data/FVP
fixtures with `prepare_streaming_assets.py`. These tools require the retained
local originals and prepared dataset, refuse some existing outputs, and are not
a standalone dataset download or model export workflow. Do not run them over the
working demo merely to verify a Git commit. Use CMSIS for embedded builds/deployment.
