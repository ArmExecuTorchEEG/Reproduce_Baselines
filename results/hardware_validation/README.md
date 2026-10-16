# Hardware validation

Start with [actual FVP execution results](../fvp-execution-20261016/README.md).
Both models executed on U85-256 with their original Alif-compiled PTEs. Four
validation windows/model give 100% host-Q/DQ prediction agreement. Model 2 meets
0.01 logit tolerance; Model 1 differs by 0.026093 and fails that tolerance.
Neither has run on the physical Alif E8.

| Stage | Preserved report | Interpretation |
|---|---|---|
| Final multi-subject E8 compilation | [Summary](../ethosu85-export-seed42-20261016-alif-e8/summary.md) | Correct final Alif SRAM/MRAM profile; 13,056-byte PTE |
| Participant 1 E8 compilation | [Report](../subject1-ethosu85-seed42-20261016/export_report.json) | Separate 13,072-byte PTE |
| Actual FVP execution, both models | [Report/commands](../fvp-execution-20261016/README.md) | Current execution evidence and numerical limits |
| Initial export attempt | [Report](../ethosu85-export-seed42-20261016/export_report.json) | Historical int8-boundary ordering failure; not the final model |
| Intermediate Corstone-profile retry | [Report](../ethosu85-export-seed42-20261016-int8io/export_report.json) | Historical successful retry; final package uses Alif profile |
| Initial FVP setup | [Notes](../fvp-integration-20261016/README.md) | Historical missing-tool/license blockers, subsequently resolved |
| Docker helper repair / compiler check | [Notes](../docker-helper-repair-20261016/README.md) | Historical signing repair and license blocker, subsequently resolved |

Training-derived host-reference test vectors are in each finished model package.
The actual four validation inputs and generated firmware remain in the ignored
FVP workspaces linked from the execution report. Keep runtime/export version
1.5.1 matched; do not regenerate the example TinyCNN over a staged EEG AI layer.
