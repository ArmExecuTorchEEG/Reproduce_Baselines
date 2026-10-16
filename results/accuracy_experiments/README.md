# Accuracy experiments

[Controlled dropout comparison](../dropout-cv-seed42-20261016/summary.md):
activation dropout 0.5 vs 0.25, subject-held-out folds {1,4}, {2,5}, {3,6,7},
training recordings only; fold-specific normalization and seed 42.

- [Selection and individual participant scores](../dropout-cv-seed42-20261016/selection.json)
- [Loss/accuracy curves](../dropout-cv-seed42-20261016/loss_accuracy_curves.png)
- [Predeclared protocol](../dropout-cv-seed42-20261016/protocol.json)
- [Final integrity audit](../dropout-cv-seed42-20261016/final_audit.json)

The comparison retained the original dropout-0.5 model. These are development
results, not unbiased final testing. Structural ablation and sensor dropout were
not implemented. No experiments were rerun during organization.
