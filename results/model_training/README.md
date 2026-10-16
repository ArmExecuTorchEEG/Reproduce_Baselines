# Model training

| Model | Report | Metrics | Curves |
|---|---|---|---|
| [01 Multi-subject package](../../models_ready_for_alif_e8/01_multi_subject_tinyeegnet/README.md) | [Baseline summary](../combined-seed42-20261016/summary.md) | [JSON](../combined-seed42-20261016/metrics.json) | [Original history](../../artifacts/combined-seed42-20261016/history.json), local artifact |
| [02 Participant 1 package](../../models_ready_for_alif_e8/02_subject_01_tinyeegnet/README.md) | [Subject-specific summary](../subject1-seed42-20261016/summary.md) | [JSON](../subject1-seed42-20261016/metrics.json) | [Loss plot](../subject1-seed42-20261016/loss_curves.png) |

Multi-subject: 1,728 train, S8 validation 93.06%, S6–7 Session 2 demonstration
62.15%, S9 cross-subject test 59.03%. Participant 1: 115 train, 29 validation
(75.86%), 144 Session 2 test (79.86%). Evaluation participants differ; these
numbers do not establish superiority. Participant 1 Session 2 was training data
for the multi-subject model, so it cannot be an unseen-session comparison for it.
Both models have 1,938 parameters and dropout 0.5. Nothing here was retrained.
