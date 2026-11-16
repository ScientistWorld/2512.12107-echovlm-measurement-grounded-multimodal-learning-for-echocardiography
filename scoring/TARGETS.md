# Evaluation Targets

The rank of a method adds, over every scored metric, the experiment weight × |coefficient| × the improvement over the reference row (`echovlm_full` in `scores_test.json`), taken in the metric's better direction and divided by that metric's spread across ranked attempts.

| experiment | weight | metric (coefficient) | reference (hidden test) | CLIP-only baseline | bound |
|---|---|---|---|---|---|
| disease_zero_shot | 0.65 | `auc` (1.0, primary) | 68.01 | 61.92 | 100 |
| | | `precision` (0.5) | 51.24 | 49.03 | 100 |
| | | `recall` (0.25) | 70.19 | 57.68 | 100 |
| image_text_retrieval | 0.10 | `recall_at_10` (1.0, primary) | 17.07 | 17.87 | 100 |
| | | `recall_at_5` (0.5) | 9.47 | 9.73 | 100 |
| view_knn | 0.25 | `f1` (1.0, primary, constraint) | 96.40 | 91.86 | 100 |

Zero-coefficient diagnostics: the per-finding AUCs and k-NN accuracy.

- **Benefit metrics**: disease AUC, precision and recall; retrieval recall.
- **Constraint metric**: view k-NN F1. It shows that the visual representation is not sacrificed.
- **Failure rule**: a method that breaks the encoder contract, or whose disease ranking is not significantly above chance (patient-level permutation test, p > 0.01), gets 0 in every metric.
- **Headroom**: no scored metric of the reference sits at its bound. Size-preserving augmentation reaches 71.3 ± 0.4 AUC and 25.9 R@10, and a 3-seed ensemble reaches 71.2 AUC. Both beat the reference beyond the paired patient-level noise.
- **Run**: `bash scripts/evaluate_test.sh` (hidden) and `bash scripts/evaluate_train.sh` (visible), with no arguments.
