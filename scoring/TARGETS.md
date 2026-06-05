# Evaluation Targets

## Primary: toy_disease_zero_shot

Tests whether image-caption embeddings distinguish abnormal from normal measurement-grounded findings on the fixed echo-style test split. This is the scored target because the paper reports directly comparable AUC, precision, and recall for the CLIP-only and full-objective settings in Table 6.

- **Primary metric**: `auc` (higher is better)
- **Supporting metrics**: `precision`, `recall` (higher is better)
- **Constraint metric**: `negation_cosine` (lower is better)
- **Run**: `bash scripts/evaluate.sh` after producing `test_embeddings.jsonl`

## Diagnostics Not Scored

The evaluator can compute retrieval recall and image-vs-negated embedding margins from the produced artifacts, but they are intentionally omitted from `scores.json`. The toy retrieval pool has far fewer candidates than the paper's full benchmark, and the paper does not report a direct image-vs-negated embedding margin, so those diagnostics are documented in `PROGRESS.md` rather than compared against paper reference values.
