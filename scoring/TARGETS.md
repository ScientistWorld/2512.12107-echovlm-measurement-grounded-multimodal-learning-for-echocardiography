# Evaluation Targets

## Primary: toy_disease_zero_shot

Tests whether image-caption embeddings distinguish abnormal from normal measurement-grounded findings on the fixed echo-style test split. This is the scored target because the original study reports directly comparable AUC, precision, and recall for the same task family.

- **Primary metric**: `auc` (higher is better)
- **Supporting metrics**: `precision`, `recall` (higher is better)
- **Constraint metric**: `negation_cosine` (lower is better)
- **Run**: `bash scripts/evaluate.sh` after producing `test_embeddings.jsonl`

The target includes every method artifact present under `checkpoints/*/test_embeddings.jsonl` or passed explicitly to `scripts/evaluate.sh`.

## Diagnostics Not Scored

The train/test slice evaluators can compute retrieval rank, balanced accuracy, hard-negative accuracy, and image-vs-negated embedding margins from produced artifacts. They have zero coefficients in `reference.json` and are intentionally omitted from the main `scores.json` contract because the toy retrieval pool has far fewer candidates than the paper's full benchmark and the paper does not report these direct diagnostics.
