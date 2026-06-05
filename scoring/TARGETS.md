# Evaluation Targets

## Primary: toy_disease_zero_shot

Tests whether image-caption embeddings distinguish abnormal from normal measurement-grounded findings on the fixed echo-style test split. This is the scored target because the paper reports directly comparable AUC, precision, and recall for the CLIP-only and full-objective settings in Table 6.

- **Primary metric**: `auc` (higher is better)
- **Supporting metrics**: `precision`, `recall` (higher is better)
- **Constraint metric**: `negation_cosine` (lower is better)
- **Run**: `bash scripts/evaluate.sh` after producing `test_embeddings.jsonl`

The target includes the Table 6 objective ablations when their artifacts are present:
`clip_only`, `echovlm_view`, `echovlm_negation`, and `echovlm_full`.

## Diagnostics Not Scored

The train/test slice evaluators can compute retrieval rank, balanced accuracy, hard-negative accuracy, and image-vs-negated embedding margins from produced artifacts. They have zero coefficients in `reference.json` and are intentionally omitted from the main `scores.json` contract because the toy retrieval pool has far fewer candidates than the paper's full benchmark and the paper does not report these direct diagnostics.
