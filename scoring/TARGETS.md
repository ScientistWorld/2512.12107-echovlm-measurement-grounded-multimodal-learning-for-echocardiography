# Evaluation Targets

## Primary: toy_disease_zero_shot

Tests whether image-caption embeddings distinguish abnormal from normal measurement-grounded findings on the fixed echo-style test split.
- **Metric**: `auc` (higher is better)
- **Run**: `bash scripts/evaluate.sh` after producing `test_embeddings.jsonl`

## Primary: toy_image_text_retrieval

Tests whether each image retrieves its paired measurement-grounded caption from the test set.
- **Metric**: `retrieval_recall_at_10` (higher is better)
- **Run**: `bash scripts/evaluate.sh`

## Constraint: toy_negation_constraint

Tests whether positive clinical statements are separated from their negated normal counterparts.
- **Metric**: `negation_cosine` (lower is better) and `positive_vs_negated_margin` (higher is better)
- **Run**: `bash scripts/evaluate.sh`
