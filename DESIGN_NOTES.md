# Design Notes

## Checks

- Confirmed `eval/train/`, `eval/test/`, `scripts/evaluate_train.sh`, and `scripts/evaluate_test.sh` exist.
- Confirmed `evaluate_test.sh` calls `eval/test/evaluate.py` and writes `scoring/scores_test.json`.
- Confirmed `evaluate_train.sh` calls `eval/train/evaluate.py` and writes `scoring/scores_train.json`.
- Confirmed there are no `scoring/scores_train.json` or `scoring/scores_test.json` stubs.
- Checked for train/test cross-imports and forbidden paper-agreement metric names; none were found.
- Checked that every scored metric in `scoring/reference.json` is emitted by both train and test evaluators.
- The paper markdown path provided in the prompt was not present on disk, so metric choices were reviewed against the existing target docs and implementation artifacts.

## Changes

- Added `caption_retrieval_mrr` to both slice evaluators. This scores whether image embeddings retrieve their paired affirmative captions within the slice, adding an image-text alignment angle beyond the visible disease threshold.
- Added `hard_negative_accuracy` to both slice evaluators. This scores whether each image is closer to its affirmative caption than its paired negated caption, making negation shortcuts visible without requiring new inference.
- Added missing `description` fields for all nonzero metrics in `scoring/reference.json`.
- Added the two new metrics to `scoring/reference.json` with moderate coefficients so the split triangulates disease ranking, threshold behavior, image-text alignment, and negation separation.
- Changed `scripts/evaluate_test.sh` to enumerate `checkpoints/*/test_embeddings.jsonl` directly instead of deriving the method list from `scores_train.json`. This prevents a stale visible-train score file from hiding a future agent's final method during held-out scoring.

## Runtime Concerns For The Next Iteration

- The split is correctly carved from the paper test artifact rather than from training data, but each slice has only seven examples. Scores may be noisy or saturate for some metrics; the runner should inspect actual train/test gaps after evaluation.
- The evaluators require each artifact row to contain `id`, `label`, `image_embedding`, `text_embedding`, and `neg_text_embedding`. Future methods must write the same artifact schema.
- The train/test fallback split by row parity is only a compatibility path for artifacts without fixed IDs. Preferred artifacts should include the fixed toy IDs so both slices evaluate the intended held-out rows.
