# Evaluation

## Metrics

Primary multimodal metrics are disease classification AUC, precision, and recall, reported as percentages. The scored negation constraint is the cosine similarity between positive and negated caption embeddings, where lower is better. Retrieval can be computed as a local diagnostic, but it is not scored in this reduced benchmark because the paper's retrieval pool is much larger and not directly comparable.

## Baselines and Targets

The original study reports disease AUC, precision, and recall for a strong measurement-grounded multimodal model and several comparison systems. This reduced gym keeps the same metric family and compares methods by the embeddings they produce on a fixed substitute split. Published reference values are recorded in `scoring/reference.json`; reproduced values are recorded in `scoring/scores.json`.

## Evaluation Protocol

Run a system to produce `test_embeddings.jsonl` artifacts containing image, caption, and negated-caption embeddings for the fixed test split. `scripts/evaluate.sh` reads only these artifacts and writes `scoring/scores.json`; it does not import or execute solution code. The original MIMIC-derived data are unavailable here, so the included runnable protocol uses a deterministic measurement-grounded substitute generated from paper images with fixed seed 7.
