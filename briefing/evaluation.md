# Evaluation

## Metrics

Primary multimodal metrics are disease classification AUC, precision, and recall, reported as percentages. Retrieval is measured by image-to-text Recall@5 and Recall@10. The negation constraint is measured by the cosine similarity between positive and negated caption embeddings, where lower is better, and the margin between image-positive and image-negated similarity, where higher is better.

## Baselines and Targets

The paper reports EchoGround-MIMIC disease AUC of 86.5 for the proposed model and 79.6 for the CLIP-only ablation, with corresponding precision/recall of 34.2/86.2 versus 29.0/73.6. For retrieval, the paper reports Recall@5/Recall@10 of 2.98/5.70 for the proposed model and 2.30/4.33 for the CLIP-only curated-caption ablation.

## Evaluation Protocol

Run the method to produce `test_embeddings.jsonl` artifacts containing image, caption, and negated-caption embeddings for the fixed test split. `scripts/evaluate.sh` reads only these artifacts and writes `scoring/scores.json`; it does not import or execute code from `method/`. The paper's exact MIMIC-derived data are unavailable here, so the included runnable protocol uses a deterministic measurement-grounded substitute generated from paper images with fixed seed 7.
