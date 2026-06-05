# Paper's Method

## Key Contribution

EchoVLM extends CLIP-style image-text pretraining for echocardiography with two clinically informed objectives on measurement-grounded echo image-caption pairs.

## Approach

The base objective is symmetric image-text contrastive learning. On top of this, the method adds a view-informed image-image contrastive loss: images from the same standardized echo view are positives and images from other views are negatives. It also adds a negation-aware text loss: each measurement-grounded caption is paired with a counterfactual negated caption, and a binary cross-entropy objective penalizes high similarity between the original and negated text embeddings. The final loss is `L_CLIP + lambda_view * L_view + lambda_neg * L_neg`, with `lambda_view = 0.5` and `lambda_neg = 0.1` in the main setting.

## Main Claims

The primary claim is that measurement-grounded multimodal pretraining with view and negation priors improves zero-shot disease classification and image-text retrieval in echocardiography. Secondary claims are that the learned visual representations transfer to view classification, segmentation, and landmark detection. This reproduction targets the central multimodal claim and the negation constraint on a deterministic scaled substitute because the full EchoGround-MIMIC release and internal initializations were not available.
