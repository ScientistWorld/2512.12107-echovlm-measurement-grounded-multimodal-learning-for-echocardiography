# Overview

- **Paper ID:** 2512.12107
- **Title:** EchoVLM: Measurement-Grounded Multi-Modal Learning for Echocardiography
- **Domain:** Medical vision-language learning
- **TL;DR:** EchoVLM adapts CLIP-style pretraining to echocardiography using measurement-grounded captions plus view-informed and negation-aware contrastive objectives.

## Short Summary

The paper introduces EchoGround-MIMIC, an echocardiography image-text dataset with standardized views, structured measurements, captions, and disease labels. EchoVLM trains a VLM with standard image-text contrast plus two echo-specific losses: one groups images by acquisition view and one separates positive clinical captions from negated counterparts. The reported main result is stronger zero-shot disease classification and retrieval than echo-specific and general VLM baselines, with additional transfer to view classification, segmentation, and landmark detection.

## Key Results

- Zero-shot disease classification on EchoGround-MIMIC: 86.5 AUC, 34.2 precision, 86.2 recall.
- Image-text retrieval on EchoGround-MIMIC: 2.98 Recall@5 and 5.70 Recall@10.
- Downstream view classification: 95.1 accuracy, 95.3 F1, 95.8 precision.
