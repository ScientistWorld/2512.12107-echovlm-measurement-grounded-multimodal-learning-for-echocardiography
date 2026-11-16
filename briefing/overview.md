# Overview

- **Paper ID:** 2512.12107
- **Title:** EchoVLM: Measurement-Grounded Multi-Modal Learning for Echocardiography
- **Domain:** Medical vision-language learning
- **TL;DR:** CLIP-style pretraining for echocardiography on measurement-grounded captions, plus view-informed and negation-aware contrastive objectives.

## Short Summary

The paper introduces EchoGround-MIMIC, an echocardiography image–text dataset with standardised views, OCR-extracted measurements, measurement-grounded captions and guideline-derived disease labels. It trains EchoVLM, a VLM with image–text contrastive learning plus two echo-specific losses. The view loss groups images by acquisition view; the negation loss separates captions from their negated counterparts. Reported gains are in zero-shot disease classification and retrieval, with transfer to view classification, segmentation and landmark detection.

## This gym

EchoGround-MIMIC is not released, so the gym rebuilds the paper's measurement-grounded setting from public CAMUS data. It has 500 patients with apical two- and four-chamber views, expert LV/LA masks and biplane LVEF. Three ASE-graded findings are labelled, and captions state each measurement with its interpretation. Methods export image and text encoders. A hidden evaluator embeds frames from 150 held-out patients together with hidden prompts and captions, and scores three things:
- zero-shot disease classification (primary),
- image-to-caption retrieval,
- frozen k-NN view classification, a constraint on the visual representation.

## Key Results

Paper (EchoGround-MIMIC):
- Zero-shot disease AUC 86.5 / precision 34.2 / recall 86.2, against CLIP-only 79.6 / 29.0 / 73.6 (Table 6).
- Retrieval R@5 2.98, R@10 5.70.
- k-NN view F1 on CAMUS 94.0.

Gym (hidden test, 150 patients, reference seed 0):
- EchoVLM AUC 68.0 / precision 51.2 / recall 70.2, R@10 17.1, view k-NN F1 96.4.
- CLIP-only AUC 61.9, R@10 17.9, view k-NN F1 91.9.
- Over 5 seeds, AUC is 68.3 ± 1.9 for EchoVLM and 64.3 ± 1.8 for CLIP-only.
