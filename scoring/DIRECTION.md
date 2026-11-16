# Research Direction

## Research Problem

Measurement-grounded vision–language learning for echocardiography. Align echo frames with clinical statements so that findings and their negations can be graded zero-shot from text, captions retrieved, and the visual representation kept faithful to echo acquisition structure.

## Core Contribution to Build On

The reference shows that echo-specific priors in the pretraining objective improve zero-shot disease classification over plain image–text contrastive learning, and that view structure in the image space is preserved. On this benchmark, though, retrieval is not improved, and the disease ranking is far from saturated: reference macro AUC 68 on hidden patients.

## Approach Scope

In scope:
- the objective, architecture and initialisation;
- text processing and the tokenizer;
- augmentation, sampling of frames and pairs, batching and optimisation;
- using several frames, views or measurements from the visible records during training;
- ensembling, and anything else that produces better stateless image and text encoders.

## Out of Bounds

- Training or selecting on CAMUS patients outside the visible split, or on any copy of CAMUS (see `CONSTRAINTS.md`).
- Encoders that read files, keep state across calls, or behave differently depending on which inputs they saw before.
- Hard-coding the evaluator's prompts or captions, or editing the evaluator.
- Improving retrieval or disease AUC by collapsing the visual representation. View k-NN is scored as a constraint.
