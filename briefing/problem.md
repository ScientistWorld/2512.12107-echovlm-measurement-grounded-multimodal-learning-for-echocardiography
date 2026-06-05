# Problem Definition

## Research Question

Can an echocardiography image-text model learn visual representations that support clinically grounded interpretation across multimodal tasks? The benchmark centers on image-caption alignment, disease finding discrimination, and view-aware transfer for echo images with structured clinical measurements and standardized views.

## Why It Matters

Echocardiography interpretation combines image view recognition, quantitative measurements, and textual diagnostic findings. A useful model should align images with clinically meaningful statements, distinguish abnormal from normal or negated findings, and provide transferable features for downstream echo workflows.

## Success Criteria

A successful system retrieves the matching clinical caption for an echo image, assigns disease findings better than generic image-text alignment, and maintains explicit separation between positive findings and normal/negated statements. Evaluation must include both benefit metrics, such as AUC and retrieval recall, and a constraint metric for negation behavior.
