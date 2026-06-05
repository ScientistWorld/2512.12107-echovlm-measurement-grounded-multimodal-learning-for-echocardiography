# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-05 03:20] - none
- Read the EchoVLM paper and identified the core algorithm: CLIP-style image-text pretraining with view-informed image contrast and negation-aware text contrast.
- Documented that the original EchoGround-MIMIC dataset, private view-classification data, and internal/EchoCLIP initializations are unavailable in this workspace.
- Implemented a deterministic measurement-grounded echo-style substitute dataset from paper images and wrote the first method/evaluation pipeline.

### [2026-06-05 03:21] - none
- Reviewed failed CPU smoke job `9d0e4379-6f3`.
- Diagnosed that training never started because `python method/train.py` could not import the top-level `data` package inside the container.
- Fixed script entry points to run the trainer as a module from `/home/user`.

### [2026-06-05 03:30] - core_claim
- CPU smoke job `ba2bbb9d-f93` completed end-to-end inside the managed container.
- The CLIP-only baseline and EchoVLM full objective both trained and wrote embedding artifacts.
- On the scaled toy substitute, EchoVLM improved disease AUC (47.92 to 54.17), disease recall (50.00 to 66.67), retrieval@5 (28.57 to 35.71), and negation cosine separation (0.745 to 0.471, lower is better).

## Stop Justification
- Not stopped.
