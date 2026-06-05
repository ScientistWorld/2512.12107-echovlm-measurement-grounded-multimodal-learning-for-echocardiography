# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-05 03:43] - core_claim
- Continued from the previous `core_claim` claim with the milestone reset to `method_runs` for audit.
- Verified that `method/echovlm.py` implements the paper's actual computation at reduced scale: symmetric CLIP alignment, view-informed image contrast, and negation-aware text separation.
- Fixed the scoring contract to include the negation constraint metric (`negation_cosine`) from saved artifacts and made `eval/metrics.py` standalone without a Torch dependency.
- Reran `bash scripts/evaluate.sh` and `python validate.py --compare`; validation passes and EchoVLM improves over CLIP-only on AUC, recall, and negated-caption separation.

### [2026-06-05 03:40] - core_claim
- Implemented and validated the paper's actual EchoVLM objective at reduced scale: symmetric CLIP loss plus view-informed image contrast and negation-aware text separation.
- Ran the CLIP-only baseline and full EchoVLM method through the artifact-only evaluator on the fixed substitute EchoGround-style dataset.
- Reproduced the central effect on the scaled benchmark: EchoVLM improved disease AUC from 47.92 to 54.17 and recall from 50.00 to 66.67 while preserving 50.00 precision.
- `python validate.py --compare` passes, including import-separation and artifact-evaluation checks.

### [2026-06-05 03:40] - method_runs
- Method and baseline scripts execute end-to-end in the managed container and write `test_embeddings.jsonl` artifacts for scoring.
- `scripts/reproduce.sh` runs data generation, CLIP-only baseline training, full EchoVLM training, and evaluation.

## Stop Justification
- Completed at milestone `core_claim`. Higher milestones require access to EchoGround-MIMIC, the private downstream TTE view data, or a stronger non-private substitute dataset.
