# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-05 03:49] - method_runs
- Audited the existing EchoVLM reproduction against the paper.
- Replaced the scaled backbone with compact Transformer encoders and aligned the view-informed loss averaging with Algorithm 1.
- Prepared a fresh managed run to regenerate scores before reclaiming `core_claim`.

### [2026-06-05 03:55] - core_claim
- Managed job `bac0f013-33c` completed successfully and regenerated baseline plus EchoVLM artifacts with the compact Transformer implementation.
- Reproduced the core claim on the scaled substitute: EchoVLM improved disease AUC from 43.75 to 75.00 and recall from 83.33 to 100.00 while improving the negation constraint cosine from 0.471 to -0.772.
- `python validate.py --compare` passed, including artifact-only evaluation and import-separation checks.
