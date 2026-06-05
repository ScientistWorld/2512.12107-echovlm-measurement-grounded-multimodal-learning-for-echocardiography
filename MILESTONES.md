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

### [2026-06-05 04:35] - core_claim
- Added first-class runners and reference entries for the Table 6 single-objective ablations.
- Prepared a managed job to regenerate `clip_only`, `echovlm_view`, `echovlm_negation`, and `echovlm_full` artifacts from scratch before claiming `core_claim_plus`.

### [2026-06-05 05:05] - core_claim
- Final verification passed with `python validate.py --compare`.
- Cleaned method-specific wording from method-agnostic evaluation and scoring documents.
- Kept milestone at `core_claim` because the validated `scores.json` contains the baseline and full method only; optional ablation runners are present but their results are not claimed.

### [2026-06-05 04:24] - core_claim
- Current package validates at `core_claim`.
- Prepared a CPU managed job to regenerate the CLIP-only baseline, view-only ablation, negation-only ablation, and full EchoVLM objective from scratch before deciding whether `core_claim_plus` is supported.

### [2026-06-05 04:27] - core_claim
- Final verifier reran `python validate.py --compare`; all scoring, structure, import-separation, and train/test checks passed.
- Kept the milestone at `core_claim` because the current scored artifacts contain the CLIP-only baseline and full method, but no validated ablation artifacts for `core_claim_plus`.
- Confirmed the main limitation is the intentionally scaled substitute data; EchoGround-MIMIC, private downstream data, and original initialization weights are unavailable.
