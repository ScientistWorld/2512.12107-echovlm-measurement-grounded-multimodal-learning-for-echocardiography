# Verification Report

Final state verified in the hardening iteration (2026-11-16); details are in `HARDENING.md`.

- `python validate.py` and `python validate.py --post-design` pass.
- The reference set (seed 0 of each Table 6 row) is in `checkpoints/` and scored by `scripts/evaluate_train.sh`, `scripts/evaluate.sh` and `scripts/evaluate_test.sh`.
- The paper's direction is reproduced on hidden-test zero-shot disease AUC: full EchoVLM 68.3 ± 1.9 vs CLIP-only 64.3 ± 1.8 over 5 seeds.
- Retrieval and the view-only ablation are not reproduced.
- Milestone: core_claim.
