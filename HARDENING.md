# Hardening report (final iteration)

There is no `REVIEW.md`; this is the first and final hardening pass. `REDTEAM.md` audited the earlier *toy* gym, whose data came from paper-figure JPEGs with index-assigned labels. A first hardening attempt in this iteration replaced that gym with public CAMUS echo data, an evaluator that runs exported encoders on hidden frames, and the paper's ViT-B / EchoCLIP backbone. Its audit, kept in `redteam/hardening/scores/`, showed that it still did not reward science. This pass fixes that.

All numbers below are on the hidden test (150 CAMUS patients, 1,500 frames) unless marked "val" (50 visible patients). Evidence:
- `redteam/hardening/summary_v2.md` and `summary_v2.json`: tables, paired bootstrap, ranks;
- `redteam/hardening/scores_v2/`: every run's score files under the hardened evaluator (`*.test.json`, `*.train.json`) and under the previous one (`*.legacy_test.json`);
- `redteam/hardening/selected_regime.json`, `smoke_report.json`, `logs_v2/`.

The code is in `redteam/hardening/audit.py`, `make_shortcuts.py`, `make_variants.py` and `summarize2.py`.

## 1. Verdict

- Better results now score higher. The paper's effect (full EchoVLM over CLIP-only) is reproduced in the paper's direction and exceeds seed noise.
- A genuinely better variant beats the reference beyond the paired test noise, so the test has headroom.
- The one paying interface shortcut (reading test answers from disk) is closed, and so is a stateful one that raised the primary metric.
- A short hand-written rule does not pass the task gate.
- What does **not** reproduce:
  - retrieval: the paper improves it, here it is slightly worse;
  - the view-only ablation: below CLIP-only here, above it in the paper.

## 2. What was wrong and what changed

| problem (before) | fix |
|---|---|
| **Reproduction failed.** The paper's recipe trained on every frame of each sequence (≈38 near-duplicate pairs per patient, 440 steps). It over-fit: reference AUC 53.9, 2 of 3 reference seeds failed the task gate, and CLIP-only beat EchoVLM by 5.6 AUC (paired, 3 seeds). | Training-setup search on the **visible validation split only**, over data size (frames per sequence), epochs and batch. The paper's lr, weight decay, warm-up, schedule, λ's and losses were unchanged. Selection rule, fixed in advance: highest mean val macro AUC of the full objective over 3 seeds. Selected **R20f5**: 5 frames per sequence (≈10 pairs per patient, close to the paper's ≈12), the paper's 20 epochs and batch 512. That is 100 steps, so the paper's 200-step warm-up is never completed and the lr peaks at 5e-5. It is now the default (`data/prepare_camus.py`; `scripts/prepare_data.sh` rebuilds stale splits). |
| **Hidden-test answers were readable while the method ran.** Labels sat in plaintext `eval/test/test_meta.json`. A TorchScript encoder could call `torch.from_file` on it and use a call counter to map each input to its record. | Images and ground truth are in one compressed archive, `eval/test/hidden_test.npz`, rebuilt from raw data and checked identical to the old set. The evaluator inspects each encoder's inlined TorchScript graph and rejects file access (`from_file`, pickle, rpc), Python state (`prim::SetAttr`), forks and opaque calls. |
| **Encoders could keep state across calls.** In-place buffer updates were possible, and inputs arrived in a fixed order with each patient's 10 frames consecutive. | Inputs are embedded in a fixed shuffled order, and a probe subset is embedded again at the end; any changed embedding fails the method. The visible evaluator applies the same rules. |
| `reference.json` lacked rows for the gym's methods and held off-scale paper numbers as gym results, so `validate.py` failed. | Every gym row is present. Off-scale paper numbers (retrieval, EchoPrime k-NN) are under `_paper_*` keys, and the paper's Table 6 numbers are kept unchanged. |
| Under the old weights (0.6 / 0.25 / 0.15), retrieval (where CLIP-only beats EchoVLM) cancelled the paper's AUC effect in the composite rank. The full − CLIP rank gap was +0.04 against a full-seed SD of 0.20. | **Weights are now 0.65 / 0.10 / 0.25** (disease / retrieval / view k-NN); see §7. |
| The briefing and scoring documents still described the toy gym. | Rewritten for the CAMUS gym and its rules. |

## 3. Do better results score higher? (variants before → after)

Rank Δ is relative to the reference row, under the stated rule with the final weights. It is shown for two cohorts that set the per-metric spread: "all" = every genuine run, including gate failures (57 runs, among them the two default-argument reruns); "pass" = only runs that pass the gate.

**Before.** In the toy gym (REDTEAM.md), the reference AUC ranged from 0 to 75 across seeds and longer training lowered scores. In the first CAMUS version (`redteam/hardening/scores/`, same 1,500 hidden frames):
- reference seed 0: AUC 53.9; seeds 1 and 2 failed the gate and scored 0;
- CLIP-only seeds 1 and 2 (60.2, 59.5) **outranked** the reference;
- view-only, negation-only, train+val, 50-epoch and batch-128 runs all failed the gate;
- the only clearly better run was a 5-epoch one (68.1), which was not the reference recipe.

**After** (selected regime, macro AUC is the raw value):

| run (seeds) | test AUC | val AUC | R@10 | view k-NN F1 | gate | rank Δ all | rank Δ pass |
|---|---|---|---|---|---|---|---|
| **reference** (full, seed 0) | 68.01 | 68.58 | 17.07 | 96.40 | pass | 0 | 0 |
| full EchoVLM (5 seeds) | 68.32 ± 1.93 | 70.75 ± 2.53 | 14.47 ± 1.89 | 96.36 ± 0.27 | 5/5 | −0.09 ± 0.14 | −0.18 ± 0.33 |
| CLIP-only baseline (5) | 64.32 ± 1.82 | 68.98 ± 1.08 | 17.35 ± 0.65 | 91.25 ± 0.58 | 5/5 | −0.29 ± 0.11 | −1.38 ± 0.45 |
| + L_neg only (3) | 68.46 ± 0.59 | 69.60 ± 1.44 | 12.58 ± 0.27 | 91.02 ± 0.61 | 3/3 | −0.14 ± 0.03 | −0.53 ± 0.12 |
| + L_view only (3) | 61.03 ± 2.15 | 63.41 ± 2.33 | 17.80 ± 0.35 | 96.07 ± 0.12 | 3/3 | −0.42 ± 0.12 | −1.66 ± 0.46 |
| untrained towers (control) | 49.8 (raw) | 49.8 | — | — | fail | −5.90 | −28.6 |
| **better:** full + size-preserving augmentation (3) | **71.33 ± 0.41** | 71.82 ± 1.05 | 25.87 ± 0.85 | 96.56 ± 0.30 | 3/3 | **+0.49 ± 0.02** | **+1.27 ± 0.12** |
| **better:** 3-seed ensemble of full | **71.21** | 72.34 | 19.40 | 96.73 | pass | **+0.23** | **+0.68** |
| full trained on train+val (3) | 66.12 ± 2.00 | (trained on val) | 17.98 ± 0.37 | 96.78 ± 0.41 | 3/3 | −0.11 ± 0.11 | −0.38 ± 0.50 |
| 3-seed ensemble of CLIP-only | 65.04 | 70.36 | 18.73 | 91.46 | pass | −0.21 | −1.16 |

**Paper's effect versus seed noise.**
- AUC: full − CLIP-only = +4.0 (5-seed means), and +5.1 ± 0.9 paired on seeds 0–2. Seed SD is ≈1.9.
- Composite rank: full − CLIP = +0.20 (all-runs cohort; full-seed SD 0.14) or +1.20 (passing cohort; SD 0.33).
- CLIP-only ranks below the reference for all 5 seeds in both cohorts.
- In the all-runs cohort the separation is only ≈1.4 seed SDs. Gate failures inflate the AUC and k-NN spreads (SD 17 and 24, against 4.4 and 2.1 among passing runs), which compresses differences between genuine methods. See §8.

**Better than the reference beyond paired noise.** The paired patient-level bootstrap (2,000 resamples of the 150 patients; every method scored on the same frames) gives these ΔAUC against the reference:

| run | ΔAUC [95% CI] |
|---|---|
| augmentation seed 0 | +3.12 [+0.62, +5.57] |
| augmentation seed 1 | +3.06 [+1.03, +5.22] |
| augmentation seed 2 | +3.79 [+1.55, +6.23] |
| 3-seed ensemble | +3.20 [+1.80, +4.62] |
| CLIP-only (5 seeds) | −6.1 to −1.3 |
| view-only (3 seeds) | −9.4 to −5.2 |

Train/test agreement: full EchoVLM scores 70.8 on val and 68.3 on test (5 seeds); the augmentation variant scores 71.8 and 71.3. Train and test measure the same thing at comparable difficulty.

## 4. Shortcuts: fix and score before → after

"Before" means the previous evaluator, or REDTEAM.md for the toy gym. "After" means the hardened evaluator, with the rank under the final rule.

| shortcut | before | fix | after |
|---|---|---|---|
| **Read test answers from disk** (`sc_file_reader`: `torch.from_file` on the plaintext test metadata, plus call counters) | **AUC 100, precision 100, recall 100, k-NN 100** (rank +2.31 all / +10.17 pass) | Answers live only inside the compressed archive; graph inspection rejects `from_file` and `SetAttr`; shuffled order | failed (forbidden ops), all 0, rank −5.90 |
| **Stateful pooling over consecutive calls** (`sc_stateful`: in-place buffers average each patient's consecutive frames) | AUC **71.24** (> reference 68.01), R@10 9.7, k-NN 84.3 (rank −0.06 / −1.01) | Shuffled order + probe re-embedding | failed ("not a pure function"), all 0 |
| **Short hand-written rule** (`sc_handcrafted`: dark-pixel fractions in fixed LV / LA regions, 3 thresholds fit on visible train, keyword text encoder) | failed gate (AUC 48.5) | none needed. A first version mis-mapped multi-token keywords; with that fixed and the mapping verified (`runs/sc_handcrafted/rule.json`) it still fails | failed gate (AUC 52.0, p = 0.27) |
| Toy-gym answer leaks (REDTEAM: visible answers, generator, brute force, forged labels, row selection), rank +1.86 / +1.78 | rank +1.86 | CAMUS redesign: the evaluator owns frames, labels, prompts, captions and pools, and runs the encoders itself | no such channel: the evaluator reads nothing a method writes besides the encoders |
| Text-only (REDTEAM: caption reading, +1.86) | +1.86 | Prompts and captions are embedded separately from the image | `sc_text_only`: failed gate (50.0) |
| Constant / huge-constant outputs (REDTEAM +0.22) | +0.22 | Evaluator L2-normalises; task gate | failed gate |
| Norm scaling (REDTEAM +9e−7) | tiny gain | Evaluator L2-normalises | exact tie (same encoder) |
| NaN, empty, wrong shape, dimension mismatch, missing export, forged score files (REDTEAM: missing artifact skipped, empty or null aborted the run) | skipped or crashed | Each is a per-method failure with worst values everywhere | failed, all 0 |
| Baseline degradation (REDTEAM: tie) | tie | Baselines are not inputs to scoring; ranks are relative to the reference row | n/a |

## 5. Shortcuts left, and why

1. **Training on the hidden patients.** They are the public complement of `data/camus_split.json`. This cannot be closed with public data. `scoring/CONSTRAINTS.md` and `scoring/DIRECTION.md` forbid it explicitly, and it is left to the grader: it requires downloading CAMUS patients outside the visible split, which is visible in a submission's code and data.
2. **Measurement-pipeline methods.** Examples: segmenting the LV / LA, or regressing the caption measurements, then applying the ASE thresholds through a keyword text encoder. These are honest alternative methods, because they learn to measure from images, not loopholes. The labels follow published guidelines, as in the paper. They would have to beat the reference on hidden patients through the same interface. A purely hand-written version fails the gate (§4).
3. **Unknown TorchScript I/O operations.** The graph check is a block-list. A future operation that reads external state would still be caught if it makes outputs order-dependent, but not otherwise. Plaintext answers are no longer on disk.

## 6. Reference score before → after

| | AUC | precision | recall | R@5 | R@10 | view k-NN F1 |
|---|---|---|---|---|---|---|
| toy gym (REDTEAM.md, 7 hidden records) | 75.00 | 42.86 | 100.00 | — | — | — |
| first CAMUS version (all frames, 20 epochs) | 53.94 | 42.42 | 53.75 | 11.73 | 21.27 | 96.87 |
| **final** (`echovlm_full`, seed 0) | **68.01** | **51.24** | **70.19** | **9.47** | **17.07** | **96.40** |

## 7. Headroom check and weights

- No scored metric of the reference sits at its bound: AUC 68, precision 51, recall 70, R@5 9.5, R@10 17, k-NN F1 96.4 (bound 100). k-NN is a constraint and is closest to its bound. Variants reach 96.6–97.1 on it.
- Variants that score above the reference beyond paired noise: augmentation (AUC +3.1 to +3.8, R@10 +8.8) and the 3-seed ensemble (+3.2). Neither reaches any bound. No new experiment was needed.
- **Weights changed from 0.6 / 0.25 / 0.15 to 0.65 / 0.10 / 0.25** (disease / retrieval / view k-NN), with metric coefficients unchanged.
  - Zero-shot disease classification is the paper's headline claim. The paper pairs it with view classification ("without compromising vision features", Table 6), so the constraint gets more weight.
  - Retrieval effects in the paper are small (Table 1: about 1 point over the best baseline) and absent from its ablations. Here retrieval moves against the paper's objectives, so a large retrieval weight let CLIP-only tie the reference in the composite.
  - The paper's reported numbers in `reference.json` are unchanged.

## 8. What could not be fixed / plainly stated

- **Retrieval is not reproduced.** Full EchoVLM retrieves worse than CLIP-only (R@10 14.5 ± 1.9 vs 17.4 ± 0.7). The negation term is the cause (+L_neg alone: 12.6), whereas the paper reports a retrieval gain.
- **Ablation ordering differs from Table 6.** +L_view alone is below CLIP-only on disease AUC (61.0 vs 64.3), though it raises view k-NN to 96. +L_neg alone matches the full objective on AUC (68.5 vs 68.3). The full objective is best overall only because it combines both: negation-only drops view k-NN to 91.
- **Composite-rank separation is modest in the all-runs cohort.** The spread of each metric is taken over all attempts, including gate failures that score 0. That inflates the AUC and k-NN spreads, so the full-vs-CLIP gap is ≈1.4 full-seed SDs in that cohort and ≈3.6 in the passing cohort. A future method needs ≳ +2 AUC to clear seed noise.
- **Schedule.** With 3,000 pairs and the paper's batch 512, 20 epochs are 100 steps. The paper's 200-step warm-up therefore never completes. This was kept rather than changing the paper's warm-up; it is a consequence of the dataset's size.
- **Substitutions.** These remain: CAMUS (3 findings, 2 views, 500 patients) instead of EchoGround-MIMIC (9 findings, 22 views); OpenAI CLIP ViT-B/16 instead of the internal echo-pretrained ViT-B; multi-sentence measurement-grounded captions from guideline rules instead of LLM-curated report sentences.

## 9. Default-argument rerun (fresh-clone check)

- `python -m data.prepare_camus` with defaults rebuilds the selected training and validation splits byte for byte.
- `python -m method.train` and `python -m baseline.clip_finetune` were rerun with every argument at its default, into `redteam/hardening/runs/repro_{full,clip}_default`.
- Both reproduce the reference and the CLIP-only rows exactly on the hidden test and the visible split:
  - EchoVLM: 68.01 / 51.24 / 70.19, R@10 17.07, k-NN 96.40;
  - CLIP-only: 61.92 / 49.03 / 57.68, R@10 17.87, k-NN 91.86.
- Official artifacts: `checkpoints/` holds seed 0 of the four Table 6 rows, and `scoring/scores_{train,test}.json` and `scoring/scores.json` were regenerated from those reference runs only. `bash scripts/run.sh` reruns the whole pipeline (data → baseline → ablations → method → all evaluators).
