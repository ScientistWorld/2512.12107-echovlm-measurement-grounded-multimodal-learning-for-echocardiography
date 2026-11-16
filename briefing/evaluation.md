# Evaluation

## What you submit

A method is a directory `checkpoints/<name>/` containing `export/`:

| file | contract |
|---|---|
| `export/image_encoder.pt` | TorchScript module: `float[B,3,112,112]` in [0,1] → `float[B,D]` |
| `export/text_encoder.pt` | TorchScript module: `(long[B,L] input_ids, long[B,L] attention_mask)` → `float[B,D]` |
| `export/tokenizer.json` | HuggingFace `tokenizers` file the evaluator uses to tokenize text |
| `export/export.json` | `{"max_length": L, "pad_id": int}` |

The evaluators load these files and run the encoders themselves, one input at a time (B = 1), on frames and text they own. Nothing else a method writes is read. Embeddings are L2-normalised by the evaluator; only directions matter.

Encoders must be pure functions of their input:
- their TorchScript graphs may not read files, keep Python-level state between calls (attribute assignment in `forward`), fork, or call opaque Python functions;
- inputs are presented in a fixed shuffled order, and a probe subset is embedded again at the end. An embedding that changed between the two passes fails the method.

A method fails, and gets the worst value (0) in every metric of every experiment, if its export is missing or does not load, if it breaks the rules above, crashes, or returns something other than a finite, non-zero `[1, D]` tensor, or if its image and text dimensions differ. It also fails if its disease discrimination is not significantly above chance: a patient-level permutation test of the macro AUC, p > 0.01 on the hidden test (p > 0.05 on the 50 visible validation patients).

## Experiments and metrics (all in %, higher is better)

1. **disease_zero_shot** (weight 0.65). For each finding, every frame's similarity to the mean embedding of the hidden positive prompts minus its similarity to the hidden negative prompts gives a score. Each frame is labelled with its patient's binary finding.
   - `auc`: macro AUC over the three findings (coefficient 1.0, primary)
   - `precision`: macro precision of the decision score > 0 (0.5)
   - `recall`: macro recall of that decision (0.25)
   - per-finding AUCs, reported as diagnostics only (0)
2. **image_text_retrieval** (weight 0.10). Image → caption retrieval within fixed pools of 50 patients (500 frame–caption pairs), with hidden caption wording. Ties count against the method.
   - `recall_at_10` (1.0, primary), `recall_at_5` (0.5)
3. **view_knn** (weight 0.25). k-NN (k = 20, temperature-weighted votes, neighbours from other patients only) on frozen image embeddings, predicting the apical view within each pool.
   - `f1` (1.0, primary); `accuracy` is a diagnostic only (0)

## Splits and scripts

| script | data | output |
|---|---|---|
| `scripts/evaluate_train.sh` (also `scripts/evaluate.sh` → `scoring/scores.json`) | 50 visible validation patients, visible prompts, one pool | `scoring/scores_train.json` |
| `scripts/evaluate_test.sh` | 150 hidden patients (3 pools of 50), hidden prompts and caption wording | `scoring/scores_test.json` |

Both scripts take no arguments and score every directory under `checkpoints/`. The hidden test is light: it only runs the exported encoders and never trains anything.

## Reference numbers

These are the hidden-test scores of the reproduced reference rows (seed 0). Means over seeds are in `HARDENING.md`.

| method | AUC | precision | recall | R@5 | R@10 | view k-NN F1 |
|---|---|---|---|---|---|---|
| image–text contrastive only (`clip_only`) | 61.92 | 49.03 | 57.68 | 9.73 | 17.87 | 91.86 |
| reference (`echovlm_full`) | 68.01 | 51.24 | 70.19 | 9.47 | 17.07 | 96.40 |

Chance is 50 AUC; the k-NN view F1 of a view-blind representation is about 50.
