# Scientist Constraints

## Data

- Train and select models only on the visible CAMUS patients listed in `data/camus_split.json` (`train`, `val`), as prepared by `scripts/prepare_data.sh`. The remaining 150 CAMUS patients are the hidden test.
- Do **not** download, read or use any other CAMUS patient, and do not use other copies or derivatives of CAMUS. The hidden patients are part of the public release, so using them is easy, and it is a rule violation that the evaluator cannot detect. Any submission found doing so is invalid.
- Other public data or pretrained weights are allowed if they contain no CAMUS images. Fetch them in `scripts/download.sh` and document them as deviations.
- You may use everything in the visible records: images, captions, negated captions, measurements, grades, labels, views and phases. You may also re-derive other frames with `python -m data.prepare_camus --train-frames all`.

## Interface

- Submit `checkpoints/<name>/export/` as described in `briefing/evaluation.md`. The evaluator runs the encoders itself on its own inputs.
- Encoders must be pure functions of one input:
  - no file access,
  - no state carried between calls, including in-place updates of buffers,
  - no dependence on the order or the identity of other inputs.
- Text must be handled through your own `tokenizer.json` and text encoder. The evaluator supplies its own prompts and captions and never reveals them.
- Do not edit `eval/`, `scripts/evaluate_*.sh` or `scoring/reference.json`.

## Compliance checklist

- [ ] Only patients in `data/camus_split.json` (train / val) were used for training and model selection.
- [ ] Encoders are deterministic, stateless TorchScript modules that do not read files.
- [ ] Any external data or weights are public, CAMUS-free, fetched by `scripts/download.sh` and documented.
- [ ] Scores come from `scripts/evaluate_train.sh` / `scripts/evaluate_test.sh`, run unmodified.
