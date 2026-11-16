# Design Notes

The current design is described in `HARDENING.md`, `briefing/evaluation.md` and `scoring/EXPERIMENTS.md`. In short:

- The hidden test owns 1,500 CAMUS frames of 150 patients, their labels, hidden zero-shot prompts, hidden caption wording and the retrieval / k-NN pools. Everything sits in one compressed archive, `eval/test/hidden_test.npz`.
- Methods submit TorchScript image and text encoders plus a tokenizer, and the evaluator runs them one input at a time:
  - encoders that read files, keep state, fork or call opaque functions are rejected;
  - inputs are shuffled and probes re-embedded;
  - invalid outputs fail the method.
- A failed method, or one whose disease ranking is not significantly above chance (patient-level permutation test), gets 0 in every metric.
- `eval/train/` runs the same protocol on the 50 visible validation patients with visible prompts. It shares no code with `eval/test/`.
