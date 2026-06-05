# Scientist Constraints

Future agents must use the fixed data split under `data/echo_toy` for the included runnable benchmark and must write artifacts in the same `test_embeddings.jsonl` format. They may change model internals and training details, but improvements must come from better echocardiography image-text representation learning rather than changing the evaluation data.

The benchmark is intentionally scaled because the paper's EchoGround-MIMIC data, private TTE view-classification data, and internal initializations are unavailable in this workspace. Do not claim comparison on the original private data unless those resources are actually acquired and wired into the same method-agnostic evaluation pipeline.

The scored constraint surface includes negation separation because the paper's central claim depends on distinguishing affirmative clinical findings from normal or negated findings. The paper does not report a direct negation embedding cosine, so `reference.json` defines the metric without paper result values; reproduced `scores.json` must still report it from artifacts. Toy retrieval remains diagnostic only because this split has a much smaller candidate pool than the paper's retrieval benchmark.

Do not use labels from the test split during training, generate extra test examples, or tune the evaluator. Larger external pretrained models are allowed only if they are documented in `PROGRESS.md`, downloaded through `scripts/download.sh`, and evaluated on the unchanged test split; model size and external data use must be reported as deviations.

## Compliance Checklist

- [ ] `data/echo_toy/test.jsonl` is unchanged for reported scores.
- [ ] `scripts/evaluate.sh` reads artifacts and does not import from `method/`.
- [ ] Test labels are not used for training or model selection.
- [ ] Any external pretrained model or dataset is documented as a deviation.
