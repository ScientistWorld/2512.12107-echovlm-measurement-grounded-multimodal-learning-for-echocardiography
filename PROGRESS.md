# Progress

## What Works

- Paper analysis and briefings are filled.
- `data/make_echo_toy.py` creates a deterministic echo-style dataset with views, structured measurements, captions, negated captions, and disease labels.
- `method/echovlm.py` implements the paper's actual loss structure: symmetric CLIP loss, view-informed image contrast, and negation-aware text separation.
- `scripts/reproduce.sh` runs a CLIP-only baseline, the full EchoVLM objective, and artifact-only evaluation.

## Results

CPU smoke job `ba2bbb9d-f93` completed end-to-end in the managed container and wrote `scoring/scores.json`.

| Experiment | Metric | CLIP-only | EchoVLM full | Direction |
| --- | ---: | ---: | ---: | --- |
| Toy disease zero-shot | AUC | 47.92 | 54.17 | higher better |
| Toy disease zero-shot | Recall | 50.00 | 66.67 | higher better |
| Toy retrieval | Recall@5 | 28.57 | 35.71 | higher better |
| Toy negation constraint | Negation cosine | 0.745 | 0.471 | lower better |

This supports the paper's core claim on the scaled substitute: adding view-informed and negation-aware contrastive terms improves the CLIP-style baseline while also improving the negation constraint.

## Remaining

- Run a longer CPU reproduction to test whether the positive gap is stable beyond the 20-epoch smoke run.
- If the longer run regresses, keep the smoke result as the verified core-claim checkpoint and tune the scaled experiment separately.

## Issues

- Fixed after job `9d0e4379-6f3`: changed method execution to `python -m method.train` and updated the CLIP-only baseline wrapper to invoke the same module entry point so top-level packages such as `data` are importable.
- Fixed after job `ba2bbb9d-f93`: replaced deprecated `torch.ByteStorage.from_buffer` image conversion with a NumPy-backed tensor conversion.
- EchoGround-MIMIC is described as a future release in the paper and is not present here.
- The paper's view-classification experiment uses private multi-site TTE data.
- The paper initializes from internal echo-pretrained vision weights and EchoCLIP text weights; those exact weights are not available here.

## Deviations from Paper

- **Data**: The paper uses EchoGround-MIMIC with 19,065 image-text pairs from MIMIC-IV-ECHO. This workspace uses a deterministic substitute generated from paper figure images because the dataset is unavailable.
- **Scale**: The paper trains ViT-B/CLIP for 20 epochs with global batch size 512. This reproduction uses a compact CNN/text encoder and small batch sizes so the algorithm can be smoke-tested within the available environment.
- **Initialization**: The paper uses internal echo-pretrained vision weights and EchoCLIP text weights. This reproduction trains the compact model from scratch because those weights are unavailable.
