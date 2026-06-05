# Progress

## What Works

- Paper analysis and briefings are filled.
- `data/make_echo_toy.py` creates a deterministic echo-style dataset with views, structured measurements, captions, negated captions, and disease labels.
- `method/echovlm.py` implements the paper's actual loss structure: symmetric CLIP loss, view-informed image contrast, and negation-aware text separation.
- `scripts/reproduce.sh` runs a CLIP-only baseline, the full EchoVLM objective, and artifact-only evaluation.

## Results

No reproduced scores yet. CPU smoke job `9d0e4379-6f3` failed before training because `method/train.py` was invoked as a file path, which hid the workspace root from Python imports inside the container.

## Remaining

- Rerun the pipeline inside the managed job/container after the module-entrypoint fix.
- Inspect `scoring/scores.json` and compare against the CLIP-only baseline.
- If the full objective does not improve the negation or disease metrics, tune the scaled experiment and resubmit.

## Issues

- Fixed after job `9d0e4379-6f3`: changed method execution to `python -m method.train` and updated the CLIP-only baseline wrapper to invoke the same module entry point so top-level packages such as `data` are importable.
- EchoGround-MIMIC is described as a future release in the paper and is not present here.
- The paper's view-classification experiment uses private multi-site TTE data.
- The paper initializes from internal echo-pretrained vision weights and EchoCLIP text weights; those exact weights are not available here.

## Deviations from Paper

- **Data**: The paper uses EchoGround-MIMIC with 19,065 image-text pairs from MIMIC-IV-ECHO. This workspace uses a deterministic substitute generated from paper figure images because the dataset is unavailable.
- **Scale**: The paper trains ViT-B/CLIP for 20 epochs with global batch size 512. This reproduction uses a compact CNN/text encoder and small batch sizes so the algorithm can be smoke-tested within the available environment.
- **Initialization**: The paper uses internal echo-pretrained vision weights and EchoCLIP text weights. This reproduction trains the compact model from scratch because those weights are unavailable.
