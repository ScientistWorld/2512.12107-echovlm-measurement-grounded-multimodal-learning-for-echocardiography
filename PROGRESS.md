# Progress

## What Works

- Paper analysis and briefings are filled.
- `data/make_echo_toy.py` creates a deterministic echo-style dataset with views, structured measurements, captions, negated captions, and disease labels.
- `method/echovlm.py` implements the paper's actual loss structure: symmetric CLIP loss, view-informed image contrast, and negation-aware text separation.
- `scripts/reproduce.sh` runs a CLIP-only baseline, the full EchoVLM objective, and artifact-only evaluation.

## Results

No reproduced scores yet. The next submitted CPU smoke job will run the full pipeline and write `scoring/scores.json`.

## Remaining

- Run the pipeline inside the managed job/container.
- Inspect `scoring/scores.json` and compare against the CLIP-only baseline.
- If the full objective does not improve the negation or disease metrics, tune the scaled experiment and resubmit.

## Issues

- EchoGround-MIMIC is described as a future release in the paper and is not present here.
- The paper's view-classification experiment uses private multi-site TTE data.
- The paper initializes from internal echo-pretrained vision weights and EchoCLIP text weights; those exact weights are not available here.

## Deviations from Paper

- **Data**: The paper uses EchoGround-MIMIC with 19,065 image-text pairs from MIMIC-IV-ECHO. This workspace uses a deterministic substitute generated from paper figure images because the dataset is unavailable.
- **Scale**: The paper trains ViT-B/CLIP for 20 epochs with global batch size 512. This reproduction uses a compact CNN/text encoder and small batch sizes so the algorithm can be smoke-tested within the available environment.
- **Initialization**: The paper uses internal echo-pretrained vision weights and EchoCLIP text weights. This reproduction trains the compact model from scratch because those weights are unavailable.
