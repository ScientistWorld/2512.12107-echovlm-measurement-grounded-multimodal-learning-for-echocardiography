# EchoVLM: Measurement-Grounded Multimodal Learning for Echocardiography

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2512.12107"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2512.12107-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2512.12107](https://arxiv.org/abs/2512.12107) |
|:---|:---|
| **Domain** | Medical vision-language learning |
| **TL;DR** | EchoVLM adapts CLIP-style pretraining to echocardiography using measurement-grounded captions plus view-informed and negation-aware contrastive objectives. |

## Summary

The paper introduces EchoGround-MIMIC, an echocardiography image-text dataset with standardized views, structured measurements, captions, and disease labels. EchoVLM trains a VLM with standard image-text contrast plus two echo-specific losses: one groups images by acquisition view and one separates positive clinical captions from negated counterparts. The reported main result is stronger zero-shot disease classification and retrieval than echo-specific and general VLM baselines, with additional transfer to view classification, segmentation, and landmark detection.

## Problem

Can an echocardiography image-text model learn visual representations that support clinically grounded interpretation across multimodal tasks? The benchmark centers on image-caption alignment, disease finding discrimination, and view-aware transfer for echo images with structured clinical measurements and standardized views.

## The paper's method

EchoVLM extends CLIP-style image-text pretraining for echocardiography with two clinically informed objectives on measurement-grounded echo image-caption pairs.

## Key results of the paper

- Zero-shot disease classification on EchoGround-MIMIC: 86.5 AUC, 34.2 precision, 86.2 recall.
- Image-text retrieval on EchoGround-MIMIC: 2.98 Recall@5 and 5.70 Recall@10.
- Downstream view classification: 95.1 accuracy, 95.3 F1, 95.8 precision.

## Run it

```bash
pip install sciworld
sw research 2512.12107 --mode innovate --agent claude-code
sw evaluate 2512.12107
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

