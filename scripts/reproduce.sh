#!/bin/bash
# Reproduce the reference end to end (inside the container, after scripts/download.sh):
#   data preparation -> CLIP-only baseline -> Table 6 ablations -> EchoVLM
#   -> visible and hidden-test scoring of every method under checkpoints/.
set -e
cd "$(dirname "$0")/.."

bash scripts/prepare_data.sh
bash scripts/baseline.sh
bash scripts/ablations.sh
bash scripts/method.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
