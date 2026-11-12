#!/bin/bash
# Train the paper's method (EchoVLM: L_CLIP + 0.5 L_view + 0.1 L_neg) and export it
# to checkpoints/echovlm_full/export/ for the evaluators.
#
# Usage: bash scripts/method.sh            (SEED / EPOCHS env vars optional)
set -e
cd "$(dirname "$0")/.."

bash scripts/prepare_data.sh
python -m method.train \
  --output-dir checkpoints/echovlm_full \
  --epochs "${EPOCHS:-20}" \
  --seed "${SEED:-0}" \
  --lambda-view 0.5 \
  --lambda-neg 0.1
