#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

if [ ! -f data/echo_toy/train.jsonl ]; then
  python -m data.make_echo_toy --output data/echo_toy --seed 7 --size 112
fi
python method/train.py \
  --data-root data/echo_toy \
  --output-dir checkpoints/echovlm_full \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --lambda-view 0.5 \
  --lambda-neg 0.1

bash scripts/evaluate.sh
