#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -e

cd /home/user

if [ ! -f data/echo_toy/train.jsonl ]; then
  python -m data.make_echo_toy --output data/echo_toy --seed 7 --size 112
fi
python baseline/random_clip.py \
  --data-root data/echo_toy \
  --output-dir checkpoints/clip_only \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-32}"
