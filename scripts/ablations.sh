#!/bin/bash
# Run EchoVLM objective ablations from Table 6.

set -e

cd /home/user

if [ ! -f data/echo_toy/train.jsonl ]; then
  python -m data.make_echo_toy --output data/echo_toy --seed 7 --size 112
fi

python -m method.train \
  --data-root data/echo_toy \
  --output-dir checkpoints/echovlm_view \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --lambda-view 0.5 \
  --lambda-neg 0.0

python -m method.train \
  --data-root data/echo_toy \
  --output-dir checkpoints/echovlm_negation \
  --epochs "${EPOCHS:-20}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --lambda-view 0.0 \
  --lambda-neg 0.1

bash scripts/evaluate.sh \
  "echovlm_view=checkpoints/echovlm_view/test_embeddings.jsonl" \
  "echovlm_negation=checkpoints/echovlm_negation/test_embeddings.jsonl"
