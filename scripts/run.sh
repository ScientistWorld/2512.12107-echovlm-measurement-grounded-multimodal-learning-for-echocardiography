#!/bin/bash

set -e

cd /home/user

rm -rf checkpoints/clip_only checkpoints/echovlm_view checkpoints/echovlm_negation checkpoints/echovlm_full

EPOCHS=20 BATCH_SIZE=32 bash scripts/reproduce.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python validate.py --compare
