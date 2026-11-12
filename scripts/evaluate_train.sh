#!/bin/bash
# Visible-split evaluation: scores every method directory under checkpoints/
# (each must contain export/) on the 50 visible validation patients.
# Writes scoring/scores_train.json. Runs with no arguments.
set -e
cd "$(dirname "$0")/.."
mkdir -p scoring
python eval/train/evaluate.py "$@"
