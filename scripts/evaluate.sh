#!/bin/bash
# Score every method under checkpoints/ on the visible validation patients
# (same protocol as the hidden test) and write scoring/scores.json.
# A method is a directory checkpoints/<name>/ containing export/ (see briefing/evaluation.md).
#
# Usage: bash scripts/evaluate.sh [--root DIR] [--only NAME ...]
set -e
cd "$(dirname "$0")/.."
mkdir -p scoring
python eval/train/evaluate.py --output scoring/scores.json "$@"
