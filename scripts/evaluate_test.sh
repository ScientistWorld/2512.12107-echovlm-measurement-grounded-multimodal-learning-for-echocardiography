#!/bin/bash
# Hidden-test evaluation: scores every method directory under checkpoints/
# (each must contain export/) on the 150 held-out patients in eval/test/.
# Writes scoring/scores_test.json. Runs with no arguments.
set -e
cd "$(dirname "$0")/.."
mkdir -p scoring
if [ ! -f eval/test/test_meta.json ]; then
  python eval/test/prepare_test.py
fi
python eval/test/evaluate.py "$@"
