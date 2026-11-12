#!/bin/bash
# Fetch the raw CAMUS patients of the hidden test split (evaluator-owned).
# Called by scripts/download.sh when eval/test/ is present. Idempotent.
set -e
cd "$(dirname "$0")/../.."
python3 eval/test/fetch_test_camus.py --split-file eval/test/test_patients.json --keys test \
  --out eval/test/downloads/camus
