#!/bin/bash
# Evaluate the held-out test slice from precomputed method artifacts.

set -e

cd /home/user

mkdir -p scoring

inputs=("$@")
if [ "${#inputs[@]}" -eq 0 ]; then
  mapfile -t inputs < <(
    python - <<'PY'
from pathlib import Path

for path in sorted(Path("checkpoints").glob("*/test_embeddings.jsonl")):
    print(f"{path.parent.name}={path}")
PY
  )
fi

if [ "${#inputs[@]}" -eq 0 ]; then
  echo "No embedding artifacts found. Run scripts/method.sh or scripts/baseline.sh first." >&2
  exit 1
fi

python eval/test/evaluate.py --inputs "${inputs[@]}" --output scoring/scores_test.json
