#!/bin/bash
# Evaluate the held-out test slice from precomputed method artifacts.

set -e

cd /home/user

mkdir -p scoring

inputs=("$@")
if [ "${#inputs[@]}" -eq 0 ] && [ -f scoring/scores_train.json ]; then
  mapfile -t inputs < <(
    python - <<'PY'
import json
from pathlib import Path

scores = json.loads(Path("scoring/scores_train.json").read_text(encoding="utf-8"))
experiments = scores.get("experiments", scores)
method_names = set()
input_paths = {}
for exp_data in experiments.values():
    if not isinstance(exp_data, dict):
        continue
    split = exp_data.get("split", {})
    if isinstance(split, dict) and isinstance(split.get("input_paths"), dict):
        input_paths.update(split["input_paths"])
    methods = exp_data.get("results", exp_data)
    if not isinstance(methods, dict):
        continue
    for name, method_scores in methods.items():
        if not name.startswith("_") and isinstance(method_scores, dict):
            method_names.add(name)

for name in sorted(method_names):
    path = Path(input_paths.get(name, Path("checkpoints") / name / "test_embeddings.jsonl"))
    if not path.is_file():
        raise SystemExit(f"Missing embedding artifact for train-scored method '{name}': {path}")
    print(f"{name}={path}")
PY
  )
fi

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
