#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script must be SELF-CONTAINED using only code in eval/.
# Do NOT import from method/ — evaluate.sh must work even when the
# paper's method code is hidden from a scientist agent.
#
# Design it so that anyone following the same output format can run
# this script to evaluate their results.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict nested by experiment, e.g.:
#   {"imagenet_classification": {"top1_accuracy": 82.1, "top5_accuracy": 94.8}}
# Experiment and metric names must match the keys in scoring/reference.json.
# This file is read by the system for automated comparison.
#
# Example usage:
#   ./evaluate.sh <method_and_run_name>                             # evaluate with defaults
#   ./evaluate.sh <method_and_run_name> checkpoints/epoch_10.pt     # evaluate a specific checkpoint
#
# Called from run.sh after training, or independently.

set -e

cd /home/user

mkdir -p scoring

inputs=("$@")
if [ "${#inputs[@]}" -eq 0 ]; then
  mapfile -t inputs < <(
    python - <<'PY'
from pathlib import Path

preferred = ["clip_only", "echovlm_view", "echovlm_negation", "echovlm_full"]
seen = set()
for name in preferred:
    path = Path("checkpoints") / name / "test_embeddings.jsonl"
    if path.exists():
        print(f"{name}={path}")
        seen.add(name)
for path in sorted(Path("checkpoints").glob("*/test_embeddings.jsonl")):
    if path.parent.name not in seen:
        print(f"{path.parent.name}={path}")
PY
  )
fi

if [ "${#inputs[@]}" -eq 0 ]; then
  echo "No embedding artifacts found. Run scripts/method.sh or scripts/baseline.sh first." >&2
  exit 1
fi

python -m eval.metrics --inputs "${inputs[@]}" --output scoring/scores.json
