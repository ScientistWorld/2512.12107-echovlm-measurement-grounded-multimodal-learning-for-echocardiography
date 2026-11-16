#!/bin/bash
# Job entry point (final hardening iteration, follow-up). Resumable.
#   1. default data preparation reproduces the selected training split byte for byte
#   2. reference / baseline retrained with every argument at its default (checks the shipped defaults)
#   3. rebuilt shortcuts (stateful pooling, hand-written rule) scored by the hardened and previous evaluator
#   4. summary with the final experiment weights
set -e
cd /home/user

bash scripts/prepare_data.sh
CHECK=redteam/hardening/runs/prep_check
if [ ! -f "$CHECK/train.jsonl" ]; then
  python -m data.prepare_camus --out "$CHECK"
fi
python - <<'PY'
import json, numpy as np
from pathlib import Path
a, b = Path("data/downloads/camus_vlm"), Path("redteam/hardening/runs/prep_check")
same = {}
for split in ("train", "val"):
    same[split] = bool(np.array_equal(np.load(a / f"{split}_images.npy"), np.load(b / f"{split}_images.npy"))) \
        and (a / f"{split}.jsonl").read_bytes() == (b / f"{split}.jsonl").read_bytes()
print(json.dumps({"default_prep_identical_to_audit_data": same}), flush=True)
PY

python redteam/hardening/audit.py --final
python redteam/hardening/summarize2.py
