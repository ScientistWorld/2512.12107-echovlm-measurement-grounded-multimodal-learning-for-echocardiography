#!/bin/bash
# Job entry point (final hardening iteration). Resumable: finished runs / score files are skipped.
#   0. data: hidden_test.npz (compressed images + ground truth) from the raw hidden patients,
#      checked identical to the previous plaintext test files; 5-frames-per-sequence training split
#   1. smoke test of every trainer / builder / evaluator (aborts on a broken evaluator)
#   2. regime search on the visible split -> selected regime -> reference set, ablations, seeds,
#      variants, controls, shortcuts -> official checkpoints/ + scoring/scores_{train,test}.json
#   3. summary (redteam/hardening/summary_v2.{md,json})
set -e
cd /home/user

python - <<'PY'
import os, torch
print({"torch": torch.__version__, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
       "cpus": os.cpu_count()}, flush=True)
PY

bash scripts/prepare_data.sh
python - <<'PY'
import json, numpy as np
from pathlib import Path
new = np.load("eval/test/hidden_test.npz", allow_pickle=False)
meta = json.loads(new["meta"].tobytes().decode())
old_dir = Path("eval/test/legacy_v1")
if (old_dir / "test_meta.json").is_file():
    old_imgs = np.load(old_dir / "test_images.npz")["images"]
    old_recs = json.loads((old_dir / "test_meta.json").read_text())["records"]
    same = bool(np.array_equal(old_imgs, new["images"])) and old_recs == meta["records"]
    print({"hidden_test_identical_to_previous": same, "n": len(meta["records"])}, flush=True)
PY
if [ ! -f data/downloads/camus_vlm_f5/train.jsonl ]; then
  python -m data.prepare_camus --train-frames eval --out data/downloads/camus_vlm_f5
fi

python redteam/hardening/audit.py
python redteam/hardening/summarize2.py
