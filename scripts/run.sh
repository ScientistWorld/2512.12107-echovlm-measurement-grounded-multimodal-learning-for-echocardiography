#!/bin/bash
# Job entry point (this iteration: smoke test of the hardened pipeline).
set -e
cd /home/user

python - <<'PY'
import os, torch, tokenizers, numpy, PIL
print({"torch": torch.__version__, "cuda": torch.cuda.is_available(), "gpus": torch.cuda.device_count(),
       "tokenizers": tokenizers.__version__, "numpy": numpy.__version__, "cpus": os.cpu_count()}, flush=True)
PY

python -m data.prepare_camus
python eval/test/prepare_test.py
python redteam/hardening/orchestrate.py --smoke
python redteam/hardening/summarize.py --reference smoke_full || true
for f in redteam/hardening/logs/*.log; do echo "=== $f"; tail -n 25 "$f"; done
