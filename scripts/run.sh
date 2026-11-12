#!/bin/bash
# Job entry point (this iteration: hardening audit). Resumable: finished runs are skipped.
#   1. reference set (paper Table 6 rows, seed 0) -> checkpoints/ -> official scoring
#   2. seeds, weaker controls, candidate improvements, shortcuts -> redteam/hardening/
set -e
cd /home/user

python - <<'PY'
import os, torch
print({"torch": torch.__version__, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
       "cpus": os.cpu_count()}, flush=True)
PY
bash scripts/prepare_data.sh

H=redteam/hardening
python $H/orchestrate.py --only echovlm_full_s0 clip_only_s0 echovlm_view_s0 echovlm_negation_s0

for pair in echovlm_full:echovlm_full_s0 clip_only:clip_only_s0 echovlm_view:echovlm_view_s0 echovlm_negation:echovlm_negation_s0; do
  dst=checkpoints/${pair%%:*}; src=$H/runs/${pair##*:}
  rm -rf "$dst"; mkdir -p "$dst"
  cp -r "$src/export" "$dst/"; cp "$src/run_config.json" "$src/history.json" "$dst/"
done
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python $H/summarize.py || true

python $H/orchestrate.py
python $H/summarize.py
