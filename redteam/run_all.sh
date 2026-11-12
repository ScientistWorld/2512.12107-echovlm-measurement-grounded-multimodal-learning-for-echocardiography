#!/bin/bash
set -uo pipefail
cd /home/user
# Limit CPU threading for the small Transformers, without changing evaluation.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
status=0
for name in reference_cached reference_seed7 reference_seed17 reference_seed27 \
  stronger_training100 stronger_training200 weaker_untrained weaker_clip_only \
  shortcut_visible_answers shortcut_generator_answers shortcut_text_only \
  shortcut_bruteforce_answers shortcut_forged_labels shortcut_norm_scaling \
  shortcut_constant_antipodes shortcut_constant_scaled_failure shortcut_empty_vectors \
  shortcut_forged_scores shortcut_baseline_degradation shortcut_row_selection \
  shortcut_missing_artifact shortcut_empty_artifact shortcut_null_vectors; do
  bash "redteam/$name/run.sh" || status=1
done
python -m redteam.summarize || status=1
exit "$status"
