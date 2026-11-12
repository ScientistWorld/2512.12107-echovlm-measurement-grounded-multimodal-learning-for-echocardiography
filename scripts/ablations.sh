#!/bin/bash
# Paper Table 6 single-objective ablations: L_CLIP + L_view and L_CLIP + L_neg.
set -e
cd "$(dirname "$0")/.."

bash scripts/prepare_data.sh
python -m method.train --output-dir checkpoints/echovlm_view \
  --epochs "${EPOCHS:-20}" --seed "${SEED:-0}" --lambda-view 0.5 --lambda-neg 0.0
python -m method.train --output-dir checkpoints/echovlm_negation \
  --epochs "${EPOCHS:-20}" --seed "${SEED:-0}" --lambda-view 0.0 --lambda-neg 0.1
