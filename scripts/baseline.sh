#!/bin/bash
# CLIP-only baseline (paper Table 6, "L_CLIP" row): same backbone, initialisation and
# optimisation recipe, image-text contrastive loss only. Self-contained (baseline/).
#
# Usage: bash scripts/baseline.sh          (SEED / EPOCHS env vars optional)
set -e
cd "$(dirname "$0")/.."

bash scripts/prepare_data.sh
python -m baseline.clip_finetune \
  --output-dir checkpoints/clip_only \
  --epochs "${EPOCHS:-20}" \
  --seed "${SEED:-0}"
