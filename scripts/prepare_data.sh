#!/bin/bash
# Build the image-caption splits from the raw CAMUS downloads (runs in the container).
#   visible:  data/downloads/camus_vlm/{train,val}_images.npy + {train,val}.jsonl
#   hidden:   eval/test/hidden_test.npz (committed; rebuilt only if missing)
set -e
cd "$(dirname "$0")/.."

if [ ! -f data/downloads/camus_vlm/val.jsonl ] || [ ! -f data/downloads/camus_vlm/train.jsonl ]; then
  python -m data.prepare_camus
fi
if [ -f eval/test/prepare_test.py ] && [ ! -f eval/test/hidden_test.npz ]; then
  python eval/test/prepare_test.py
fi
