#!/bin/bash
# Build the image-caption splits from the raw CAMUS downloads (runs in the container).
#   visible:  data/downloads/camus_vlm/{train,val}_images.npy + {train,val}.jsonl
#             (5 evenly spaced frames per ED->ES sequence, ~10 image-caption pairs per patient)
#   hidden:   eval/test/hidden_test.npz (committed; rebuilt only if missing)
set -e
cd "$(dirname "$0")/.."

# Rebuild when missing or when built with a different training-frame sampling than the default.
if [ ! -f data/downloads/camus_vlm/val.jsonl ] || [ ! -f data/downloads/camus_vlm/train.jsonl ] \
   || ! grep -q '"frames": "eval"' data/downloads/camus_vlm/summary.json 2>/dev/null; then
  python -m data.prepare_camus
fi
if [ -f eval/test/prepare_test.py ] && [ ! -f eval/test/hidden_test.npz ]; then
  python eval/test/prepare_test.py
fi
