#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container or
# submitting jobs. Idempotent: every fetch is skipped when its file exists.
# Uses only bash, curl and the Python standard library.
#
# Estimated total download size: ~4.7 GB
#   CAMUS public release, 500 patients, half-sequences + masks + info  ~3.5 GB
#     (350 visible patients -> data/downloads/camus,
#      150 hidden-test patients -> eval/test/downloads/camus)
#   OpenAI CLIP ViT-B/16 (open_clip safetensors)                        ~0.6 GB
#   EchoCLIP (open_clip weights + tokenizer)                            ~0.6 GB
# Disk after preprocessing: + ~0.2 GB (data/downloads/camus_vlm)

set -e

cd "$(dirname "$0")/.."

# 1. CAMUS (CC BY-NC-SA 4.0, https://humanheart-project.creatis.insa-lyon.fr/database/)
#    visible train / val patients
python3 data/fetch_camus.py --split-file data/camus_split.json --keys train val --out data/downloads/camus

# 2. hidden-test patients: only fetched when the evaluator-owned eval/test/ is present
if [ -f eval/test/fetch_test_data.sh ]; then
  bash eval/test/fetch_test_data.sh
fi

# 3. pretrained weights (HuggingFace)
fetch() {
  local url="$1" dst="$2"
  if [ -s "$dst" ]; then return 0; fi
  mkdir -p "$(dirname "$dst")"
  curl -fL --retry 5 -o "$dst.part" "$url"
  mv "$dst.part" "$dst"
}
HF=https://huggingface.co
fetch "$HF/timm/vit_base_patch16_clip_224.openai/resolve/main/open_clip_model.safetensors" \
  data/downloads/models/openai-clip-vit-b-16/open_clip_model.safetensors
fetch "$HF/timm/vit_base_patch16_clip_224.openai/resolve/main/open_clip_config.json" \
  data/downloads/models/openai-clip-vit-b-16/open_clip_config.json
fetch "$HF/mkaichristensen/echo-clip/resolve/main/open_clip_pytorch_model.bin" \
  data/downloads/models/echo-clip/open_clip_pytorch_model.bin
fetch "$HF/mkaichristensen/echo-clip/resolve/main/open_clip_config.json" \
  data/downloads/models/echo-clip/open_clip_config.json
fetch "$HF/mkaichristensen/echo-clip/resolve/main/tokenizer.json" \
  data/downloads/models/echo-clip/tokenizer.json

echo "downloads complete"
