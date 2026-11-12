#!/bin/bash
# Install Python packages into the container overlay (system Python, no venv).
# torch: models/training; numpy + pillow: data preparation; tokenizers: the
# evaluator tokenizes evaluator-owned text with each method's tokenizer.json.
set -e

uv pip install --system torch pillow numpy tokenizers
