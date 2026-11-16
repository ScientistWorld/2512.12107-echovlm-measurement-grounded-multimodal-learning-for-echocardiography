#!/bin/bash
# Job entry point (final hardening iteration, last check): retrain the reference and the
# CLIP-only baseline with every argument at its default into redteam/hardening/runs/repro_*,
# score them with both evaluators and refresh the audit summary.
set -e
cd /home/user
bash scripts/prepare_data.sh
python redteam/hardening/audit.py --final
python redteam/hardening/summarize2.py
