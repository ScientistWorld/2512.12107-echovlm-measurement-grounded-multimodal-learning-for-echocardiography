#!/bin/bash
# Job entry point: reproduce the reference end to end inside the container
# (data preparation -> CLIP-only baseline -> Table 6 ablations -> EchoVLM -> all evaluators).
# The hardening audit that selected and checked this setup is redteam/hardening/audit.py
# (see HARDENING.md).
set -e
cd /home/user
bash scripts/reproduce.sh
