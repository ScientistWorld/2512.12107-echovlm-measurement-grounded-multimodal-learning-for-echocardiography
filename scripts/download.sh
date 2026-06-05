#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: <50 MB
#
# This reproduction uses deterministic local data generation and does not fetch
# external datasets or model weights.

set -e

cd /home/user

# The exact EchoGround-MIMIC dataset and internal/EchoCLIP initialization
# weights are not available in this workspace. The reproducible substitute data
# is generated deterministically from paper-provided images and requires no
# network download.
python -m data.make_echo_toy --output data/echo_toy --seed 7 --size 112
