#!/bin/bash

set -e

cd /home/user

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
