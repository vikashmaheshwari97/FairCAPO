#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

test -s scripts/run_adult_v3_mocapo_ablation.py
test -s scripts/run_adult_v3_nsga2_po.py
test -s scripts/evaluate_adult_v3_on_test.py

# Ensure every held-out job saves the full instruction + ordered demonstrations.
export EVAL_RUNNER="scripts/evaluate_adult_v3_on_test.py"

exec bash scripts/hpc/submit_adult_7p5m_v3_pipeline.sh "$@"
