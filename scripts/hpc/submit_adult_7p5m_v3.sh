#!/usr/bin/env bash
set -euo pipefail

# Ensure every held-out job saves the full instruction + ordered demonstrations.
export EVAL_RUNNER="scripts/evaluate_adult_v3_on_test.py"

exec bash scripts/hpc/submit_adult_7p5m_v3_pipeline.sh "$@"
