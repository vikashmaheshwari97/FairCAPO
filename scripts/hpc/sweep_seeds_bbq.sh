#!/bin/bash
# ---------------------------------------------------------------------------
# Portable (non-SLURM) seed sweep for paper-scale FairCAPO / ablation on BBQ.
# Runs the seeds SEQUENTIALLY into per-seed output dirs. Use on an interactive
# HPC node or a workstation; for a real cluster prefer run_bbq_hpc.slurm
# (which runs the seeds in parallel as a job array).
#
# Usage:
#   bash scripts/hpc/sweep_seeds_bbq.sh [CONFIG] [RUN_TAG] [SEEDS...]
# Examples:
#   bash scripts/hpc/sweep_seeds_bbq.sh
#   bash scripts/hpc/sweep_seeds_bbq.sh configs/HPC_Config/phase2_budgeted_mocapo_bbq_HPC.yaml bbq_faircapo 0 1 2
#   bash scripts/hpc/sweep_seeds_bbq.sh configs/HPC_Config/mocapo_baseline_bbq_HPC.yaml bbq_ablation 0 1 2
#
# Assumes an OpenAI-compatible server is already up at the config's api_url.
# ---------------------------------------------------------------------------
set -euo pipefail

CONFIG="${1:-configs/HPC_Config/phase2_budgeted_mocapo_bbq_HPC.yaml}"
RUN_TAG="${2:-bbq_faircapo}"
shift 2 2>/dev/null || true
SEEDS=("${@:-0 1 2}")
# Normalize when SEEDS came through as a single "0 1 2" string.
if [ "${#SEEDS[@]}" -eq 1 ]; then read -r -a SEEDS <<< "${SEEDS[0]}"; fi

mkdir -p "outputs/hpc/${RUN_TAG}"

for SEED in "${SEEDS[@]}"; do
  OUT_DIR="outputs/hpc/${RUN_TAG}/seed_${SEED}"
  echo "=== seed ${SEED} -> ${OUT_DIR} ==="
  PYTHONPATH=. python -u scripts/run_phase2_budgeted_mocapo.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --output-dir "$OUT_DIR" \
    2>&1 | tee "outputs/hpc/${RUN_TAG}/seed_${SEED}.log"
done

echo "All seeds done: outputs/hpc/${RUN_TAG}/seed_*"
