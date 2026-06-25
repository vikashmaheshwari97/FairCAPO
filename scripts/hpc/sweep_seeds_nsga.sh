#!/bin/bash
# ---------------------------------------------------------------------------
# Portable (non-SLURM) seed sweep for paper-scale NSGA-II-PO + fairness on BBQ.
# Uses scripts/run_baseline_nsga2_po.py (DIFFERENT runner than FairCAPO/ablation).
# Runs the seeds SEQUENTIALLY into per-seed output dirs.
#
# Usage:
#   bash scripts/hpc/sweep_seeds_nsga.sh [CONFIG] [RUN_TAG] [SEEDS...]
# Examples:
#   bash scripts/hpc/sweep_seeds_nsga.sh
#   bash scripts/hpc/sweep_seeds_nsga.sh configs/HPC_Config/nsga2_po_bbq_HPC.yaml bbq_nsga2po 0 1 2
#
# Assumes an OpenAI-compatible server is already up at the config's api_url.
# ---------------------------------------------------------------------------
set -euo pipefail

CONFIG="${1:-configs/HPC_Config/nsga2_po_bbq_HPC.yaml}"
RUN_TAG="${2:-bbq_nsga2po}"
shift 2 2>/dev/null || true
SEEDS=("${@:-0 1 2}")
if [ "${#SEEDS[@]}" -eq 1 ]; then read -r -a SEEDS <<< "${SEEDS[0]}"; fi

mkdir -p "outputs/hpc/${RUN_TAG}"

for SEED in "${SEEDS[@]}"; do
  OUT_DIR="outputs/hpc/${RUN_TAG}/seed_${SEED}"
  echo "=== NSGA-II-PO seed ${SEED} -> ${OUT_DIR} ==="
  PYTHONPATH=. python -u scripts/run_baseline_nsga2_po.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --output-dir "$OUT_DIR" \
    2>&1 | tee "outputs/hpc/${RUN_TAG}/seed_${SEED}.log"
done

echo "All NSGA seeds done: outputs/hpc/${RUN_TAG}/seed_*"
