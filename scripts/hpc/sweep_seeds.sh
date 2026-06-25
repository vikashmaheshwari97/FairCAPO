#!/bin/bash
# ---------------------------------------------------------------------------
# Legacy portable (non-SLURM) seed sweep for paper-scale HEAL-CAPO / MO-CAPO.
# For current BBQ Rocket jobs, prefer run_bbq_hpc.slurm and run_bbq_nsga_hpc.slurm.
# Runs the seeds SEQUENTIALLY into per-seed output dirs. Use on an interactive
# HPC node or a workstation; for a real cluster prefer run_mocapo_hpc.slurm
# (which runs the seeds in parallel as a job array).
#
# Usage:
#   bash scripts/hpc/sweep_seeds.sh [CONFIG] [RUN_TAG] [SEEDS...]
# Examples:
#   bash scripts/hpc/sweep_seeds.sh
#   bash scripts/hpc/sweep_seeds.sh configs/HPC_Config/phase2_budgeted_mocapo_subj_HPC.yaml subj_mistral 0
#   bash scripts/hpc/sweep_seeds.sh configs/HPC_Config/phase2_budgeted_mocapo_subj_HPC.yaml subj_mistral 0 1 2
#
# Assumes an OpenAI-compatible server is already up at the config's api_url.
# ---------------------------------------------------------------------------
set -euo pipefail

CONFIG="${1:-configs/HPC_Config/phase2_budgeted_mocapo_subj_HPC.yaml}"
RUN_TAG="${2:-subj_mistral}"
shift 2 2>/dev/null || true
SEEDS=("${@:-0}")
# Normalize when SEEDS came through as a single "0 1 2" string.
if [ "${#SEEDS[@]}" -eq 1 ]; then read -r -a SEEDS <<< "${SEEDS[0]}"; fi

mkdir -p "outputs/hpc/${RUN_TAG}"

for SEED in "${SEEDS[@]}"; do
  OUT_DIR="outputs/hpc/${RUN_TAG}/seed_${SEED}"
  echo "=== seed ${SEED} -> ${OUT_DIR} ==="
  PYTHONPATH=. python scripts/run_phase2_budgeted_mocapo.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --output-dir "$OUT_DIR" \
    2>&1 | tee "outputs/hpc/${RUN_TAG}/seed_${SEED}.log"
done

echo "All seeds done: outputs/hpc/${RUN_TAG}/seed_*"
