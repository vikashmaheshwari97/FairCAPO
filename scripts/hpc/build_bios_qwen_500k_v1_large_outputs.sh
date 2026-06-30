#!/usr/bin/env bash
set -euo pipefail

# Build Bias-in-Bios Qwen FairCAPO seed-0 comparison table and figures.

cd "$(dirname "$0")/../.."

TABLE_CONFIG="${TABLE_CONFIG:-configs/HPC_Config/bios_experiment_table_qwen_500k_v1_large_HPC.yaml}"
TABLE_CSV="${TABLE_CSV:-outputs/experiment_table/bios_qwen_hpc_500k_v1_large_seed0/experiment_table.csv}"
FIG_DIR="${FIG_DIR:-outputs/figures/paper_bios_qwen_hpc_500k_v1_large_seed0}"
TITLE="${TITLE:-Bias-in-Bios / Qwen3-30B-A3B-Instruct-2507 / Rocket 500k seed 0 (large held-out)}"

echo "Checking Bias-in-Bios Qwen large-held-out eval output..."
test -f outputs/hpc/evaluation_large/seed_0/bios_faircapo_qwen_500k_v1/test_eval_summary.json

echo "Building Bias-in-Bios Qwen experiment table..."
PYTHONPATH=. python scripts/build_experiment_table.py --config "${TABLE_CONFIG}"

mkdir -p "${FIG_DIR}"

echo "Building Bias-in-Bios Qwen paper figures..."
python scripts/visualize_paper_figures.py \
  --run outputs/hpc/bios_faircapo_qwen_500k_v1/seed_0 \
  --run-csv outputs/hpc/evaluation_large/seed_0/bios_faircapo_qwen_500k_v1/test_eval_candidates.csv \
  --table "${TABLE_CSV}" \
  --title "${TITLE}" \
  --out "${FIG_DIR}"

echo "Building Bias-in-Bios Qwen staircase..."
python scripts/visualize_staircase.py \
  --fair outputs/hpc/evaluation_large/seed_0/bios_faircapo_qwen_500k_v1/test_eval_candidates.csv \
  --portfolio outputs/hpc/bios_faircapo_qwen_500k_v1/seed_0/phase2_prompt_portfolio.csv \
  --mocapo "" \
  --title "${TITLE}" \
  --out "${FIG_DIR}/fig_pareto_staircase.png" \
  --color-fairness

echo "Bias-in-Bios Qwen table:"
echo "  ${TABLE_CSV}"
echo "Bias-in-Bios Qwen figures:"
ls -lh "${FIG_DIR}"
