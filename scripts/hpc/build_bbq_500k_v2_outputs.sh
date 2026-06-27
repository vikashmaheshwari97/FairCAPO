#!/usr/bin/env bash
set -euo pipefail

# Build held-out table and figures for the active Rocket 500k v2 seed-0 run.
# Run this on a login node after the three standard eval jobs and the
# re-evaluated post-hoc job have completed. This script does not start vLLM.

cd "$(dirname "$0")/../.."

TABLE_CONFIG="${TABLE_CONFIG:-configs/HPC_Config/experiment_table_bbq_HPC.yaml}"
AGG_CONFIG="${AGG_CONFIG:-configs/HPC_Config/aggregate_multiseed_bbq_HPC.yaml}"
TABLE_CSV="${TABLE_CSV:-outputs/experiment_table/bbq_mistral_hpc_500k_v2_seed0/experiment_table.csv}"
FIG_DIR="${FIG_DIR:-outputs/figures/paper_bbq_hpc_500k_v2_seed0}"
TITLE="${TITLE:-BBQ / Mistral-Small-3.2 / Rocket 500k v2 seed 0 (held-out)}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"

echo "Checking 500k v2 held-out eval outputs..."
test -f outputs/hpc/evaluation/seed_0/bbq_faircapo_500k_v2/test_eval_summary.json
test -f outputs/hpc/evaluation/seed_0/bbq_ablation_500k_v2/test_eval_summary.json
test -f outputs/hpc/evaluation/seed_0/bbq_nsga2po_500k_v2/test_eval_summary.json
test -f outputs/hpc/evaluation/seed_0/bbq_posthoc_500k_v2/test_eval_summary.json
test -f outputs/hpc/evaluation/seed_0/bbq_posthoc_500k_v2/test_eval_candidates.csv

echo "Building 500k v2 held-out experiment table..."
PYTHONPATH=. python scripts/build_experiment_table.py --config "${TABLE_CONFIG}"

if [[ "${RUN_AGGREGATE}" == "1" ]]; then
  echo "Building 500k v2 aggregate summary..."
  PYTHONPATH=. python scripts/aggregate_multiseed.py --config "${AGG_CONFIG}"
fi

mkdir -p "${FIG_DIR}"

echo "Building 500k v2 held-out figures..."
python scripts/visualize_paper_figures.py \
  --run outputs/hpc/bbq_faircapo_500k_v2/seed_0 \
  --run-csv outputs/hpc/evaluation/seed_0/bbq_faircapo_500k_v2/test_eval_candidates.csv \
  --table "${TABLE_CSV}" \
  --title "${TITLE}" \
  --out "${FIG_DIR}"

echo "Building 500k v2 held-out staircase..."
python scripts/visualize_staircase.py \
  --fair outputs/hpc/evaluation/seed_0/bbq_faircapo_500k_v2/test_eval_candidates.csv \
  --portfolio outputs/hpc/bbq_faircapo_500k_v2/seed_0/phase2_prompt_portfolio.csv \
  --mocapo "" \
  --title "${TITLE}" \
  --out "${FIG_DIR}/fig_pareto_staircase.png" \
  --color-fairness

echo "500k v2 table:"
echo "  ${TABLE_CSV}"
echo "500k v2 figures:"
ls -lh "${FIG_DIR}"
