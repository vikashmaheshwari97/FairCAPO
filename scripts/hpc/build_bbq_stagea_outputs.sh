#!/usr/bin/env bash
set -euo pipefail

# Build Stage A large-held-out tables and figures for the completed Rocket 500k
# seed-0 portfolios. Run this on a login node only after the three Stage A eval
# SLURM jobs have completed; it does not start vLLM and does not use a GPU.

cd "$(dirname "$0")/../.."

TABLE_CONFIG="${TABLE_CONFIG:-configs/HPC_Config/experiment_table_bbq_500k_large_HPC.yaml}"
AGG_CONFIG="${AGG_CONFIG:-configs/HPC_Config/aggregate_multiseed_bbq_500k_large_HPC.yaml}"
TABLE_CSV="${TABLE_CSV:-outputs/experiment_table/bbq_mistral_hpc_500k_large/experiment_table.csv}"
FIG_DIR="${FIG_DIR:-outputs/figures/paper_bbq_hpc_500k_large}"
TITLE="${TITLE:-BBQ / Mistral-Small-3.2 / Rocket 500k seed 0 (large held-out)}"
RUN_AGGREGATE="${RUN_AGGREGATE:-0}"
RUN_SEARCH_FIGURE="${RUN_SEARCH_FIGURE:-0}"

echo "Checking Stage A eval outputs..."
test -f outputs/hpc/evaluation_large/seed_0/bbq_faircapo/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/bbq_ablation/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/bbq_nsga2po/test_eval_summary.json

echo "Building Stage A experiment table..."
PYTHONPATH=. python scripts/build_experiment_table.py --config "${TABLE_CONFIG}"

if [[ "${RUN_AGGREGATE}" == "1" ]]; then
  echo "Building Stage A aggregate summary..."
  PYTHONPATH=. python scripts/aggregate_multiseed.py --config "${AGG_CONFIG}"
fi

mkdir -p "${FIG_DIR}"
rm -f "${FIG_DIR}/fig_front_richness_bbq.png"

echo "Building Stage A paper figures..."
python scripts/visualize_paper_figures.py \
  --run outputs/hpc/bbq_faircapo/seed_0 \
  --run-csv outputs/hpc/evaluation_large/seed_0/bbq_faircapo/test_eval_candidates.csv \
  --table "${TABLE_CSV}" \
  --title "${TITLE}" \
  --out "${FIG_DIR}"

if [[ "${RUN_SEARCH_FIGURE}" == "1" ]]; then
  echo "Building search-basis front richness figure..."
  python scripts/visualize_front_richness.py \
    --faircapo outputs/hpc/bbq_faircapo/seed_0/phase2_all_candidates.csv \
    --nsga outputs/hpc/bbq_nsga2po/seed_0/nsga2_po_all_candidates.csv \
    --ablation outputs/hpc/bbq_ablation/seed_0/phase2_all_candidates.csv \
    --title "BBQ / Mistral-Small-3.2 / Rocket 500k seed 0 (search basis; not Stage A held-out)" \
    --out "${FIG_DIR}/fig_front_richness_search_basis.png"
fi

echo "Building Stage A large-held-out staircase..."
python scripts/visualize_staircase.py \
  --fair outputs/hpc/evaluation_large/seed_0/bbq_faircapo/test_eval_candidates.csv \
  --portfolio outputs/hpc/bbq_faircapo/seed_0/phase2_prompt_portfolio.csv \
  --mocapo "" \
  --title "${TITLE}" \
  --out "${FIG_DIR}/fig_pareto_staircase.png" \
  --color-fairness

echo "Stage A table:"
echo "  ${TABLE_CSV}"
echo "Stage A figures:"
ls -lh "${FIG_DIR}"
