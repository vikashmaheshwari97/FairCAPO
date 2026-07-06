#!/usr/bin/env bash
set -euo pipefail

# Build Bias-in-Bios 500k_v2 seed-0 large-held-out table and figures.
# Run on login node after the search/eval jobs complete. This does not start vLLM.

cd "$(dirname "$0")/../.."

TABLE_CONFIG="${TABLE_CONFIG:-configs/HPC_Config/bios_experiment_table_500k_v2_large_HPC.yaml}"
AGG_CONFIG="${AGG_CONFIG:-configs/HPC_Config/bios_aggregate_500k_v2_large_HPC.yaml}"
TABLE_CSV="${TABLE_CSV:-outputs/experiment_table/bios_mistral_hpc_500k_v2_large_seed0/experiment_table.csv}"
FIG_DIR="${FIG_DIR:-outputs/figures/paper_bios_hpc_500k_v2_large_seed0}"
TITLE="${TITLE:-Bias-in-Bios / Mistral-Small-3.2 / Rocket 500k v2 seed 0 (reason-then-label, large held-out)}"

echo "Checking Bias-in-Bios 500k_v2 large-held-out eval outputs..."
test -f outputs/hpc/evaluation_large/seed_0/bios_faircapo_500k_v2/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/bios_ablation_500k_v2/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/bios_nsga2po_500k_v2/test_eval_summary.json

echo "Building Bias-in-Bios 500k_v2 experiment table..."
PYTHONPATH=. python scripts/build_experiment_table.py --config "${TABLE_CONFIG}"

echo "Building Bias-in-Bios 500k_v2 aggregate summary..."
PYTHONPATH=. python scripts/aggregate_multiseed.py --config "${AGG_CONFIG}"

mkdir -p "${FIG_DIR}"

echo "Building Bias-in-Bios 500k_v2 paper figures..."
python scripts/visualize_paper_figures.py \
  --run outputs/hpc/bios_faircapo_500k_v2/seed_0 \
  --run-csv outputs/hpc/evaluation_large/seed_0/bios_faircapo_500k_v2/test_eval_candidates.csv \
  --table "${TABLE_CSV}" \
  --title "${TITLE}" \
  --out "${FIG_DIR}"

echo "Building Bias-in-Bios 500k_v2 staircase..."
python scripts/visualize_staircase.py \
  --fair outputs/hpc/evaluation_large/seed_0/bios_faircapo_500k_v2/test_eval_candidates.csv \
  --portfolio outputs/hpc/bios_faircapo_500k_v2/seed_0/phase2_prompt_portfolio.csv \
  --mocapo "" \
  --title "${TITLE}" \
  --out "${FIG_DIR}/fig_pareto_staircase.png" \
  --color-fairness

echo "Building Bias-in-Bios 500k_v2 search-basis front richness figure..."
python scripts/visualize_front_richness.py \
  --faircapo outputs/hpc/bios_faircapo_500k_v2/seed_0/phase2_all_candidates.csv \
  --nsga outputs/hpc/bios_nsga2po_500k_v2/seed_0/nsga2_po_all_candidates.csv \
  --ablation outputs/hpc/bios_ablation_500k_v2/seed_0/phase2_all_candidates.csv \
  --title "Bias-in-Bios / Mistral-Small-3.2 / Rocket 500k v2 seed 0 (search basis)" \
  --out "${FIG_DIR}/fig_front_richness_search_basis.png"

FAIR_TRAJ="outputs/hpc/bios_faircapo_500k_v2/seed_0/budgeted_mocapo_trajectory.json"
ABL_TRAJ="outputs/hpc/bios_ablation_500k_v2/seed_0/budgeted_mocapo_trajectory.json"
if [[ -f "${FAIR_TRAJ}" && -f "${ABL_TRAJ}" ]]; then
  echo "Building Bias-in-Bios 500k_v2 search-basis trajectory figure..."
  PYTHONPATH=. python scripts/visualize_trajectory.py \
    --trajectory "${FAIR_TRAJ}" \
    --label FairCAPO \
    --trajectory "${ABL_TRAJ}" \
    --label "MO-CAPO (fairness off)" \
    --title "Bias-in-Bios / Mistral-Small-3.2 / Rocket 500k v2 seed 0 (search trajectory)" \
    --out "${FIG_DIR}/fig_trajectory_search_basis.png"
fi

echo "Building Bias-in-Bios 500k_v2 Pareto diagnostic figures..."
python scripts/visualize_pareto_front.py \
  --run outputs/hpc/bios_faircapo_500k_v2/seed_0 \
  --csv outputs/hpc/evaluation_large/seed_0/bios_faircapo_500k_v2/test_eval_candidates.csv \
  --title "${TITLE}" \
  --out "${FIG_DIR}/pareto_diagnostics"

echo "Bias-in-Bios 500k_v2 table:"
echo "  ${TABLE_CSV}"
echo "Bias-in-Bios 500k_v2 figures:"
ls -lh "${FIG_DIR}"
