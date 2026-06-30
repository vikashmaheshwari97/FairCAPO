#!/usr/bin/env bash
set -euo pipefail

# Build the table/figures for the Bias-in-Bios label-scoring FairCAPO seed-0
# diagnostic against the frozen Mistral v1 baselines.

cd "$(dirname "$0")/../.."

TABLE_CONFIG="${TABLE_CONFIG:-configs/HPC_Config/bios_experiment_table_labelscore_500k_v1_large_HPC.yaml}"
AGG_CONFIG="${AGG_CONFIG:-configs/HPC_Config/bios_aggregate_labelscore_500k_v1_large_HPC.yaml}"
TABLE_CSV="${TABLE_CSV:-outputs/experiment_table/bios_mistral_labelscore_500k_v1_large_seed0/experiment_table.csv}"
FIG_DIR="${FIG_DIR:-outputs/figures/paper_bios_labelscore_500k_v1_large_seed0}"
TITLE="${TITLE:-Bias-in-Bios / Mistral-Small-3.2 / Label scoring / Rocket 500k v1 seed 0}"

echo "Checking Bias-in-Bios label-score and frozen baseline eval outputs..."
test -f outputs/hpc/evaluation_large/seed_0/bios_faircapo_labelscore_500k_v1/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/bios_faircapo_500k_v1/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/bios_ablation_500k_v1/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/bios_nsga2po_500k_v1/test_eval_summary.json

echo "Building Bias-in-Bios label-score comparison table..."
PYTHONPATH=. python scripts/build_experiment_table.py --config "${TABLE_CONFIG}"

echo "Building Bias-in-Bios label-score aggregate summary..."
PYTHONPATH=. python scripts/aggregate_multiseed.py --config "${AGG_CONFIG}"

mkdir -p "${FIG_DIR}"

echo "Building Bias-in-Bios label-score paper figures..."
python scripts/visualize_paper_figures.py \
  --run outputs/hpc/bios_faircapo_labelscore_500k_v1/seed_0 \
  --run-csv outputs/hpc/evaluation_large/seed_0/bios_faircapo_labelscore_500k_v1/test_eval_candidates.csv \
  --table "${TABLE_CSV}" \
  --title "${TITLE}" \
  --out "${FIG_DIR}"

echo "Building Bias-in-Bios label-score staircase..."
python scripts/visualize_staircase.py \
  --fair outputs/hpc/evaluation_large/seed_0/bios_faircapo_labelscore_500k_v1/test_eval_candidates.csv \
  --portfolio outputs/hpc/bios_faircapo_labelscore_500k_v1/seed_0/phase2_prompt_portfolio.csv \
  --mocapo "" \
  --title "${TITLE}" \
  --out "${FIG_DIR}/fig_pareto_staircase.png" \
  --color-fairness

echo "Building Bias-in-Bios label-score search-basis front richness figure..."
python scripts/visualize_front_richness.py \
  --faircapo outputs/hpc/bios_faircapo_labelscore_500k_v1/seed_0/phase2_all_candidates.csv \
  --nsga outputs/hpc/bios_nsga2po_500k_v1/seed_0/nsga2_po_all_candidates.csv \
  --ablation outputs/hpc/bios_ablation_500k_v1/seed_0/phase2_all_candidates.csv \
  --title "Bias-in-Bios / Mistral-Small-3.2 / Label scoring vs frozen baselines" \
  --out "${FIG_DIR}/fig_front_richness_search_basis.png"

FAIR_TRAJ="outputs/hpc/bios_faircapo_labelscore_500k_v1/seed_0/budgeted_mocapo_trajectory.json"
if [[ -f "${FAIR_TRAJ}" ]]; then
  echo "Building Bias-in-Bios label-score search trajectory figure..."
  PYTHONPATH=. python scripts/visualize_trajectory.py \
    --trajectory "${FAIR_TRAJ}" \
    --label "FairCAPO label-score" \
    --title "Bias-in-Bios / Mistral-Small-3.2 / Label scoring search trajectory" \
    --out "${FIG_DIR}/fig_trajectory_search_basis.png"
fi

echo "Building Bias-in-Bios label-score Pareto diagnostic figures..."
python scripts/visualize_pareto_front.py \
  --run outputs/hpc/bios_faircapo_labelscore_500k_v1/seed_0 \
  --csv outputs/hpc/evaluation_large/seed_0/bios_faircapo_labelscore_500k_v1/test_eval_candidates.csv \
  --title "${TITLE}" \
  --out "${FIG_DIR}/pareto_diagnostics"

echo "Bias-in-Bios label-score table:"
echo "  ${TABLE_CSV}"
echo "Bias-in-Bios label-score figures:"
ls -lh "${FIG_DIR}"
