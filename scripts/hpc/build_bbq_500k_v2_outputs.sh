#!/usr/bin/env bash
set -euo pipefail

# Build LARGE-held-out table and figures for the active Rocket 500k v2 seed-0 run.
# Run this on a login node after the three large eval jobs, post-hoc scoring,
# and re-evaluated post-hoc large eval job have completed. This script does not
# start vLLM.

cd "$(dirname "$0")/../.."

TABLE_CONFIG="${TABLE_CONFIG:-configs/HPC_Config/experiment_table_bbq_HPC.yaml}"
AGG_CONFIG="${AGG_CONFIG:-configs/HPC_Config/aggregate_multiseed_bbq_HPC.yaml}"
TABLE_CSV="${TABLE_CSV:-outputs/experiment_table/bbq_mistral_hpc_500k_v2_large_seed0/experiment_table.csv}"
FIG_DIR="${FIG_DIR:-outputs/figures/paper_bbq_hpc_500k_v2_large_seed0}"
TITLE="${TITLE:-BBQ / Mistral-Small-3.2 / Rocket 500k v2 seed 0 (large held-out)}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"
RUN_RICHNESS="${RUN_RICHNESS:-1}"
RUN_TRAJECTORY="${RUN_TRAJECTORY:-1}"
RUN_PARETO_DIAGNOSTICS="${RUN_PARETO_DIAGNOSTICS:-1}"

echo "Checking 500k v2 LARGE held-out eval outputs..."
test -f outputs/hpc/evaluation_large/seed_0/bbq_faircapo_500k_v2/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/bbq_ablation_500k_v2/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/bbq_nsga2po_500k_v2/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/bbq_posthoc_500k_v2/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/bbq_posthoc_500k_v2/test_eval_candidates.csv

echo "Building 500k v2 LARGE held-out experiment table..."
PYTHONPATH=. python scripts/build_experiment_table.py --config "${TABLE_CONFIG}"

if [[ "${RUN_AGGREGATE}" == "1" ]]; then
  echo "Building 500k v2 LARGE held-out aggregate summary..."
  PYTHONPATH=. python scripts/aggregate_multiseed.py --config "${AGG_CONFIG}"
fi

mkdir -p "${FIG_DIR}"

echo "Building 500k v2 LARGE held-out figures..."
python scripts/visualize_paper_figures.py \
  --run outputs/hpc/bbq_faircapo_500k_v2/seed_0 \
  --run-csv outputs/hpc/evaluation_large/seed_0/bbq_faircapo_500k_v2/test_eval_candidates.csv \
  --table "${TABLE_CSV}" \
  --title "${TITLE}" \
  --out "${FIG_DIR}"

echo "Building 500k v2 LARGE held-out staircase..."
python scripts/visualize_staircase.py \
  --fair outputs/hpc/evaluation_large/seed_0/bbq_faircapo_500k_v2/test_eval_candidates.csv \
  --portfolio outputs/hpc/bbq_faircapo_500k_v2/seed_0/phase2_prompt_portfolio.csv \
  --mocapo "" \
  --title "${TITLE}" \
  --out "${FIG_DIR}/fig_pareto_staircase.png" \
  --color-fairness

if [[ "${RUN_RICHNESS}" == "1" ]]; then
  echo "Building 500k v2 search-basis front richness figure..."
  python scripts/visualize_front_richness.py \
    --faircapo outputs/hpc/bbq_faircapo_500k_v2/seed_0/phase2_all_candidates.csv \
    --nsga outputs/hpc/bbq_nsga2po_500k_v2/seed_0/nsga2_po_all_candidates.csv \
    --ablation outputs/hpc/bbq_ablation_500k_v2/seed_0/phase2_all_candidates.csv \
    --title "BBQ / Mistral-Small-3.2 / Rocket 500k v2 seed 0 (search basis; not large held-out)" \
    --out "${FIG_DIR}/fig_front_richness_search_basis.png"
fi

if [[ "${RUN_TRAJECTORY}" == "1" ]]; then
  FAIR_TRAJ="outputs/hpc/bbq_faircapo_500k_v2/seed_0/budgeted_mocapo_trajectory.json"
  ABL_TRAJ="outputs/hpc/bbq_ablation_500k_v2/seed_0/budgeted_mocapo_trajectory.json"
  if [[ -f "${FAIR_TRAJ}" && -f "${ABL_TRAJ}" ]]; then
    echo "Building 500k v2 search-basis trajectory figure..."
    PYTHONPATH=. python scripts/visualize_trajectory.py \
      --trajectory "${FAIR_TRAJ}" \
      --label FairCAPO \
      --trajectory "${ABL_TRAJ}" \
      --label "MO-CAPO (fairness off)" \
      --title "BBQ / Mistral-Small-3.2 / Rocket 500k v2 seed 0 (search trajectory)" \
      --out "${FIG_DIR}/fig_trajectory_search_basis.png"
  else
    echo "Skipping trajectory figure; missing ${FAIR_TRAJ} or ${ABL_TRAJ}"
  fi
fi

if [[ "${RUN_PARETO_DIAGNOSTICS}" == "1" ]]; then
  echo "Building 500k v2 LARGE held-out Pareto diagnostic figures..."
  python scripts/visualize_pareto_front.py \
    --run outputs/hpc/bbq_faircapo_500k_v2/seed_0 \
    --csv outputs/hpc/evaluation_large/seed_0/bbq_faircapo_500k_v2/test_eval_candidates.csv \
    --title "${TITLE}" \
    --out "${FIG_DIR}/pareto_diagnostics"
fi

echo "500k v2 table:"
echo "  ${TABLE_CSV}"
echo "500k v2 figures:"
ls -lh "${FIG_DIR}"
