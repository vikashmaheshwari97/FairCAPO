#!/usr/bin/env bash
set -euo pipefail

# Build LARGE-held-out table and figures for the FairCAPO 500k v4 decision run.
# This compares FairCAPO v4 against the already-completed 500k v2 baselines.
# Run on a login node after the FairCAPO v4 search and large eval complete.

cd "$(dirname "$0")/../.."

TABLE_CONFIG="${TABLE_CONFIG:-configs/HPC_Config/experiment_table_bbq_HPC.yaml}"
AGG_CONFIG="${AGG_CONFIG:-configs/HPC_Config/aggregate_multiseed_bbq_HPC.yaml}"
TABLE_CSV="${TABLE_CSV:-outputs/experiment_table/bbq_mistral_hpc_500k_v4_vs_v2_large_seed0/experiment_table.csv}"
FIG_DIR="${FIG_DIR:-outputs/figures/paper_bbq_hpc_500k_v4_vs_v2_large_seed0}"
TITLE="${TITLE:-BBQ / Mistral-Small-3.2 / Rocket FairCAPO 500k v4 vs v2 baselines (large held-out)}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"
RUN_RICHNESS="${RUN_RICHNESS:-1}"
RUN_TRAJECTORY="${RUN_TRAJECTORY:-1}"
RUN_PARETO_DIAGNOSTICS="${RUN_PARETO_DIAGNOSTICS:-1}"

echo "Checking FairCAPO 500k v4 LARGE held-out eval and existing v2 baselines..."
test -f outputs/hpc/evaluation_large/seed_0/bbq_faircapo_500k_v4/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/bbq_faircapo_500k_v4/test_eval_candidates.csv
test -f outputs/hpc/evaluation_large/seed_0/bbq_ablation_500k_v2/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/bbq_nsga2po_500k_v2/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/bbq_posthoc_500k_v2/test_eval_summary.json

echo "Building FairCAPO 500k v4 vs v2 baseline experiment table..."
PYTHONPATH=. python scripts/build_experiment_table.py --config "${TABLE_CONFIG}"

if [[ "${RUN_AGGREGATE}" == "1" ]]; then
  echo "Building FairCAPO 500k v4 vs v2 baseline aggregate summary..."
  PYTHONPATH=. python scripts/aggregate_multiseed.py --config "${AGG_CONFIG}"
fi

mkdir -p "${FIG_DIR}"

echo "Building FairCAPO 500k v4 vs v2 baseline figures..."
python scripts/visualize_paper_figures.py \
  --run outputs/hpc/bbq_faircapo_500k_v4/seed_0 \
  --run-csv outputs/hpc/evaluation_large/seed_0/bbq_faircapo_500k_v4/test_eval_candidates.csv \
  --table "${TABLE_CSV}" \
  --title "${TITLE}" \
  --out "${FIG_DIR}"

echo "Building FairCAPO 500k v4 LARGE held-out staircase..."
python scripts/visualize_staircase.py \
  --fair outputs/hpc/evaluation_large/seed_0/bbq_faircapo_500k_v4/test_eval_candidates.csv \
  --portfolio outputs/hpc/bbq_faircapo_500k_v4/seed_0/phase2_prompt_portfolio.csv \
  --mocapo "" \
  --title "${TITLE}" \
  --out "${FIG_DIR}/fig_pareto_staircase.png" \
  --color-fairness

if [[ "${RUN_RICHNESS}" == "1" ]]; then
  echo "Building FairCAPO v4 vs v2 baseline search-basis front richness figure..."
  python scripts/visualize_front_richness.py \
    --faircapo outputs/hpc/bbq_faircapo_500k_v4/seed_0/phase2_all_candidates.csv \
    --nsga outputs/hpc/bbq_nsga2po_500k_v2/seed_0/nsga2_po_all_candidates.csv \
    --ablation outputs/hpc/bbq_ablation_500k_v2/seed_0/phase2_all_candidates.csv \
    --title "BBQ / Mistral-Small-3.2 / FairCAPO 500k v4 vs v2 baselines (search basis)" \
    --out "${FIG_DIR}/fig_front_richness_search_basis.png"
fi

if [[ "${RUN_TRAJECTORY}" == "1" ]]; then
  FAIR_TRAJ="outputs/hpc/bbq_faircapo_500k_v4/seed_0/budgeted_mocapo_trajectory.json"
  ABL_TRAJ="outputs/hpc/bbq_ablation_500k_v2/seed_0/budgeted_mocapo_trajectory.json"
  if [[ -f "${FAIR_TRAJ}" && -f "${ABL_TRAJ}" ]]; then
    echo "Building FairCAPO v4 vs ablation v2 search-basis trajectory figure..."
    PYTHONPATH=. python scripts/visualize_trajectory.py \
      --trajectory "${FAIR_TRAJ}" \
      --label "FairCAPO 500k v4" \
      --trajectory "${ABL_TRAJ}" \
      --label "MO-CAPO fairness off 500k v2" \
      --title "BBQ / Mistral-Small-3.2 / FairCAPO 500k v4 vs ablation v2 trajectory" \
      --out "${FIG_DIR}/fig_trajectory_search_basis.png"
  else
    echo "Skipping trajectory figure; missing ${FAIR_TRAJ} or ${ABL_TRAJ}"
  fi
fi

if [[ "${RUN_PARETO_DIAGNOSTICS}" == "1" ]]; then
  echo "Building FairCAPO 500k v4 LARGE held-out Pareto diagnostic figures..."
  python scripts/visualize_pareto_front.py \
    --run outputs/hpc/bbq_faircapo_500k_v4/seed_0 \
    --csv outputs/hpc/evaluation_large/seed_0/bbq_faircapo_500k_v4/test_eval_candidates.csv \
    --title "${TITLE}" \
    --out "${FIG_DIR}/pareto_diagnostics"
fi

echo "FairCAPO 500k v4 table:"
echo "  ${TABLE_CSV}"
echo "FairCAPO 500k v4 figures:"
ls -lh "${FIG_DIR}"
