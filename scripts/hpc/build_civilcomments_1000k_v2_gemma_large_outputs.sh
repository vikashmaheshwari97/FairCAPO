#!/usr/bin/env bash
set -euo pipefail

# Build CivilComments-WILDS 1000k_v2 seed-0 (Gemma-2-27B) large-held-out table and
# figures. Run on the login node after the search/eval jobs complete. No GPU.

cd "$(dirname "$0")/../.."

TABLE_CONFIG="${TABLE_CONFIG:-configs/HPC_Config/civilcomments_experiment_table_1000k_v2_gemma_large_HPC.yaml}"
AGG_CONFIG="${AGG_CONFIG:-configs/HPC_Config/civilcomments_aggregate_1000k_v2_gemma_large_HPC.yaml}"
TABLE_CSV="${TABLE_CSV:-outputs/experiment_table/civilcomments_gemma_hpc_1000k_v2_large_seed0/experiment_table.csv}"
FIG_DIR="${FIG_DIR:-outputs/figures/paper_civilcomments_gemma_hpc_1000k_v2_large_seed0}"
TITLE="${TITLE:-CivilComments-WILDS / Gemma-2-27B / Rocket 1000k v2 seed 0 (direct-label, large held-out)}"

echo "Checking CivilComments 1000k_v2 large-held-out eval outputs..."
test -f outputs/hpc/evaluation_large/seed_0/civilcomments_faircapo_1000k_v2_gemma/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/civilcomments_ablation_1000k_v2_gemma/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/civilcomments_nsga2po_1000k_v2_gemma/test_eval_summary.json

echo "Building CivilComments 1000k_v2 experiment table..."
PYTHONPATH=. python scripts/build_experiment_table.py --config "${TABLE_CONFIG}"

echo "Building CivilComments 1000k_v2 aggregate summary..."
PYTHONPATH=. python scripts/aggregate_multiseed.py --config "${AGG_CONFIG}"

mkdir -p "${FIG_DIR}"

echo "Building CivilComments 1000k_v2 paper figures..."
python scripts/visualize_paper_figures.py \
  --run outputs/hpc/civilcomments_faircapo_1000k_v2_gemma/seed_0 \
  --run-csv outputs/hpc/evaluation_large/seed_0/civilcomments_faircapo_1000k_v2_gemma/test_eval_candidates.csv \
  --table "${TABLE_CSV}" \
  --title "${TITLE}" \
  --out "${FIG_DIR}"

echo "Building CivilComments 1000k_v2 staircase..."
python scripts/visualize_staircase.py \
  --fair outputs/hpc/evaluation_large/seed_0/civilcomments_faircapo_1000k_v2_gemma/test_eval_candidates.csv \
  --portfolio outputs/hpc/civilcomments_faircapo_1000k_v2_gemma/seed_0/phase2_prompt_portfolio.csv \
  --mocapo "" \
  --title "${TITLE}" \
  --out "${FIG_DIR}/fig_pareto_staircase.png" \
  --color-fairness

echo "Building CivilComments 1000k_v2 search-basis front richness figure..."
python scripts/visualize_front_richness.py \
  --faircapo outputs/hpc/civilcomments_faircapo_1000k_v2_gemma/seed_0/phase2_all_candidates.csv \
  --nsga outputs/hpc/civilcomments_nsga2po_1000k_v2_gemma/seed_0/nsga2_po_all_candidates.csv \
  --ablation outputs/hpc/civilcomments_ablation_1000k_v2_gemma/seed_0/phase2_all_candidates.csv \
  --title "CivilComments-WILDS / Gemma-2-27B / Rocket 1000k v2 seed 0 (search basis)" \
  --out "${FIG_DIR}/fig_front_richness_search_basis.png"

FAIR_TRAJ="outputs/hpc/civilcomments_faircapo_1000k_v2_gemma/seed_0/budgeted_mocapo_trajectory.json"
ABL_TRAJ="outputs/hpc/civilcomments_ablation_1000k_v2_gemma/seed_0/budgeted_mocapo_trajectory.json"
if [[ -f "${FAIR_TRAJ}" && -f "${ABL_TRAJ}" ]]; then
  echo "Building CivilComments 1000k_v2 search-basis trajectory figure..."
  PYTHONPATH=. python scripts/visualize_trajectory.py \
    --trajectory "${FAIR_TRAJ}" \
    --label FairCAPO \
    --trajectory "${ABL_TRAJ}" \
    --label "MO-CAPO (fairness off)" \
    --title "CivilComments-WILDS / Gemma-2-27B / Rocket 1000k v2 seed 0 (search trajectory)" \
    --out "${FIG_DIR}/fig_trajectory_search_basis.png"
fi

echo "Building CivilComments 1000k_v2 Pareto diagnostic figures..."
python scripts/visualize_pareto_front.py \
  --run outputs/hpc/civilcomments_faircapo_1000k_v2_gemma/seed_0 \
  --csv outputs/hpc/evaluation_large/seed_0/civilcomments_faircapo_1000k_v2_gemma/test_eval_candidates.csv \
  --title "${TITLE}" \
  --out "${FIG_DIR}/pareto_diagnostics"

echo "CivilComments 1000k_v2 table:"
echo "  ${TABLE_CSV}"
echo "CivilComments 1000k_v2 figures:"
ls -lh "${FIG_DIR}"
