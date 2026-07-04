#!/usr/bin/env bash
set -euo pipefail

# Build only scientifically valid Adult v3 seed-0 outputs. The old envelope
# table mixed independently best accuracy/cost/fairness candidates and computed
# method-local self-reference nR2, so it is intentionally not used here.

cd "$(dirname "$0")/../.."

TABLE_CONFIG="configs/HPC_Config/adult_experiment_table_7p5m_v3_large_HPC.yaml"
OUT_DIR="outputs/experiment_table/adult_mistral_hpc_7p5m_v3_large_seed0"
TABLE_CSV="${OUT_DIR}/representative_experiment_table.csv"
FIG_DIR="outputs/figures/paper_adult_hpc_7p5m_v3_large_seed0"
TITLE="Adult / Mistral-Small-3.2 / Rocket 7.5M v3 seed 0"

echo "Validating completed Adult v3 searches and held-out evaluations..."
PYTHONPATH=. python scripts/validate_adult_v3_outputs.py

echo "Building one-real-candidate-per-method table with shared-reference nR2..."
PYTHONPATH=. python scripts/build_representative_experiment_table.py \
  --config "${TABLE_CONFIG}"

test -s "${TABLE_CSV}"
mkdir -p "${FIG_DIR}"

echo "Building FairCAPO held-out Pareto figures..."
python scripts/visualize_paper_figures.py \
  --run outputs/hpc/adult_faircapo_7p5m_v3/seed_0 \
  --run-csv outputs/hpc/evaluation_large/seed_0/adult_faircapo_7p5m_v3/test_eval_candidates.csv \
  --table "${TABLE_CSV}" \
  --title "${TITLE} (large held-out)" \
  --out "${FIG_DIR}"

python scripts/visualize_staircase.py \
  --fair outputs/hpc/evaluation_large/seed_0/adult_faircapo_7p5m_v3/test_eval_candidates.csv \
  --portfolio outputs/hpc/adult_faircapo_7p5m_v3/seed_0/phase2_prompt_portfolio.csv \
  --mocapo "" \
  --title "${TITLE} (large held-out)" \
  --out "${FIG_DIR}/fig_pareto_staircase.png" \
  --color-fairness

# Cross-method front richness must use a common evaluation basis. Search CSV
# costs are per-example for blockwise MO-CAPO but full-dev totals for NSGA, so
# the earlier search-basis comparison was not valid. All held-out CSVs below use
# the same 2,000 test records and therefore have comparable objective scales.
python scripts/visualize_front_richness.py \
  --faircapo outputs/hpc/evaluation_large/seed_0/adult_faircapo_7p5m_v3/test_eval_candidates.csv \
  --nsga outputs/hpc/evaluation_large/seed_0/adult_nsga2po_7p5m_v3/test_eval_candidates.csv \
  --ablation outputs/hpc/evaluation_large/seed_0/adult_ablation_7p5m_v3/test_eval_candidates.csv \
  --title "${TITLE} (shared held-out basis)" \
  --out "${FIG_DIR}/fig_front_richness_heldout.png"

# FairCAPO and MO-CAPO both use BlockEvaluator's per-example cost and identical
# fixed bounds, so their search trajectories are directly comparable.
FAIR_TRAJ="outputs/hpc/adult_faircapo_7p5m_v3/seed_0/budgeted_mocapo_trajectory.json"
MO_TRAJ="outputs/hpc/adult_ablation_7p5m_v3/seed_0/budgeted_mocapo_trajectory.json"
if [[ -s "${FAIR_TRAJ}" && -s "${MO_TRAJ}" ]]; then
  PYTHONPATH=. python scripts/visualize_trajectory.py \
    --trajectory "${FAIR_TRAJ}" --label FairCAPO \
    --trajectory "${MO_TRAJ}" --label "MO-CAPO (fairness objective off)" \
    --title "${TITLE} (search trajectory)" \
    --out "${FIG_DIR}/fig_trajectory_search_basis.png"
fi

python scripts/visualize_pareto_front.py \
  --run outputs/hpc/adult_faircapo_7p5m_v3/seed_0 \
  --csv outputs/hpc/evaluation_large/seed_0/adult_faircapo_7p5m_v3/test_eval_candidates.csv \
  --title "${TITLE} (large held-out)" \
  --out "${FIG_DIR}/pareto_diagnostics"

echo "Representative table (use this for claims): ${TABLE_CSV}"
echo "Figures: ${FIG_DIR}"
find "${FIG_DIR}" -maxdepth 2 -type f -printf '%p\n' | sort
