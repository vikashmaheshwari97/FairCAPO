#!/usr/bin/env bash
set -euo pipefail

# Active Adult v3 1M pilot output builder.
cd "$(dirname "$0")/../.."

TABLE_CONFIG="configs/HPC_Config/adult_experiment_table_7p5m_v3_large_HPC.yaml"
OUT_DIR="outputs/experiment_table/adult_mistral_hpc_1m_v3_seed0"
TABLE_CSV="${OUT_DIR}/representative_experiment_table.csv"
FIG_DIR="outputs/figures/paper_adult_hpc_1m_v3_seed0"
TITLE="Adult / Mistral-Small-3.2 / Rocket 1M v3 seed 0"

PYTHONPATH=. python scripts/validate_adult_v3_outputs.py
PYTHONPATH=. python scripts/build_representative_experiment_table.py \
  --config "${TABLE_CONFIG}"

test -s "${TABLE_CSV}"
mkdir -p "${FIG_DIR}"

python scripts/visualize_paper_figures.py \
  --run outputs/hpc/adult_faircapo_1m_v3/seed_0 \
  --run-csv outputs/hpc/evaluation_1m/seed_0/adult_faircapo_1m_v3/test_eval_candidates.csv \
  --table "${TABLE_CSV}" \
  --title "${TITLE} (1,000-record held-out)" \
  --out "${FIG_DIR}"

python scripts/visualize_staircase.py \
  --fair outputs/hpc/evaluation_1m/seed_0/adult_faircapo_1m_v3/test_eval_candidates.csv \
  --portfolio outputs/hpc/adult_faircapo_1m_v3/seed_0/phase2_prompt_portfolio.csv \
  --mocapo "" \
  --title "${TITLE} (1,000-record held-out)" \
  --out "${FIG_DIR}/fig_pareto_staircase.png"

python scripts/visualize_adult_v3_fronts.py \
  --faircapo outputs/hpc/evaluation_1m/seed_0/adult_faircapo_1m_v3/test_eval_candidates.csv \
  --mocapo outputs/hpc/evaluation_1m/seed_0/adult_ablation_1m_v3/test_eval_candidates.csv \
  --nsga outputs/hpc/evaluation_1m/seed_0/adult_nsga2po_1m_v3/test_eval_candidates.csv \
  --title "${TITLE} (shared held-out basis)" \
  --out "${FIG_DIR}/fig_front_richness_heldout.png"

FAIR_TRAJ="outputs/hpc/adult_faircapo_1m_v3/seed_0/budgeted_mocapo_trajectory.json"
MO_TRAJ="outputs/hpc/adult_ablation_1m_v3/seed_0/budgeted_mocapo_trajectory.json"
if [[ -s "${FAIR_TRAJ}" && -s "${MO_TRAJ}" ]]; then
  PYTHONPATH=. python scripts/visualize_trajectory.py \
    --trajectory "${FAIR_TRAJ}" --label FairCAPO \
    --trajectory "${MO_TRAJ}" --label "MO-CAPO (fairness objective off)" \
    --title "${TITLE} (search trajectory)" \
    --out "${FIG_DIR}/fig_trajectory_search_basis.png"
fi

python scripts/visualize_pareto_front.py \
  --run outputs/hpc/adult_faircapo_1m_v3/seed_0 \
  --csv outputs/hpc/evaluation_1m/seed_0/adult_faircapo_1m_v3/test_eval_candidates.csv \
  --title "${TITLE} (1,000-record held-out)" \
  --out "${FIG_DIR}/pareto_diagnostics"

echo "Representative table: ${TABLE_CSV}"
echo "Figures: ${FIG_DIR}"
find "${FIG_DIR}" -maxdepth 2 -type f -printf '%p\n' | sort
