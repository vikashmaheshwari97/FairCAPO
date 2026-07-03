#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

TABLE_CONFIG="configs/HPC_Config/adult_experiment_table_7p5m_v3_large_HPC.yaml"
AGG_CONFIG="configs/HPC_Config/adult_aggregate_7p5m_v3_large_HPC.yaml"
OUT_DIR="outputs/experiment_table/adult_mistral_hpc_7p5m_v3_large_seed0"
TABLE_CSV="${OUT_DIR}/representative_experiment_table.csv"
FIG_DIR="outputs/figures/paper_adult_hpc_7p5m_v3_large_seed0"
TITLE="Adult / Mistral-Small-3.2 / Rocket 7.5M v3 seed 0"

echo "Checking Adult v3 held-out outputs..."
test -f outputs/hpc/evaluation_large/seed_0/adult_faircapo_7p5m_v3/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/adult_ablation_7p5m_v3/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/adult_nsga2po_7p5m_v3/test_eval_summary.json

echo "Building objective-envelope and representative tables..."
PYTHONPATH=. python scripts/build_experiment_table.py --config "${TABLE_CONFIG}"
PYTHONPATH=. python scripts/build_representative_experiment_table.py --config "${TABLE_CONFIG}"
PYTHONPATH=. python scripts/aggregate_multiseed.py --config "${AGG_CONFIG}"

mkdir -p "${FIG_DIR}"

echo "Building paper figures..."
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

python scripts/visualize_front_richness.py \
  --faircapo outputs/hpc/adult_faircapo_7p5m_v3/seed_0/phase2_all_candidates.csv \
  --nsga outputs/hpc/adult_nsga2po_7p5m_v3/seed_0/nsga2_po_all_candidates.csv \
  --ablation outputs/hpc/adult_ablation_7p5m_v3/seed_0/phase2_all_candidates.csv \
  --title "${TITLE} (search basis)" \
  --out "${FIG_DIR}/fig_front_richness_search_basis.png"

FAIR_TRAJ="outputs/hpc/adult_faircapo_7p5m_v3/seed_0/budgeted_mocapo_trajectory.json"
MO_TRAJ="outputs/hpc/adult_ablation_7p5m_v3/seed_0/budgeted_mocapo_trajectory.json"
if [[ -f "${FAIR_TRAJ}" && -f "${MO_TRAJ}" ]]; then
  PYTHONPATH=. python scripts/visualize_trajectory.py \
    --trajectory "${FAIR_TRAJ}" --label FairCAPO \
    --trajectory "${MO_TRAJ}" --label "MO-CAPO (fairness off)" \
    --title "${TITLE} (search trajectory)" \
    --out "${FIG_DIR}/fig_trajectory_search_basis.png"
fi

python scripts/visualize_pareto_front.py \
  --run outputs/hpc/adult_faircapo_7p5m_v3/seed_0 \
  --csv outputs/hpc/evaluation_large/seed_0/adult_faircapo_7p5m_v3/test_eval_candidates.csv \
  --title "${TITLE} (large held-out)" \
  --out "${FIG_DIR}/pareto_diagnostics"

echo "Objective-envelope table: ${OUT_DIR}/experiment_table.csv"
echo "Representative table: ${TABLE_CSV}"
echo "Figures: ${FIG_DIR}"
ls -lh "${FIG_DIR}"
