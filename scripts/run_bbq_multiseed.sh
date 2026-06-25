#!/usr/bin/env bash
# S12 paper-parity: 3-seed BBQ sweep (FairCAPO + ablation + NSGA-II-PO), each with a
# held-out Dtest, so the experiment table can report mean +/- std (Buessing et al. run
# every config with 3 seeds). The runner --seed overrides config['seed'], which seeds
# BOTH the data split and the evolutionary/NSGA RNG, and the eval --seed seeds the
# held-out test split to match.
#
# Usage:  bash scripts/run_bbq_multiseed.sh [SEEDS...]
#   default seeds: 0 1 2
#
# Requirements: LM Studio @ :1234 with mistralai/mistral-small-3.2 loaded.
# Keep the laptop awake:  MSYS_NO_PATHCONV=1 powercfg /change standby-timeout-ac 0
#
# Output layout (per seed S):
#   outputs/seed_S/phase2_budgeted_mocapo_bbq_local/   (FairCAPO)
#   outputs/seed_S/mocapo_baseline_bbq_local/          (ablation)
#   outputs/seed_S/baselines/nsga2_po_bbq_local/       (NSGA)
#   outputs/seed_S/evaluation/bbq_{mistral,ablation,nsga}_local/  (3 Dtests)
#
# NOTE: this does NOT build the table -- aggregation across seeds is a separate step
# (scripts/aggregate_multiseed.py, TODO) so we can compute mean +/- std per method.
set -euo pipefail

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then SEEDS=(0 1 2); fi

cd "$(dirname "$0")/.."
export PYTHONPATH=.

run() { echo; echo ">>> $*"; "$@"; }

for S in "${SEEDS[@]}"; do
  BASE="outputs/seed_${S}"
  FAIRCAPO_DIR="${BASE}/phase2_budgeted_mocapo_bbq_local"
  ABLATION_DIR="${BASE}/mocapo_baseline_bbq_local"
  NSGA_DIR="${BASE}/baselines/nsga2_po_bbq_local"

  echo "==================== SEED ${S} ===================="

  # --- search runs (meta-LLM ON, sDIS fold, tuned, 500k iso-budget) ---
  run python -u scripts/run_phase2_budgeted_mocapo.py \
    --config configs/phase2_budgeted_mocapo_bbq_local.yaml \
    --seed "${S}" --output-dir "${FAIRCAPO_DIR}" \
    2>&1 | tee "outputs/_bbq_faircapo_seed${S}.log"

  run python -u scripts/run_phase2_budgeted_mocapo.py \
    --config configs/mocapo_baseline_bbq_local.yaml \
    --seed "${S}" --output-dir "${ABLATION_DIR}" \
    2>&1 | tee "outputs/_bbq_ablation_seed${S}.log"

  run python -u scripts/run_baseline_nsga2_po.py \
    --config configs/nsga2_po_bbq.yaml \
    --seed "${S}" --output-dir "${NSGA_DIR}" \
    2>&1 | tee "outputs/_bbq_nsga_seed${S}.log"

  # --- held-out Dtest for each method (same seed for the test split) ---
  run python scripts/evaluate_pareto_on_test.py \
    --config configs/evaluate_pareto_bbq.yaml --seed "${S}" \
    --portfolio-csv "${FAIRCAPO_DIR}/phase2_prompt_portfolio.csv" \
    --output-dir "${BASE}/evaluation/bbq_mistral_local"

  run python scripts/evaluate_pareto_on_test.py \
    --config configs/evaluate_pareto_bbq_ablation.yaml --seed "${S}" \
    --portfolio-csv "${ABLATION_DIR}/phase2_prompt_portfolio.csv" \
    --output-dir "${BASE}/evaluation/bbq_ablation_local"

  run python scripts/evaluate_pareto_on_test.py \
    --config configs/evaluate_pareto_bbq_nsga.yaml --seed "${S}" \
    --portfolio-csv "${NSGA_DIR}/nsga2_po_pareto_portfolio.csv" \
    --output-dir "${BASE}/evaluation/bbq_nsga_local"
done

echo
echo "All seeds done: ${SEEDS[*]}"
echo "Next: python scripts/aggregate_multiseed.py (TODO) -> mean +/- std table."
