#!/usr/bin/env bash
# S14 recovery: the FairCAPO held-out Dtest (step 3 of run_bbq_seed0_pipeline.sh)
# HUNG at startup and was killed, so its output is stale. The ablation/NSGA Dtests
# completed. This re-runs ONLY the FairCAPO Dtest, then rebuilds the cross-method
# table + figures so they pick up the fresh FairCAPO held-out row.
#
# Run this AFTER the main pipeline (run_bbq_seed0_pipeline.sh) has finished.
set -uo pipefail
export PATH="/usr/bin:/bin:$PATH"
cd ~/PycharmProjects/Tri_CAPO
source .venv/Scripts/activate
export PYTHONPATH=.

echo "===== RECOVER: held-out Dtest — FairCAPO (re-run) ====="
python -u scripts/evaluate_pareto_on_test.py \
  --config configs/evaluate_pareto_bbq.yaml --seed 0
echo "faircapo Dtest rc=$?"

echo "===== rebuild experiment table ====="
python scripts/build_experiment_table.py \
  --config configs/experiment_table_bbq.yaml
echo "table rc=$?"

echo "===== rebuild paper figures ====="
python scripts/visualize_paper_figures.py \
  --run outputs/seed_0/phase2_budgeted_mocapo_bbq_local \
  --out outputs/figures/paper_bbq_local
echo "figures rc=$?"

echo "===== RECOVERY COMPLETE ====="
