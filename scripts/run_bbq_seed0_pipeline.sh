#!/usr/bin/env bash
# S14: finish the seed-0 BBQ comparison after the FairCAPO search completed.
# Runs ablation + NSGA searches, the 3 held-out Dtests, then table + figures.
# FairCAPO outputs were already synced into the default dir. Unbuffered; the
# python runners are quiet mid-loop (results write at the end) — a quiet log = working.
set -uo pipefail
cd ~/PycharmProjects/Tri_CAPO
source .venv/Scripts/activate
export PYTHONPATH=.
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] ===== STEP 1/6: MO-CAPO fairness-OFF ablation (seed 0) ====="
python -u scripts/run_phase2_budgeted_mocapo.py \
  --config configs/mocapo_baseline_bbq_local.yaml --seed 0
echo "[$(ts)] step1 rc=$?"

echo "[$(ts)] ===== STEP 2/6: NSGA-II-PO + fairness (seed 0, unbudgeted, pop16/off4) ====="
python -u scripts/run_baseline_nsga2_po.py \
  --config configs/nsga2_po_bbq.yaml --seed 0
echo "[$(ts)] step2 rc=$?"

echo "[$(ts)] ===== STEP 3/6: held-out Dtest — FairCAPO ====="
python -u scripts/evaluate_pareto_on_test.py \
  --config configs/evaluate_pareto_bbq.yaml --seed 0
echo "[$(ts)] step3 rc=$?"

echo "[$(ts)] ===== STEP 4/6: held-out Dtest — ablation ====="
python -u scripts/evaluate_pareto_on_test.py \
  --config configs/evaluate_pareto_bbq_ablation.yaml --seed 0
echo "[$(ts)] step4 rc=$?"

echo "[$(ts)] ===== STEP 5/6: held-out Dtest — NSGA ====="
python -u scripts/evaluate_pareto_on_test.py \
  --config configs/evaluate_pareto_bbq_nsga.yaml --seed 0
echo "[$(ts)] step5 rc=$?"

echo "[$(ts)] ===== STEP 6/6: experiment table + paper figures ====="
python scripts/build_experiment_table.py \
  --config configs/experiment_table_bbq.yaml
echo "[$(ts)] table rc=$?"
python scripts/visualize_paper_figures.py \
  --run outputs/seed_0/phase2_budgeted_mocapo_bbq_local \
  --out outputs/figures/paper_bbq_local
echo "[$(ts)] figures rc=$?"

echo "[$(ts)] ===== PIPELINE COMPLETE ====="
