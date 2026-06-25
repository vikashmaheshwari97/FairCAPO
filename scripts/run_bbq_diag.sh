#!/usr/bin/env bash
# Stage A fairness diagnostic: re-run the 3 held-out Dtests against the LARGER
# disjoint fairness set (data/fairness_bbq_holdout_large.jsonl, 180 items / 90
# disambiguated) with eval_pairs=180, so sDIS is no longer quantized to ~k/11.
# Writes to *_diag output dirs; leaves the original 36-pair eval dirs untouched.
set +e  # never abort the whole sweep if one method hangs/fails
cd "$(dirname "$0")/.." || exit 1
source .venv/Scripts/activate

run () {
  local name="$1" cfg="$2"
  echo "=========================================================="
  echo "[$name] starting  cfg=$cfg"
  PYTHONPATH=. python -u scripts/evaluate_pareto_on_test.py --config "$cfg" \
    2>&1 | tee "outputs/_bbq_diag_${name}.log"
  echo "[$name] done (rc=${PIPESTATUS[0]})"
}

run ablation configs/evaluate_pareto_bbq_ablation_diag.yaml
run nsga     configs/evaluate_pareto_bbq_nsga_diag.yaml
run faircapo configs/evaluate_pareto_bbq_diag.yaml

echo "ALL DIAGNOSTIC EVALS COMPLETE"
