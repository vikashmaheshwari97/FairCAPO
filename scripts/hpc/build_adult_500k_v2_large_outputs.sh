#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONFIG="${CONFIG:-configs/HPC_Config/adult_experiment_table_500k_v2_large_HPC.yaml}"
OUT="outputs/experiment_table/adult_mistral_hpc_500k_v2_large_seed0"

test -f outputs/hpc/evaluation_large/seed_0/adult_faircapo_500k_v2/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/adult_ablation_500k_v1/test_eval_summary.json
test -f outputs/hpc/evaluation_large/seed_0/adult_nsga2po_500k_v1/test_eval_summary.json

PYTHONPATH=. python scripts/build_experiment_table.py --config "${CONFIG}"
PYTHONPATH=. python scripts/build_representative_experiment_table.py --config "${CONFIG}"

echo "Envelope table: ${OUT}/experiment_table.csv"
echo "Representative table: ${OUT}/representative_experiment_table.csv"
