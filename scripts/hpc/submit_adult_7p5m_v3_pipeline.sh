#!/usr/bin/env bash
set -euo pipefail

# Seed-0 Adult v3 1M pilot. The larger 7.5M configs remain in the repository for
# later scaling, but this active pipeline uses the smaller validation profile.

cd "$(dirname "$0")/../.."
mkdir -p outputs/hpc/logs

# Prefer rebuilding the semantic file from the canonical raw Adult CSV. If the
# raw CSV is unavailable but a previously generated semantic CSV exists, use it
# only after the full preflight validates row count, split balance, leakage and
# prompt rendering.
if [[ -s data/adult.csv ]]; then
  PYTHONPATH=. python scripts/prepare_adult_semantic_csv.py \
    --input data/adult.csv \
    --output data/adult_semantic_v3.csv
elif [[ -s data/adult_semantic_v3.csv ]]; then
  echo "data/adult.csv is missing; using existing data/adult_semantic_v3.csv after preflight validation."
else
  echo "Missing both data/adult.csv and data/adult_semantic_v3.csv" >&2
  exit 3
fi

PYTHONPATH=. python scripts/preflight_adult_v3.py \
  --data data/adult_semantic_v3.csv

RUN_DIRS=(
  outputs/hpc/adult_faircapo_1m_v3/seed_0
  outputs/hpc/adult_ablation_1m_v3/seed_0
  outputs/hpc/adult_nsga2po_1m_v3/seed_0
  outputs/hpc/evaluation_1m/seed_0/adult_faircapo_1m_v3
  outputs/hpc/evaluation_1m/seed_0/adult_ablation_1m_v3
  outputs/hpc/evaluation_1m/seed_0/adult_nsga2po_1m_v3
)
if [[ "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
  for run_dir in "${RUN_DIRS[@]}"; do
    if [[ -d "${run_dir}" ]] && find "${run_dir}" -mindepth 1 -print -quit | grep -q .; then
      echo "Refusing to mix outputs: ${run_dir} is not empty." >&2
      echo "Archive/remove the old 1M outputs or use ALLOW_OVERWRITE=1." >&2
      exit 4
    fi
  done
fi

COMMON_EXPORT="ALL,FAIRCAPO_ADULT_REASONING_SHOTS=1,EVAL_RUNNER=scripts/evaluate_adult_v3_on_test.py"

echo "Submitting Adult FairCAPO v3 1M search..."
fair_search_job=$(sbatch --parsable --time=06:00:00 \
  --job-name=adult-fair-1m-v3 --array=0 \
  --export="${COMMON_EXPORT},CONFIG=configs/HPC_Config/adult_faircapo_1m_v3_HPC.yaml,RUN_TAG=adult_faircapo_1m_v3" \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting Adult MO-CAPO strict fairness-objective-off v3 1M search..."
ablation_search_job=$(sbatch --parsable --time=06:00:00 \
  --job-name=adult-mo-1m-v3 --dependency=afterok:${fair_search_job} --array=0 \
  --export="${COMMON_EXPORT},CONFIG=configs/HPC_Config/adult_ablation_1m_v3_HPC.yaml,RUN_TAG=adult_ablation_1m_v3,MOCAPO_RUNNER=scripts/run_adult_v3_mocapo_ablation.py" \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting Adult NSGA-II-PO v3 1M search..."
nsga_search_job=$(sbatch --parsable --time=06:00:00 \
  --job-name=adult-nsga-1m-v3 --dependency=afterok:${ablation_search_job} --array=0 \
  --export="${COMMON_EXPORT},CONFIG=configs/HPC_Config/adult_nsga2po_1m_v3_HPC.yaml,RUN_TAG=adult_nsga2po_1m_v3,NSGA_RUNNER=scripts/run_adult_v3_nsga2_po.py" \
  scripts/hpc/run_bios_nsga_hpc.slurm)

EVAL_CONFIG="configs/HPC_Config/adult_eval_large_1m_v3_HPC.yaml"

echo "Submitting Adult FairCAPO v3 1M held-out evaluation..."
fair_eval_job=$(sbatch --parsable --time=04:00:00 \
  --job-name=adult-eval-fair-1m-v3 --dependency=afterok:${nsga_search_job} --array=0 \
  --export="${COMMON_EXPORT},METHOD=adult_faircapo,CONFIG=${EVAL_CONFIG},PORTFOLIO_CSV=outputs/hpc/adult_faircapo_1m_v3/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_1m/seed_0/adult_faircapo_1m_v3" \
  scripts/hpc/run_bios_eval_hpc.slurm)

echo "Submitting Adult MO-CAPO v3 1M held-out evaluation..."
ablation_eval_job=$(sbatch --parsable --time=04:00:00 \
  --job-name=adult-eval-mo-1m-v3 --dependency=afterok:${fair_eval_job} --array=0 \
  --export="${COMMON_EXPORT},METHOD=adult_ablation,CONFIG=${EVAL_CONFIG},PORTFOLIO_CSV=outputs/hpc/adult_ablation_1m_v3/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_1m/seed_0/adult_ablation_1m_v3" \
  scripts/hpc/run_bios_eval_hpc.slurm)

echo "Submitting Adult NSGA-II-PO v3 1M held-out evaluation..."
nsga_eval_job=$(sbatch --parsable --time=04:00:00 \
  --job-name=adult-eval-nsga-1m-v3 --dependency=afterok:${ablation_eval_job} --array=0 \
  --export="${COMMON_EXPORT},METHOD=adult_nsga,CONFIG=${EVAL_CONFIG},PORTFOLIO_CSV=outputs/hpc/adult_nsga2po_1m_v3/seed_0/nsga2_po_pareto_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_1m/seed_0/adult_nsga2po_1m_v3" \
  scripts/hpc/run_bios_eval_hpc.slurm)

cat <<EOF
Submitted Adult v3 1M seed-0 pipeline:
  FairCAPO search:      ${fair_search_job}
  MO-CAPO search:       ${ablation_search_job}
  NSGA-II-PO search:    ${nsga_search_job}
  FairCAPO eval:        ${fair_eval_job}
  MO-CAPO eval:         ${ablation_eval_job}
  NSGA-II-PO eval:      ${nsga_eval_job}

Monitor:
  squeue -u \$USER
  sacct -j ${fair_search_job},${ablation_search_job},${nsga_search_job},${fair_eval_job},${ablation_eval_job},${nsga_eval_job} --format=JobID,JobName%28,State,ExitCode,Elapsed,NodeList

Build tables and figures after all six jobs complete:
  bash scripts/hpc/build_adult_1m_v3_outputs.sh
EOF
