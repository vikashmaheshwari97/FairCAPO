#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p outputs/hpc/logs

CONFIG="configs/HPC_Config/adult_faircapo_accuracy_4m_v3_HPC.yaml"
RUN_DIR="outputs/hpc/adult_faircapo_accuracy_4m_v3/seed_0"
EVAL_DIR="outputs/hpc/evaluation_accuracy/seed_0/adult_faircapo_accuracy_4m_v3"
PORTFOLIO="${RUN_DIR}/phase2_prompt_portfolio.csv"
EVAL_CONFIG="configs/HPC_Config/adult_eval_large_1m_v3_HPC.yaml"

if [[ -s data/adult.csv ]]; then
  PYTHONPATH=. python scripts/prepare_adult_semantic_csv.py \
    --input data/adult.csv \
    --output data/adult_semantic_v3.csv
fi

test -s data/adult_semantic_v3.csv
PYTHONPATH=. python scripts/preflight_adult_accuracy_v3.py \
  --config "${CONFIG}" \
  --data data/adult_semantic_v3.csv

if [[ "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
  for run_dir in "${RUN_DIR}" "${EVAL_DIR}"; do
    if [[ -d "${run_dir}" ]] && find "${run_dir}" -mindepth 1 -print -quit | grep -q .; then
      echo "Refusing to mix outputs: ${run_dir} is not empty." >&2
      echo "Archive/remove it or use ALLOW_OVERWRITE=1." >&2
      exit 4
    fi
  done
fi

COMMON_EXPORT="ALL,FAIRCAPO_ADULT_REASONING_SHOTS=0,EVAL_RUNNER=scripts/evaluate_adult_v3_on_test.py"

echo "Submitting Adult FairCAPO accuracy-discovery 4M search..."
search_job=$(sbatch --parsable --time=10:00:00 \
  --job-name=adult-acc-4m-v3 --array=0 \
  --export="${COMMON_EXPORT},CONFIG=${CONFIG},RUN_TAG=adult_faircapo_accuracy_4m_v3" \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting held-out evaluation for the discovered FairCAPO accuracy portfolio..."
eval_job=$(sbatch --parsable --time=04:00:00 \
  --job-name=adult-eval-acc-4m-v3 --dependency=afterok:${search_job} --array=0 \
  --export="${COMMON_EXPORT},METHOD=adult_faircapo_accuracy,CONFIG=${EVAL_CONFIG},PORTFOLIO_CSV=${PORTFOLIO},OUT_DIR=${EVAL_DIR}" \
  scripts/hpc/run_bios_eval_hpc.slurm)

cat <<EOF
Submitted Adult accuracy-discovery v3:
  Search: ${search_job}
  Eval:   ${eval_job}

Monitor:
  squeue -u \$USER
  sacct -j ${search_job},${eval_job} --format=JobID,JobName%30,State,ExitCode,Elapsed,NodeList

Result after eval:
  ${EVAL_DIR}/test_eval_candidates.csv
EOF
