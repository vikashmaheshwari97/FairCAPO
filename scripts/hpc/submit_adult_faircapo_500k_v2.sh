#!/usr/bin/env bash
set -euo pipefail

# Submit only the improved Adult FairCAPO v2 search and held-out evaluation.
# Existing Adult v1 MO-CAPO and NSGA-II-PO outputs remain the comparison baselines.

cd "$(dirname "$0")/../.."
mkdir -p outputs/hpc/logs

if [ ! -s data/adult.csv ]; then
  echo "Missing data/adult.csv" >&2
  exit 3
fi

ARRAY_SPEC="${ARRAY_SPEC:-0}"

search_job=$(sbatch --parsable --job-name=adult-fair-v2 --array="${ARRAY_SPEC}" \
  --export=ALL,CONFIG=configs/HPC_Config/adult_faircapo_500k_v2_HPC.yaml,RUN_TAG=adult_faircapo_500k_v2 \
  scripts/hpc/run_bios_hpc.slurm)

eval_job=$(sbatch --parsable --job-name=adult-eval-fair-v2 \
  --dependency=afterok:${search_job} --array="${ARRAY_SPEC}" \
  --export=ALL,METHOD=adult_faircapo,CONFIG=configs/HPC_Config/adult_eval_large_500k_v2_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/adult_faircapo_500k_v2/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/adult_faircapo_500k_v2 \
  scripts/hpc/run_bios_eval_hpc.slurm)

cat <<EOF
Submitted Adult FairCAPO 500k v2:
  Array spec:       ${ARRAY_SPEC}
  FairCAPO search:  ${search_job}
  Held-out eval:    ${eval_job}  (afterok:${search_job})

Monitor:
  squeue -u \$USER
  sacct -j ${search_job},${eval_job} --format=JobID,JobName%24,State,ExitCode,Elapsed,NodeList

Build the comparison after evaluation completes:
  bash scripts/hpc/build_adult_500k_v2_large_outputs.sh
EOF
