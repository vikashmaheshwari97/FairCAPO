#!/usr/bin/env bash
set -euo pipefail

# Submit only the Bias-in-Bios 500k_v1 large-held-out eval jobs.
# Use this if the search outputs already exist.

cd "$(dirname "$0")/../.."

mkdir -p outputs/hpc/logs

fair_eval_job=$(sbatch --parsable --array=0 \
  --export=ALL,METHOD=faircapo \
  scripts/hpc/run_bios_eval_hpc.slurm)

ablation_eval_job=$(sbatch --parsable --dependency=afterok:${fair_eval_job} --array=0 \
  --export=ALL,METHOD=ablation \
  scripts/hpc/run_bios_eval_hpc.slurm)

nsga_eval_job=$(sbatch --parsable --dependency=afterok:${ablation_eval_job} --array=0 \
  --export=ALL,METHOD=nsga \
  scripts/hpc/run_bios_eval_hpc.slurm)

cat <<EOF
Submitted Bias-in-Bios 500k_v1 large eval jobs:
  FairCAPO eval: ${fair_eval_job}
  Ablation eval: ${ablation_eval_job}  (afterok:${fair_eval_job})
  NSGA eval:     ${nsga_eval_job}  (afterok:${ablation_eval_job})

After all jobs finish:
  bash scripts/hpc/build_bios_500k_v1_large_outputs.sh
EOF
