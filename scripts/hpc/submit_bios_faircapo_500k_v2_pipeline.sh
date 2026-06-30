#!/usr/bin/env bash
set -euo pipefail

# Submit only Bias-in-Bios FairCAPO 500k_v2 search and its large-held-out eval.
# Use this before spending GPU time rerunning baselines.

cd "$(dirname "$0")/../.."

mkdir -p outputs/hpc/logs

echo "Submitting Bias-in-Bios FairCAPO 500k v2 search..."
search_job=$(sbatch --parsable --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/bios_faircapo_500k_v2_HPC.yaml,RUN_TAG=bios_faircapo_500k_v2 \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting Bias-in-Bios FairCAPO 500k v2 large-held-out eval..."
eval_job=$(sbatch --parsable --dependency=afterok:${search_job} --array=0 \
  --export=ALL,METHOD=faircapo,CONFIG=configs/HPC_Config/bios_eval_large_500k_v2_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bios_faircapo_500k_v2/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bios_faircapo_500k_v2 \
  scripts/hpc/run_bios_eval_hpc.slurm)

cat <<EOF
Submitted Bias-in-Bios FairCAPO 500k_v2:
  Search:          ${search_job}
  Large eval:      ${eval_job}  (afterok:${search_job})

Monitor:
  squeue -u \$USER
  ls -lt outputs/hpc/logs | head

After eval finishes, compare against existing v1 baselines:
  bash scripts/hpc/build_bios_500k_v2_large_outputs.sh
EOF
