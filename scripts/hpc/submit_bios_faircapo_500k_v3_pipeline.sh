#!/usr/bin/env bash
set -euo pipefail

# Submit only Bias-in-Bios FairCAPO 500k_v3 search and large-held-out eval.
# v3 is the v1-based controlled attempt after v2 over-corrected toward cost.

cd "$(dirname "$0")/../.."

mkdir -p outputs/hpc/logs

echo "Submitting Bias-in-Bios FairCAPO 500k v3 search..."
search_job=$(sbatch --parsable --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/bios_faircapo_500k_v3_HPC.yaml,RUN_TAG=bios_faircapo_500k_v3 \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting Bias-in-Bios FairCAPO 500k v3 large-held-out eval..."
eval_job=$(sbatch --parsable --dependency=afterok:${search_job} --array=0 \
  --export=ALL,METHOD=faircapo,CONFIG=configs/HPC_Config/bios_eval_large_500k_v3_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bios_faircapo_500k_v3/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bios_faircapo_500k_v3 \
  scripts/hpc/run_bios_eval_hpc.slurm)

cat <<EOF
Submitted Bias-in-Bios FairCAPO 500k_v3:
  Search:          ${search_job}
  Large eval:      ${eval_job}  (afterok:${search_job})

Monitor:
  squeue -u \$USER
  ls -lt outputs/hpc/logs | head

After eval finishes:
  bash scripts/hpc/build_bios_500k_v3_large_outputs.sh
EOF
