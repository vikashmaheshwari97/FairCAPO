#!/usr/bin/env bash
set -euo pipefail

# Submit only the Bias-in-Bios enhanced-prompt FairCAPO 500k seed-0 pilot:
#   FairCAPO search -> large-held-out eval
#
# This keeps frozen BIOS 500k_v1 intact while testing whether a richer
# profession-disambiguation prompt pool improves exact-label accuracy.

cd "$(dirname "$0")/../.."

mkdir -p outputs/hpc/logs

echo "Submitting Bias-in-Bios enhanced-prompt FairCAPO 500k v1 search..."
fair_search_job=$(sbatch --parsable --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/bios_faircapo_enhancedprompts_500k_v1_HPC.yaml,RUN_TAG=bios_faircapo_enhancedprompts_500k_v1 \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting Bias-in-Bios enhanced-prompt FairCAPO 500k v1 large-held-out eval..."
fair_eval_job=$(sbatch --parsable --dependency=afterok:${fair_search_job} --array=0 \
  --export=ALL,METHOD=faircapo,CONFIG=configs/HPC_Config/bios_eval_large_enhancedprompts_500k_v1_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bios_faircapo_enhancedprompts_500k_v1/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bios_faircapo_enhancedprompts_500k_v1 \
  scripts/hpc/run_bios_eval_hpc.slurm)

cat <<EOF
Submitted Bias-in-Bios enhanced-prompt FairCAPO 500k v1 pilot:
  FairCAPO search:     ${fair_search_job}
  FairCAPO large eval: ${fair_eval_job}  (afterok:${fair_search_job})

Monitor:
  squeue -u \$USER
  ls -lt outputs/hpc/logs | head

After it finishes:
  cat outputs/hpc/evaluation_large/seed_0/bios_faircapo_enhancedprompts_500k_v1/test_eval_summary.json
  ls -lh outputs/hpc/bios_faircapo_enhancedprompts_500k_v1/seed_0
EOF
