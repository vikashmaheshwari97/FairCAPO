#!/usr/bin/env bash
set -euo pipefail

# Submit only the Bias-in-Bios two-stage label-scoring FairCAPO seed-0 run.
# This keeps the frozen Mistral v1 baselines untouched. Run baselines again only
# if the label-scoring FairCAPO result is worth comparing across methods.

cd "$(dirname "$0")/../.."

mkdir -p outputs/hpc/logs

echo "Submitting Bias-in-Bios FairCAPO label-score 500k v1 search..."
search_job=$(sbatch --parsable --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/bios_faircapo_labelscore_500k_v1_HPC.yaml,RUN_TAG=bios_faircapo_labelscore_500k_v1 \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting Bias-in-Bios FairCAPO label-score large-held-out eval after search..."
eval_job=$(sbatch --parsable --dependency=afterok:${search_job} --array=0 \
  --export=ALL,METHOD=faircapo,CONFIG=configs/HPC_Config/bios_eval_large_labelscore_500k_v1_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bios_faircapo_labelscore_500k_v1/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bios_faircapo_labelscore_500k_v1 \
  scripts/hpc/run_bios_eval_hpc.slurm)

cat <<EOF
Submitted Bias-in-Bios label-score FairCAPO seed-0 pipeline:
  Search:     ${search_job}
  Large eval: ${eval_job}  (afterok:${search_job})

Monitor:
  squeue -u \$USER
  ls -lt outputs/hpc/logs | head

After both jobs finish:
  bash scripts/hpc/build_bios_labelscore_500k_v1_large_outputs.sh
EOF
