#!/usr/bin/env bash
set -euo pipefail

# Submit only Qwen FairCAPO seed-0 search and large-held-out eval.
# Do this before any Qwen multi-seed run.

cd "$(dirname "$0")/../.."

mkdir -p outputs/hpc/logs

echo "Submitting Bias-in-Bios FairCAPO Qwen 500k v1 search..."
search_job=$(sbatch --parsable --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/bios_faircapo_qwen_500k_v1_HPC.yaml,RUN_TAG=bios_faircapo_qwen_500k_v1 \
  scripts/hpc/run_bios_qwen_hpc.slurm)

echo "Submitting Bias-in-Bios FairCAPO Qwen 500k v1 large-held-out eval..."
eval_job=$(sbatch --parsable --dependency=afterok:${search_job} --array=0 \
  scripts/hpc/run_bios_qwen_eval_hpc.slurm)

cat <<EOF
Submitted Bias-in-Bios FairCAPO Qwen 500k_v1:
  Search:          ${search_job}
  Large eval:      ${eval_job}  (afterok:${search_job})

Monitor:
  squeue -u \$USER
  ls -lt outputs/hpc/logs | head

After eval finishes:
  bash scripts/hpc/build_bios_qwen_500k_v1_large_outputs.sh
EOF
