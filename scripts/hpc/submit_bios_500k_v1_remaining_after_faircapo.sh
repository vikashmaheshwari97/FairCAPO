#!/usr/bin/env bash
set -euo pipefail

# Submit the remaining Bias-in-Bios 500k_v1 seed-0 pipeline after
# FairCAPO search has already completed successfully.
#
# Chain:
#   ablation search -> NSGA search -> FairCAPO eval -> ablation eval -> NSGA eval

cd "$(dirname "$0")/../.."

fair_portfolio="outputs/hpc/bios_faircapo_500k_v1/seed_0/phase2_prompt_portfolio.csv"
if [ ! -f "${fair_portfolio}" ]; then
  echo "Missing completed FairCAPO portfolio: ${fair_portfolio}" >&2
  echo "Run FairCAPO search first, or check RUN_TAG/output path." >&2
  exit 3
fi

mkdir -p outputs/hpc/logs

echo "Submitting Bias-in-Bios ablation 500k v1 search..."
ablation_search_job=$(sbatch --parsable --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/bios_ablation_500k_v1_HPC.yaml,RUN_TAG=bios_ablation_500k_v1 \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting Bias-in-Bios NSGA-II-PO 500k v1 search after ablation..."
nsga_search_job=$(sbatch --parsable --dependency=afterok:${ablation_search_job} --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/bios_nsga2po_500k_v1_HPC.yaml,RUN_TAG=bios_nsga2po_500k_v1 \
  scripts/hpc/run_bios_nsga_hpc.slurm)

echo "Submitting Bias-in-Bios FairCAPO large-held-out eval after searches..."
fair_eval_job=$(sbatch --parsable --dependency=afterok:${nsga_search_job} --array=0 \
  --export=ALL,METHOD=faircapo \
  scripts/hpc/run_bios_eval_hpc.slurm)

echo "Submitting Bias-in-Bios ablation large-held-out eval after FairCAPO eval..."
ablation_eval_job=$(sbatch --parsable --dependency=afterok:${fair_eval_job} --array=0 \
  --export=ALL,METHOD=ablation \
  scripts/hpc/run_bios_eval_hpc.slurm)

echo "Submitting Bias-in-Bios NSGA-II-PO large-held-out eval after ablation eval..."
nsga_eval_job=$(sbatch --parsable --dependency=afterok:${ablation_eval_job} --array=0 \
  --export=ALL,METHOD=nsga \
  scripts/hpc/run_bios_eval_hpc.slurm)

cat <<EOF
Submitted remaining Bias-in-Bios 500k_v1 seed-0 pipeline:
  Ablation search:       ${ablation_search_job}
  NSGA search:           ${nsga_search_job}  (afterok:${ablation_search_job})
  FairCAPO large eval:   ${fair_eval_job}  (afterok:${nsga_search_job})
  Ablation large eval:   ${ablation_eval_job}  (afterok:${fair_eval_job})
  NSGA large eval:       ${nsga_eval_job}  (afterok:${ablation_eval_job})

Monitor:
  squeue -u \$USER
  ls -lt outputs/hpc/logs | head

After all jobs finish:
  bash scripts/hpc/build_bios_500k_v1_large_outputs.sh
EOF
