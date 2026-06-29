#!/usr/bin/env bash
set -euo pipefail

# Submit the full Bias-in-Bios 500k_v1 seed-0 pipeline.
#
# Conservative Rocket mode: one GPU job at a time via afterok dependencies.
# This avoids loading several A100 jobs at once and keeps failures from
# cascading into expensive downstream runs.

cd "$(dirname "$0")/../.."

mkdir -p outputs/hpc/logs

echo "Submitting Bias-in-Bios FairCAPO 500k v1 search..."
fair_search_job=$(sbatch --parsable --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/bios_faircapo_500k_v1_HPC.yaml,RUN_TAG=bios_faircapo_500k_v1 \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting Bias-in-Bios ablation 500k v1 search after FairCAPO..."
ablation_search_job=$(sbatch --parsable --dependency=afterok:${fair_search_job} --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/bios_ablation_500k_v1_HPC.yaml,RUN_TAG=bios_ablation_500k_v1 \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting Bias-in-Bios NSGA-II-PO 500k v1 search after ablation..."
nsga_search_job=$(sbatch --parsable --dependency=afterok:${ablation_search_job} --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/bios_nsga2po_500k_v1_HPC.yaml,RUN_TAG=bios_nsga2po_500k_v1 \
  scripts/hpc/run_bios_nsga_hpc.slurm)

echo "Submitting Bias-in-Bios FairCAPO large-held-out eval after all searches..."
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
Submitted Bias-in-Bios 500k_v1 seed-0 pipeline:
  FairCAPO search:       ${fair_search_job}
  Ablation search:       ${ablation_search_job}  (afterok:${fair_search_job})
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
