#!/usr/bin/env bash
set -euo pipefail

# Submit the guided-classifier Bias-in-Bios 500k_v1 pipeline for seeds 0,1,2.
#
# The array throttle (%1) keeps the run conservative: all three seeds are
# submitted now, but Slurm runs one seed task at a time for each stage.

cd "$(dirname "$0")/../.."

mkdir -p outputs/hpc/logs

ARRAY_SPEC="${ARRAY_SPEC:-0-2%1}"

echo "Submitting guided Bias-in-Bios FairCAPO search array ${ARRAY_SPEC}..."
fair_search_job=$(sbatch --parsable --array="${ARRAY_SPEC}" \
  --export=ALL,CONFIG=configs/HPC_Config/bios_faircapo_guided_500k_v1_HPC.yaml,RUN_TAG=bios_faircapo_guided_500k_v1 \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting guided Bias-in-Bios ablation search after FairCAPO..."
ablation_search_job=$(sbatch --parsable --dependency=afterok:${fair_search_job} --array="${ARRAY_SPEC}" \
  --export=ALL,CONFIG=configs/HPC_Config/bios_ablation_guided_500k_v1_HPC.yaml,RUN_TAG=bios_ablation_guided_500k_v1 \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting guided Bias-in-Bios NSGA-II-PO search after ablation..."
nsga_search_job=$(sbatch --parsable --dependency=afterok:${ablation_search_job} --array="${ARRAY_SPEC}" \
  --export=ALL,CONFIG=configs/HPC_Config/bios_nsga2po_guided_500k_v1_HPC.yaml,RUN_TAG=bios_nsga2po_guided_500k_v1 \
  scripts/hpc/run_bios_nsga_hpc.slurm)

echo "Submitting guided Bias-in-Bios FairCAPO large-held-out eval after all searches..."
fair_eval_job=$(sbatch --parsable --dependency=afterok:${nsga_search_job} --array="${ARRAY_SPEC}" \
  --export=ALL,METHOD=faircapo_guided \
  scripts/hpc/run_bios_eval_hpc.slurm)

echo "Submitting guided Bias-in-Bios ablation large-held-out eval after FairCAPO eval..."
ablation_eval_job=$(sbatch --parsable --dependency=afterok:${fair_eval_job} --array="${ARRAY_SPEC}" \
  --export=ALL,METHOD=ablation_guided \
  scripts/hpc/run_bios_eval_hpc.slurm)

echo "Submitting guided Bias-in-Bios NSGA-II-PO large-held-out eval after ablation eval..."
nsga_eval_job=$(sbatch --parsable --dependency=afterok:${ablation_eval_job} --array="${ARRAY_SPEC}" \
  --export=ALL,METHOD=nsga_guided \
  scripts/hpc/run_bios_eval_hpc.slurm)

cat <<EOF
Submitted guided Bias-in-Bios 500k_v1 multiseed pipeline:
  Array spec:             ${ARRAY_SPEC}
  FairCAPO search:        ${fair_search_job}
  Ablation search:        ${ablation_search_job}  (afterok:${fair_search_job})
  NSGA search:            ${nsga_search_job}  (afterok:${ablation_search_job})
  FairCAPO large eval:    ${fair_eval_job}  (afterok:${nsga_search_job})
  Ablation large eval:    ${ablation_eval_job}  (afterok:${fair_eval_job})
  NSGA large eval:        ${nsga_eval_job}  (afterok:${ablation_eval_job})

Monitor:
  squeue -u \$USER
  ls -lt outputs/hpc/logs | head

Expected search outputs:
  outputs/hpc/bios_faircapo_guided_500k_v1/seed_{0,1,2}/
  outputs/hpc/bios_ablation_guided_500k_v1/seed_{0,1,2}/
  outputs/hpc/bios_nsga2po_guided_500k_v1/seed_{0,1,2}/

Expected large-held-out eval outputs:
  outputs/hpc/evaluation_large/seed_{0,1,2}/bios_faircapo_guided_500k_v1/
  outputs/hpc/evaluation_large/seed_{0,1,2}/bios_ablation_guided_500k_v1/
  outputs/hpc/evaluation_large/seed_{0,1,2}/bios_nsga2po_guided_500k_v1/
EOF
