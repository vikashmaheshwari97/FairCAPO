#!/usr/bin/env bash
set -euo pipefail

# Submit the Adult 500k_v1 pipeline.
#
# Default is seed 0 only. For conservative multiseed:
#   ARRAY_SPEC=0-2%1 bash scripts/hpc/submit_adult_500k_v1_pipeline.sh

cd "$(dirname "$0")/../.."

mkdir -p outputs/hpc/logs

if [ ! -s data/adult.csv ]; then
  echo "Missing data/adult.csv" >&2
  echo "Copy the Adult CSV to data/adult.csv before submitting." >&2
  exit 3
fi

ARRAY_SPEC="${ARRAY_SPEC:-0}"

echo "Submitting Adult FairCAPO search array ${ARRAY_SPEC}..."
fair_search_job=$(sbatch --parsable --job-name=adult-faircapo --array="${ARRAY_SPEC}" \
  --export=ALL,CONFIG=configs/HPC_Config/adult_faircapo_500k_v1_HPC.yaml,RUN_TAG=adult_faircapo_500k_v1 \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting Adult MO-CAPO fairness-off search after FairCAPO..."
ablation_search_job=$(sbatch --parsable --job-name=adult-ablation --dependency=afterok:${fair_search_job} --array="${ARRAY_SPEC}" \
  --export=ALL,CONFIG=configs/HPC_Config/adult_ablation_500k_v1_HPC.yaml,RUN_TAG=adult_ablation_500k_v1 \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting Adult NSGA-II-PO search after ablation..."
nsga_search_job=$(sbatch --parsable --job-name=adult-nsga2po --dependency=afterok:${ablation_search_job} --array="${ARRAY_SPEC}" \
  --export=ALL,CONFIG=configs/HPC_Config/adult_nsga2po_500k_v1_HPC.yaml,RUN_TAG=adult_nsga2po_500k_v1 \
  scripts/hpc/run_bios_nsga_hpc.slurm)

echo "Submitting Adult FairCAPO large-held-out eval after all searches..."
fair_eval_job=$(sbatch --parsable --job-name=adult-eval-fair --dependency=afterok:${nsga_search_job} --array="${ARRAY_SPEC}" \
  --export=ALL,METHOD=adult_faircapo \
  scripts/hpc/run_bios_eval_hpc.slurm)

echo "Submitting Adult MO-CAPO fairness-off large-held-out eval after FairCAPO eval..."
ablation_eval_job=$(sbatch --parsable --job-name=adult-eval-abl --dependency=afterok:${fair_eval_job} --array="${ARRAY_SPEC}" \
  --export=ALL,METHOD=adult_ablation \
  scripts/hpc/run_bios_eval_hpc.slurm)

echo "Submitting Adult NSGA-II-PO large-held-out eval after ablation eval..."
nsga_eval_job=$(sbatch --parsable --job-name=adult-eval-nsga --dependency=afterok:${ablation_eval_job} --array="${ARRAY_SPEC}" \
  --export=ALL,METHOD=adult_nsga \
  scripts/hpc/run_bios_eval_hpc.slurm)

cat <<EOF
Submitted Adult 500k_v1 pipeline:
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

Build outputs after all eval jobs finish:
  bash scripts/hpc/build_adult_500k_v1_large_outputs.sh
EOF
