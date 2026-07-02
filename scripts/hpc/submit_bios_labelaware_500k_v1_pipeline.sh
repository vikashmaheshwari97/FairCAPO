#!/usr/bin/env bash
set -euo pipefail

# Submit the full Bias-in-Bios label-aware 500k_v1 seed-0 pipeline.
#
# This is the clean rerun after the objective/cost/fairness-probe fixes:
#   FairCAPO label-aware search
#     -> MO-CAPO fairness-off search
#       -> NSGA-II-PO + fairness search
#         -> large-held-out evals for all three
#
# Conservative Rocket mode: one GPU job at a time via afterok dependencies.

cd "$(dirname "$0")/../.."

mkdir -p outputs/hpc/logs

echo "Submitting Bias-in-Bios label-aware FairCAPO 500k v1 search..."
fair_search_job=$(sbatch --parsable --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/bios_faircapo_labelaware_500k_v1_HPC.yaml,RUN_TAG=bios_faircapo_labelaware_500k_v1 \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting Bias-in-Bios MO-CAPO fairness-off 500k v1 search after FairCAPO..."
ablation_search_job=$(sbatch --parsable --dependency=afterok:${fair_search_job} --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/bios_ablation_500k_v1_HPC.yaml,RUN_TAG=bios_ablation_500k_v1 \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting Bias-in-Bios NSGA-II-PO + fairness 500k v1 search after ablation..."
nsga_search_job=$(sbatch --parsable --dependency=afterok:${ablation_search_job} --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/bios_nsga2po_500k_v1_HPC.yaml,RUN_TAG=bios_nsga2po_500k_v1 \
  scripts/hpc/run_bios_nsga_hpc.slurm)

echo "Submitting label-aware FairCAPO large-held-out eval after all searches..."
fair_eval_job=$(sbatch --parsable --dependency=afterok:${nsga_search_job} --array=0 \
  --export=ALL,METHOD=faircapo,CONFIG=configs/HPC_Config/bios_eval_large_labelaware_500k_v1_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bios_faircapo_labelaware_500k_v1/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bios_faircapo_labelaware_500k_v1 \
  scripts/hpc/run_bios_eval_hpc.slurm)

echo "Submitting MO-CAPO fairness-off large-held-out eval after FairCAPO eval..."
ablation_eval_job=$(sbatch --parsable --dependency=afterok:${fair_eval_job} --array=0 \
  --export=ALL,METHOD=ablation,CONFIG=configs/HPC_Config/bios_eval_ablation_large_500k_v1_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bios_ablation_500k_v1/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bios_ablation_500k_v1 \
  scripts/hpc/run_bios_eval_hpc.slurm)

echo "Submitting NSGA-II-PO + fairness large-held-out eval after ablation eval..."
nsga_eval_job=$(sbatch --parsable --dependency=afterok:${ablation_eval_job} --array=0 \
  --export=ALL,METHOD=nsga,CONFIG=configs/HPC_Config/bios_eval_nsga_large_500k_v1_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bios_nsga2po_500k_v1/seed_0/nsga2_po_pareto_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bios_nsga2po_500k_v1 \
  scripts/hpc/run_bios_eval_hpc.slurm)

cat <<EOF
Submitted Bias-in-Bios label-aware 500k_v1 seed-0 pipeline:
  FairCAPO label-aware search:  ${fair_search_job}
  MO-CAPO fairness-off search:  ${ablation_search_job}  (afterok:${fair_search_job})
  NSGA-II-PO + fairness search: ${nsga_search_job}  (afterok:${ablation_search_job})
  FairCAPO large eval:          ${fair_eval_job}  (afterok:${nsga_search_job})
  MO-CAPO large eval:           ${ablation_eval_job}  (afterok:${fair_eval_job})
  NSGA large eval:              ${nsga_eval_job}  (afterok:${ablation_eval_job})

Monitor:
  squeue -u \$USER
  ls -lt outputs/hpc/logs | head

After all jobs finish:
  bash scripts/hpc/build_bios_labelaware_500k_v1_large_outputs.sh
EOF
