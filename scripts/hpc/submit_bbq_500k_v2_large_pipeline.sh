#!/usr/bin/env bash
set -euo pipefail

# Submit the active BBQ 500k_v2 large-held-out evaluation pipeline.
#
# Conservative default: run at most two GPU jobs at a time on Pegasus2. UT HPC
# recommends 16 CPU cores per A100-80GB GPU on pegasus2; limiting concurrency is
# the safer way to reduce cluster load without breaking CPU/GPU binding.
#
# Dependency chain:
#   wave 1: FairCAPO eval + ablation eval
#   wave 2: NSGA eval + post-hoc fairness scoring, after wave 1 succeeds
#   wave 3: post-hoc Dtest eval, after post-hoc scoring succeeds

cd "$(dirname "$0")/../.."

mkdir -p outputs/hpc/logs

echo "Submitting FairCAPO large-held-out eval..."
fair_job=$(sbatch --parsable --array=0 \
  --export=ALL,METHOD=faircapo,CONFIG=configs/HPC_Config/evaluate_pareto_bbq_large_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bbq_faircapo_500k_v2/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bbq_faircapo_500k_v2 \
  scripts/hpc/run_bbq_eval_hpc.slurm)

echo "Submitting ablation large-held-out eval..."
ablation_job=$(sbatch --parsable --array=0 \
  --export=ALL,METHOD=ablation,CONFIG=configs/HPC_Config/evaluate_pareto_bbq_ablation_large_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bbq_ablation_500k_v2/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bbq_ablation_500k_v2 \
  scripts/hpc/run_bbq_eval_hpc.slurm)

wave1_dep="${fair_job}:${ablation_job}"

echo "Submitting NSGA-II-PO large-held-out eval after wave 1..."
nsga_job=$(sbatch --parsable --dependency=afterok:${wave1_dep} --array=0 \
  --export=ALL,METHOD=nsga,CONFIG=configs/HPC_Config/evaluate_pareto_bbq_nsga_large_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bbq_nsga2po_500k_v2/seed_0/nsga2_po_pareto_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bbq_nsga2po_500k_v2 \
  scripts/hpc/run_bbq_eval_hpc.slurm)

echo "Submitting post-hoc large fairness scoring after wave 1..."
posthoc_score_job=$(sbatch --parsable --dependency=afterok:${wave1_dep} --array=0 \
  --export=ALL,FAIRNESS_CONFIG=configs/HPC_Config/evaluate_pareto_bbq_ablation_large_HPC.yaml,INPUT_CSV=outputs/hpc/bbq_ablation_500k_v2/seed_0/phase2_prompt_portfolio.csv,OUTPUT_SUFFIX=_bbqfair_large \
  scripts/hpc/run_bbq_posthoc_hpc.slurm)

echo "Submitting post-hoc large-held-out eval after post-hoc scoring completes..."
posthoc_eval_job=$(sbatch --parsable --dependency=afterok:${posthoc_score_job} --array=0 \
  --export=ALL,METHOD=ablation,CONFIG=configs/HPC_Config/evaluate_pareto_bbq_ablation_large_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bbq_ablation_500k_v2/seed_0/phase2_prompt_portfolio_bbqfair_large.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bbq_posthoc_500k_v2 \
  scripts/hpc/run_bbq_eval_hpc.slurm)

cat <<EOF
Submitted jobs:
  Wave 1 FairCAPO eval:       ${fair_job}
  Wave 1 ablation eval:       ${ablation_job}
  Wave 2 NSGA eval:           ${nsga_job}  (afterok:${wave1_dep})
  Wave 2 post-hoc scoring:    ${posthoc_score_job}  (afterok:${wave1_dep})
  Wave 3 post-hoc eval:       ${posthoc_eval_job}  (afterok:${posthoc_score_job})

Monitor:
  squeue -u \$USER
  ls -lt outputs/hpc/logs | head

After all jobs finish:
  bash scripts/hpc/build_bbq_500k_v2_outputs.sh
EOF
