#!/usr/bin/env bash
set -euo pipefail

# Submit the active BBQ 500k_v2 large-held-out evaluation pipeline.
#
# The three method evals and the post-hoc fairness scoring are independent and
# can run at the same time on different GPUs if SLURM has capacity. The final
# post-hoc Dtest eval depends on the post-hoc scoring job because it needs the
# *_bbqfair_large.csv portfolio first.

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

echo "Submitting NSGA-II-PO large-held-out eval..."
nsga_job=$(sbatch --parsable --array=0 \
  --export=ALL,METHOD=nsga,CONFIG=configs/HPC_Config/evaluate_pareto_bbq_nsga_large_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bbq_nsga2po_500k_v2/seed_0/nsga2_po_pareto_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bbq_nsga2po_500k_v2 \
  scripts/hpc/run_bbq_eval_hpc.slurm)

echo "Submitting post-hoc large fairness scoring..."
posthoc_score_job=$(sbatch --parsable --array=0 \
  --export=ALL,FAIRNESS_CONFIG=configs/HPC_Config/evaluate_pareto_bbq_ablation_large_HPC.yaml,INPUT_CSV=outputs/hpc/bbq_ablation_500k_v2/seed_0/phase2_prompt_portfolio.csv,OUTPUT_SUFFIX=_bbqfair_large \
  scripts/hpc/run_bbq_posthoc_hpc.slurm)

echo "Submitting post-hoc large-held-out eval after post-hoc scoring completes..."
posthoc_eval_job=$(sbatch --parsable --dependency=afterok:${posthoc_score_job} --array=0 \
  --export=ALL,METHOD=ablation,CONFIG=configs/HPC_Config/evaluate_pareto_bbq_ablation_large_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bbq_ablation_500k_v2/seed_0/phase2_prompt_portfolio_bbqfair_large.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bbq_posthoc_500k_v2 \
  scripts/hpc/run_bbq_eval_hpc.slurm)

cat <<EOF
Submitted jobs:
  FairCAPO eval:       ${fair_job}
  Ablation eval:       ${ablation_job}
  NSGA eval:           ${nsga_job}
  Post-hoc scoring:    ${posthoc_score_job}
  Post-hoc eval:       ${posthoc_eval_job}  (afterok:${posthoc_score_job})

Monitor:
  squeue -u \$USER
  ls -lt outputs/hpc/logs | head

After all jobs finish:
  bash scripts/hpc/build_bbq_500k_v2_outputs.sh
EOF
