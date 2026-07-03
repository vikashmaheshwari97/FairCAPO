#!/usr/bin/env bash
set -euo pipefail

# Seed-0 Adult v3 pipeline. All three optimizers use the same semantic dataset,
# ten initial prompts, reasoning-bearing few-shots, and 7.5M-token search cap.

cd "$(dirname "$0")/../.."
mkdir -p outputs/hpc/logs

if [[ ! -s data/adult.csv ]]; then
  echo "Missing data/adult.csv" >&2
  exit 3
fi

PYTHONPATH=. python scripts/prepare_adult_semantic_csv.py \
  --input data/adult.csv \
  --output data/adult_semantic_v3.csv

test -s data/adult_semantic_v3.csv

COMMON_EXPORT="ALL,FAIRCAPO_ADULT_REASONING_SHOTS=1"

echo "Submitting Adult FairCAPO v3 search..."
fair_search_job=$(sbatch --parsable --job-name=adult-fair-v3 --array=0 \
  --export="${COMMON_EXPORT},CONFIG=configs/HPC_Config/adult_faircapo_7p5m_v3_HPC.yaml,RUN_TAG=adult_faircapo_7p5m_v3" \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting Adult MO-CAPO fairness-off v3 search..."
ablation_search_job=$(sbatch --parsable --job-name=adult-mo-v3 \
  --dependency=afterok:${fair_search_job} --array=0 \
  --export="${COMMON_EXPORT},CONFIG=configs/HPC_Config/adult_ablation_7p5m_v3_HPC.yaml,RUN_TAG=adult_ablation_7p5m_v3" \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting Adult NSGA-II-PO v3 search..."
nsga_search_job=$(sbatch --parsable --job-name=adult-nsga-v3 \
  --dependency=afterok:${ablation_search_job} --array=0 \
  --export="${COMMON_EXPORT},CONFIG=configs/HPC_Config/adult_nsga2po_7p5m_v3_HPC.yaml,RUN_TAG=adult_nsga2po_7p5m_v3" \
  scripts/hpc/run_bios_nsga_hpc.slurm)

EVAL_CONFIG="configs/HPC_Config/adult_eval_large_7p5m_v3_HPC.yaml"

echo "Submitting Adult FairCAPO v3 held-out evaluation..."
fair_eval_job=$(sbatch --parsable --job-name=adult-eval-fair-v3 \
  --dependency=afterok:${nsga_search_job} --array=0 \
  --export="${COMMON_EXPORT},METHOD=adult_faircapo,CONFIG=${EVAL_CONFIG},PORTFOLIO_CSV=outputs/hpc/adult_faircapo_7p5m_v3/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/adult_faircapo_7p5m_v3" \
  scripts/hpc/run_bios_eval_hpc.slurm)

echo "Submitting Adult MO-CAPO v3 held-out evaluation..."
ablation_eval_job=$(sbatch --parsable --job-name=adult-eval-mo-v3 \
  --dependency=afterok:${fair_eval_job} --array=0 \
  --export="${COMMON_EXPORT},METHOD=adult_ablation,CONFIG=${EVAL_CONFIG},PORTFOLIO_CSV=outputs/hpc/adult_ablation_7p5m_v3/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/adult_ablation_7p5m_v3" \
  scripts/hpc/run_bios_eval_hpc.slurm)

echo "Submitting Adult NSGA-II-PO v3 held-out evaluation..."
nsga_eval_job=$(sbatch --parsable --job-name=adult-eval-nsga-v3 \
  --dependency=afterok:${ablation_eval_job} --array=0 \
  --export="${COMMON_EXPORT},METHOD=adult_nsga,CONFIG=${EVAL_CONFIG},PORTFOLIO_CSV=outputs/hpc/adult_nsga2po_7p5m_v3/seed_0/nsga2_po_pareto_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/adult_nsga2po_7p5m_v3" \
  scripts/hpc/run_bios_eval_hpc.slurm)

cat <<EOF
Submitted Adult v3 seed-0 pipeline:
  FairCAPO search:      ${fair_search_job}
  MO-CAPO search:       ${ablation_search_job}
  NSGA-II-PO search:    ${nsga_search_job}
  FairCAPO eval:        ${fair_eval_job}
  MO-CAPO eval:         ${ablation_eval_job}
  NSGA-II-PO eval:      ${nsga_eval_job}

Monitor:
  squeue -u \$USER
  sacct -j ${fair_search_job},${ablation_search_job},${nsga_search_job},${fair_eval_job},${ablation_eval_job},${nsga_eval_job} --format=JobID,JobName%24,State,ExitCode,Elapsed,NodeList

Build tables and figures after all evaluations complete:
  bash scripts/hpc/build_adult_7p5m_v3_large_outputs.sh
EOF
