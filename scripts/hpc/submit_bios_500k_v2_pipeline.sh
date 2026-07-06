#!/usr/bin/env bash
set -euo pipefail

# Submit the full Bias-in-Bios 500k_v2 seed-0 pipeline (reason-then-label CoT).
#
# v2 = the frozen label-conditioned search regime + a reasoning-then-answer
# classification evaluator (classification_mode: reason_then_label, num_predict
# 384). Targets the ~78% v1 accuracy ceiling without changing the fairness
# objective.
#
#   FairCAPO v2 search
#     -> MO-CAPO fairness-off v2 search
#       -> NSGA-II-PO + fairness v2 search
#         -> large-held-out evals for all three
#
# Conservative Rocket mode: one GPU job at a time via afterok dependencies.

cd "$(dirname "$0")/../.."

mkdir -p outputs/hpc/logs

# The FairCAPO and NSGA v2 searches use the shared search-only fairness probe.
# Build it once on the login node before submitting (same probe as label-aware v1).
test -s data/fairness_bios_probe_search_seed0.jsonl || {
  echo "Missing BIOS fairness probe: data/fairness_bios_probe_search_seed0.jsonl" >&2
  echo "Build it first with:" >&2
  echo "  PYTHONPATH=. python scripts/build_bios_fairness_probe.py --out data/fairness_bios_probe_search_seed0.jsonl --split train --seed 0 --examples-per-group 5 --max-labels 8" >&2
  exit 3
}

probe_lines=$(wc -l < data/fairness_bios_probe_search_seed0.jsonl)
if [[ "${probe_lines}" -ne 80 ]]; then
  echo "BIOS fairness probe should contain 80 lines, found ${probe_lines}." >&2
  exit 3
fi

echo "Submitting Bias-in-Bios FairCAPO 500k v2 search..."
fair_search_job=$(sbatch --parsable --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/bios_faircapo_500k_v2_HPC.yaml,RUN_TAG=bios_faircapo_500k_v2 \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting Bias-in-Bios MO-CAPO fairness-off 500k v2 search after FairCAPO..."
ablation_search_job=$(sbatch --parsable --dependency=afterok:${fair_search_job} --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/bios_ablation_500k_v2_HPC.yaml,RUN_TAG=bios_ablation_500k_v2 \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting Bias-in-Bios NSGA-II-PO + fairness 500k v2 search after ablation..."
nsga_search_job=$(sbatch --parsable --dependency=afterok:${ablation_search_job} --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/bios_nsga2po_500k_v2_HPC.yaml,RUN_TAG=bios_nsga2po_500k_v2 \
  scripts/hpc/run_bios_nsga_hpc.slurm)

echo "Submitting FairCAPO v2 large-held-out eval after all searches..."
fair_eval_job=$(sbatch --parsable --dependency=afterok:${nsga_search_job} --array=0 \
  --export=ALL,METHOD=faircapo,CONFIG=configs/HPC_Config/bios_eval_large_500k_v2_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bios_faircapo_500k_v2/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bios_faircapo_500k_v2 \
  scripts/hpc/run_bios_eval_hpc.slurm)

echo "Submitting MO-CAPO fairness-off v2 large-held-out eval after FairCAPO eval..."
ablation_eval_job=$(sbatch --parsable --dependency=afterok:${fair_eval_job} --array=0 \
  --export=ALL,METHOD=ablation,CONFIG=configs/HPC_Config/bios_eval_ablation_large_500k_v2_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bios_ablation_500k_v2/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bios_ablation_500k_v2 \
  scripts/hpc/run_bios_eval_hpc.slurm)

echo "Submitting NSGA-II-PO + fairness v2 large-held-out eval after ablation eval..."
nsga_eval_job=$(sbatch --parsable --dependency=afterok:${ablation_eval_job} --array=0 \
  --export=ALL,METHOD=nsga,CONFIG=configs/HPC_Config/bios_eval_nsga_large_500k_v2_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bios_nsga2po_500k_v2/seed_0/nsga2_po_pareto_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bios_nsga2po_500k_v2 \
  scripts/hpc/run_bios_eval_hpc.slurm)

cat <<EOF
Submitted Bias-in-Bios 500k_v2 seed-0 pipeline (reason-then-label CoT):
  FairCAPO v2 search:           ${fair_search_job}
  MO-CAPO fairness-off search:  ${ablation_search_job}  (afterok:${fair_search_job})
  NSGA-II-PO + fairness search: ${nsga_search_job}  (afterok:${ablation_search_job})
  FairCAPO v2 large eval:       ${fair_eval_job}  (afterok:${nsga_search_job})
  MO-CAPO v2 large eval:        ${ablation_eval_job}  (afterok:${fair_eval_job})
  NSGA v2 large eval:           ${nsga_eval_job}  (afterok:${ablation_eval_job})

Monitor:
  squeue -u \$USER
  ls -lt outputs/hpc/logs | head

After all jobs finish:
  bash scripts/hpc/build_bios_500k_v2_large_outputs.sh
EOF
