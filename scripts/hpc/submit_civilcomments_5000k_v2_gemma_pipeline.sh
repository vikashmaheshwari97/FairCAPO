#!/usr/bin/env bash
set -euo pipefail

# Submit the full CivilComments-WILDS 5000k_v2 seed-0 pipeline on Gemma-2-27B.
#
# Clean-trade-off showcase (complements the hard 28-way BIOS showcase):
# binary toxicity, direct-label evaluator (classification_mode: generate),
# label-conditioned identity fairness. Few-shot count is the cost lever, so the
# 5000k v2 budget yields a rich multi-point Pareto front.
#
#   FairCAPO v2 search
#     -> MO-CAPO fairness-off v2 search
#       -> NSGA-II-PO + fairness v2 search
#         -> large-held-out evals for all three
#
# Conservative Rocket mode: one GPU job at a time via afterok dependencies.
# Gemma is served by vLLM via the MODEL_PATH/SERVED_MODEL_NAME/MODEL_FAMILY
# overrides below (the BIOS SLURM scripts are model-agnostic: MODEL_FAMILY=gemma
# skips the Mistral-only vLLM flags).

cd "$(dirname "$0")/../.."

mkdir -p outputs/hpc/logs

# The FairCAPO and NSGA searches use the shared search-only fairness probe.
# Build it once on the login node before submitting:
#   PYTHONPATH=. python scripts/build_civilcomments_fairness_probe.py \
#     --out data/fairness_civilcomments_probe_search_seed0.jsonl \
#     --split train --seed 0 --examples-per-group 5
PROBE=data/fairness_civilcomments_probe_search_seed0.jsonl
test -s "${PROBE}" || {
  echo "Missing CivilComments fairness probe: ${PROBE}" >&2
  echo "Build it first with:" >&2
  echo "  PYTHONPATH=. python scripts/build_civilcomments_fairness_probe.py --out ${PROBE} --split train --seed 0 --examples-per-group 5" >&2
  exit 3
}

probe_lines=$(wc -l < "${PROBE}")
if [[ "${probe_lines}" -ne 80 ]]; then
  echo "CivilComments fairness probe should contain 80 lines, found ${probe_lines}." >&2
  echo "(8 identities x 2 toxicity labels x 5 examples). Rebuild with --examples-per-group 5." >&2
  exit 3
fi

# Gemma serving overrides applied to every GPU job in the chain.
GEMMA_ENV="MODEL_PATH=google/gemma-2-27b-it,SERVED_MODEL_NAME=google/gemma-2-27b-it,MODEL_FAMILY=gemma"

# NOTE: the reused run_bios_*.slurm scripts hardcode #SBATCH --job-name=bios-*,
# so squeue shows "bios-..." by default. --job-name below overrides that header
# per job (SLURM honors the CLI flag over the script header) purely for readable
# monitoring -- it changes nothing about the workload, which is driven by CONFIG
# and the Gemma MODEL_* exports.
echo "Submitting CivilComments FairCAPO 5000k v2 (Gemma) search..."
fair_search_job=$(sbatch --parsable --array=0 --job-name=cc5000k-faircapo \
  --export=ALL,${GEMMA_ENV},CONFIG=configs/HPC_Config/civilcomments_faircapo_5000k_v2_gemma_HPC.yaml,RUN_TAG=civilcomments_faircapo_5000k_v2_gemma \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting CivilComments MO-CAPO fairness-off 5000k v2 (Gemma) search after FairCAPO..."
ablation_search_job=$(sbatch --parsable --dependency=afterok:${fair_search_job} --array=0 --job-name=cc5000k-ablation \
  --export=ALL,${GEMMA_ENV},CONFIG=configs/HPC_Config/civilcomments_ablation_5000k_v2_gemma_HPC.yaml,RUN_TAG=civilcomments_ablation_5000k_v2_gemma \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting CivilComments NSGA-II-PO + fairness 5000k v2 (Gemma) search after ablation..."
nsga_search_job=$(sbatch --parsable --dependency=afterok:${ablation_search_job} --array=0 --job-name=cc5000k-nsga \
  --export=ALL,${GEMMA_ENV},CONFIG=configs/HPC_Config/civilcomments_nsga2po_5000k_v2_gemma_HPC.yaml,RUN_TAG=civilcomments_nsga2po_5000k_v2_gemma \
  scripts/hpc/run_bios_nsga_hpc.slurm)

echo "Submitting FairCAPO large-held-out eval after all searches..."
fair_eval_job=$(sbatch --parsable --dependency=afterok:${nsga_search_job} --array=0 --job-name=cc5000k-fair-eval \
  --export=ALL,${GEMMA_ENV},METHOD=faircapo,CONFIG=configs/HPC_Config/civilcomments_eval_large_5000k_v2_gemma_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/civilcomments_faircapo_5000k_v2_gemma/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/civilcomments_faircapo_5000k_v2_gemma \
  scripts/hpc/run_bios_eval_hpc.slurm)

echo "Submitting MO-CAPO fairness-off large-held-out eval after FairCAPO eval..."
ablation_eval_job=$(sbatch --parsable --dependency=afterok:${fair_eval_job} --array=0 --job-name=cc5000k-abl-eval \
  --export=ALL,${GEMMA_ENV},METHOD=ablation,CONFIG=configs/HPC_Config/civilcomments_eval_ablation_large_5000k_v2_gemma_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/civilcomments_ablation_5000k_v2_gemma/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/civilcomments_ablation_5000k_v2_gemma \
  scripts/hpc/run_bios_eval_hpc.slurm)

echo "Submitting NSGA-II-PO + fairness large-held-out eval after ablation eval..."
nsga_eval_job=$(sbatch --parsable --dependency=afterok:${ablation_eval_job} --array=0 --job-name=cc5000k-nsga-eval \
  --export=ALL,${GEMMA_ENV},METHOD=nsga,CONFIG=configs/HPC_Config/civilcomments_eval_nsga_large_5000k_v2_gemma_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/civilcomments_nsga2po_5000k_v2_gemma/seed_0/nsga2_po_pareto_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/civilcomments_nsga2po_5000k_v2_gemma \
  scripts/hpc/run_bios_eval_hpc.slurm)

cat <<EOF
Submitted CivilComments 5000k_v2 seed-0 pipeline (Gemma-2-27B, direct-label):
  FairCAPO search:              ${fair_search_job}
  MO-CAPO fairness-off search:  ${ablation_search_job}  (afterok:${fair_search_job})
  NSGA-II-PO + fairness search: ${nsga_search_job}  (afterok:${ablation_search_job})
  FairCAPO large eval:          ${fair_eval_job}  (afterok:${nsga_search_job})
  MO-CAPO large eval:           ${ablation_eval_job}  (afterok:${fair_eval_job})
  NSGA large eval:              ${nsga_eval_job}  (afterok:${ablation_eval_job})

Monitor:
  squeue -u \$USER
  ls -lt outputs/hpc/logs | head

After all jobs finish:
  bash scripts/hpc/build_civilcomments_5000k_v2_gemma_large_outputs.sh
EOF
