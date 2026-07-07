#!/usr/bin/env bash
set -euo pipefail

# Submit the REMAINING 4 jobs of the CivilComments-WILDS 1000k_v2 seed-0 pipeline
# on Gemma-2-27B, AFTER the FairCAPO smoke test (search + eval) has succeeded:
#   scripts/hpc/submit_civilcomments_1000k_v2_gemma_faircapo_smoke.sh
#
# This adds the two baseline arms and their held-out evals:
#   MO-CAPO fairness-off v2 search
#     -> NSGA-II-PO + fairness v2 search
#       -> MO-CAPO fairness-off large eval
#         -> NSGA-II-PO + fairness large eval
#
# The FairCAPO search + eval are NOT resubmitted -- their outputs are reused when
# you build the final table/figures. One GPU job at a time via afterok.
#
# Precondition: the FairCAPO smoke test finished OK, i.e. these exist:
#   outputs/hpc/civilcomments_faircapo_1000k_v2_gemma/seed_0/phase2_prompt_portfolio.csv
#   outputs/hpc/evaluation_large/seed_0/civilcomments_faircapo_1000k_v2_gemma/test_eval_summary.json

cd "$(dirname "$0")/../.."

mkdir -p outputs/hpc/logs

# Guard: refuse to submit the baselines if the FairCAPO smoke test didn't land.
FAIR_PORTFOLIO=outputs/hpc/civilcomments_faircapo_1000k_v2_gemma/seed_0/phase2_prompt_portfolio.csv
FAIR_EVAL=outputs/hpc/evaluation_large/seed_0/civilcomments_faircapo_1000k_v2_gemma/test_eval_summary.json
test -s "${FAIR_PORTFOLIO}" || {
  echo "Missing FairCAPO portfolio: ${FAIR_PORTFOLIO}" >&2
  echo "Run the FairCAPO smoke test first and let it finish." >&2
  exit 3
}
test -s "${FAIR_EVAL}" || {
  echo "Missing FairCAPO held-out eval summary: ${FAIR_EVAL}" >&2
  echo "Let the FairCAPO smoke-test eval finish before submitting the baselines." >&2
  exit 3
}

# The NSGA search reuses the shared search-only fairness probe (same as FairCAPO).
PROBE=data/fairness_civilcomments_probe_search_seed0.jsonl
test -s "${PROBE}" || { echo "Missing fairness probe: ${PROBE}" >&2; exit 3; }
probe_lines=$(wc -l < "${PROBE}")
if [[ "${probe_lines}" -ne 80 ]]; then
  echo "CivilComments fairness probe should contain 80 lines, found ${probe_lines}." >&2
  exit 3
fi

GEMMA_ENV="MODEL_PATH=google/gemma-2-27b-it,SERVED_MODEL_NAME=google/gemma-2-27b-it,MODEL_FAMILY=gemma"

echo "Submitting CivilComments MO-CAPO fairness-off 1000k v2 (Gemma) search..."
ablation_search_job=$(sbatch --parsable --array=0 --job-name=cc1000k-ablation \
  --export=ALL,${GEMMA_ENV},CONFIG=configs/HPC_Config/civilcomments_ablation_1000k_v2_gemma_HPC.yaml,RUN_TAG=civilcomments_ablation_1000k_v2_gemma \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting CivilComments NSGA-II-PO + fairness 1000k v2 (Gemma) search after ablation..."
nsga_search_job=$(sbatch --parsable --dependency=afterok:${ablation_search_job} --array=0 --job-name=cc1000k-nsga \
  --export=ALL,${GEMMA_ENV},CONFIG=configs/HPC_Config/civilcomments_nsga2po_1000k_v2_gemma_HPC.yaml,RUN_TAG=civilcomments_nsga2po_1000k_v2_gemma \
  scripts/hpc/run_bios_nsga_hpc.slurm)

echo "Submitting MO-CAPO fairness-off large-held-out eval after NSGA search..."
ablation_eval_job=$(sbatch --parsable --dependency=afterok:${nsga_search_job} --array=0 --job-name=cc1000k-abl-eval \
  --export=ALL,${GEMMA_ENV},METHOD=ablation,CONFIG=configs/HPC_Config/civilcomments_eval_ablation_large_1000k_v2_gemma_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/civilcomments_ablation_1000k_v2_gemma/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/civilcomments_ablation_1000k_v2_gemma \
  scripts/hpc/run_bios_eval_hpc.slurm)

echo "Submitting NSGA-II-PO + fairness large-held-out eval after ablation eval..."
nsga_eval_job=$(sbatch --parsable --dependency=afterok:${ablation_eval_job} --array=0 --job-name=cc1000k-nsga-eval \
  --export=ALL,${GEMMA_ENV},METHOD=nsga,CONFIG=configs/HPC_Config/civilcomments_eval_nsga_large_1000k_v2_gemma_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/civilcomments_nsga2po_1000k_v2_gemma/seed_0/nsga2_po_pareto_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/civilcomments_nsga2po_1000k_v2_gemma \
  scripts/hpc/run_bios_eval_hpc.slurm)

cat <<EOF
Submitted the remaining CivilComments 1000k_v2 seed-0 baseline jobs (Gemma-2-27B):
  MO-CAPO fairness-off search:  ${ablation_search_job}
  NSGA-II-PO + fairness search: ${nsga_search_job}  (afterok:${ablation_search_job})
  MO-CAPO large eval:           ${ablation_eval_job}  (afterok:${nsga_search_job})
  NSGA large eval:              ${nsga_eval_job}  (afterok:${ablation_eval_job})

(FairCAPO search + eval from the smoke test are reused, not recomputed.)

Monitor:
  squeue -u \$USER
  ls -lt outputs/hpc/logs | head

After all jobs finish -- table + figures (login node, no GPU):
  bash scripts/hpc/build_civilcomments_1000k_v2_gemma_large_outputs.sh
EOF
