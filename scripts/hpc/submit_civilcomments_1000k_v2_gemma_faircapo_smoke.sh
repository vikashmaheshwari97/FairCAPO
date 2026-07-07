#!/usr/bin/env bash
set -euo pipefail

# SMOKE TEST: submit ONLY the FairCAPO search + its large-held-out eval for the
# CivilComments-WILDS 1000k_v2 seed-0 experiment on Gemma-2-27B.
#
# Purpose: validate the whole path end-to-end cheaply (does Gemma serve, does the
# search explore a real front at 1M, does the held-out eval produce a sane
# accuracy/fairness number) BEFORE spending GPU on the ablation + NSGA arms.
# If this pair succeeds, submit the full pipeline for the remaining 4 jobs with:
#   bash scripts/hpc/submit_civilcomments_1000k_v2_gemma_remaining_after_faircapo.sh
# (or just re-run the full pipeline script -- the FairCAPO outputs are reused).
#
# Two GPU jobs, one at a time via an afterok dependency:
#   FairCAPO v2 search  ->  FairCAPO large-held-out eval
#
# Gemma is served by vLLM via MODEL_PATH/SERVED_MODEL_NAME/MODEL_FAMILY (the BIOS
# SLURM scripts are model-agnostic: MODEL_FAMILY=gemma skips Mistral-only flags).

cd "$(dirname "$0")/../.."

mkdir -p outputs/hpc/logs

# The FairCAPO search uses the shared search-only fairness probe. Build it once
# on the login node before submitting:
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

# Gemma serving overrides applied to both GPU jobs.
GEMMA_ENV="MODEL_PATH=google/gemma-2-27b-it,SERVED_MODEL_NAME=google/gemma-2-27b-it,MODEL_FAMILY=gemma"

echo "Submitting CivilComments FairCAPO 1000k v2 (Gemma) search..."
fair_search_job=$(sbatch --parsable --array=0 --job-name=cc1000k-faircapo \
  --export=ALL,${GEMMA_ENV},CONFIG=configs/HPC_Config/civilcomments_faircapo_1000k_v2_gemma_HPC.yaml,RUN_TAG=civilcomments_faircapo_1000k_v2_gemma \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting FairCAPO large-held-out eval after the FairCAPO search..."
fair_eval_job=$(sbatch --parsable --dependency=afterok:${fair_search_job} --array=0 --job-name=cc1000k-fair-eval \
  --export=ALL,${GEMMA_ENV},METHOD=faircapo,CONFIG=configs/HPC_Config/civilcomments_eval_large_1000k_v2_gemma_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/civilcomments_faircapo_1000k_v2_gemma/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/civilcomments_faircapo_1000k_v2_gemma \
  scripts/hpc/run_bios_eval_hpc.slurm)

cat <<EOF
Submitted CivilComments 1000k_v2 seed-0 FairCAPO SMOKE TEST (Gemma-2-27B):
  FairCAPO search:      ${fair_search_job}
  FairCAPO large eval:  ${fair_eval_job}  (afterok:${fair_search_job})

Monitor:
  squeue -u \$USER
  ls -lt outputs/hpc/logs | head

When both finish, sanity-check the held-out number:
  cat outputs/hpc/evaluation_large/seed_0/civilcomments_faircapo_1000k_v2_gemma/test_eval_summary.json

If it looks good, submit the remaining 4 jobs (ablation + NSGA searches + their
evals). The FairCAPO search/eval outputs are reused, not recomputed.
EOF
