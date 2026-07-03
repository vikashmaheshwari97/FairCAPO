#!/usr/bin/env bash
set -euo pipefail

# Submit only the Bias-in-Bios label-aware FairCAPO 1M seed-0 pilot:
#   FairCAPO search -> large-held-out eval
#
# Use this before spending GPU time on 1M ablation/NSGA baselines. If this
# pilot does not improve over BIOS 500k_v1, keep 500k_v1 as the main result.

cd "$(dirname "$0")/../.."

mkdir -p outputs/hpc/logs

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

echo "Submitting Bias-in-Bios label-aware FairCAPO 1M v1 search..."
fair_search_job=$(sbatch --parsable --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/bios_faircapo_labelaware_1m_v1_HPC.yaml,RUN_TAG=bios_faircapo_labelaware_1m_v1 \
  scripts/hpc/run_bios_hpc.slurm)

echo "Submitting Bias-in-Bios label-aware FairCAPO 1M v1 large-held-out eval..."
fair_eval_job=$(sbatch --parsable --dependency=afterok:${fair_search_job} --array=0 \
  --export=ALL,METHOD=faircapo,CONFIG=configs/HPC_Config/bios_eval_large_labelaware_1m_v1_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bios_faircapo_labelaware_1m_v1/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bios_faircapo_labelaware_1m_v1 \
  scripts/hpc/run_bios_eval_hpc.slurm)

cat <<EOF
Submitted Bias-in-Bios label-aware FairCAPO 1M v1 pilot:
  FairCAPO search:     ${fair_search_job}
  FairCAPO large eval: ${fair_eval_job}  (afterok:${fair_search_job})

Monitor:
  squeue -u \$USER
  ls -lt outputs/hpc/logs | head

After it finishes:
  cat outputs/hpc/evaluation_large/seed_0/bios_faircapo_labelaware_1m_v1/test_eval_summary.json
  ls -lh outputs/hpc/bios_faircapo_labelaware_1m_v1/seed_0
EOF
