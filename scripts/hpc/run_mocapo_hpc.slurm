#!/bin/bash
#SBATCH --job-name=heal-capo-mocapo
#SBATCH --output=outputs/hpc/logs/%x_%A_seed%a.out
#SBATCH --error=outputs/hpc/logs/%x_%A_seed%a.err
#SBATCH --array=0-2                 # one task per seed -> seeds 0,1,2 (paper uses 3 seeds)
#SBATCH --gres=gpu:1                # mistral-small-3.2 (24B) fits on 1 large GPU; bump for 120B
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# ---------------------------------------------------------------------------
# Paper-scale HEAL-CAPO (MO-CAPO + in-loop fairness) on SUBJ.
# Runs ONE seed per array task (SLURM_ARRAY_TASK_ID), each into its own
# output dir, so the three seeds run in parallel and never overwrite.
#
# ADAPT TO YOUR CLUSTER: the marked (#>>>) blocks are site-specific.
# This job assumes an OpenAI-compatible inference server is reachable at the
# api_url in the config (default http://localhost:1234/v1). The MO-CAPO authors
# served local models with vLLM; an equivalent OpenAI-compatible endpoint works
# because our llm.backend is "lmstudio" (OpenAI Chat Completions protocol).
# ---------------------------------------------------------------------------
set -euo pipefail

CONFIG="${CONFIG:-configs/HPC_Config/phase2_budgeted_mocapo_subj_HPC.yaml}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
RUN_TAG="${RUN_TAG:-subj_mistral}"
OUT_DIR="outputs/hpc/${RUN_TAG}/seed_${SEED}"

mkdir -p "$(dirname "$OUT_DIR")" outputs/hpc/logs

#>>> SITE-SPECIFIC: environment ------------------------------------------------
# module load cuda/12.4 python/3.10            # example
# source .venv/bin/activate                    # project venv (pip install -e .)
#>>> ---------------------------------------------------------------------------

#>>> SITE-SPECIFIC: start an OpenAI-compatible server (uncomment + adapt) -------
# Serve the eval model on port 1234 so it matches llm.api_url in the config.
# python -m vllm.entrypoints.openai.api_server \
#   --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
#   --port 1234 --max-model-len 4096 --dtype auto &
# SERVER_PID=$!
# trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
# # wait for health
# until curl -sf http://localhost:1234/v1/models >/dev/null; do sleep 5; done
#>>> ---------------------------------------------------------------------------

echo "[$(hostname)] config=$CONFIG seed=$SEED out=$OUT_DIR"

PYTHONPATH=. python scripts/run_phase2_budgeted_mocapo.py \
  --config "$CONFIG" \
  --seed "$SEED" \
  --output-dir "$OUT_DIR"

echo "DONE seed=$SEED -> $OUT_DIR"
