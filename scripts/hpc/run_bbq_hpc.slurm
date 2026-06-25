#!/bin/bash
#SBATCH --job-name=bbq-faircapo
#SBATCH --output=outputs/hpc/logs/%x_%A_seed%a.out
#SBATCH --error=outputs/hpc/logs/%x_%A_seed%a.err
#SBATCH --array=0-2
#SBATCH --partition=gpu
#SBATCH --nodelist=pegasus2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# FairCAPO / MO-CAPO fairness-off BBQ search on UT Rocket Pegasus 2.
#
# UT GPU guidance for pegasus2:
#   - partition: gpu
#   - node: pegasus2
#   - 1 GPU per task
#   - 16 CPU cores per GPU
#
# One SLURM array task runs one seed. Each task starts its own local vLLM server
# on a unique localhost port, then points the project LLM client at that port via
# FAIRCAPO_LLM_API_URL. This avoids port collisions when several array tasks land
# on the same 8-GPU pegasus2 node.
#
# Usage:
#   sbatch scripts/hpc/run_bbq_hpc.slurm
#   sbatch --array=0 --export=CONFIG=configs/HPC_Config/phase2_budgeted_mocapo_bbq_HPC.yaml,RUN_TAG=bbq_faircapo scripts/hpc/run_bbq_hpc.slurm
#   sbatch --export=CONFIG=configs/HPC_Config/mocapo_baseline_bbq_HPC.yaml,RUN_TAG=bbq_ablation scripts/hpc/run_bbq_hpc.slurm

set -euo pipefail

CONFIG="${CONFIG:-configs/HPC_Config/phase2_budgeted_mocapo_bbq_HPC.yaml}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
RUN_TAG="${RUN_TAG:-bbq_faircapo}"
OUT_DIR="outputs/hpc/${RUN_TAG}/seed_${SEED}"

MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-Small-3.2-24B-Instruct-2506}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mistralai/mistral-small-3.2}"
SERVER_PORT="${SERVER_PORT:-$((12000 + (${SLURM_JOB_ID:-0} % 1000) + SEED))}"

mkdir -p "$(dirname "$OUT_DIR")" outputs/hpc/logs

module purge || true
module load cuda || true
module load python || true

if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  python -m venv .venv
  source .venv/bin/activate
fi

python -m pip install --upgrade pip
python -m pip install -e .
if [ -d "third_party/promptolution" ]; then
  python -m pip install -e third_party/promptolution
fi
python -m pip install \
  "vllm>=0.8" "huggingface_hub>=0.24" \
  pandas requests pyyaml numpy pydantic matplotlib altair streamlit

export PYTHONPATH=.
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export FAIRCAPO_LLM_API_URL="http://127.0.0.1:${SERVER_PORT}/v1"
export FAIRCAPO_MODEL_ID="${SERVED_MODEL_NAME}"

echo "Starting vLLM on $(hostname):${SERVER_PORT}"
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port "${SERVER_PORT}" \
  --max-model-len 8192 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 &

SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

until curl -sf "${FAIRCAPO_LLM_API_URL}/models" >/dev/null; do
  echo "Waiting for vLLM at ${FAIRCAPO_LLM_API_URL} ..."
  sleep 10
done

echo "vLLM ready."
echo "config=${CONFIG}"
echo "seed=${SEED}"
echo "out=${OUT_DIR}"
echo "api=${FAIRCAPO_LLM_API_URL}"

python -u scripts/run_phase2_budgeted_mocapo.py \
  --config "${CONFIG}" \
  --seed "${SEED}" \
  --output-dir "${OUT_DIR}"

echo "DONE seed=${SEED} -> ${OUT_DIR}"
