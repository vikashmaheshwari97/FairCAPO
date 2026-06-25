#!/bin/bash
#SBATCH --job-name=bbq-faircapo
#SBATCH --output=outputs/hpc/logs/%x_%A_seed%a.out
#SBATCH --error=outputs/hpc/logs/%x_%A_seed%a.err
#SBATCH --array=0
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
# One SLURM array task runs one seed. The default is seed 0 only for cost
# control; later submit with --array=0-2 for the three-seed paper sweep.
# Each task starts its own local vLLM server
# on a unique localhost port, then points the project LLM client at that port via
# FAIRCAPO_LLM_API_URL. This avoids port collisions when several array tasks land
# on the same 8-GPU pegasus2 node.
#
# Usage:
#   sbatch scripts/hpc/run_bbq_hpc.slurm
#   sbatch --array=0 --export=ALL,CONFIG=configs/HPC_Config/phase2_budgeted_mocapo_bbq_HPC.yaml,RUN_TAG=bbq_faircapo scripts/hpc/run_bbq_hpc.slurm
#   sbatch --array=0-2 --export=ALL,CONFIG=configs/HPC_Config/phase2_budgeted_mocapo_bbq_HPC.yaml,RUN_TAG=bbq_faircapo scripts/hpc/run_bbq_hpc.slurm
#   sbatch --array=0 --export=ALL,CONFIG=configs/HPC_Config/mocapo_baseline_bbq_HPC.yaml,RUN_TAG=bbq_ablation scripts/hpc/run_bbq_hpc.slurm

set -euo pipefail

CONFIG="${CONFIG:-configs/HPC_Config/phase2_budgeted_mocapo_bbq_HPC.yaml}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
RUN_TAG="${RUN_TAG:-bbq_faircapo}"
OUT_DIR="outputs/hpc/${RUN_TAG}/seed_${SEED}"

MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-Small-3.2-24B-Instruct-2506}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mistralai/mistral-small-3.2}"
SERVER_PORT="${SERVER_PORT:-$((12000 + (${SLURM_JOB_ID:-0} % 1000) + SEED))}"
VLLM_LIMIT_MM_IMAGE="${VLLM_LIMIT_MM_IMAGE:-0}"

mkdir -p "$(dirname "$OUT_DIR")" outputs/hpc/logs

module purge || true
module load cuda/12.1.0
module load python/3.10.19

LOCK_DIR=".venv_install.lock"
until mkdir "${LOCK_DIR}" 2>/dev/null; do
  echo "Waiting for another array task to finish .venv setup..."
  sleep 10
done
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true; kill ${SERVER_PID:-} 2>/dev/null || true' EXIT

if [ ! -f ".venv/.faircapo_hpc_ready" ]; then
  python -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -e .
  if [ -d "third_party/promptolution" ]; then
    python -m pip install -e third_party/promptolution
  fi
  python -m pip install \
    "vllm>=0.8" "huggingface_hub>=0.24" \
    pandas requests pyyaml numpy pydantic matplotlib altair streamlit
  mkdir -p .venv
  date > .venv/.faircapo_hpc_ready
else
  source .venv/bin/activate
fi

rmdir "${LOCK_DIR}" 2>/dev/null || true

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
  --tokenizer-mode mistral \
  --config-format mistral \
  --load-format mistral \
  --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --limit-mm-per-prompt "{\"image\":${VLLM_LIMIT_MM_IMAGE}}" \
  --max-model-len 8192 \
  --dtype auto \
  --enforce-eager \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 &

SERVER_PID=$!

until curl -sf "${FAIRCAPO_LLM_API_URL}/models" >/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "vLLM exited before becoming ready. Check this job's .err log above." >&2
    wait "${SERVER_PID}" || true
    exit 4
  fi
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
