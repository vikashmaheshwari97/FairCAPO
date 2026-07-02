#!/bin/bash
#SBATCH --job-name=bios-qwen
#SBATCH --output=outputs/hpc/logs/%x_%A_seed%a.out
#SBATCH --error=outputs/hpc/logs/%x_%A_seed%a.err
#SBATCH --array=0
#SBATCH --partition=gpu
## Previous Firefly target:
##SBATCH --nodelist=firefly1
#SBATCH --nodelist=firefly2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=256G
#SBATCH --time=24:00:00

set -euo pipefail

CONFIG="${CONFIG:-configs/HPC_Config/bios_faircapo_qwen_500k_v1_HPC.yaml}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
RUN_TAG="${RUN_TAG:-bios_faircapo_qwen_500k_v1}"
OUT_DIR="outputs/hpc/${RUN_TAG}/seed_${SEED}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
SERVER_PORT="${SERVER_PORT:-$((18000 + (${SLURM_JOB_ID:-0} % 1000) + SEED))}"

mkdir -p "$(dirname "$OUT_DIR")" outputs/hpc/logs

module purge || true
module load cuda/12.1.0
module load python/3.10.19

LOCK_DIR=".venv_install.lock"
until mkdir "${LOCK_DIR}" 2>/dev/null; do
  echo "Waiting for another job to finish .venv setup..."
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
    "vllm>=0.8.5" "transformers>=4.51.0" "huggingface_hub>=0.24" \
    "datasets>=2.20" pandas requests pyyaml numpy pydantic matplotlib altair streamlit
  mkdir -p .venv
  date > .venv/.faircapo_hpc_ready
else
  source .venv/bin/activate
fi

python - <<'PY' >/dev/null 2>&1 || python -m pip install "vllm>=0.8.5" "transformers>=4.51.0" "datasets>=2.20"
import datasets
import transformers
import vllm
PY

rmdir "${LOCK_DIR}" 2>/dev/null || true

export PYTHONPATH=.
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export FAIRCAPO_LLM_API_URL="http://127.0.0.1:${SERVER_PORT}/v1"
export FAIRCAPO_MODEL_ID="${SERVED_MODEL_NAME}"

echo "Starting Qwen vLLM on $(hostname):${SERVER_PORT}"
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port "${SERVER_PORT}" \
  --max-model-len 8192 \
  --dtype auto \
  --enforce-eager \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 &

SERVER_PID=$!

until curl -sf "${FAIRCAPO_LLM_API_URL}/models" >/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "vLLM exited before becoming ready. Check this job's .err log." >&2
    wait "${SERVER_PID}" || true
    exit 4
  fi
  echo "Waiting for Qwen vLLM at ${FAIRCAPO_LLM_API_URL} ..."
  sleep 10
done

echo "Qwen vLLM ready."
echo "config=${CONFIG}"
echo "seed=${SEED}"
echo "out=${OUT_DIR}"
echo "api=${FAIRCAPO_LLM_API_URL}"

python -u scripts/run_phase2_budgeted_mocapo.py \
  --config "${CONFIG}" \
  --seed "${SEED}" \
  --output-dir "${OUT_DIR}"

echo "DONE seed=${SEED} -> ${OUT_DIR}"
