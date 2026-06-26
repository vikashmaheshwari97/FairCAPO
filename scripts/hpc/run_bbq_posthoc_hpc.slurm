#!/bin/bash
#SBATCH --job-name=bbq-posthoc
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
#SBATCH --time=12:00:00

# Post-hoc BBQ fairness measurement on UT Rocket Pegasus 2.
#
# This starts a local vLLM server inside the GPU job, then re-scores an existing
# MO-CAPO fairness-off portfolio with the canonical BBQ bias-score evaluator.
#
# Defaults target the active 1M ablation portfolio:
#   outputs/hpc/bbq_ablation_1m/seed_0/phase2_prompt_portfolio.csv
#
# Useful overrides:
#   FAIRNESS_CONFIG=...
#   INPUT_CSV=...
#   OUTPUT_SUFFIX=_bbqfair_large
#
# 1M standard held-out:
#   sbatch --array=0 scripts/hpc/run_bbq_posthoc_hpc.slurm
#
# Completed 500k standard held-out:
#   sbatch --array=0 --export=ALL,INPUT_CSV=outputs/hpc/bbq_ablation/seed_0/phase2_prompt_portfolio.csv scripts/hpc/run_bbq_posthoc_hpc.slurm
#
# Stage A large-held-out for completed 500k:
#   sbatch --array=0 --export=ALL,FAIRNESS_CONFIG=configs/HPC_Config/evaluate_pareto_bbq_ablation_large_HPC.yaml,INPUT_CSV=outputs/hpc/bbq_ablation/seed_0/phase2_prompt_portfolio.csv,OUTPUT_SUFFIX=_bbqfair_large scripts/hpc/run_bbq_posthoc_hpc.slurm

set -euo pipefail

SEED="${SLURM_ARRAY_TASK_ID:-0}"
FAIRNESS_CONFIG="${FAIRNESS_CONFIG:-configs/HPC_Config/evaluate_pareto_bbq_HPC.yaml}"
INPUT_CSV="${INPUT_CSV:-outputs/hpc/bbq_ablation_1m/seed_${SEED}/phase2_prompt_portfolio.csv}"
PROMPT_COLUMN="${PROMPT_COLUMN:-prompt}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-_bbqfair}"

MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-Small-3.2-24B-Instruct-2506}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mistralai/mistral-small-3.2}"
SERVER_PORT="${SERVER_PORT:-$((15000 + (${SLURM_JOB_ID:-0} % 1000) + SEED))}"
VLLM_LIMIT_MM_IMAGE="${VLLM_LIMIT_MM_IMAGE:-0}"

mkdir -p outputs/hpc/logs

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

if [ ! -f "${FAIRNESS_CONFIG}" ]; then
  echo "Missing fairness config: ${FAIRNESS_CONFIG}" >&2
  exit 3
fi

if [ ! -f "${INPUT_CSV}" ]; then
  echo "Missing input portfolio CSV: ${INPUT_CSV}" >&2
  exit 4
fi

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
    exit 5
  fi
  echo "Waiting for vLLM at ${FAIRCAPO_LLM_API_URL} ..."
  sleep 10
done

echo "vLLM ready."
echo "seed=${SEED}"
echo "fairness_config=${FAIRNESS_CONFIG}"
echo "input_csv=${INPUT_CSV}"
echo "output_suffix=${OUTPUT_SUFFIX}"
echo "api=${FAIRCAPO_LLM_API_URL}"

python -u scripts/measure_baseline_fairness.py \
  --fairness-config "${FAIRNESS_CONFIG}" \
  --inputs "${INPUT_CSV}" \
  --prompt-column "${PROMPT_COLUMN}" \
  --output-suffix "${OUTPUT_SUFFIX}"

echo "DONE posthoc seed=${SEED} input=${INPUT_CSV}"
