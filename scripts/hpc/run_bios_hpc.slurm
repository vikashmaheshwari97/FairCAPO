#!/bin/bash
#SBATCH --job-name=bios-faircapo
#SBATCH --output=outputs/hpc/logs/%x_%A_seed%a.out
#SBATCH --error=outputs/hpc/logs/%x_%A_seed%a.err
#SBATCH --array=0
#SBATCH --partition=gpu
#SBATCH --nodelist=firefly2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --mem=256G
#SBATCH --time=24:00:00

set -euo pipefail

CONFIG="${CONFIG:-configs/HPC_Config/bios_faircapo_500k_v1_HPC.yaml}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
RUN_TAG="${RUN_TAG:-bios_faircapo_500k_v1}"
OUT_DIR="${OUT_DIR:-outputs/hpc/${RUN_TAG}/seed_${SEED}}"
MOCAPO_RUNNER="${MOCAPO_RUNNER:-scripts/run_phase2_budgeted_mocapo.py}"

MODEL_PATH="${MODEL_PATH:-mistralai/Mistral-Small-3.2-24B-Instruct-2506}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mistralai/mistral-small-3.2}"
MODEL_FAMILY="${MODEL_FAMILY:-mistral}"
MODEL_REVISION="${MODEL_REVISION:-}"
SERVER_PORT="${SERVER_PORT:-$((15000 + (${SLURM_JOB_ID:-0} % 1000) + SEED))}"
VLLM_LIMIT_MM_IMAGE="${VLLM_LIMIT_MM_IMAGE:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

mkdir -p "${OUT_DIR}" outputs/hpc/logs

if [ ! -f "${CONFIG}" ]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 3
fi
if [ ! -f "${MOCAPO_RUNNER}" ]; then
  echo "Missing runner: ${MOCAPO_RUNNER}" >&2
  exit 3
fi

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
  date > .venv/.faircapo_hpc_ready
else
  source .venv/bin/activate
fi

python - <<'PY' >/dev/null 2>&1 || python -m pip install "datasets>=2.20"
import datasets
PY

if [[ "${MODEL_FAMILY}" == "qwen" ]]; then
  python - <<'PY'
from importlib.metadata import version
from packaging.version import Version
assert Version(version("vllm")) >= Version("0.8.5")
assert Version(version("transformers")) >= Version("4.51.0")
PY
fi

rmdir "${LOCK_DIR}" 2>/dev/null || true

export PYTHONPATH=.
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export FAIRCAPO_LLM_API_URL="http://127.0.0.1:${SERVER_PORT}/v1"
export FAIRCAPO_MODEL_ID="${SERVED_MODEL_NAME}"
export CONFIG OUT_DIR MOCAPO_RUNNER MODEL_PATH SERVED_MODEL_NAME MODEL_FAMILY MODEL_REVISION
export SEED MAX_MODEL_LEN MAX_NUM_SEQS GPU_MEMORY_UTILIZATION

# CPU-only preflight and immutable run provenance. This executes before model load
# so a stale manifest, wrong CSV, missing prompt pool, or malformed YAML does not
# consume GPU allocation time.
python - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml


def digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            state.update(chunk)
    return state.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None

config_path = Path(os.environ["CONFIG"])
config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
dev = config.get("dev") or {}
prompt_pool = Path(str(config.get("prompt_pool", ""))) if config.get("prompt_pool") else None
data_path = Path(str(dev.get("data_path", ""))) if dev.get("data_path") else None
manifest_path = Path(str(dev.get("split_manifest", ""))) if dev.get("split_manifest") else None

if prompt_pool is not None and not prompt_pool.is_file():
    raise FileNotFoundError(f"Missing prompt pool: {prompt_pool}")

if str(config.get("dataset") or dev.get("dataset") or "").lower() == "adult":
    if not str(config.get("task_description", "")).strip():
        raise ValueError("Adult config is missing task_description.")
    if data_path is None or manifest_path is None:
        raise ValueError("Adult config must define dev.data_path and dev.split_manifest.")
    from heal_capo.adult_v4_data import load_manifest
    load_manifest(str(manifest_path), data_path=str(data_path), require_fingerprint=True)

try:
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--porcelain"], text=True).splitlines()
except Exception:
    git_sha, git_status = None, []

provenance = {
    "git_sha": git_sha,
    "git_dirty_paths": git_status,
    "config_path": str(config_path),
    "config_sha256": digest(config_path),
    "prompt_pool_path": str(prompt_pool) if prompt_pool else None,
    "prompt_pool_sha256": digest(prompt_pool) if prompt_pool else None,
    "data_path": str(data_path) if data_path else None,
    "data_sha256": digest(data_path) if data_path else None,
    "manifest_path": str(manifest_path) if manifest_path else None,
    "manifest_sha256": digest(manifest_path) if manifest_path else None,
    "runner": os.environ["MOCAPO_RUNNER"],
    "runner_sha256": digest(Path(os.environ["MOCAPO_RUNNER"])),
    "model_path": os.environ["MODEL_PATH"],
    "served_model_name": os.environ["SERVED_MODEL_NAME"],
    "model_family": os.environ["MODEL_FAMILY"],
    "model_revision": os.environ.get("MODEL_REVISION") or None,
    "max_model_len": int(os.environ["MAX_MODEL_LEN"]),
    "max_num_seqs": int(os.environ["MAX_NUM_SEQS"]),
    "gpu_memory_utilization": float(os.environ["GPU_MEMORY_UTILIZATION"]),
    "seed": int(os.environ["SEED"]),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "packages": {
        "python": os.sys.version,
        "vllm": package_version("vllm"),
        "transformers": package_version("transformers"),
        "datasets": package_version("datasets"),
        "torch": package_version("torch"),
    },
}
output = Path(os.environ["OUT_DIR"]) / "run_provenance.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
print(json.dumps(provenance, indent=2))
PY

nvidia-smi -q > "${OUT_DIR}/nvidia_smi.txt" 2>&1 || true

MODEL_ARGS=()
if [[ "${MODEL_FAMILY}" == "mistral" ]]; then
  MODEL_ARGS+=(--tokenizer-mode mistral --config-format mistral --load-format mistral)
  MODEL_ARGS+=(--tool-call-parser mistral --enable-auto-tool-choice)
fi
if [[ -n "${MODEL_REVISION}" ]]; then
  MODEL_ARGS+=(--revision "${MODEL_REVISION}")
fi
if [[ "${VLLM_LIMIT_MM_IMAGE}" =~ ^[1-9][0-9]*$ ]]; then
  MODEL_ARGS+=(--limit-mm-per-prompt "{\"image\":${VLLM_LIMIT_MM_IMAGE}}")
fi

echo "Starting vLLM model_family=${MODEL_FAMILY} on $(hostname):${SERVER_PORT}"
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port "${SERVER_PORT}" \
  "${MODEL_ARGS[@]}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype auto \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" &

SERVER_PID=$!

until curl -sf "${FAIRCAPO_LLM_API_URL}/models" >/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "vLLM exited before becoming ready. Check this job's .err log." >&2
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
echo "runner=${MOCAPO_RUNNER}"
echo "api=${FAIRCAPO_LLM_API_URL}"

python -u "${MOCAPO_RUNNER}" \
  --config "${CONFIG}" \
  --seed "${SEED}" \
  --output-dir "${OUT_DIR}"

echo "DONE seed=${SEED} -> ${OUT_DIR}"
