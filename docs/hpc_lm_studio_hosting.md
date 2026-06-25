# Hosting mistral-small-3.2 (24B) on University HPC

_Last updated: 2026-06-23_

This guide covers serving `mistralai/mistral-small-3.2` (24B parameters) via an
OpenAI-compatible API on a university HPC cluster, so the FairCAPO pipeline can
use it as its evaluation model. The model is served on a GPU node and the
Python scripts connect to it over HTTP (OpenAI Chat Completions protocol).

---

## Quick reference

| Item | Value |
|---|---|
| Model | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` |
| Size | 24B parameters (~48 GB in FP16, ~15 GB in Q4_K_M) |
| Protocol | OpenAI Chat Completions (same as LM Studio) |
| Default port | 1234 |
| Min GPU memory | 48 GB (FP16), 24 GB (4-bit quantized) |
| Min CPU RAM | 64 GB (128 GB recommended with BBQ contexts) |
| Context length | 4096–8192 tokens (BBQ contexts are ~500–1500 chars each) |

---

## Option A: vLLM (recommended for HPC)

vLLM is the production-grade option used by the MO-CAPO authors. It handles
production throughput well and supports PagedAttention for efficient memory use.

### 1. Install

```bash
pip install vllm
```

### 2. Start the server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --port 1234 \
  --max-model-len 8192 \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 4
```

Key flags:
- `--max-model-len 8192`: BBQ prompts with few-shot demos can be long; 8192 is safe.
- `--gpu-memory-utilization 0.90`: leaves 10% headroom.
- `--max-num-seqs 4`: limits concurrency (our pipeline is sequential; keeps memory free).

### 3. Verify

```bash
curl -s http://localhost:1234/v1/models | python -m json.tool
# Should show "mistralai/Mistral-Small-3.2-24B-Instruct-2506"

curl -s http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mistralai/Mistral-Small-3.2-24B-Instruct-2506","messages":[{"role":"user","content":"Say hello."}],"max_tokens":10}'
# Should return a valid completion.
```

---

## Option B: llama.cpp server (lighter weight)

If vLLM is not available on your cluster, llama.cpp provides a lighter-weight
server. You need a quantized GGUF of the model.

### 1. Download a GGUF quant

Download a Q4_K_M quant from HuggingFace (e.g., from bartowski or TheBloke):

```bash
# Example: download via huggingface-cli
huggingface-cli download bartowski/Mistral-Small-3.2-24B-Instruct-2506-GGUF \
  Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf \
  --local-dir /path/to/models/
```

### 2. Build or install llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && make -j8 llama-server
```

### 3. Start the server

```bash
./llama-server \
  -m /path/to/Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf \
  --port 1234 \
  --n-gpu-layers 99 \
  --ctx-size 8192 \
  --host 0.0.0.0
```

Key flags:
- `--n-gpu-layers 99`: offloads 99 layers to GPU (all layers for this model).
- `--ctx-size 8192`: BBQ contexts need headroom.
- `--host 0.0.0.0`: accepts connections from other nodes (not just localhost).

### 4. Verify

Same curl commands as vLLM above. The `/v1/models` endpoint works identically.

---

## Option C: Text Generation Inference (TGI)

Another alternative if your cluster has Docker/Podman + GPU support:

```bash
docker run --gpus all -p 1234:80 \
  -e HF_TOKEN=$HF_TOKEN \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --max-total-tokens 8192
```

---

## HPC-specific considerations

### GPU requirements

| Precision | GPU memory | Example GPU |
|---|---|---|
| FP16 / BF16 | ~48 GB | 1× A100 (80 GB), 1× L40S (48 GB) |
| Q4_K_M (GGUF) | ~15 GB | 1× RTX 6000 Ada (48 GB), 1× A40 (48 GB), 1× L40S |

**Note:** This model does NOT require multi-GPU; it fits on a single modern
datacenter GPU. If your cluster only has smaller GPUs (e.g., RTX 3090 with
24 GB), use the Q4_K_M quant with llama.cpp.

### Memory

Request at least **64 GB CPU RAM with FP16, 128 GB recommended** — BBQ contexts
are long (500–1500 characters each) and few-shot demos accumulate tokens quickly.
The SLURM scripts request 128 GB by default.

### Port / networking

By default the server binds to `localhost:1234`. If your compute nodes cannot
reach each other:

- **Same-node setup (simplest):** Run the inference server AND the python
  pipeline on the SAME compute node. Submit a single SLURM job that starts the
  server in the background, then runs the pipeline. The existing SLURM scripts
  (`scripts/hpc/run_bbq_hpc.slurm`) already have the server-start block
  commented out for this pattern.

- **Multi-node setup:** Start the server on one node with `--host 0.0.0.0`,
  note its hostname, and set `api_url` in the config to
  `http://<server-hostname>:1234/v1`. This requires the nodes to be on the same
  network segment.

### Firewall

If your cluster uses a firewall between nodes, ensure port 1234 is allowed
(or use a different port and update all configs). Most university clusters
allow same-node loopback (`localhost`) without restriction.

### Starting the server inside the SLURM job

The `run_bbq_hpc.slurm` and `run_bbq_nsga_hpc.slurm` scripts include a
commented-out server-start block. Uncomment and adapt it for your cluster:

```bash
# Start the inference server on the allocated GPU
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --port 1234 --max-model-len 8192 --dtype auto &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# Wait for the server to be ready
until curl -sf http://localhost:1234/v1/models >/dev/null; do
  echo "Waiting for inference server..."
  sleep 5
done
echo "Inference server ready."
```

This is the recommended pattern: one SLURM job = one seed run with its own
server. The server starts, the run completes, the server is cleaned up.

### Running the server separately (interactive)

If you prefer to start the server once and run multiple jobs against it:

```bash
# In an interactive session on a GPU node:
srun --gres=gpu:1 --mem=128G --time=48:00:00 --pty bash

# Inside the interactive session:
module load cuda/12.4 python/3.10
source .venv/bin/activate
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --port 1234 --max-model-len 8192
```

Then from another terminal, submit SLURM jobs with the `api_url` pointing to
that node. The server node must remain running for the duration of all jobs.

---

## Pointing the config at the server

All HPC configs in `configs/HPC_Config/` have:

```yaml
llm:
  backend: lmstudio
  model_id: mistralai/mistral-small-3.2
  api_url: http://localhost:1234/v1   # <-- CHANGE THIS for your cluster
```

If your server is on a different host, change `localhost` to the server's
hostname. The `backend: lmstudio` uses the OpenAI Chat Completions protocol,
which all three server options (vLLM, llama.cpp, TGI) speak.

---

## Full BBQ HPC pipeline

Once the inference server is up:

```bash
cd ~/PycharmProjects/Tri_CAPO
source .venv/bin/activate  # or module load ...

# --- Option 1: SLURM (parallel seeds) ---
# FairCAPO (3 seeds)
sbatch scripts/hpc/run_bbq_hpc.slurm

# Ablation (3 seeds)
sbatch --export=CONFIG=configs/HPC_Config/mocapo_baseline_bbq_HPC.yaml,RUN_TAG=bbq_ablation \
  scripts/hpc/run_bbq_hpc.slurm

# NSGA-II-PO (3 seeds)
sbatch scripts/hpc/run_bbq_nsga_hpc.slurm

# --- Option 2: Interactive node (sequential seeds) ---
bash scripts/hpc/sweep_seeds_bbq.sh \
  configs/HPC_Config/phase2_budgeted_mocapo_bbq_HPC.yaml bbq_faircapo 0 1 2

bash scripts/hpc/sweep_seeds_bbq.sh \
  configs/HPC_Config/mocapo_baseline_bbq_HPC.yaml bbq_ablation 0 1 2

bash scripts/hpc/sweep_seeds_nsga.sh \
  configs/HPC_Config/nsga2_po_bbq_HPC.yaml bbq_nsga2po 0 1 2

# --- Held-out evaluation (after all searches complete) ---
python scripts/evaluate_pareto_on_test.py \
  --config configs/HPC_Config/evaluate_pareto_bbq_HPC.yaml

python scripts/evaluate_pareto_on_test.py \
  --config configs/HPC_Config/evaluate_pareto_bbq_ablation_HPC.yaml

python scripts/evaluate_pareto_on_test.py \
  --config configs/HPC_Config/evaluate_pareto_bbq_nsga_HPC.yaml

# --- Aggregate + table + figures (LLM-free) ---
python scripts/aggregate_multiseed.py \
  --base-dirs outputs/hpc/bbq_faircapo outputs/hpc/bbq_ablation \
  outputs/hpc/bbq_nsga2po --seeds 0 1 2

python scripts/build_experiment_table.py \
  --config configs/HPC_Config/experiment_table_bbq_HPC.yaml

python scripts/visualize_paper_figures.py \
  --run outputs/hpc/bbq_faircapo/seed_0 \
  --out outputs/figures/paper_bbq_hpc
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `curl: connection refused` | Server not started | Check server process, wait for `/v1/models` to return 200 |
| `CUDA out of memory` | Model too big for GPU | Use a quantized version (Q4_K_M) or request a bigger GPU |
| `context length exceeded` | BBQ prompt too long | Raise `--max-model-len` to 8192 or reduce `max_few_shot_examples` |
| `timeout` (900s) | Model overloaded or hung | Check GPU utilization (`nvidia-smi`); reduce `max-num-seqs` |
| Run dies silently mid-way | Out of memory (OOM killer) | Increase `--mem` in SLURM script (128 GB minimum for BBQ) |
| `ModuleNotFoundError: No module named 'heal_capo'` | Missing venv / PYTHONPATH | `source .venv/bin/activate && pip install -e .` |

---

## Model download (first time only)

If your cluster restricts internet access from compute nodes, download the model
on a login node first:

```bash
# vLLM / HuggingFace Transformers path:
huggingface-cli download mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --local-dir /scratch/$USER/models/mistral-small-3.2-24b/

# For llama.cpp, download a GGUF quant:
huggingface-cli download bartowski/Mistral-Small-3.2-24B-Instruct-2506-GGUF \
  Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf \
  --local-dir /scratch/$USER/models/
```

Then point the server at the local path instead of the HuggingFace model ID.

---

## Related files

| What | Path |
|---|---|
| FairCAPO HPC config | `configs/HPC_Config/phase2_budgeted_mocapo_bbq_HPC.yaml` |
| Ablation HPC config | `configs/HPC_Config/mocapo_baseline_bbq_HPC.yaml` |
| NSGA-II-PO HPC config | `configs/HPC_Config/nsga2_po_bbq_HPC.yaml` |
| Held-out eval configs | `configs/HPC_Config/evaluate_pareto_bbq{,_nsga,_ablation}_HPC.yaml` |
| Experiment table | `configs/HPC_Config/experiment_table_bbq_HPC.yaml` |
| SLURM — FairCAPO/ablation | `scripts/hpc/run_bbq_hpc.slurm` |
| SLURM — NSGA-II-PO | `scripts/hpc/run_bbq_nsga_hpc.slurm` |
| Interactive sweep — FairCAPO/ablation | `scripts/hpc/sweep_seeds_bbq.sh` |
| Interactive sweep — NSGA | `scripts/hpc/sweep_seeds_nsga.sh` |
| SUBJ HPC config (existing) | `configs/HPC_Config/phase2_budgeted_mocapo_subj_HPC.yaml` |
| SUBJ SLURM / sweep (existing) | `scripts/hpc/run_mocapo_hpc.slurm`, `scripts/hpc/sweep_seeds.sh` |
