# FairCAPO Status

Last updated: 2026-07-01.

## Active Decision

Freeze **Bias-in-Bios / Mistral / 500k v1 / seed 0** as the current main result.
Do not replace it with BIOS v2/v3, Qwen seed 0, or label-scoring seed 0.

Main BIOS v1 large-held-out result:

- FairCAPO: accuracy `0.782`, cost `5085.52`, fairness_risk `0.000952`, HV `0.390932`
- MO-CAPO fairness off: accuracy `0.780`, cost `4765.52`, fairness_risk `0.005632`, HV `0.401523`
- NSGA-II-PO + fairness: accuracy `0.782`, cost `4725.84`, fairness_risk `0.010037`, HV `0.406183`

Interpretation: FairCAPO v1 is the strongest fairness result, but accuracy is
stuck around 78%. The next improvement should target accuracy without losing the
fairness advantage.

## Qwen Seed-0 Decision

Qwen3-30B-A3B-Instruct-2507 FairCAPO seed 0 improved accuracy slightly but hurt
fairness:

- Qwen FairCAPO: accuracy `0.788`, fairness_risk `0.016318`
- Mistral FairCAPO v1: accuracy `0.782`, fairness_risk `0.000952`

Decision: **do not run more Qwen jobs yet**. Qwen did not preserve the FairCAPO
fairness advantage.

## Label-Scoring Decision

Two-stage label-scoring FairCAPO seed 0 did not work:

- Label-score FairCAPO: accuracy `0.760`, cost `21628.24`, fairness_risk `0.003082`
- Mistral FairCAPO v1: accuracy `0.782`, cost `5085.52`, fairness_risk `0.000952`

Decision: **do not run more label-scoring jobs yet**. The implementation is now
diagnostic-only. Its result is worse in accuracy, cost, fairness, and HV.

## Current Next Step

Before any new HPC spending, run the config audit locally or on the Rocket login
node:

```bash
python scripts/audit_hpc_configs.py
```

Warnings about Qwen and label-scoring are expected because those files are kept
as diagnostic records. Errors must be fixed before submitting jobs.

The new controlled diagnostic is **BIOS FairCAPO fair-shots 500k v1 seed 0**.
It keeps the frozen Mistral v1 search scale/model and changes only the
few-shot pool selection: label/group balanced examples with diversity-aware
mutation. This is inspired by fairness-guided few-shot prompting, but it is not
the main result unless it beats Mistral v1 on large-held-out evaluation.

The optimizer now also records MO-CAPO-style prompt/block cache reuse in
`budget_summary.json -> block_history`. Use that to check whether real runs are
deepening candidates and reusing repeated prompt/block evaluations as intended.

After the fair-shots diagnostic, the scientifically clean next experiment is
still **multi-seed BIOS v1** if v1 remains best.

## Rocket Commands

Pull the latest code on Rocket:

```bash
cd ~/FairCAPO
git pull origin main
```

Recommended next HPC direction, when ready: run one BIOS FairCAPO fair-shots
diagnostic job, then evaluate on large-held-out only. Do not submit Qwen or
label-scoring jobs.

Search:

```bash
sbatch --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/bios_faircapo_fairshots_500k_v1_HPC.yaml,RUN_TAG=bios_faircapo_fairshots_500k_v1 \
  scripts/hpc/run_bios_hpc.slurm
```

Evaluate after the search succeeds:

```bash
sbatch --array=0 \
  --export=ALL,METHOD=faircapo,CONFIG=configs/HPC_Config/bios_eval_large_fairshots_500k_v1_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bios_faircapo_fairshots_500k_v1/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bios_faircapo_fairshots_500k_v1 \
  scripts/hpc/run_bios_eval_hpc.slurm
```

Monitor:

```bash
squeue -u $USER
ls -lt outputs/hpc/logs | head
```

After fair-shots eval finishes, build the diagnostic table:

```bash
PYTHONPATH=. python scripts/build_experiment_table.py \
  --config configs/HPC_Config/bios_experiment_table_fairshots_500k_v1_large_HPC.yaml
```

Inspect MO-CAPO-style block/cache behavior:

```bash
python - <<'PY'
import json
p = "outputs/hpc/bios_faircapo_fairshots_500k_v1/seed_0/budget_summary.json"
d = json.load(open(p))
print(json.dumps(d.get("block_history", {}), indent=2))
PY
```

Diagnostic table:

```text
outputs/experiment_table/bios_mistral_hpc_fairshots_500k_v1_large_seed0/experiment_table.csv
```

## Decision Rule

- If FairCAPO v1 keeps a lower fairness_risk than NSGA-II-PO across seeds 0/1/2,
  report BIOS v1 as the main fairness result.
- If fair-shots improves accuracy or HV while keeping fairness_risk near v1
  (`~0.001`), promote fair-shots to the new BIOS candidate and rerun baselines.
- If fair-shots worsens fairness like Qwen/label-score did, keep frozen v1.
- If the fairness advantage disappears across seeds, stop and debug the fairness
  objective before trying new models or scorers.
- If accuracy remains near 78%, treat higher accuracy as a separate evaluator
  problem. Do not keep spending HPC on prompt-search tweaks that do not alter
  the evaluator.

## Active BIOS Files

Frozen Mistral v1:

- `configs/HPC_Config/bios_faircapo_500k_v1_HPC.yaml`
- `configs/HPC_Config/bios_ablation_500k_v1_HPC.yaml`
- `configs/HPC_Config/bios_nsga2po_500k_v1_HPC.yaml`
- `configs/HPC_Config/bios_eval_large_500k_v1_HPC.yaml`
- `configs/HPC_Config/bios_eval_ablation_large_500k_v1_HPC.yaml`
- `configs/HPC_Config/bios_eval_nsga_large_500k_v1_HPC.yaml`
- `scripts/hpc/submit_bios_500k_v1_pipeline.sh`
- `scripts/hpc/build_bios_500k_v1_large_outputs.sh`

Diagnostic-only files, not active main-result files:

- `configs/HPC_Config/bios_faircapo_labelscore_500k_v1_HPC.yaml`
- `configs/HPC_Config/bios_eval_large_labelscore_500k_v1_HPC.yaml`
- `configs/HPC_Config/bios_experiment_table_labelscore_500k_v1_large_HPC.yaml`
- `configs/HPC_Config/bios_aggregate_labelscore_500k_v1_large_HPC.yaml`
- `scripts/hpc/submit_bios_labelscore_faircapo_500k_v1_pipeline.sh`
- `scripts/hpc/build_bios_labelscore_500k_v1_large_outputs.sh`
- `configs/HPC_Config/bios_faircapo_qwen_500k_v1_HPC.yaml`
- `configs/HPC_Config/bios_eval_large_qwen_500k_v1_HPC.yaml`
- `configs/HPC_Config/bios_experiment_table_qwen_500k_v1_large_HPC.yaml`
- `scripts/hpc/submit_bios_qwen_faircapo_500k_v1_pipeline.sh`
- `scripts/hpc/build_bios_qwen_500k_v1_large_outputs.sh`
- `configs/HPC_Config/bios_faircapo_fairshots_500k_v1_HPC.yaml`
- `configs/HPC_Config/bios_eval_large_fairshots_500k_v1_HPC.yaml`
- `configs/HPC_Config/bios_experiment_table_fairshots_500k_v1_large_HPC.yaml`

## Archived Context

BBQ work remains in the repository, but current reporting focus is
Bias-in-Bios. BBQ standard held-out tables should not be used for claims; use
large-held-out only when revisiting BBQ.
