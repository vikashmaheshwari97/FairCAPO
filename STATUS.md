# FairCAPO Status

Last updated: 2026-06-30.

## Active Decision

Freeze **Bias-in-Bios / Mistral / 500k v1 / seed 0** as the current main result.
Do not replace it with BIOS v2/v3 or Qwen seed 0.

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

## Current Next Step

Run a controlled two-stage label-scoring FairCAPO seed-0 experiment on Mistral.
This is a new diagnostic, not a replacement for BIOS v1.

Behavior:

1. Stage 1 asks the model for the top 3 likely profession labels.
2. Stage 2 scores only those 3 candidate labels for evidence support.
3. The final prediction is the highest-supported label.

Expected benefit: better Bias-in-Bios classification accuracy and cleaner output
parsing than free-form label generation, while keeping cost below full 28-label
scoring.

## Rocket Commands

Pull the latest code on Rocket:

```bash
cd ~/FairCAPO
git pull origin main
```

Submit only the label-scoring FairCAPO search + large-held-out eval:

```bash
bash scripts/hpc/submit_bios_labelscore_faircapo_500k_v1_pipeline.sh
```

Monitor:

```bash
squeue -u $USER
ls -lt outputs/hpc/logs | head
```

After both jobs finish, build table and figures:

```bash
bash scripts/hpc/build_bios_labelscore_500k_v1_large_outputs.sh
```

Primary table:

```text
outputs/experiment_table/bios_mistral_labelscore_500k_v1_large_seed0/experiment_table.csv
```

Primary figures:

```text
outputs/figures/paper_bios_labelscore_500k_v1_large_seed0/
```

## Decision Rule

- If label-scoring accuracy improves and fairness_risk stays near BIOS v1
  (`~0.001`), keep label scoring and then run baselines or seeds.
- If label-scoring accuracy improves but fairness worsens badly, tune the
  scoring prompts before any multi-seed run.
- If label scoring does not improve accuracy, keep BIOS Mistral v1 as the main
  result and run multi-seed stability for v1.

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

Two-stage label-scoring diagnostic:

- `configs/HPC_Config/bios_faircapo_labelscore_500k_v1_HPC.yaml`
- `configs/HPC_Config/bios_eval_large_labelscore_500k_v1_HPC.yaml`
- `configs/HPC_Config/bios_experiment_table_labelscore_500k_v1_large_HPC.yaml`
- `configs/HPC_Config/bios_aggregate_labelscore_500k_v1_large_HPC.yaml`
- `scripts/hpc/submit_bios_labelscore_faircapo_500k_v1_pipeline.sh`
- `scripts/hpc/build_bios_labelscore_500k_v1_large_outputs.sh`

## Archived Context

BBQ work remains in the repository, but current reporting focus is
Bias-in-Bios. BBQ standard held-out tables should not be used for claims; use
large-held-out only when revisiting BBQ.
