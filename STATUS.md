# FairCAPO Status

Last updated: 2026-06-28.

## Active Decision

Use **only the large-held-out BBQ diagnostic** for Rocket reporting. Do not cite
or rebuild the standard `outputs/hpc/evaluation/...` BBQ tables/figures because
that smaller eval has coarse fairness resolution and creates misleading ties.

The latest large-held-out `500k_v2` result showed FairCAPO was slightly fairer
than NSGA-II-PO but lost on hypervolume because its Pareto front was too
expensive. Therefore:

- Do **not** run 1M yet.
- Do **not** run seeds 1/2 yet.
- Run **FairCAPO 500k_v3 only** first.
- Compare FairCAPO v3 against the already-completed v2 baselines.
- Rerun ablation/NSGA only if FairCAPO v3 improves enough to justify more GPU.

## Active Reporting Layout

- FairCAPO v3 search: `outputs/hpc/bbq_faircapo_500k_v3/seed_0/`
- FairCAPO v3 large eval: `outputs/hpc/evaluation_large/seed_0/bbq_faircapo_500k_v3/`
- Existing ablation baseline: `outputs/hpc/evaluation_large/seed_0/bbq_ablation_500k_v2/`
- Existing NSGA baseline: `outputs/hpc/evaluation_large/seed_0/bbq_nsga2po_500k_v2/`
- Existing post-hoc baseline: `outputs/hpc/evaluation_large/seed_0/bbq_posthoc_500k_v2/`
- Table output: `outputs/experiment_table/bbq_mistral_hpc_500k_v3_vs_v2_large_seed0/`
- Figure output: `outputs/figures/paper_bbq_hpc_500k_v3_vs_v2_large_seed0/`

## What Changed For FairCAPO 500k_v3

- Lower few-shot pressure:
  - `few_shot_probability: 0.15`
  - `max_few_shot_examples: 2`
- Added short fairness-aware seed prompts.
- Added optional weighted parent-selection tie-breaks so otherwise-incomparable
  candidates prefer cheaper fair prompts.
- Added low-cost protection in environmental selection so cheap candidates are
  not removed before they can be intensified.

## Run FairCAPO 500k_v3

```bash
sbatch --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/phase2_budgeted_mocapo_bbq_HPC.yaml,RUN_TAG=bbq_faircapo_500k_v3 \
  scripts/hpc/run_bbq_hpc.slurm
```

Monitor:

```bash
squeue -u $USER
ls -lt outputs/hpc/logs | head
tail -f outputs/hpc/logs/bbq-faircapo_JobID_seed0.out
```

Expected search output:

```text
outputs/hpc/bbq_faircapo_500k_v3/seed_0/
```

## Evaluate FairCAPO 500k_v3 On Large Held-Out

Run only after the v3 search succeeds:

```bash
sbatch --array=0 --export=ALL,METHOD=faircapo scripts/hpc/run_bbq_eval_hpc.slurm
```

Expected eval output:

```text
outputs/hpc/evaluation_large/seed_0/bbq_faircapo_500k_v3/
```

## Build V3 Table And Figures

Run on the login node after the v3 large eval completes:

```bash
bash scripts/hpc/build_bbq_500k_v3_outputs.sh
```

This compares FairCAPO v3 against existing v2 baselines and writes:

```text
outputs/experiment_table/bbq_mistral_hpc_500k_v3_vs_v2_large_seed0/experiment_table.csv
outputs/figures/paper_bbq_hpc_500k_v3_vs_v2_large_seed0/
```

## Decision Rule

- If FairCAPO v3 beats NSGA v2 on HV or gives a clearly better
  fairness-cost tradeoff, then rerun ablation/NSGA under the same v3 prompt pool
  and reporting path.
- If FairCAPO v3 still loses to NSGA v2, do not run 1M or seeds 1/2. Improve
  the FairCAPO search/intensification further.

## Active Files

- `configs/HPC_Config/phase2_budgeted_mocapo_bbq_HPC.yaml`
- `configs/phase2_prompt_pool_bbq.yaml`
- `configs/HPC_Config/evaluate_pareto_bbq_large_HPC.yaml`
- `configs/HPC_Config/experiment_table_bbq_HPC.yaml`
- `configs/HPC_Config/aggregate_multiseed_bbq_HPC.yaml`
- `scripts/run_phase2_budgeted_mocapo.py`
- `heal_capo/optimizers/parent_selection.py`
- `heal_capo/optimizers/environmental_selection.py`
- `scripts/hpc/run_bbq_hpc.slurm`
- `scripts/hpc/run_bbq_eval_hpc.slurm`
- `scripts/hpc/build_bbq_500k_v3_outputs.sh`
