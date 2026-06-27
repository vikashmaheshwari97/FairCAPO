# FairCAPO Status

Last updated: 2026-06-28.

## Active Decision

Use **only the large-held-out BBQ diagnostic** for Rocket reporting. Do not cite
or rebuild standard `outputs/hpc/evaluation/...` BBQ results.

The `500k_v2` and `500k_v3` FairCAPO runs did not beat NSGA-II-PO on large
held-out hypervolume because FairCAPO kept an expensive front. The active path
is now **FairCAPO 500k_v4**, which adds a stronger cost-repair stage.

Do not run 1M or seeds 1/2 until FairCAPO v4 improves the seed-0 large-held-out
comparison.

## Active Layout

- FairCAPO v4 search: `outputs/hpc/bbq_faircapo_500k_v4/seed_0/`
- FairCAPO v4 large eval: `outputs/hpc/evaluation_large/seed_0/bbq_faircapo_500k_v4/`
- Retained comparison baselines:
  - `outputs/hpc/evaluation_large/seed_0/bbq_ablation_500k_v2/`
  - `outputs/hpc/evaluation_large/seed_0/bbq_nsga2po_500k_v2/`
  - `outputs/hpc/evaluation_large/seed_0/bbq_posthoc_500k_v2/`
- Table output: `outputs/experiment_table/bbq_mistral_hpc_500k_v4_vs_v2_large_seed0/`
- Figure output: `outputs/figures/paper_bbq_hpc_500k_v4_vs_v2_large_seed0/`

## What Changed For FairCAPO 500k_v4

- Explicit final cost-repair stage:
  - reserves 120k tokens / 20% of budget,
  - creates zero-shot, one-shot, and compact zero-shot variants from the current
    front,
  - evaluates those repair candidates before the final Pareto front is saved.
- Few-shot pressure is lowered to `few_shot_probability: 0.10`.
- v3 cost-aware parent tie-breaks and low-cost environmental protection remain.

## Run FairCAPO 500k_v4

```bash
cd ~/FairCAPO
git pull origin main

sbatch --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/phase2_budgeted_mocapo_bbq_HPC.yaml,RUN_TAG=bbq_faircapo_500k_v4 \
  scripts/hpc/run_bbq_hpc.slurm
```

Expected search output:

```text
outputs/hpc/bbq_faircapo_500k_v4/seed_0/
```

## Evaluate FairCAPO 500k_v4 On Large Held-Out

Run only after the v4 search succeeds:

```bash
sbatch --array=0 --export=ALL,METHOD=faircapo scripts/hpc/run_bbq_eval_hpc.slurm
```

Expected eval output:

```text
outputs/hpc/evaluation_large/seed_0/bbq_faircapo_500k_v4/
```

## Build V4 Table And Figures

```bash
bash scripts/hpc/build_bbq_500k_v4_outputs.sh
```

## Decision Rule

- If FairCAPO v4 beats NSGA v2 on large-held-out HV or gives a clearly better
  fairness-cost tradeoff, then rerun ablation/NSGA under the active v4 code path.
- If FairCAPO v4 still loses to NSGA v2, do not run 1M or seeds 1/2.

## Optional Rocket Cleanup After v4 Is Verified

Do this only after confirming v4 outputs are complete:

```bash
rm -rf outputs/hpc/bbq_faircapo_500k_v3
rm -rf outputs/hpc/evaluation_large/seed_0/bbq_faircapo_500k_v3
rm -rf outputs/experiment_table/bbq_mistral_hpc_500k_v3_vs_v2_large_seed0
rm -rf outputs/figures/paper_bbq_hpc_500k_v3_vs_v2_large_seed0
```

Keep the `500k_v2` ablation/NSGA/post-hoc outputs for comparison until new v4
baselines are explicitly rerun.
