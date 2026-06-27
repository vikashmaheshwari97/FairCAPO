# FairCAPO Status

Last updated: 2026-06-27.

## Active Decision

Use **only the large-held-out BBQ diagnostic** for Rocket reporting. Do not cite
or rebuild the standard `outputs/hpc/evaluation/...` BBQ tables/figures because
that smaller eval has coarse fairness resolution and creates misleading ties.

Active reporting layout:

- Search outputs: `outputs/hpc/*_500k_v2/seed_0/`
- Large held-out eval outputs: `outputs/hpc/evaluation_large/seed_0/*_500k_v2/`
- Table output: `outputs/experiment_table/bbq_mistral_hpc_500k_v2_large_seed0/`
- Figure output: `outputs/figures/paper_bbq_hpc_500k_v2_large_seed0/`

Do not move to 1M until the 500k_v2 large-held-out comparison is clear.

## Current Plan

1. Run FairCAPO v2 large-held-out eval:

```bash
sbatch --array=0 \
  --export=ALL,METHOD=faircapo,CONFIG=configs/HPC_Config/evaluate_pareto_bbq_large_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bbq_faircapo_500k_v2/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bbq_faircapo_500k_v2 \
  scripts/hpc/run_bbq_eval_hpc.slurm
```

2. Run ablation v2 large-held-out eval:

```bash
sbatch --array=0 \
  --export=ALL,METHOD=ablation,CONFIG=configs/HPC_Config/evaluate_pareto_bbq_ablation_large_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bbq_ablation_500k_v2/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bbq_ablation_500k_v2 \
  scripts/hpc/run_bbq_eval_hpc.slurm
```

3. Run NSGA-II-PO v2 large-held-out eval:

```bash
sbatch --array=0 \
  --export=ALL,METHOD=nsga,CONFIG=configs/HPC_Config/evaluate_pareto_bbq_nsga_large_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bbq_nsga2po_500k_v2/seed_0/nsga2_po_pareto_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bbq_nsga2po_500k_v2 \
  scripts/hpc/run_bbq_eval_hpc.slurm
```

4. Run post-hoc fairness scoring on the ablation v2 portfolio using the large
   held-out fairness config:

```bash
sbatch --array=0 \
  --export=ALL,FAIRNESS_CONFIG=configs/HPC_Config/evaluate_pareto_bbq_ablation_large_HPC.yaml,INPUT_CSV=outputs/hpc/bbq_ablation_500k_v2/seed_0/phase2_prompt_portfolio.csv,OUTPUT_SUFFIX=_bbqfair_large \
  scripts/hpc/run_bbq_posthoc_hpc.slurm
```

Expected post-hoc portfolio:

```text
outputs/hpc/bbq_ablation_500k_v2/seed_0/phase2_prompt_portfolio_bbqfair_large.csv
```

5. Re-evaluate the post-hoc portfolio on the same large-held-out Dtest basis:

```bash
sbatch --array=0 \
  --export=ALL,METHOD=ablation,CONFIG=configs/HPC_Config/evaluate_pareto_bbq_ablation_large_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bbq_ablation_500k_v2/seed_0/phase2_prompt_portfolio_bbqfair_large.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bbq_posthoc_500k_v2 \
  scripts/hpc/run_bbq_eval_hpc.slurm
```

6. Build the large-held-out v2 table and all figures:

```bash
bash scripts/hpc/build_bbq_500k_v2_outputs.sh
```

This writes:

```text
outputs/experiment_table/bbq_mistral_hpc_500k_v2_large_seed0/experiment_table.csv
outputs/figures/paper_bbq_hpc_500k_v2_large_seed0/
```

## Interpretation Rules

- If FairCAPO large-held-out HV is clearly above NSGA-II-PO and fairness is no
  worse, proceed to seed 1 and seed 2.
- If FairCAPO and NSGA-II-PO are still tied, do not run 1M yet. Improve the
  FairCAPO search operators/intensification first.
- If FairCAPO loses to NSGA-II-PO, stop the Rocket sweep and debug the search.

## Code-Level Improvement Levers To Check Next

These are not changed yet; inspect after the large-held-out v2 table is built.

- Initial prompt pool: add explicitly fairness-aware BBQ instructions if the
  current seed pool is too generic.
- Variation operators: bias mutation/crossover prompts toward reducing
  `max(|sAMB|, |sDIS|)` instead of generic accuracy/cost rewrites.
- Selection/intensification: ensure fairness-first candidates are not pruned
  when accuracy is saturated near 0.98-1.00.
- Fairness eval allocation: spend more in-loop fairness budget on promising
  low-cost candidates where methods currently tie.
- Reporting: use large-held-out `evaluation_large` only; keep search-basis
  richness/trajectory figures separate and clearly labeled.

## Active Files

- `configs/HPC_Config/phase2_budgeted_mocapo_bbq_HPC.yaml`
- `configs/HPC_Config/mocapo_baseline_bbq_HPC.yaml`
- `configs/HPC_Config/nsga2_po_bbq_HPC.yaml`
- `configs/HPC_Config/evaluate_pareto_bbq_large_HPC.yaml`
- `configs/HPC_Config/evaluate_pareto_bbq_ablation_large_HPC.yaml`
- `configs/HPC_Config/evaluate_pareto_bbq_nsga_large_HPC.yaml`
- `configs/HPC_Config/experiment_table_bbq_HPC.yaml`
- `configs/HPC_Config/aggregate_multiseed_bbq_HPC.yaml`
- `scripts/hpc/run_bbq_hpc.slurm`
- `scripts/hpc/run_bbq_eval_hpc.slurm`
- `scripts/hpc/run_bbq_nsga_hpc.slurm`
- `scripts/hpc/run_bbq_posthoc_hpc.slurm`
- `scripts/hpc/build_bbq_500k_v2_outputs.sh`
