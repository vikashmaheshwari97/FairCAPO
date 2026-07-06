# FairCAPO — Project Status

_Last updated: 2026-07-06._

> **HEADLINE.** FairCAPO = **MO-CAPO + an in-loop fairness objective** (accuracy↑, cost↓, risk↓,
> fairness_risk↓). MO-CAPO tunes a prompt for accuracy + low cost and returns a *menu* (Pareto front)
> of trade-offs; FairCAPO adds fairness as a 4th goal measured with the real model **during** search,
> so fairness genuinely decides which prompts survive.
>
> **Current showcase = Bias-in-Bios (BIOS)** on Mistral-Small-3.2 (24B), run on the **Rocket HPC**
> cluster (SLURM + vLLM). BBQ is archived; SUBJ is the no-harm control. Adult-income was removed.

**Naming.** "FairCAPO" is the method/display name. The Python package is still `heal_capo/`
(renaming buys nothing). Older notes say "HEAL-CAPO" — treat as a synonym.

---

## The frozen main result — BIOS / Mistral / 500k v1 / seed 0

| method | accuracy | cost | fairness_risk | HV |
|---|---|---|---|---|
| FairCAPO | **0.782** | 5085.52 | **0.000952** | 0.390932 |
| MO-CAPO (fairness off) | 0.780 | 4765.52 | 0.005632 | 0.401523 |
| NSGA-II-PO + fairness | 0.782 | 4725.84 | 0.010037 | 0.406183 |

**Read:** FairCAPO is the clear fairness winner (~10× lower fairness_risk than NSGA), but **accuracy is
stuck ~78%** and NSGA edges it on HV. The open problem = **raise accuracy without losing the fairness
advantage.**

**Ruled-out dead ends (do NOT re-run — kept as diagnostic-only configs):** Qwen3-30B (accuracy ↑ a
little, fairness ↓ to 0.016), two-stage label-scoring (worse everywhere), fair-shots (better cost/HV
but lost the fairness edge).

---

## ▶️ Active work: BIOS 500k **v2** (reason-then-label CoT) — pushed, not yet run

Diagnosis of the 78% ceiling: v1 forced a 24B model to do **28-way** classification with a **32-token
output cap and no reasoning**, plus a fragile label extractor. v2 attacks the *evaluator*, the lever
STATUS.md called out ("higher accuracy is a separate evaluator problem").

**What v2 changes (committed `3e9b4da3`):**
1. **`classification_mode: reason_then_label`** in the runner — the model reasons briefly about
   occupational evidence, then commits the label inside `<final_answer>` tags. `num_predict` 32 → 384.
2. **Hardened `extract_label`** — reads only the `<final_answer>` span, word-boundary matching (no
   incidental `model`/`poet` substring hits), prefers last-asserted/longer label, tolerates a
   truncated closing tag. 12 new tests; 439 pass (1 pre-existing unrelated failure).
3. **Enhanced prompt pool** (`phase2_prompt_pool_bios_enhanced.yaml`) — v1's 12 prompts + 8
   accuracy-focused seeds (confusion pairs, exact boundary rules, label taxonomy, negative evidence).
4. **v2 config family** templated on the label-aware v1 regime: label-conditioned group fairness +
   shared search probe + `stratify_group_key: gender`, split 280/112/500, held-out `test_size 1000`.
   All pass `audit_hpc_configs.py` with zero findings. NSGA shares FairCAPO's evaluator → identical
   CoT path → fair comparison.

**Caveat:** CoT spends more output tokens, so under the same 500k budget the search explores fewer
candidates. If the front looks thin, raise budget to 1M. Held identical to v1 otherwise for a clean
single-variable comparison.

**Next:** run the v2 pipeline on Rocket, compare `performance` vs 0.782. If it clears ~85%, extend to
3 seeds. Ideas parked (not started): try **Gemma-2-27B**, and add **CivilComments/WILDS** as a
binary toxicity+subgroup dataset for a richer multi-point Pareto front.

---

## How the fairness extension works (plain English)

MO-CAPO tunes a prompt for **accuracy + low cost** and returns a *menu* (Pareto front) of trade-offs.
**FairCAPO adds fairness as a 4th goal.**

**Measuring `fairness_risk` (0 = perfectly fair):**
- **Group tasks (Bias-in-Bios, the showcase):** the model classifies a biography into one of 28
  professions; **gender** is a protected attribute used only for evaluation. `fairness_risk` =
  **group accuracy gap** (difference in accuracy between gender groups), optionally
  **label-conditioned** (per-profession gap, so an aggregate gap can't hide that e.g. *surgeon* or
  *nurse* is unfair while totals look balanced). Code: `heal_capo/fairness.py`.
- **Counterfactual tasks (SUBJ):** swap one demographic detail ("He/She is a nurse"); if the answer
  flips when it shouldn't, that's an unfair flip. Blend: flip 0.50 / group-gap 0.25 / bias-language
  0.15 / decayed debt 0.10 → one `fairness_risk`. Code: `heal_capo/fairness.py`.
- **BBQ (archived):** canonical BBQ bias score (sAMB = ambiguous-context bias, sDIS = disambiguated).
  `fairness_risk` distilled via `fairness_bbq.bbq_score`. Code: `heal_capo/fairness_bbq.py`.

**In-loop steering:** during search, every candidate's `fairness_risk` is measured with the real model
(cached, cost charged to budget) and added as the 4th objective. The SAME MO-CAPO
Pareto/intensification/selection machinery runs in 4-D, so fairness genuinely decides which prompts
survive. Wired in `LLMObjectiveEvaluator` (`scripts/run_phase2_budgeted_mocapo.py`) behind
`fairness.in_loop`. For BIOS the search uses a fixed, support-balanced **probe** set
(`data/fairness_bios_probe_search_seed0.jsonl`, 80 items); the held-out eval measures fairness on the
joint test split instead.

**Why BIOS is the showcase:** its group accuracy gap moves with the prompt and there is real headroom.
SUBJ is near the 24B ceiling (`fairness_risk ≈ 0`) so it's the no-harm control; BBQ pins accuracy at
1.0 so it can't show an accuracy/fairness/cost trade-off.

---

## Comparison design (DECIDED — don't re-litigate)

- **FairCAPO = MO-CAPO + in-loop fairness = the METHOD**, not a baseline.
- Baselines: **MO-CAPO fairness-OFF** (ablation, isolates the fairness objective) + **NSGA-II-PO +
  fairness** (primary algorithmic baseline = MO-CAPO §4.2: NSGA-II + CAPO operators on the SAME
  objectives, only the search algorithm differs) + **post-hoc fairness** (in-loop vs post-process).
- **Self-contained:** all methods run by us on the same model/budget/eval set; we compare among our
  own runs, NOT against the MO-CAPO paper's numbers (they have no fairness).
- **Integrity guardrail:** never handicap the baseline (no lowering NSGA accuracy / inflating cost).
  Legit moves only: make the comparison genuinely FAIR, or genuinely IMPROVE FairCAPO.

---

## How to run the BIOS v2 pipeline on Rocket (HPC)

```bash
cd ~/FairCAPO && git pull origin main

# 1. Build the shared search-only fairness probe (login node, once; ~1 min, no GPU)
PYTHONPATH=. python scripts/build_bios_fairness_probe.py \
  --out data/fairness_bios_probe_search_seed0.jsonl \
  --split train --seed 0 --examples-per-group 5 --max-labels 8
wc -l data/fairness_bios_probe_search_seed0.jsonl        # must be 80

# 2. Audit (v2 rows must be clean; pre-existing _guided v1 errors are unrelated)
python scripts/audit_hpc_configs.py
python scripts/audit_hpc_configs.py 2>&1 | grep 500k_v2   # should print nothing

# 3. Submit the full pipeline (6 chained jobs: 3 searches -> 3 held-out evals, one GPU at a time)
bash scripts/hpc/submit_bios_500k_v2_pipeline.sh
squeue -u $USER ; ls -lt outputs/hpc/logs | head

# 4. After all 6 jobs finish — table + figures (login node, no GPU)
bash scripts/hpc/build_bios_500k_v2_large_outputs.sh
column -s, -t outputs/experiment_table/bios_mistral_hpc_500k_v2_large_seed0/experiment_table.csv | less -S
```

SLURM headers pin `--partition=gpu --nodelist=firefly2`; confirm those are still valid on Rocket
before submitting or the jobs sit pending.

---

## Key files & configs

| What | Path |
|---|---|
| Budgeted optimizer (runner) | `scripts/run_phase2_budgeted_mocapo.py` + `heal_capo/optimizers/*` |
| NSGA-II-PO baseline | `scripts/run_baseline_nsga2_po.py`, `baselines/nsga2_po_runner.py` (shares the runner's evaluator for BIOS) |
| Held-out evaluator | `scripts/evaluate_pareto_on_test.py` (`--seed`, `--portfolio-csv`) |
| Experiment table / aggregate | `scripts/build_experiment_table.py`, `scripts/aggregate_multiseed.py` |
| Fairness / risk core | `heal_capo/fairness.py` (group_accuracy_gap, label_conditioned_group_accuracy_gap, DSP/EO/EOdds), `heal_capo/risk.py` |
| BBQ bias score (archived) | `heal_capo/fairness_bbq.py` |
| BIOS loader | `experiments/datasets.py` (`load_bias_in_bios`, 28 profession labels, gender metadata) |
| BIOS fairness probe builder | `scripts/build_bios_fairness_probe.py` → `data/fairness_bios_probe_search_seed0.jsonl` |
| BIOS zero-HPC fairness diagnostic | `scripts/diagnose_bios_fairness.py` (label-conditioned gap, worst_label) |
| Config auditor | `scripts/audit_hpc_configs.py` (gate before GPU spend) |
| **BIOS v2 configs** | `configs/HPC_Config/bios_{faircapo,ablation,nsga2po}_500k_v2_HPC.yaml`, `bios_eval_{large,ablation_large,nsga_large}_500k_v2_HPC.yaml`, `bios_experiment_table_500k_v2_large_HPC.yaml`, `bios_aggregate_500k_v2_large_HPC.yaml` |
| **BIOS prompt pools** | `configs/phase2_prompt_pool_bios.yaml`, `configs/phase2_prompt_pool_bios_enhanced.yaml` (v2 uses enhanced) |
| **BIOS v2 HPC pipeline** | `scripts/hpc/submit_bios_500k_v2_pipeline.sh`, `scripts/hpc/build_bios_500k_v2_large_outputs.sh` |
| SLURM job scripts | `scripts/hpc/run_bios_hpc.slurm`, `run_bios_nsga_hpc.slurm`, `run_bios_eval_hpc.slurm` |
| MO metrics (HV opt/pes, nR2, Gap) | `heal_capo/evaluation/mo_metrics.py` |
| Reference gap analysis vs paper | `docs/mocapo_gap_analysis_S12.md` |

---

## Environment

- **HPC (Rocket):** SLURM + vLLM serving Mistral-Small-3.2-24B on GPU (`firefly2`, partition `gpu`);
  configs in `configs/HPC_Config/`, jobs in `scripts/hpc/`. Outputs live under `outputs/hpc/` on the
  cluster (not committed).
- **Local (dev laptop):** LM Studio @ `localhost:1234` for quick checks. Project venv is
  `.venv/Scripts/python.exe` (Python 3.10; point PyCharm's interpreter here).
- Cost weights: input 0.08 / output 0.32 (Mistral OpenRouter average, paper A.5).
- pytest: `testpaths = ["tests"]`; ~440 tests, 1 known pre-existing failure
  (`test_portfolio_rows_to_candidates_restores_few_shot_examples`, a stale wording assert).

## Live-run gotchas

- **Keep the model server up the whole run** (LM Studio locally, or vLLM on the SLURM node) — closing
  it mid-run crashes with a 400.
- The runner prints little mid-loop — a quiet log usually = working (results write at the end).
- Always pass a scratch `--output-dir` for `--no-llm` dry runs (a dry run once clobbered live outputs).
- Build tables/figures AFTER all runs (LLM-free).

## History (condensed)

- **S1–S8** built the budgeted MO-CAPO optimizer (intensification, few-shot ops, held-out Dtest),
  renamed HEAL-CAPO→FairCAPO, first SUBJ + table/figures.
- **S9–S15** integrated **BBQ** end-to-end; found the in-loop fairness win was in-sample only on BBQ
  (held-out metric saturation), added the `max_amb_dis` sDIS fold + paper-parity pass; BBQ seed-0
  ended in a held-out tie with NSGA → BBQ archived in favor of a cleaner showcase.
- **Bias-in-Bios era** became the main track: 28-way occupation classification, gender group fairness,
  label-conditioned option, HPC pipeline on Rocket, frozen **v1 seed-0** result above (FairCAPO
  fairness 0.000952 vs NSGA 0.010, accuracy ~78%). Qwen / label-scoring / fair-shots tried and shelved.
- **2026-07-05** removed all Adult-income dataset files (48) from the repo.
- **2026-07-06** added BIOS **v2** (reason-then-label CoT + hardened extraction + enhanced pool) to
  break the 78% accuracy ceiling; committed + pushed, pending the Rocket run.
