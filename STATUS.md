# FairCAPO — Project Status

## Active Rocket Status — 2026-06-27

Do not cite the old 500k_v2 table whose `source` columns are
`outputs/hpc/*_500k_v2/seed_0/*all_candidates.csv`. That table mixed search/dev
objective values with held-out inference-cost columns, which is why it showed
near-perfect or perfect accuracy and produced weak method-comparison figures.

The valid reporting basis is held-out evaluation CSVs:

- Standard 500k_v2 held-out: `outputs/hpc/evaluation/seed_0/.../test_eval_candidates.csv`
- Stage A large held-out: `outputs/hpc/evaluation_large/seed_0/.../test_eval_candidates.csv`

Stage A large-held-out already shows the important diagnostic result:

- FairCAPO: HV 0.4658, perf 0.980, fairness_risk 0.0575
- NSGA-II-PO + fairness: HV 0.4630, perf 0.980, fairness_risk 0.0682
- MO-CAPO fairness off: HV 0.3733, perf 0.955, fairness_risk 0.0145
- Post-hoc fair. (held-out): same held-out Dtest basis as the ablation-derived
  post-hoc portfolio; keep it as a diagnostic row, not as a separate search
  method.

Interpretation: FairCAPO has a real but small large-held-out HV edge over NSGA
at 500k seed 0. This is not strong enough to justify moving to 1M yet. First
finish the clean 500k_v2 held-out table/figures from eval CSVs, then decide
whether to improve FairCAPO operators/intensification before spending more GPU
time.

Current next commands on Rocket after `git pull origin main`:

```bash
# If the post-hoc v2 portfolio has not yet been re-evaluated on the same Dtest:
sbatch --array=0 \
  --export=ALL,METHOD=ablation,CONFIG=configs/HPC_Config/evaluate_pareto_bbq_ablation_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bbq_ablation_500k_v2/seed_0/phase2_prompt_portfolio_bbqfair.csv,OUT_DIR=outputs/hpc/evaluation/seed_0/bbq_posthoc_500k_v2 \
  scripts/hpc/run_bbq_eval_hpc.slurm

# After the four standard 500k_v2 eval dirs exist:
bash scripts/hpc/build_bbq_500k_v2_outputs.sh
```

That helper now builds the clean held-out table plus the full figure bundle:
three paper-style figures, held-out staircase, search-basis front richness,
search-basis trajectory when both trajectory JSONs exist, and held-out Pareto
diagnostics under `pareto_diagnostics/`.

Use `bash scripts/hpc/build_bbq_stagea_outputs.sh` only for the older Stage A
large-held-out diagnostic outputs.

_Last updated: 2026-06-26 — UT Rocket/Pegasus2 path is the active plan. The local LM Studio diagnostic path is superseded. GitHub export `vikashmaheshwari97/FairCAPO` has Rocket-ready vLLM SLURM wrappers for search, eval, NSGA, and post-hoc BBQ fairness; active BBQ HPC configs default to seed 0 for cost control._
_Prior result to remember: seed-0 local comparison closed at S14 — held-out HV FairCAPO 0.201 == NSGA 0.201 > ablation 0.182. FairCAPO beats the fairness-OFF ablation but ties NSGA on held-out, so the next expensive step must be run carefully on HPC._

> **HEADLINE.** FairCAPO = **MO-CAPO + an in-loop fairness objective** (perf↑, cost↓, risk↓,
> fairness_risk↓). The BBQ contribution was validated **in-sample** at S10 (FairCAPO `fairness 0.000`
> vs fairness-OFF ablation `0.120` at equal accuracy); S11 found the **held-out** comparison was
> inconclusive (|sAMB| saturates to ~0 → ties the baseline); **S12 fixed the root causes** (metric
> saturation via the `max_amb_dis` sDIS fold, narrow search, cost mis-reporting) + paper-parity.
> **S14 (THIS session) finally produced FRESH post-S12 seed-0 numbers** — the seed-0 FairCAPO search,
> the ablation + NSGA searches, and the ablation + NSGA held-out Dtests all completed today.
> **Key win: held-out `fairness_risk` now VARIES** (ablation 0.455–0.75, was saturated-to-~0 at S11)
> → the sDIS fold works on held-out data, the comparison is discriminative again. Only the FairCAPO
> held-out Dtest + the cross-method table/figures remain (re-running now after a one-off startup
> hang). Numbers in the "ON DISK — STALE" table further down are still pre-S12; the fresh S14 numbers
> live in `outputs/seed_0/...`, `outputs/evaluation/bbq_{ablation,nsga}_local/`. Nothing committed yet.

> **▶️ S15 (THIS session, 2026-06-24 ~7:15 PM) — STAGE A FAIRNESS-RESOLUTION DIAGNOSTIC, IN
> PROGRESS / HUNG.** Goal: explain (and ideally break) the S14 **held-out tie with NSGA** (both HV
> 0.201, |P|=2, best fairness 0.455). Working hypothesis: the held-out fairness eval used only **36
> pairs** (6 disambiguated), so **sDIS is quantized to ~1/6 steps** — too coarse to separate FairCAPO
> from NSGA on the fairness axis. Stage A re-runs the **3 held-out Dtests on a LARGER disjoint set** so
> sDIS is finely resolved; if the methods still tie at high resolution the tie is real, if they
> separate the S14 tie was a quantization artifact.
>
> **Done in S15:** (1) built `data/fairness_bbq_holdout_large.jsonl` — **180 items / 90 disambiguated**,
> disjoint from the in-loop set (created 17:21). (2) Cloned 3 diagnostic eval configs —
> `configs/evaluate_pareto_bbq_{ablation,nsga,}_diag.yaml` — pointing at the large set with
> **`eval_pairs: 180`**, `test_size: 200`, `bbq_score: max_amb_dis`, writing to **`*_diag` output dirs**
> so the original 36-pair eval dirs are left untouched. (3) Wrote **`scripts/run_bbq_diag.sh`** (sweeps
> ablation → nsga → faircapo, `set +e`, tees per-method logs). All read the SAME seed-0 portfolios from
> `outputs/{phase2_budgeted_mocapo_bbq_local, mocapo_baseline_bbq_local, baselines/nsga2_po_bbq_local}/`.
>
> **⚠️ HUNG — CONFIRMED, not a measurement glitch.** The **ablation** diag eval was launched directly
> (NOT via the sweep script — no tee log) and is **wedged at startup**: two python workers from
> 18:01:21, 1h+ old, sit at **400 KB / ~15 MB** with **CPU frozen at 8.8 CPU-sec** (re-sampled 20s
> apart → ZERO change; a healthy eval climbs to 100–170 MB and ticks CPU up continuously). Same S14
> FairCAPO-Dtest startup hang. **LM Studio is HEALTHY** — a fresh `/v1/chat/completions` probe returned
> `HTTP 200 in 5.1s` — so "LM Studio is running" is true but **does NOT mean the eval is working**: the
> eval client never issued a single request (that's *why* LM Studio is idle; a live eval would keep it
> busy ~5s × 200 items). **No `outputs/_bbq_diag_*.log`, no `outputs/evaluation/*_diag` dirs → zero
> Stage A results.** **Fix:** kill the stuck pythons (PIDs 21676 + 33008), relaunch the full sweep via
> `bash scripts/run_bbq_diag.sh` (all 3 methods + tee logs). Then the open task: **rebuild BBQ figures
> in MO-CAPO style with held-out test accuracy.** Nothing committed.

> **SUPERSEDED BY ROCKET PLAN (2026-06-25).** The old S15 local diagnostic checklist required LM
> Studio on the laptop. Do not use that as the next step now. The active path is the UT Rocket
> seed-0 smoke run through `scripts/hpc/run_bbq_hpc.slurm`, which starts vLLM inside the SLURM job.
> The large-held-out diagnostic remains useful later, but only after the Rocket seed-0 search/eval is
> stable.

> **ROCKET 500k STATUS (2026-06-26).** The 500k seed-0 Rocket smoke is complete for all three methods:
> FairCAPO search/eval, MO-CAPO fairness-off search/eval, and NSGA-II-PO + fairness search/eval.
> The post-hoc fairness row is now supported on Rocket via `scripts/hpc/run_bbq_posthoc_hpc.slurm`,
> which starts its own vLLM server inside the SLURM job. The completed 500k post-hoc command wrote
> `outputs/hpc/bbq_ablation/seed_0/phase2_prompt_portfolio_bbqfair.csv`. For 500k reporting, use the
> temporary Rocket configs created on the cluster:
> `configs/HPC_Config/experiment_table_bbq_500k_seed0_TMP.yaml` and
> `configs/HPC_Config/aggregate_multiseed_bbq_500k_seed0_TMP.yaml`. The active committed table configs now point to `_500k_v2` outputs.
>
> **500k figures:** use `scripts/visualize_paper_figures.py`, `scripts/visualize_front_richness.py`,
> and `scripts/visualize_staircase.py`, writing to `outputs/figures/paper_bbq_hpc_500k_seed0/`.
> `visualize_staircase.py` was fixed so `--mocapo ""` correctly means no overlay file for BBQ.
>
> **Stage A table/figures are committed now.** After the three large-held-out eval jobs finish, do
> not write ad-hoc Python/YAML on the login node. Run:
> `bash scripts/hpc/build_bbq_stagea_outputs.sh`. It uses
> `configs/HPC_Config/experiment_table_bbq_500k_large_HPC.yaml` and
> `configs/HPC_Config/aggregate_multiseed_bbq_500k_large_HPC.yaml`, then writes
> `outputs/experiment_table/bbq_mistral_hpc_500k_large/` and
> `outputs/figures/paper_bbq_hpc_500k_large/`.
> Corrected on 2026-06-26: the Stage A table and default figures now use
> `outputs/hpc/evaluation_large/.../test_eval_candidates.csv` as the comparison basis. The
> search-basis front-richness figure is skipped by default to avoid mixing dev/search and held-out
> evidence; set `RUN_SEARCH_FIGURE=1` only if you explicitly want that separate search-basis figure.

> **✅ SEED-0 RUN COMPLETED (S14, 2026-06-24).** Unlike the S13 death, the relaunched seed-0 FairCAPO
> search **wrote full output** to `outputs/seed_0/phase2_budgeted_mocapo_bbq_local/` (10 files, 17:15).
> Results: **budget 413,713 / 500,000 tok (82.7%, not exhausted), 43 candidates / 6 blocks, 162 total
> candidates → 18 Pareto, in-search HV 0.619, Pareto cost 362.6→1400.3.** Meta-LLM active (93/162
> `used_meta_llm=True`), in-sample fairness varies {0.333,0.5,0.714,1.0}. Synced into the default dir
> `outputs/phase2_budgeted_mocapo_bbq_local/` (overwrote the stale Jun-22 recovery copy) so the
> existing eval/table configs read it.
>
> **What S14 ran after (all seed 0, via `scripts/run_bbq_seed0_pipeline.sh`):** ablation search
> (`outputs/mocapo_baseline_bbq_local/`, 19-prompt portfolio) ✅; NSGA search
> (`outputs/baselines/nsga2_po_bbq_local/`, 3-prompt portfolio) ✅; ablation held-out Dtest
> (HV 0.182, 4 Pareto, fairness 0.455–0.75) ✅; NSGA held-out Dtest (HV 0.201, 2 Pareto,
> fairness 0.455) ✅. **NSGA was bumped pop 8→16 / offspring 3→4** in `configs/nsga2_po_bbq.yaml` to
> match FairCAPO's mu/c (MO-CAPO §4.2 parity; bumping UP is the allowed direction under the integrity
> guardrail). The runner's `date`-not-found / `rc=127` log noise is **cosmetic** (a venv PATH quirk in
> the timestamp helper clobbered `$?`); every step's real output was written.
>
> **⚠️ THE FairCAPO held-out Dtest HUNG once (S14).** Step 3 of the pipeline (FairCAPO Dtest) stalled
> at startup — worker flat at 11 MB, LM Studio idle 10+ min, no output (a healthy eval climbs to
> ~100–170 MB and serves within ~1 min). Killed the stuck pythons; the pipeline (no `set -e`)
> auto-advanced through steps 4→6, so the table/figures it built carry a **STALE Jun-20 FairCAPO
> held-out row**. Recovery staged in `scripts/run_bbq_seed0_recover_step3.sh` (re-run FairCAPO Dtest →
> rebuild table → rebuild figures); **launched and confirmed healthy this time** (worker 100 MB,
> serving). The hang was transient (identical eval code ran fine for ablation + NSGA), not a bug.
>
> **✅ RECOVERY DONE + 4-SIGNAL CHECK CLOSED (S14, 2026-06-24 ~12:00).** The relaunched FairCAPO
> held-out Dtest ran healthy (HV 0.201, 2 Pareto, fresh 06-24 11:43). **But the recovery's
> table-rebuild step had crashed silently** with `_csv.Error: field larger than field limit (131072)`
> — `build_experiment_table.py:load_csv_rows` used Python's default 128 KB CSV field limit, which the
> held-out candidate CSVs (full prompt + few-shot block in one field) exceed. **FIXED:** added
> `csv.field_size_limit(10 MB)` at module top; rebuilt the table cleanly; **14/14 table tests pass.**
> (The table's `HV` column is computed from the in-search Pareto front, not the Dtest, so it was
> unaffected by the hang — only the `inference_cost_*` columns read the holdout CSV.)
>
> **4-signal verdict (seed 0, held-out Dtest):** (a) front non-degenerate ✅ (cost 362→1400, 18
> Pareto); (b) meta-LLM ✅ (93/162); (c) held-out fairness VARIES ✅ (span 0.455–0.75, sDIS fold works);
> **(d) held-out HV: FairCAPO 0.201 == NSGA 0.201 > ablation 0.182.** Honest read: FairCAPO **beats its
> fairness-OFF ablation** (0.201 > 0.182) but **TIES the NSGA algorithmic baseline** (identical HV,
> |P|=2, and best held-out fairness 0.455). Echoes the S11 pattern — the in-loop edge over the ablation
> survives held-out, but does NOT beat NSGA there. The win is now over a *discriminative* (non-
> saturated) metric, which is real progress, but it's a top-end tie with NSGA, not a clear victory.
> **DECISION POINT for the user:** with seed 0 showing a tie-vs-NSGA, decide whether the 10–15h 3-seed
> sweep is worth running as-is, or whether to first improve FairCAPO / pick a cleaner showcase
> (Bias-in-Bios) where all three axes move. Still UNCOMMITTED.
>
> **⚠️ RUNTIME REALITY (measured S12).** One FairCAPO method-seed at the current settings took
> **>1h40m** (meta-LLM ON adds 2 LLM calls/offspring, k_max=3 = longer prompts, budget 500k = ~2×
> candidates). So the FULL 3-seed × 3-method sweep ≈ **10–15+ hours** — an overnight job, possibly
> impractical on this laptop. **Before launching the sweep, RIGHT-SIZE cost:** the biggest multiplier
> is the meta-LLM; cheaper levers = lower `offspring_per_iteration`, drop budget back to 250k, or run
> fewer seeds (2 not 3). Decide this with the user, don't just launch the 15h job.

**Naming.** "FairCAPO" is the method/display name (figures, paper). The Python package is still
`heal_capo/` (renaming touches the whole test suite for zero paper benefit). Older notes say
"HEAL-CAPO" — treat as a synonym.

**Dataset scope (locked):** **BBQ + Bias-in-Bios** (fairness core) + **SUBJ** (no-harm control).
GSM8K / AG News / MBPP are **DROPPED** (no fairness signal).

---

## How the fairness extension works (plain English)

MO-CAPO tunes a prompt for **accuracy + low cost** and returns a *menu* (Pareto front) of trade-offs.
**FairCAPO adds fairness as a 4th goal.**

**Measuring `fairness_risk` (0 = perfectly fair):**
- **Counterfactual tasks (SUBJ):** swap one demographic detail ("He/She is a nurse"); if the answer
  flips when it shouldn't, that's an unfair flip. Blend: flip 0.50 / group-gap 0.25 / bias-language
  0.15 / decayed debt 0.10 → one `fairness_risk`. Code: `heal_capo/fairness.py`.
- **BBQ (the showcase):** the canonical BBQ **bias score** (sAMB = ambiguous-context bias, sDIS =
  disambiguated-context bias). `fairness_risk` is distilled from these via `fairness.bbq_score`
  (S12): `samb` (old default) / `sdis` / **`max_amb_dis`** (= max(|sAMB|,|sDIS|), now used) /
  `mean_amb_dis`. Code: `heal_capo/fairness_bbq.py`.

**In-loop steering:** during search, every candidate's `fairness_risk` is measured with the real model
(cached, cost charged to budget) and added as the 4th objective. The SAME MO-CAPO
Pareto/intensification/selection machinery runs in 4-D, so fairness genuinely decides which prompts
survive. Wired in `LLMObjectiveEvaluator` (`scripts/run_phase2_budgeted_mocapo.py`) behind
`fairness.in_loop`.

**Why BBQ, not SUBJ, is the showcase:** mistral-24B rarely flips on SUBJ counterfactuals
(`fairness_risk ≈ 0`), so SUBJ is the **no-harm control**. On BBQ the bias score moves with the prompt.

---

## ✅ SUBJ — no-harm control (DONE, prior sessions)

Near the 24B ceiling and counterfactually robust, so it can't *showcase* fairness — it proves FairCAPO
does no harm on a fairness-neutral task. Intensified run + held-out Dtest (staircase 0.60→0.96, HV
0.4619, one prompt hit `fairness_risk 0.333`) + table + figures all on disk
(`outputs/phase2_budgeted_mocapo_subj_ddev30_intensified/`, `outputs/evaluation/subj_mistral_ddev30/`).

---

## BBQ results ON DISK — ⚠️ STALE (pre-S12, superseded by the running re-run)

Kept for reference. These came from the OLD regime (60k–250k budget, `|sAMB|`-only metric, no
meta-LLM, k_max=2). **Do not cite — the S12 re-run replaces them.**

| method | perf | cost* | fairness | HV | \|P\| |
|---|---|---|---|---|---|
| FairCAPO | 1.000 | 361.9 | 0.000 | 0.879 | 5 |
| MO-CAPO (fairness OFF) | 1.000 | 268.0 | 0.120 | 0.797 | 5 |
| NSGA-II-PO + fairness | 1.000 | 350.6 | 0.000 | **0.883** | 1 |

*`cost` here is the mislabeled token-weighted **sum** over the dev set, not $/1M-calls (S12 fix below).
**The problem this exposed:** NSGA's single point dominated all 5 FairCAPO points (HV 0.883 > 0.879) —
the "richer front" win was hollow. S12 attacks exactly this.

**S11 post-hoc finding (still valid):** the fairness-OFF front re-scored on the **held-out** set also
reaches `|sAMB| {0.0,0.04}` → ties FairCAPO. The `0.120→0.000` advantage is **in-sample only** because
held-out `|sAMB|` saturates (mistral abstains); real residual bias lives in `sDIS ≈ 0.45–0.60`. Full
writeup: `docs/bbq_posthoc_fairness_finding.md`. **S12's `max_amb_dis` fix folds sDIS in to make the
held-out comparison discriminative again.**

---

## ✅ S12 changes (this session) — all UNCOMMITTED, 380 pytest green

Full reasoning in `docs/mocapo_gap_analysis_S12.md` (read against MO-CAPO §5, §6.2, A.1–A.8).

**Root-cause fixes (why FairCAPO looked dominated):**
1. **Cost was mis-reported, not mis-earned.** Plotted `cost` = token-weighted SUM over the dev set
   (off from "$/1M-calls" by factor N) and on the search basis also folds one-time fairness-eval
   tokens. Fixed the two axis labels in `visualize_paper_figures.py`; added honest **per-call
   inference cost** (fairness-audit cost separated out) to the table via `holdout_inference_cost`
   in `build_experiment_table.py` (FairCAPO held-out ≈ **7.75/call** vs the old folded 727–1118).
2. **sDIS metric fix (highest-leverage):** `fairness.bbq_score: max_amb_dis` in all BBQ configs, so
   the objective stays discriminative when `|sAMB|` saturates. Threaded through runner + held-out
   eval + post-hoc. 4 tests.
3. **Meta-LLM ENABLED** (`use_meta_llm: true`) on FairCAPO + ablation + NSGA. Previously OFF → search
   only recombined the 11 seed prompts + few-shot counts (a narrow space; likely why NSGA's cheap
   point was unreachable). ON, the meta-LLM **writes new instruction text** — faithful to CAPO §4.
   Our crossover/mutation meta-prompts already match paper Table 10. ⚠️ Meta-LLM tokens are NOT
   metered against budget, but symmetrically across all 3 methods (same operator) → parity holds.

**Paper-parity (Büssing et al. §5, A.3.4):**
4. **k_max 1→3** (`max_few_shot_examples`) — paper k_max=5; the accuracy/cost staircase is *driven* by
   varying few-shot count, so k_max=1 had flattened our own staircase.
5. **Search tuning:** `population_size 8→16`, `max_iterations 5→8`, `offspring 3→4`
   (≈48 candidate-gens vs 23), `few_shot_probability 0.6→0.3`. (Pool cap: gen-0 starts at 11 seeds.)
6. **Budget 250k→500k** iso (FairCAPO + ablation). **NSGA reverted to UNBUDGETED** (MO-CAPO §4.2).
7. **Dtest 60→200** + **nR2 over 500 preference vectors** (was 50) in all eval configs (we have 13.7k
   BBQ rows; held-out eval is outside the search budget so this is free).
8. **HV_opt / HV_pes / Gap columns** in the experiment table (already computed in `mo_metrics.py`,
   just unexposed) — paper Table 2 reports robustness, not a single HV.
9. **3-seed orchestration:** added `--seed` to the NSGA + evaluate runners and `--portfolio-csv` to
   the evaluate runner; wrote **`scripts/run_bbq_multiseed.sh`** (3 seeds × 3 methods × Dtests) and
   **`scripts/aggregate_multiseed.py`** (mean±std table — validated against the existing eval).

**Held-out Dtest configs** added for NSGA + ablation (`configs/evaluate_pareto_bbq_{nsga,ablation}.yaml`)
so all 3 methods share one held-out basis.

---

## ▶️ NEXT STEPS (in order)

> **Integrity guardrail:** never handicap the baseline (no lowering NSGA accuracy / inflating its cost).
> Only legitimate moves: make the comparison genuinely FAIR, or genuinely IMPROVE FairCAPO.

**Detailed Rocket run plan (current).**

2. **Run Stage A large-held-out diagnostic on existing 500k results.** Purpose: check whether the
   FairCAPO-vs-NSGA tie is caused by too-small held-out fairness evaluation.

   Run FairCAPO large eval:
   `sbatch --array=0 --export=ALL,METHOD=faircapo,CONFIG=configs/HPC_Config/evaluate_pareto_bbq_large_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bbq_faircapo/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bbq_faircapo scripts/hpc/run_bbq_eval_hpc.slurm`.

   Run ablation large eval:
   `sbatch --array=0 --export=ALL,METHOD=ablation,CONFIG=configs/HPC_Config/evaluate_pareto_bbq_ablation_large_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bbq_ablation/seed_0/phase2_prompt_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bbq_ablation scripts/hpc/run_bbq_eval_hpc.slurm`.

   Run NSGA large eval:
   `sbatch --array=0 --export=ALL,METHOD=nsga,CONFIG=configs/HPC_Config/evaluate_pareto_bbq_nsga_large_HPC.yaml,PORTFOLIO_CSV=outputs/hpc/bbq_nsga2po/seed_0/nsga2_po_pareto_portfolio.csv,OUT_DIR=outputs/hpc/evaluation_large/seed_0/bbq_nsga2po scripts/hpc/run_bbq_eval_hpc.slurm`.

3. **Inspect Stage A results.** Check all three completed:
   `ls -lh outputs/hpc/evaluation_large/seed_0/bbq_faircapo`;
   `ls -lh outputs/hpc/evaluation_large/seed_0/bbq_ablation`;
   `ls -lh outputs/hpc/evaluation_large/seed_0/bbq_nsga2po`.

   Read summaries:
   `cat outputs/hpc/evaluation_large/seed_0/bbq_faircapo/test_eval_summary.json`;
   `cat outputs/hpc/evaluation_large/seed_0/bbq_ablation/test_eval_summary.json`;
   `cat outputs/hpc/evaluation_large/seed_0/bbq_nsga2po/test_eval_summary.json`.

   Decision: if FairCAPO > NSGA, the tie was likely fairness-resolution noise. If FairCAPO ~= NSGA,
   the tie is probably real at 500k. If FairCAPO < NSGA, improve FairCAPO before running more seeds.

   Then build the Stage A table and figures without writing code on the login node:
   `bash scripts/hpc/build_bbq_stagea_outputs.sh`.

4. **Run 500k v2 FairCAPO seed 0.** Stage A showed only a tiny FairCAPO edge over NSGA, so do
   not spend on 1M yet. The active configs now use Ddev=75, Dshots=25, Dtest=100, budget=500k,
   block_size=15, population_size=16, max_iterations=8, offspring_per_iteration=4, and a stronger
   BBQ fairness signal (`eval_pairs=24`) plus extra disambiguated-bias prompts.
   `sbatch --array=0 --export=ALL,CONFIG=configs/HPC_Config/phase2_budgeted_mocapo_bbq_HPC.yaml,RUN_TAG=bbq_faircapo_500k_v2 scripts/hpc/run_bbq_hpc.slurm`.

   Monitor:
   `squeue -u $USER`;
   `ls -lt outputs/hpc/logs | head`;
   `tail -f outputs/hpc/logs/bbq-faircapo_<JOBID>_seed0.out`.
   Expected output: `outputs/hpc/bbq_faircapo_500k_v2/seed_0`.

5. **Evaluate 500k v2 FairCAPO.** Only after the search succeeds:
   `sbatch --array=0 --export=ALL,METHOD=faircapo scripts/hpc/run_bbq_eval_hpc.slurm`.
   Expected output: `outputs/hpc/evaluation/seed_0/bbq_faircapo_500k_v2`.

6. **Run 500k v2 ablation seed 0.**
   `sbatch --array=0 --export=ALL,CONFIG=configs/HPC_Config/mocapo_baseline_bbq_HPC.yaml,RUN_TAG=bbq_ablation_500k_v2 scripts/hpc/run_bbq_hpc.slurm`.
   Expected output: `outputs/hpc/bbq_ablation_500k_v2/seed_0`.

7. **Evaluate 500k v2 ablation.**
   `sbatch --array=0 --export=ALL,METHOD=ablation scripts/hpc/run_bbq_eval_hpc.slurm`.
   Expected output: `outputs/hpc/evaluation/seed_0/bbq_ablation_500k_v2`.

8. **Run 500k v2 NSGA-II-PO seed 0.**
   `sbatch --array=0 scripts/hpc/run_bbq_nsga_hpc.slurm`.
   Expected output: `outputs/hpc/bbq_nsga2po_500k_v2/seed_0`.

9. **Evaluate 500k v2 NSGA-II-PO.**
   `sbatch --array=0 --export=ALL,METHOD=nsga scripts/hpc/run_bbq_eval_hpc.slurm`.
   Expected output: `outputs/hpc/evaluation/seed_0/bbq_nsga2po_500k_v2`.

10. **Run 500k v2 post-hoc fairness before building the table.** This is required because the active
    experiment table includes `Post-hoc fair. (held-out)`.
    `sbatch --array=0 scripts/hpc/run_bbq_posthoc_hpc.slurm`.
    Expected output: `outputs/hpc/bbq_ablation_500k_v2/seed_0/phase2_prompt_portfolio_bbqfair.csv`.

11. **Build 500k v2 table and figures.**
    `PYTHONPATH=. python scripts/build_experiment_table.py --config configs/HPC_Config/experiment_table_bbq_HPC.yaml`;
    `PYTHONPATH=. python scripts/aggregate_multiseed.py --config configs/HPC_Config/aggregate_multiseed_bbq_HPC.yaml`;
    `mkdir -p outputs/figures/paper_bbq_hpc_500k_v2_seed0`;
    `python scripts/visualize_paper_figures.py --run outputs/hpc/bbq_faircapo_500k_v2/seed_0 --table outputs/experiment_table/bbq_mistral_hpc_500k_v2_seed0/experiment_table.csv --title "BBQ / Mistral-Small-3.2 (Rocket 500k v2 seed 0)" --out outputs/figures/paper_bbq_hpc_500k_v2_seed0`.

12. **Decision after 500k v2.** If FairCAPO clearly beats NSGA on held-out HV/fairness tradeoff,
    then consider the commented 1M settings. If FairCAPO still ties NSGA, improve FairCAPO
    operators/intensification before any 1M or 3-seed sweep. If FairCAPO loses to NSGA, do not run
    more seeds yet.

1. **✅ Rocket 500k seed-0 pipeline completed.** FairCAPO, MO-CAPO fairness-off, and NSGA-II-PO
   searches and held-out evals all completed on Pegasus2. Held-out HV: FairCAPO ≈ NSGA ≫
   fairness-off ablation. Treat this as a successful systems smoke plus an unresolved FairCAPO-vs-NSGA
   tie, not a final paper result.
2. **✅ 500k post-hoc fairness row is wired and measured.** Rocket wrapper:
   `scripts/hpc/run_bbq_posthoc_hpc.slurm`. Completed 500k output:
   `outputs/hpc/bbq_ablation/seed_0/phase2_prompt_portfolio_bbqfair.csv`. Rebuild the 500k table
   with the cluster temp config before citing `Post-hoc fair. (held-out)`:
   `python scripts/build_experiment_table.py --config configs/HPC_Config/experiment_table_bbq_500k_seed0_TMP.yaml`.
3. **▶️ Rebuild/check 500k figures with post-hoc included in the method table.** Commands:
   `python scripts/visualize_paper_figures.py --run outputs/hpc/bbq_faircapo/seed_0 --table outputs/experiment_table/bbq_mistral_hpc_500k_seed0/experiment_table.csv --title "BBQ / Mistral-Small-3.2 / Rocket 500k seed 0" --out outputs/figures/paper_bbq_hpc_500k_seed0`;
   `python scripts/visualize_front_richness.py --faircapo outputs/hpc/bbq_faircapo/seed_0/phase2_all_candidates.csv --nsga outputs/hpc/bbq_nsga2po/seed_0/nsga2_po_all_candidates.csv --ablation outputs/hpc/bbq_ablation/seed_0/phase2_all_candidates.csv --title "BBQ / Mistral-Small-3.2 / Rocket 500k seed 0 (search basis)" --out outputs/figures/paper_bbq_hpc_500k_seed0/fig_front_richness_bbq.png`;
   `python scripts/visualize_staircase.py --fair outputs/hpc/evaluation/seed_0/bbq_faircapo/test_eval_candidates.csv --portfolio outputs/hpc/bbq_faircapo/seed_0/phase2_prompt_portfolio.csv --mocapo "" --title "BBQ / Mistral-Small-3.2 / Rocket 500k seed 0 (held-out)" --out outputs/figures/paper_bbq_hpc_500k_seed0/fig_pareto_staircase.png --color-fairness`.
4. **▶️ Stage A large-held-out diagnostic is now unpaused.** Run the completed 500k seed-0 portfolios
   against `data/fairness_bbq_holdout_large.jsonl` using:
   `configs/HPC_Config/evaluate_pareto_bbq_large_HPC.yaml`,
   `configs/HPC_Config/evaluate_pareto_bbq_ablation_large_HPC.yaml`, and
   `configs/HPC_Config/evaluate_pareto_bbq_nsga_large_HPC.yaml`. These write to
   `outputs/hpc/evaluation_large/seed_0/...`.
5. **▶️ Next search scale stays 500k, not 1M.** Active Rocket configs now use Ddev=75, Dshots=25, Dtest=100, budget=500k, block_size=15, z_max=5, and write to `_500k_v2` output dirs: `outputs/hpc/bbq_faircapo_500k_v2/seed_0`, `outputs/hpc/bbq_ablation_500k_v2/seed_0`, and `outputs/hpc/bbq_nsga2po_500k_v2/seed_0`. The 1M values remain as comments in the configs.
6. **Run 500k v2 FairCAPO seed 0 first.** Command:
   `sbatch --array=0 --export=ALL,CONFIG=configs/HPC_Config/phase2_budgeted_mocapo_bbq_HPC.yaml,RUN_TAG=bbq_faircapo_500k_v2 scripts/hpc/run_bbq_hpc.slurm`.
   Inspect logs/output before launching baselines.
7. **If FairCAPO succeeds, run 500k v2 baselines one at a time.** Ablation:
   `sbatch --array=0 --export=ALL,CONFIG=configs/HPC_Config/mocapo_baseline_bbq_HPC.yaml,RUN_TAG=bbq_ablation_500k_v2 scripts/hpc/run_bbq_hpc.slurm`.
   NSGA-II-PO:
   `sbatch --array=0 scripts/hpc/run_bbq_nsga_hpc.slurm`.
8. **Evaluate 500k v2 seed 0 after each search exists.** Standard eval wrapper now defaults to `_500k_v2` paths:
   `sbatch --array=0 --export=ALL,METHOD=faircapo scripts/hpc/run_bbq_eval_hpc.slurm`,
   then `METHOD=ablation`, then `METHOD=nsga`.
9. **Run 500k v2 post-hoc fairness after the ablation search exists.** Command:
   `sbatch --array=0 scripts/hpc/run_bbq_posthoc_hpc.slurm`. Expected output:
   `outputs/hpc/bbq_ablation_500k_v2/seed_0/phase2_prompt_portfolio_bbqfair.csv`.
10. **Build 500k v2 seed-0 tables after all three evals plus the post-hoc CSV exist.** Use
   `configs/HPC_Config/experiment_table_bbq_HPC.yaml` and
   `configs/HPC_Config/aggregate_multiseed_bbq_HPC.yaml`; both point to `_500k_v2` output dirs and
   aggregate only `seeds: [0]`.
11. **Do not run seeds 1 and 2 yet.** Run `--array=0-2` only after the revised 500k v2 seed-0 pipeline shows stronger FairCAPO-vs-NSGA separation.
12. **Improvement path if the tie remains real:** improve FairCAPO directly, not by handicapping NSGA:
   increase final intensification depth/budget, improve fairness-specific prompt mutation/crossover
   operators, or move to the second fairness dataset.
13. **Then Bias-in-Bios = second fairness dataset.** This remains a later research step: occupation
   classification, gender groups, `group_accuracy_gap`/`equal_opportunity`, and a new loader/config
   path. Do not start it until BBQ Rocket execution is settled.

---

## Comparison design (DECIDED — don't re-litigate)

- **FairCAPO = MO-CAPO + in-loop fairness = the METHOD**, not a baseline. Baselines: MO-CAPO
  **fairness-OFF** (ablation, isolates the fairness objective) + **NSGA-II-PO +fairness** (primary
  algorithmic baseline = MO-CAPO §4.2: NSGA-II + CAPO operators on the SAME objectives, only the
  search algorithm differs — it also serves as the intensification ablation) + **post-hoc fairness**
  (in-loop vs post-process).
- **Self-contained:** we run all methods ourselves on the same model/budget/eval set and compare among
  our runs — NOT against the MO-CAPO paper's numbers (they have no fairness).

## Live-run gotchas

- **Rocket/HPC:** do not use LM Studio. The SLURM wrappers start vLLM inside the allocated Pegasus2
  GPU job and export `FAIRCAPO_LLM_API_URL=http://127.0.0.1:<port>/v1`.
- Mistral Small 3.2 loads as a Pixtral/multimodal architecture in vLLM. BBQ is text-only, so the
  SLURM wrappers pass Mistral-format load flags and `--limit-mm-per-prompt '{"image":0}'` to avoid
  the image processor startup path.
- vLLM eager mode is enabled on Rocket (`--enforce-eager`) because job `66935498` reached 45.6GB
  VRAM allocated by `VLLM::EngineCore` but stalled before `/v1/models` became ready.
- Active Rocket search scale is now the 500k v2 seed-0 scale: Ddev=75, Dshots=25, Dtest=100, budget=500k tokens. The completed 500k smoke outputs are preserved in the original non-v2 folders; 1M values are kept as comments in the configs.
- **Local-only fallback:** if running on the laptop, keep LM Studio loaded the whole run. Confirm:
  `curl http://localhost:1234/v1/models`.
- **Local-only sleep guard:** `MSYS_NO_PATHCONV=1 powercfg /change standby-timeout-ac 0` (long laptop
  runs were silently killed by sleep). Run unbuffered (`python -u … | tee`).
- The runner prints nothing mid-loop — **a quiet log = working** (results write at the end).
- On Rocket, logs are `outputs/hpc/logs/<job>_<JOBID>_seed0.out` and `.err`. Watch both.
- **Always pass a scratch `--output-dir` for `--no-llm` dry runs** (an S11 dry run clobbered live
  gitignored outputs; recovered from the log). Memory: `dryrun-clobbers-live-outputs`.
- With meta-LLM ON + k_max=3, expect runs noticeably longer than the old ~30–45 min.
- Build the table/figures AFTER all runs (LLM-free; HV is fast since the `dedupe_by_objective` fix).

---

## Key files & configs

| What | Path |
|---|---|
| Budgeted optimizer (runner) | `scripts/run_phase2_budgeted_mocapo.py` + `heal_capo/optimizers/*` |
| NSGA-II-PO baseline | `scripts/run_baseline_nsga2_po.py`, `baselines/nsga2_po_runner.py` |
| Held-out evaluator | `scripts/evaluate_pareto_on_test.py` (`--seed`, `--portfolio-csv`) |
| Experiment table | `scripts/build_experiment_table.py` (HV_opt/pes/Gap, per-call inference cost) |
| **Multi-seed sweep + aggregate** | `scripts/run_bbq_multiseed.sh`, `scripts/aggregate_multiseed.py` |
| Paper figures | `scripts/visualize_paper_figures.py`, `visualize_front_richness.py`, `visualize_staircase.py` |
| Fairness / risk core | `heal_capo/fairness.py`, `heal_capo/risk.py` |
| **BBQ bias score (sAMB/sDIS)** | `heal_capo/fairness_bbq.py` (`bbq_score` mode → `fairness_risk`) |
| BBQ loader / multiple_choice | `experiments/datasets.py` (`load_bbq`), runner (`multiple_choice`, `_evaluate_candidate_fairness_bbq`) |
| BBQ fairness-set builder | `scripts/build_bbq_fairness_set.py` → `data/fairness_bbq.jsonl` (+`--exclude` holdout) |
| BBQ post-hoc fairness | `scripts/measure_baseline_fairness.py` |
| **BBQ configs** | `configs/phase2_budgeted_mocapo_bbq_local.yaml`, `mocapo_baseline_bbq_local.yaml` (ablation), `nsga2_po_bbq.yaml`, `evaluate_pareto_bbq{,_nsga,_ablation}.yaml`, `phase2_prompt_pool_bbq.yaml`, `experiment_table_bbq.yaml` |
| MO metrics (HV opt/pes, nR2, Gap) | `heal_capo/evaluation/mo_metrics.py` |
| Meta-prompt templates | `heal_capo/optimizers/evolutionary_ops.py` (`make_crossover/mutation_meta_prompt`) |
| Gap analysis vs paper | `docs/mocapo_gap_analysis_S12.md` |
| Reference figure style | `docs/MO-CAPO Figures/` (`MO-CAPO 1.png` = the staircase to match) |

---

## How to run (Rocket/HPC) — BBQ seed 0 first

```bash
cd ~/FairCAPO
git pull
mkdir -p outputs/hpc/logs

# 1. FairCAPO seed 0 search. This starts vLLM inside the SLURM job.
sbatch --array=0 \
  --export=ALL,CONFIG=configs/HPC_Config/phase2_budgeted_mocapo_bbq_HPC.yaml,RUN_TAG=bbq_faircapo \
  scripts/hpc/run_bbq_hpc.slurm

# 2. Watch queue/logs.
squeue -u $USER
ls outputs/hpc/logs
tail -f outputs/hpc/logs/bbq-faircapo_<JOBID>_seed0.out
tail -f outputs/hpc/logs/bbq-faircapo_<JOBID>_seed0.err

# 3. After search completes, evaluate FairCAPO seed 0.
sbatch --array=0 --export=ALL,METHOD=faircapo scripts/hpc/run_bbq_eval_hpc.slurm
```

Run ablation and NSGA seed 0 only after the FairCAPO search and eval are healthy.

---

## How to run (local fallback) — BBQ pipeline

```bash
# 0. LM Studio: load mistralai/mistral-small-3.2, server on :1234. KEEP IT OPEN.
curl http://localhost:1234/v1/models
MSYS_NO_PATHCONV=1 powercfg /change standby-timeout-ac 0   # keep laptop awake

# 1. env + tests
cd ~/PycharmProjects/Tri_CAPO && source .venv/Scripts/activate && pip install -e . && pytest tests

# 2a. single-seed validation (one method shown; --output-dir keeps seeds separate)
PYTHONPATH=. python -u scripts/run_phase2_budgeted_mocapo.py \
  --config configs/phase2_budgeted_mocapo_bbq_local.yaml \
  --seed 0 --output-dir outputs/seed_0/phase2_budgeted_mocapo_bbq_local | tee outputs/_bbq_faircapo_seed0.log

# 2b. OR the full 3-seed sweep (FairCAPO + ablation + NSGA + 3 Dtests, all seeds)
bash scripts/run_bbq_multiseed.sh 0 1 2

# 3. aggregate + table + figures (LLM-free)
python scripts/aggregate_multiseed.py --seeds 0 1 2
python scripts/build_experiment_table.py --config configs/experiment_table_bbq.yaml
python scripts/visualize_paper_figures.py --run outputs/seed_0/phase2_budgeted_mocapo_bbq_local --out outputs/figures/paper_bbq_local
```

---

## Architecture rollup (implemented)

| Area | Files | Notes |
|---|---|---|
| Budgeted MO-CAPO optimizer | `scripts/run_phase2_budgeted_mocapo.py`, `heal_capo/optimizers/*` | block_evaluator, budget_allocator, intensification, parent/environmental selection, evolutionary_ops, advance_incumbents; few-shot ops; `--no-llm`, `--seed`, `--output-dir` |
| Meta-LLM offspring ops | `heal_capo/optimizers/evolutionary_ops.py` | CAPO-style crossover/mutation via meta-LLM (`use_meta_llm`); meta-prompts match paper Table 10; deterministic fallback on error |
| In-loop fairness | runner + `heal_capo/fairness.py` | real cached `fairness_risk` drives the 4-objective front (`fairness.in_loop`) |
| Combined fairness_risk | `heal_capo/fairness.py` | flip 0.50 / group-gap 0.25 / bias 0.15 / debt 0.10; DSP/EO/EOdds opt-in |
| BBQ fairness path | `heal_capo/fairness_bbq.py`, `load_bbq`, runner `multiple_choice` | sAMB/sDIS; `bbq_score` mode → `fairness_risk`; in-loop + at Dtest |
| Baselines | `baselines/`, `scripts/run_baseline_*` | GEPA, NSGA-II-PO, CAPO, EvoPromptGA/DE, OPRO, MO-CAPO-style |
| MO metrics + export | `heal_capo/evaluation/mo_metrics.py` | HV (optimistic/pessimistic), approximation_gap, nR2 + Chebychev prefs |
| Held-out eval / table / figures | `evaluate_pareto_on_test.py`, `build_experiment_table.py`, `visualize_paper_figures.py` | few-shot restored at Dtest; per-call inference cost separated from search overhead |
| HPC scaffolding | `scripts/hpc/run_bbq_hpc.slurm`, `run_bbq_nsga_hpc.slurm`, `run_bbq_eval_hpc.slurm`, `configs/HPC_Config/*` | UT Rocket/Pegasus2 vLLM-in-job path; defaults to seed 0 |

---

## Environment

- Rocket/HPC active path: vLLM started by SLURM on Pegasus2, serving
  `mistralai/Mistral-Small-3.2-24B-Instruct-2506` as `mistralai/mistral-small-3.2`.
- Local fallback path: LM Studio @ `localhost:1234` serving `mistralai/mistral-small-3.2`.
- Cost weights: input 0.08 / output 0.32 (exact Mistral OpenRouter average, paper A.5).
- pytest: `testpaths = ["tests"]` (third_party excluded — missing dep + filename collision).

## History (condensed; full detail in `.claude/.../memory/`)

- **S2–S6** few-shot operators + real Ddev split; intensification block-bookkeeping fix + front
  deepening; fixed the "final front = 1-block flukes" bug; long runs silently killed by laptop sleep
  (→ disable sleep, run unbuffered).
- **S7–S8** intensified live re-run verified; held-out Dtest staircase 0.60→0.96; renamed
  HEAL-CAPO→FairCAPO; rebuilt table + method-comparison figures (FairCAPO 0.51→0.967, HV 0.736, |P| 21).
- **S9** integrated **BBQ** end-to-end (loader, `multiple_choice`, canonical sAMB/sDIS bias score,
  NSGA+fairness, ablation, post-hoc, configs). Caught a real bug: `stereotyped_groups` uses CODES
  (`"F"`) vs `answer_info` surface labels (`"woman"`) — fixed in `_group_is_stereotyped`.
- **S10** ran the FULL live BBQ pipeline; headline **FairCAPO fairness 0.000 vs ablation 0.120**;
  efficiency story reproduced (5-pt vs 1-pt front); fixed exponential-HV hang via `dedupe_by_objective`.
- **S11** post-hoc baseline → **KEY FINDING:** held-out `|sAMB|` saturates → ties FairCAPO; the
  `0.120→0.000` win is **in-sample only**; real bias in `sDIS`. Data incident: a `--no-llm` dry run
  clobbered live outputs (recovered from log). End-of-S11: discovered FairCAPO's front is **dominated
  by NSGA** (HV 0.883 > 0.879) → S12 mandate to tune fairly.
- **S12** diagnosed the domination (cost mis-reporting + narrow search, not budget) and did a full
  **paper-parity pass**: sDIS fold (`max_amb_dis`), **meta-LLM enabled**, k_max 1→3, search tuning,
  budget 250k→500k, NSGA unbudgeted, per-call inference-cost reporting, HV_opt/pes/Gap columns, Dtest
  60→200 + nR2 500 vectors, **3-seed orchestration** (`run_bbq_multiseed.sh` + `aggregate_multiseed.py`,
  `--seed`/`--portfolio-csv` flags). Gap analysis in `docs/mocapo_gap_analysis_S12.md`. **380 pytest
  green.** Seed-0 validation run IN PROGRESS. All work UNCOMMITTED.
- **S13** confirmed the S12 seed-0 run **DIED before writing output** (no `outputs/seed_0/` dir, 0-byte
  `tee` log, no `budget_summary.json`). Diagnosed from the 4 **LM Studio server logs** in `docs/`
  (`2026-06-22.4.log`, `2026-06-23.{1,2,3}.log`): continuous serving 23:59→01:47 (~1h48m, matches the
  >1h40m S12 runtime) ending clean with no server-side error → **python client killed at ~01:47 during
  the silent final write** (likely the laptop-sleep kill; sleep guard not set). Run results LOST,
  nothing recoverable. **Relaunch deferred to tomorrow (S14)** with the sleep guard set first (see the
  💀 block + NEXT STEPS #1). Still UNCOMMITTED.
- **S14** (2026-06-24) set the sleep guard first, **relaunched seed 0 → it COMPLETED** (413k/500k tok,
  43 cands, 18 Pareto, in-search HV 0.619, cost 362→1400, meta-LLM 93/162). Synced it into the default
  dir and ran the rest of the seed-0 comparison via `scripts/run_bbq_seed0_pipeline.sh`: ablation +
  NSGA searches ✅, ablation held-out Dtest (HV 0.182) ✅, NSGA held-out Dtest (HV 0.201) ✅. Bumped
  NSGA pop 8→16 / off 3→4 for mu/c parity (allowed direction). **KEY: held-out `fairness_risk` now
  VARIES** (ablation 0.455–0.75) — the S12 `max_amb_dis` sDIS fold finally rescues the held-out
  fairness axis from the S11 saturation. The **FairCAPO held-out Dtest HUNG once at startup** (11 MB
  flat, LM Studio idle 10+ min); killed + re-running via `scripts/run_bbq_seed0_recover_step3.sh`
  (table/figures the pipeline built carry a stale FairCAPO held-out row until recovery rebuilds them).
  Also created the **HPC config + script set** (`configs/HPC_Config/`, `scripts/hpc/run_bbq_*.slurm`,
  `sweep_seeds_*.sh`, `docs/hpc_lm_studio_hosting.md`) for the eventual 3-seed sweep on UT HPC. Open
  item: the d-signal HV comparison (FairCAPO held-out HV) once recovery finishes. Still UNCOMMITTED.
- **S15** (2026-06-24) opened the **Stage A fairness-resolution diagnostic** to test whether the S14
  held-out tie-vs-NSGA is real or a metric-quantization artifact (36-pair held-out eval → sDIS in ~1/6
  steps). Built `data/fairness_bbq_holdout_large.jsonl` (**180 items / 90 disambiguated**, disjoint
  from in-loop), cloned 3 `*_diag` eval configs (`eval_pairs=180`, `test_size=200`, `max_amb_dis`,
  `*_diag` output dirs), wrote `scripts/run_bbq_diag.sh` (sweeps all 3 held-out Dtests on the large
  set). **The ablation diag eval was launched directly and HUNG at startup** (two workers from 18:01,
  1h+ at 400 KB / ~15 MB, ~8s CPU — S14 hang signature) → **zero diagnostic results yet**, no logs/
  output dirs. Recovery = kill stuck pythons, relaunch via `run_bbq_diag.sh`. Still UNCOMMITTED.
