# MO-CAPO paper vs. FairCAPO — gap analysis (S12)

Source: `docs/_mocapo_extracted.txt` (full paper). Read §5, §6.2 ablations, A.1, A.2, A.3,
A.3.4, A.4.1–A.4.3, A.5, A.7, A.8. This lists what the paper does that we don't yet, what
we already match, and what we should *not* copy (because our task differs).

Our scope is BBQ + Bias-in-Bios (fairness) + Subj (no-harm control), 4 objectives
(perf, cost, risk, fairness_risk). MO-CAPO is 2 objectives (perf, cost), 4 datasets
(AG News, GSM8K, Subj, MBPP), 3 models. So some divergence is correct by design;
the gaps below are the ones that actually matter for credibility.

---

## A. Hyperparameter parity (A.3.4) — biggest single fix

Paper defaults (Mistral-3.2-24B, the model we use):

| HP | Paper symbol | Paper value | Our BBQ config | Verdict |
|---|---|---|---|---|
| crossovers/offspring per iter | c | **4** | 4 | ✅ match (after S12) |
| population size | μ | **10** (fixed by 10 sampled init prompts) | 16 | ⚠️ we went higher; paper A.7 says μ=12 best but gains tiny |
| max few-shot | k_max | **5** | 1 | ⚠️ we cut hard for cost; see note |
| block size | b | **30** | 6 | ✅ correct scaling (our Ddev=36 vs paper 300) |
| D_dev | — | **300** | 36 | ⚠️ small — see §C |
| D_shots | — | **100** | (pool_size 20) | ⚠️ small |
| D_test | — | **500** | 10 | ❌ far too small — see §C |
| budget | — | **7.5M tokens** Φeval | 500k | ⚠️ see §F |
| seeds | — | **3** (seeds control init sampling, partition, decoding, stochastic ops) | 1 (seed:0) | ❌ no error bars — see §B |
| iteration cap | — | 2000 steps | (max_iterations 8) | ✅ fine |
| max output length | — | 3000 | (not set) | check |
| input/output weights | w_in/w_out | **0.08 / 0.32** | 0.08 / 0.32 | ✅ exact match (A.5 Mistral avg) |

**k_max divergence (k_max=5 → we use 1):** the paper's accuracy/cost staircase is *driven*
by varying few-shot count 0..5 (A.8 examples: best-perf prompt has 3+ shots, lowest-cost has
0). By capping at 1 we *flatten our own staircase* — the very trade-off MO-CAPO showcases.
We cut it to push cheap prompts into NSGA's region, but k_max=1 likely throws away the
high-accuracy end of the front. **Recommend k_max=3 (compromise), not 1.** Let the cost
objective penalize long prompts rather than hard-capping shots.

---

## B. Three random seeds + error bars (§5) — **the most important credibility gap**

The paper runs **every config with 3 seeds** and reports **mean ± std** for every number
(Tables 2, 4, 5, 16, 17, 18). Seeds control: init-instruction sampling, dataset partition,
LLM decoding, and stochastic algorithm components. We run **seed:0 only** → single point
estimates, no error bars. A reviewer will reject single-seed claims for a stochastic
optimizer, especially a "FairCAPO beats NSGA" headline that currently hinges on HV 0.879
vs 0.883 (a difference far inside one seed's noise).

**Action:** run FairCAPO + ablation + NSGA at **≥3 seeds** (the runner already takes
`--seed`). Report mean ± std. This is non-negotiable for publication and directly addresses
the "FairCAPO dominated by NSGA" worry — the gap may vanish or reverse within noise.

---

## C. Dataset partition sizes (A.3.2) — ours are ~10× too small

Paper: D_dev=300, D_shots=100, **D_test=500**. Ours (BBQ): dev_size=36, test_size=10,
fairness holdout=18. Our **D_test=10** cannot support a generalization claim — 10 items
gives accuracy resolution of 0.1 and saturates instantly. This is *why* our held-out
accuracy pins at 1.0 and |sAMB|→0 (the S11 saturation finding is partly a sample-size
artifact, not only a model-strength one).

**Action:** enlarge BBQ Dtest to the largest disjoint holdout BBQ supports (ideally ~100+),
and Ddev toward ~100–300 if the budget allows. Bias-in-Bios should be sized to the paper's
300/100/500 from the start.

---

## D. Multi-objective metrics & normalization (§5 + A.1) — partially missing

Paper reports, per method: **nR2** (Chebychev utility, averaged over **500 uniform preference
vectors**), **HV_opt** (optimistic), **HV_pes** (pessimistic), and **Gap = HV_opt − HV_pes**.
Plus **global min-max normalization per dataset**, bounds from the *union of all incumbents
across dev+test, all optimizers, all seeds*. Plus **EAS plots** (2/3-EAS median line, 1/3 &
3/3 bands over 3 seeds).

We have: HV (single), nR2 implemented (`mo_metrics.py`). We're **missing in the table**:
optimistic/pessimistic HV split, the Gap, and the cross-method/seed normalization. We report
a single HV, which the paper explicitly argues is insufficient (robustness needs opt/pes).

**Action:** (1) add HV_opt / HV_pes / Gap columns to the experiment table (we already compute
optimistic/pessimistic per a CLAUDE.md note — surface them). (2) Implement the union-based
global normalization once seeds exist. (3) nR2 with 500 preference vectors as the primary
convergence metric, not just HV.

---

## E. Answer/prompt extraction parity (A.3.4) — verify we match

Paper uses **marker-based extraction**: prompts between `<prompt></prompt>`, predictions
between `<final_answer></final_answer>`, and these markers are embedded in initial
instructions + task descriptions. We have `require_prompt_tags: true` and
`preserve_output_format: true`, which suggests we match — **verify** our BBQ initial
instructions and task description actually carry the `<final_answer>` markers, since BBQ is
multiple-choice (the paper never ran MC). If our extraction differs, the cost/accuracy
numbers aren't comparable to the paper's basis.

---

## F. Budget: is 500k enough? (§5 says 7.5M; A.7 + Table 4 give the real guidance)

Paper budget = **7.5M tokens** (Φeval only). We use 500k (~15× smaller). BUT — and this is
the key insight from Table 4 + the trajectory analysis — **MO-CAPO reaches near-final
performance well before the full budget**, often <1/3 of it, and on Mistral-Subj the first
solution set appears at ~109k tokens. Table 4 (Mistral): MO-CAPO evaluates **168 cands on
GSM8K / 442 on Subj** at 7.5M.

What this means for us:
- 500k is **enough to get a valid front** (the first set forms in ~100–330k on Mistral).
- 500k is **NOT enough to match the paper's candidate counts** (we'd see ~30–50 cands vs
  their 168–442) → our front will be sparser and our "richer front than NSGA" claim weaker.
- The honest move: **report the trajectory** (HV/nR2 vs budget), as the paper does (Fig 2),
  rather than a single end-of-budget point. That reframes 500k as "competitive early" instead
  of "under-budget."

**Recommendation on your question:** 500k is fine **for a first valid S12 result with the
meta-LLM on**. But for the *paper*, plan a **higher budget (≥2M, ideally toward 7.5M) on the
real run**, because (a) the meta-LLM now consumes more per iteration, (b) candidate count
drives the front-richness argument, and (c) you need 3 seeds × that budget. Do the cheap 500k
× (FairCAPO/ablation/NSGA) run now to validate the pipeline + meta-LLM + sDIS fix; then scale
budget and seeds for the publishable numbers. Don't scale budget before confirming this run's
front is non-degenerate.

---

## G. Cost-objective ablation (§6.2 "Effect of Cost Weights") — we should add it

The paper isolates the cost objective by setting w_in=0, w_out=0 and shows input-token weight
is the dominant lever (Table 5). This is a clean, expected reviewer ask: "does your cost
objective actually do anything?" We have the machinery (cost weights are config). **Action:**
add a w_in=0 / w_out=0 / both-zero ablation for our chosen showcase dataset. Bonus: both-zero
reduces FairCAPO to a 3-objective (perf, risk, fairness) optimizer — a natural ablation.

---

## H. Single-objective baselines (§5, A.6) — partially present, not wired for BBQ

Paper compares max-accuracy front member against **CAPO, EvoPromptGA, GEPA** (single-obj).
We have runners (`baselines/`, `scripts/run_baseline_*`, `run_phase1_baselines.py`) but they
are not in the BBQ pipeline/table. For the fairness story these are lower priority (they have
no fairness objective), but the paper measures fairness *post-hoc* on single-obj winners — we
do exactly this with the post-hoc baseline (S11), so we're conceptually covered. **Optional:**
add GEPA/CAPO max-accuracy points to the BBQ table for completeness.

---

## I. Meta-prompt templates (A.4.3) — adopt the paper's exact wording

The paper's crossover + mutation meta-prompts (Table 10) are short and specific:
- **Crossover:** "You receive two prompts for the following task: <task_description>. Please
  merge the two prompts into a single coherent prompt. Maintain the key linguistic features
  from both ... Prompt 1: <mother> Prompt 2: <father>. Return ... <prompt>new prompt</prompt>."
- **Mutation:** "You receive a prompt for ... <task_description>. Please rephrase the prompt,
  preserving its core meaning while substantially varying the linguistic style. Prompt:
  <instruction>. Return ... <prompt>new prompt</prompt>."

Now that we've enabled `use_meta_llm: true` (S12), **verify our meta-prompt templates match
this wording** (in `heal_capo/optimizers/evolutionary_ops.py` / prompt_generator). Divergent
meta-prompts = divergent search behavior, and matching them keeps us faithful to CAPO.
Also note A.4.1: task descriptions **must embed the `<final_answer>` marker instruction** —
check our BBQ task description does.

---

## J. Few-shot creation with reasoning (A.2.1) — likely a divergence

CAPO/MO-CAPO create few-shot examples by **prompting the eval-LLM to solve the input and using
its (reasoning + prediction) as the example output**; only on failure do they fall back to the
bare label. This gives "richer information than a label alone." Our few-shot pool for BBQ —
verify whether we attach reasoning or just label. For multiple-choice BBQ this matters less,
but it's a parity point worth a one-line check.

---

## What we already match (good — don't touch)

- ✅ Cost weights 0.08/0.32 (exact Mistral OpenRouter average, A.5).
- ✅ Block-based intensification, advance-incumbents, environmental selection, binary
  tournament with weaker dominance — all implemented (CLAUDE.md architecture rollup).
- ✅ block_size scaled correctly to our Ddev (6 for Ddev=36 ≈ paper's 30 for 300).
- ✅ NSGA-II-PO as the isolation-of-intensification baseline — exactly the paper's framing
  (§6.2: "NSGA-II-PO ... also serves as an ablation, since its main difference ... is the
  intensification mechanism").
- ✅ c=4 offspring per iteration (after S12 tuning).
- ✅ Marker-based extraction infrastructure (`require_prompt_tags`, `preserve_output_format`).
- ✅ Post-hoc fairness on single-obj/ablation winners (our S11 baseline = paper's "measure
  fairness post-hoc on the max-accuracy solution" idea, applied to fairness).

---

## Priority-ordered action list (for the paper, beyond the current S12 run)

1. **3 seeds + mean±std** everywhere (§B). Single-seed is the #1 reviewer-killer.
2. **D_test ≫ 10** (§C). Our generalization claim needs ~100–500 held-out items;
   current saturation is partly a sample-size artifact.
3. **k_max=1 → 3** (§A). We flattened our own accuracy/cost staircase; restore it.
4. **Report HV_opt / HV_pes / Gap + nR2(500 prefs)** in the table, with cross-seed global
   normalization (§D). Single HV is what the paper argues against.
5. **Trajectory plots** (HV/nR2 vs budget) (§F) — reframes our smaller budget as
   "competitive early," the paper's own efficiency argument.
6. **Cost-weight ablation** (w_in=0/w_out=0) (§G) — cheap, expected.
7. **Verify meta-prompt wording + few-shot-with-reasoning + <final_answer> markers** match
   the paper (§E, §I, §J).
8. **Scale budget toward 2M–7.5M** for the publishable run (§F) — only after the 500k run
   confirms a non-degenerate front.

## Bottom line on the budget question

500k is **adequate to validate** the S12 changes (meta-LLM + sDIS fold + tuned search) and
get a real front — the paper shows Mistral forms its first solution set at ~100–330k. It is
**not adequate for the final paper numbers**: you'll want ≥2M (toward 7.5M) AND 3 seeds so the
candidate count and error bars match the paper's basis. Keep the current 500k run going to
validate; budget the publishable re-run higher.
