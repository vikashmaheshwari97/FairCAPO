# BBQ post-hoc fairness baseline — finding (session 11)

_Date: 2026-06-22_

## What was run

The "post-hoc fairness" baseline answers the reviewer question **"why not just
optimize for accuracy/cost and pick the fairest prompt at the end?"** We took the
**MO-CAPO fairness-OFF** front (the fairness-blind ablation,
`outputs/mocapo_baseline_bbq_local/`) and re-scored each of its 7 Pareto prompts
with the canonical BBQ bias scorer on the **held-out, disjoint** fairness set
(`data/fairness_bbq_holdout.jsonl`, 36 ambiguous + 18 disambiguated items) via:

```
PYTHONPATH=. python scripts/measure_baseline_fairness.py \
  --fairness-config configs/evaluate_pareto_bbq.yaml \
  --inputs outputs/mocapo_baseline_bbq_local/phase2_prompt_portfolio.csv
# -> outputs/mocapo_baseline_bbq_local/phase2_prompt_portfolio_bbqfair.csv
```

This makes LLM calls (7 prompts × the held-out items), no new search.

## The finding (this is important — it qualifies the headline)

The **same fairness-blind front** scores very differently depending on which
fairness set you measure on:

| Fairness set | `fairness_risk` = \|sAMB\| over the fairness-OFF front | min |
|---|---|---|
| **In-loop** `data/fairness_bbq.jsonl` (what FairCAPO optimized on) | {0.12, 0.18, 0.30} | **0.12** |
| **Held-out** `data/fairness_bbq_holdout.jsonl` (disjoint, = Dtest set) | {0.0, 0.04} | **0.00** |

Consequences:

1. **The "in-loop 0.0 vs ablation 0.12" headline is an IN-SAMPLE result.** It holds
   on `data/fairness_bbq.jsonl` — the very set FairCAPO's in-loop objective
   optimized against. The "MO-CAPO (fairness off)" row in the experiment table
   (0.12) is exactly the in-sample post-hoc number.

2. **On held-out data the post-hoc baseline TIES FairCAPO.** FairCAPO's held-out
   Dtest fairness was {0.0, 0.04}; the fairness-blind front re-scored on the *same*
   held-out set is *also* {0.0, 0.04}. So **"in-loop fairness beats
   post-processing" does NOT generalize on this BBQ setup** — both reach ~0.

3. **`|sAMB|` saturates to ~0 on held-out for every prompt.** On the held-out
   ambiguous items mistral-24B abstains ("unknown") correctly regardless of the
   prompt, so the optimized metric cannot distinguish methods there. The real
   residual bias shows up in **sDIS ≈ 0.45–0.60** (disambiguated contexts), which
   `fairness_risk` (= |sAMB|) neither optimizes nor reports in the headline.

## How it is presented in the experiment table

`configs/experiment_table_bbq.yaml` now has BOTH rows, so the table itself shows
both measurement bases:

| method | perf | cost | fairness | basis |
|---|---|---|---|---|
| FairCAPO | 1.000 | 361.9 | 0.000 | in-loop set |
| MO-CAPO (fairness off) | 1.000 | 268.0 | **0.120** | in-loop set (= in-sample post-hoc) |
| NSGA-II-PO + fairness | 1.000 | 350.6 | 0.000 | in-loop set |
| **Post-hoc fair. (held-out)** | 1.000 | 268.0 | **0.000** | **held-out set** |

> Caveat on the post-hoc row's HV (0.910): its perf/cost come from the ablation's
> dev measurement while its fairness comes from the held-out set, so its HV is not
> directly comparable to the other rows — read this row for its **fairness column
> only** (0.0 on held-out).

## Honest takeaways for the paper

- **In-sample**, the in-loop fairness objective does measurably reduce `|sAMB|`
  (0.12 → 0.0 at equal accuracy) where post-hoc selection on a fairness-blind front
  cannot.
- **Out-of-sample (held-out)**, that advantage disappears: `|sAMB|` is ~0 for both,
  because the metric saturates on these held-out ambiguous items.
- Therefore the defensible claim is currently **in-sample**, plus a clear caveat
  that the held-out `|sAMB|` comparison is inconclusive (metric saturation) and that
  meaningful residual bias persists in **sDIS**.

## Suggested next steps (to make the held-out comparison discriminative)

1. Build a **harder / larger held-out fairness set** where mistral-24B does NOT
   abstain trivially on the ambiguous items (so `|sAMB|` varies across prompts).
2. Consider **folding sDIS into `fairness_risk`** (e.g. `max(|sAMB|, |sDIS|)` or a
   weighted blend) so the optimized objective captures disambiguated bias too.
3. Re-run the held-out comparison; only then can "in-loop beats post-hoc" be
   claimed out-of-sample.
