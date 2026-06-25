"""
Aggregate a multi-seed BBQ sweep into a mean +/- std table.

Büssing et al. (MO-CAPO) run every config with 3 seeds and report mean +/- std on
every metric. `scripts/run_bbq_multiseed.sh` produces per-seed held-out Dtest outputs
under `outputs/seed_{S}/evaluation/<method-subdir>/`. This script reads those
`test_eval_summary.json` + `test_eval_candidates.csv` files across seeds and emits a
per-method table with mean +/- std for each metric.

Metrics aggregated per method (over seeds):
  - performance        : max performance over the Pareto front (best accuracy point)
  - fairness_risk      : min fairness_risk over the front (fairest point)
  - hypervolume / optimistic_hypervolume / pessimistic_hypervolume / approximation_gap
  - nr2
  - inference_cost_per_call : honest per-call inference cost (fairness-audit cost
                              separated out), via build_experiment_table.holdout_inference_cost

Usage:
  python scripts/aggregate_multiseed.py --config configs/aggregate_multiseed_bbq.yaml
or with defaults baked for the BBQ sweep:
  python scripts/aggregate_multiseed.py --base outputs/seed_{seed} --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from scripts.build_experiment_table import (
    filter_pareto_rows,
    holdout_inference_cost,
    load_csv_rows,
    parse_float,
    save_csv,
    save_json,
)

# Metrics pulled straight from test_eval_summary.json["summary"].
SUMMARY_METRICS = [
    "hypervolume",
    "optimistic_hypervolume",
    "pessimistic_hypervolume",
    "approximation_gap",
    "nr2",
]

# Metrics we aggregate in addition to the summary ones.
ALL_METRICS = [
    "performance",
    "fairness_risk",
    *SUMMARY_METRICS,
    "inference_cost_per_call",
]

DEFAULT_METHODS = [
    {"name": "FairCAPO", "eval_subdir": "evaluation/bbq_mistral_local"},
    {"name": "MO-CAPO (fairness off)", "eval_subdir": "evaluation/bbq_ablation_local"},
    {"name": "NSGA-II-PO + fairness", "eval_subdir": "evaluation/bbq_nsga_local"},
]


def load_summary(eval_dir: Path) -> dict | None:
    path = eval_dir / "test_eval_summary.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def per_seed_metrics(eval_dir: Path) -> dict | None:
    """One seed's metrics for a method, or None if its outputs are missing."""
    summary_doc = load_summary(eval_dir)
    candidates_csv = eval_dir / "test_eval_candidates.csv"
    if summary_doc is None or not candidates_csv.exists():
        return None

    summary = summary_doc.get("summary", {})
    out: dict[str, float | None] = {m: summary.get(m) for m in SUMMARY_METRICS}

    rows = filter_pareto_rows(load_csv_rows(str(candidates_csv)), only_pareto=True)
    perfs = [parse_float(r.get("performance"), None) for r in rows]
    fairs = [parse_float(r.get("fairness_risk"), None) for r in rows]
    perfs = [p for p in perfs if p is not None]
    fairs = [f for f in fairs if f is not None]
    out["performance"] = max(perfs) if perfs else None
    out["fairness_risk"] = min(fairs) if fairs else None

    inf = holdout_inference_cost(str(candidates_csv))
    out["inference_cost_per_call"] = inf.get("inference_cost_per_call")
    return out


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)  # sample std
    return mean, math.sqrt(var)


def aggregate_method(name: str, eval_subdir: str, base: str, seeds: list[int]) -> dict:
    per_seed: list[dict] = []
    missing: list[int] = []
    for s in seeds:
        eval_dir = Path(base.replace("{seed}", str(s))) / eval_subdir
        m = per_seed_metrics(eval_dir)
        if m is None:
            missing.append(s)
        else:
            per_seed.append(m)

    row: dict[str, Any] = {"method": name, "n_seeds": len(per_seed)}
    if missing:
        row["missing_seeds"] = ",".join(str(s) for s in missing)
    for metric in ALL_METRICS:
        mean, std = mean_std([m.get(metric) for m in per_seed])
        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std
    return row


def _fmt(value: Any, fmt: str = "{:.3f}") -> str:
    if value is None:
        return "--"
    try:
        return fmt.format(float(value))
    except Exception:
        return str(value)


def print_table(rows: list[dict]) -> None:
    print("\nMulti-seed BBQ aggregate (mean +/- std over seeds)")
    print("-" * 96)
    print(
        f"{'method':<26}{'seeds':>6}{'perf':>14}{'fair':>14}"
        f"{'HV_opt':>14}{'HV_pes':>14}{'nR2':>14}"
    )
    for r in rows:
        def ms(metric: str) -> str:
            return f"{_fmt(r.get(metric + '_mean'))}±{_fmt(r.get(metric + '_std'))}"

        print(
            f"{str(r.get('method','')):<26}{r.get('n_seeds',0):>6}"
            f"{ms('performance'):>14}{ms('fairness_risk'):>14}"
            f"{ms('optimistic_hypervolume'):>14}{ms('pessimistic_hypervolume'):>14}"
            f"{ms('nr2'):>14}"
        )
    print("-" * 96)
    print("perf=max accuracy over front; fair=min fairness_risk over front (lower=fairer).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML: methods/seeds/base/output_dir.")
    parser.add_argument("--base", default="outputs/seed_{seed}", help="Per-seed base dir; {seed} is substituted.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", default="outputs/experiment_table/bbq_mistral_multiseed")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        methods = cfg.get("methods", DEFAULT_METHODS)
        seeds = cfg.get("seeds", args.seeds)
        base = cfg.get("base", args.base)
        output_dir = cfg.get("output_dir", args.output_dir)
    else:
        methods, seeds, base, output_dir = DEFAULT_METHODS, args.seeds, args.base, args.output_dir

    rows = [aggregate_method(m["name"], m["eval_subdir"], base, seeds) for m in methods]

    print_table(rows)
    save_csv(rows, f"{output_dir}/multiseed_aggregate.csv")
    save_json({"seeds": seeds, "base": base, "rows": rows}, f"{output_dir}/multiseed_aggregate.json")
    print(f"\nSaved -> {output_dir}/multiseed_aggregate.{{csv,json}}")


if __name__ == "__main__":
    main()
