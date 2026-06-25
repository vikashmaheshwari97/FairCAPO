from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

# Held-out candidate CSVs store the full prompt + few-shot block in one field,
# which can exceed Python's default 128 KB csv field limit (raised an
# `_csv.Error: field larger than field limit` on the seed-0 FairCAPO holdout).
# 10 MB is plenty and stays within the C long range on Windows.
csv.field_size_limit(10 * 1024 * 1024)

from heal_capo.core import EvaluationResult
from heal_capo.evaluation.mo_metrics import (
    DEFAULT_OBJECTIVE_SPECS,
    ObjectiveSpec,
    fixed_bounds_from_config,
    summarize_mo_metrics,
)


# Columns reported per method, in order.
TABLE_COLUMNS = [
    "method",
    "performance",
    "cost",
    # Held-out INFERENCE cost, with the one-time fairness-audit cost separated out
    # (only populated for methods that declare a `holdout_csv`). The plain `cost`
    # column above is the search-basis summed dev cost (fairness folded in); these
    # three are the honest "what does deploying this prompt cost per query" view.
    "inference_cost_per_call",
    "inference_cost_total",
    "fairness_eval_cost",
    "risk",
    "fairness_risk",
    "hypervolume",
    # Paper-parity MO metrics (Büssing et al. Table 2): optimistic & pessimistic HV
    # and their Gap measure Pareto-front ROBUSTNESS (a single HV cannot). nR2 (over
    # 500 preference vectors) is the primary convergence metric. All four come from
    # summarize_mo_metrics; only single-HV was surfaced before.
    "optimistic_hypervolume",
    "pessimistic_hypervolume",
    "approximation_gap",
    "nr2",
    "portfolio_size",
    "budget_used",
    "source",
]


def _json_default(obj: Any):
    try:
        return obj.item()
    except AttributeError:
        return str(obj)


def load_yaml(path: str) -> dict:
    yaml_path = Path(path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(data: Any, path: str):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def save_csv(rows: list[dict], path: str):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"No rows to save for {path}")

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv_rows(path: str) -> list[dict]:
    csv_path = Path(path)

    if not csv_path.exists():
        return []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json_if_exists(path: str | None) -> dict:
    if not path:
        return {}

    json_path = Path(path)

    if not json_path.exists():
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None or value == "":
        return default

    try:
        return float(value)
    except Exception:
        return default


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def objective_specs_from_config(config: dict) -> tuple[ObjectiveSpec, ...]:
    objectives = config.get("objectives")

    if not objectives:
        return DEFAULT_OBJECTIVE_SPECS

    specs = []

    for item in objectives:
        if isinstance(item, str):
            if item == "performance":
                specs.append(ObjectiveSpec(item, "maximize"))
            else:
                specs.append(ObjectiveSpec(item, "minimize"))
            continue

        specs.append(
            ObjectiveSpec(
                name=str(item["name"]),
                direction=str(item.get("direction", "minimize")),
            )
        )

    return tuple(specs)


def empty_row(name: str, source: str = "missing") -> dict:
    return {
        "method": name,
        "performance": None,
        "cost": None,
        "inference_cost_per_call": None,
        "inference_cost_total": None,
        "fairness_eval_cost": None,
        "risk": None,
        "fairness_risk": None,
        "hypervolume": None,
        "optimistic_hypervolume": None,
        "pessimistic_hypervolume": None,
        "approximation_gap": None,
        "nr2": None,
        "portfolio_size": 0,
        "budget_used": None,
        "source": source,
    }


def single_point_row(method_cfg: dict) -> dict:
    """
    Pull one row from a phase1-style baseline_table.csv / final summary CSV.

    These baselines optimize performance + cost only, so risk/fairness default
    to 0.0 and portfolio_size is 1. HV/nR2 are left blank (a single point has
    no meaningful Pareto-set quality).
    """
    name = str(method_cfg["name"])
    csv_path = str(method_cfg.get("csv", ""))
    match = str(method_cfg.get("match", "")).strip().lower()

    rows = load_csv_rows(csv_path)

    if not rows:
        print(f"[warn] {name}: CSV not found or empty: {csv_path}")
        return empty_row(name)

    selected = None
    for row in rows:
        if str(row.get("method", "")).strip().lower() == match:
            selected = row
            break

    if selected is None:
        print(f"[warn] {name}: no row with method=='{match}' in {csv_path}")
        return empty_row(name, source=csv_path)

    # Prefer test_score/test_cost; fall back to dev or generic columns.
    performance = (
        parse_float(selected.get("test_score"), None)
        if selected.get("test_score") not in (None, "")
        else None
    )
    if performance is None:
        performance = parse_float(
            selected.get("performance", selected.get("dev_score")), None
        )

    cost = (
        parse_float(selected.get("test_cost"), None)
        if selected.get("test_cost") not in (None, "")
        else None
    )
    if cost is None:
        cost = parse_float(selected.get("cost", selected.get("dev_cost")), None)

    return {
        "method": name,
        "performance": performance,
        "cost": cost,
        "risk": parse_float(selected.get("risk"), 0.0),
        "fairness_risk": parse_float(selected.get("fairness_risk"), 0.0),
        "hypervolume": None,
        "nr2": None,
        "portfolio_size": 1,
        "budget_used": parse_float(
            selected.get("budget_used", selected.get("test_cost")), None
        ),
        "source": csv_path,
    }


def filter_pareto_rows(rows: list[dict], only_pareto: bool) -> list[dict]:
    if not only_pareto:
        return rows

    if not any("is_pareto" in row for row in rows):
        # No is_pareto column (e.g. mocapo_style baseline_table); use all rows.
        return rows

    pareto = [row for row in rows if parse_bool(row.get("is_pareto", False))]
    return pareto or rows


def row_to_result(row: dict, index: int) -> EvaluationResult:
    candidate_id = str(
        row.get("candidate_id")
        or row.get("method")
        or row.get("prompt_id")
        or f"row_{index}"
    )

    # Performance: prefer explicit performance, else test_score/dev_score.
    performance = parse_float(
        row.get("performance"),
        None,
    )
    if performance is None:
        performance = parse_float(
            row.get("test_score", row.get("dev_score")), 0.0
        )

    cost = parse_float(row.get("cost"), None)
    if cost is None:
        cost = parse_float(row.get("test_cost", row.get("dev_cost")), 0.0)

    return EvaluationResult(
        candidate_id=candidate_id,
        performance=performance or 0.0,
        cost=cost or 0.0,
        risk=parse_float(row.get("risk"), 0.0),
        fairness_risk=parse_float(row.get("fairness_risk"), 0.0),
        drift=parse_float(row.get("drift"), 0.0),
    )


def dedupe_by_objective(
    results: list[EvaluationResult],
) -> list[EvaluationResult]:
    """Drop results with a duplicate objective vector.

    A Pareto front can contain many candidates that share an identical
    (performance, cost, risk, fairness_risk) vector. Duplicates do not change
    the hypervolume, but they blow up the inclusion-exclusion HV computation
    (cost ~2^n in the number of points), so a front of e.g. 36 duplicate rows
    that collapses to 5 unique points would otherwise hang. Deduplicate first.
    """
    seen: set[tuple[float, float, float, float]] = set()
    unique: list[EvaluationResult] = []
    for r in results:
        key = (
            round(r.performance, 9),
            round(r.cost, 9),
            round(r.risk, 9),
            round(r.fairness_risk, 9),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def holdout_inference_cost(holdout_csv: str, only_pareto: bool = True) -> dict:
    """
    Honest per-call INFERENCE cost from a held-out Dtest CSV.

    The held-out evaluator logs each candidate's total ``cost`` (summed over the
    test set with the one-time fairness audit folded in), the separable
    ``detail_fairness_eval_cost``, and the call count ``detail_total``. A deployed
    prompt pays only inference per query, NOT the fairness audit, so:

        inference_cost       = cost - detail_fairness_eval_cost
        inference_per_call   = inference_cost / detail_total

    Returns the cheapest-to-deploy front member's per-call + total inference cost
    and the mean fairness-audit cost (reported separately, not hidden in cost).
    Returns empty dict if the file is missing or lacks the needed columns.
    """
    rows = load_csv_rows(holdout_csv)
    if not rows:
        return {}
    rows = filter_pareto_rows(rows, only_pareto=only_pareto)

    per_call = []
    audit_costs = []
    for row in rows:
        cost = parse_float(row.get("cost"), None)
        n = parse_float(row.get("detail_total"), None)
        if cost is None or not n:
            continue
        audit = parse_float(row.get("detail_fairness_eval_cost"), 0.0) or 0.0
        inference = max(0.0, cost - audit)
        per_call.append((inference / n, inference, n))
        audit_costs.append(audit)

    if not per_call:
        return {}

    cheapest = min(per_call, key=lambda t: t[0])
    return {
        "inference_cost_per_call": cheapest[0],
        "inference_cost_total": cheapest[1],
        "fairness_eval_cost": sum(audit_costs) / len(audit_costs),
    }


def best_per_objective(results: list[EvaluationResult]) -> dict:
    """Best value per objective across the set (perf=max, others=min)."""
    return {
        "performance": max(r.performance for r in results),
        "cost": min(r.cost for r in results),
        "risk": min(r.risk for r in results),
        "fairness_risk": min(r.fairness_risk for r in results),
    }


def pareto_set_row(method_cfg: dict, table_config: dict) -> dict:
    """
    Build a row from a candidates CSV (NSGA-II-PO, MO-CAPO-style, HEAL-CAPO).

    Reports best-per-objective over the Pareto set plus HV and nR2.
    """
    name = str(method_cfg["name"])
    csv_path = str(method_cfg.get("candidates_csv", ""))
    only_pareto = bool(method_cfg.get("only_pareto", True))

    rows = load_csv_rows(csv_path)

    if not rows:
        print(f"[warn] {name}: candidates CSV not found or empty: {csv_path}")
        return empty_row(name)

    rows = filter_pareto_rows(rows, only_pareto=only_pareto)
    results = [row_to_result(row, idx) for idx, row in enumerate(rows)]
    # Collapse duplicate objective vectors before MO metrics: identical points
    # do not affect HV/nR2 but make the inclusion-exclusion HV blow up.
    results = dedupe_by_objective(results)

    if not results:
        print(f"[warn] {name}: no usable candidate rows in {csv_path}")
        return empty_row(name, source=csv_path)

    best = best_per_objective(results)

    objective_specs = objective_specs_from_config(table_config)
    metric_cfg = table_config.get("metrics", {})
    num_preference_vectors = int(metric_cfg.get("num_preference_vectors", 50))
    seed = int(metric_cfg.get("seed", table_config.get("seed", 0)))

    bounds = fixed_bounds_from_config(
        bounds_config=table_config.get("bounds"),
        objective_specs=objective_specs,
    )

    summary = summarize_mo_metrics(
        candidate_results=results,
        reference_results=results,
        objective_specs=objective_specs,
        num_preference_vectors=num_preference_vectors,
        seed=seed,
        bounds=bounds,
    ).to_dict()

    budget_summary = load_json_if_exists(method_cfg.get("budget_json"))
    budget_used = parse_float(
        budget_summary.get("used_budget", budget_summary.get("total_tokens")),
        None,
    )

    # Optional held-out inference-cost view (search-overhead separated out).
    holdout_csv = str(method_cfg.get("holdout_csv", "")).strip()
    inference = holdout_inference_cost(holdout_csv) if holdout_csv else {}

    return {
        "method": name,
        "performance": best["performance"],
        "cost": best["cost"],
        "inference_cost_per_call": inference.get("inference_cost_per_call"),
        "inference_cost_total": inference.get("inference_cost_total"),
        "fairness_eval_cost": inference.get("fairness_eval_cost"),
        "risk": best["risk"],
        "fairness_risk": best["fairness_risk"],
        "hypervolume": summary.get("hypervolume"),
        "optimistic_hypervolume": summary.get("optimistic_hypervolume"),
        "pessimistic_hypervolume": summary.get("pessimistic_hypervolume"),
        "approximation_gap": summary.get("approximation_gap"),
        "nr2": summary.get("nr2"),
        "portfolio_size": len(results),
        "budget_used": budget_used,
        "source": csv_path,
    }


def build_method_row(method_cfg: dict, table_config: dict) -> dict:
    method_type = str(method_cfg.get("type", "single_point")).strip().lower()

    if method_type == "pareto_set":
        return pareto_set_row(method_cfg, table_config)

    return single_point_row(method_cfg)


def build_experiment_table(config: dict) -> list[dict]:
    methods = config.get("methods", [])

    if not methods:
        raise ValueError("Config must list at least one method under 'methods'.")

    rows = [build_method_row(method_cfg, config) for method_cfg in methods]

    # Ensure consistent column ordering.
    ordered = []
    for row in rows:
        ordered.append({col: row.get(col) for col in TABLE_COLUMNS})

    return ordered


def _fmt(value: Any, fmt: str = "{:.3f}") -> str:
    if value is None or value == "":
        return "--"

    try:
        return fmt.format(float(value))
    except Exception:
        return str(value)


def make_latex_table(rows: list[dict], config: dict) -> str:
    dataset = config.get("dataset", "")
    model = config.get("model", "")

    # Paper Table 2 layout: optimistic & pessimistic HV + Gap (robustness) + nR2.
    header_cells = [
        "Method",
        "Perf.",
        "Cost",
        "Risk",
        "Fair.",
        r"HV$_{opt}$",
        r"HV$_{pes}$",
        "Gap",
        "nR2",
        "$|P|$",
        "Budget",
    ]
    header_line = " & ".join(header_cells) + r" \\"

    value_lines = []
    for row in rows:
        cells = [
            str(row.get("method", "")),
            _fmt(row.get("performance")),
            _fmt(row.get("cost"), "{:.2f}"),
            _fmt(row.get("risk")),
            _fmt(row.get("fairness_risk")),
            _fmt(row.get("optimistic_hypervolume")),
            _fmt(row.get("pessimistic_hypervolume")),
            _fmt(row.get("approximation_gap")),
            _fmt(row.get("nr2")),
            str(row.get("portfolio_size", 0)),
            _fmt(row.get("budget_used"), "{:.1f}"),
        ]
        value_lines.append(" & ".join(cells) + r" \\")

    caption = (
        f"Comparison of prompt optimization methods on {dataset} ({model}). "
        "Multi-prompt methods report best-per-objective over the Pareto set. "
        r"HV$_{opt}$/HV$_{pes}$ (optimistic/pessimistic hypervolume) and their Gap "
        "measure Pareto-front quality and robustness (higher HV / lower Gap better); "
        "nR2 measures convergence (lower better)."
    )

    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrrrrrrrrrr}",
            r"\toprule",
            header_line,
            r"\midrule",
            *value_lines,
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            r"\label{tab:experiment_comparison}",
            r"\end{table}",
        ]
    )


def print_table(rows: list[dict]):
    print("Experiment comparison table")
    print("-" * 80)
    header = (
        f"{'method':<16}{'perf':>8}{'cost':>9}{'risk':>8}"
        f"{'fair':>8}{'HV':>8}{'nR2':>8}{'|P|':>6}"
    )
    print(header)
    for row in rows:
        print(
            f"{str(row.get('method','')):<16}"
            f"{_fmt(row.get('performance')):>8}"
            f"{_fmt(row.get('cost'), '{:.2f}'):>9}"
            f"{_fmt(row.get('risk')):>8}"
            f"{_fmt(row.get('fairness_risk')):>8}"
            f"{_fmt(row.get('hypervolume')):>8}"
            f"{_fmt(row.get('nr2')):>8}"
            f"{str(row.get('portfolio_size', 0)):>6}"
        )
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Build a cross-method experiment comparison table."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment_table.yaml",
        help="Experiment table YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)

    output_dir = args.output_dir or config.get(
        "output_dir",
        "outputs/experiment_table/subj_mistral",
    )

    rows = build_experiment_table(config)
    latex_table = make_latex_table(rows, config)

    save_csv(rows, f"{output_dir}/experiment_table.csv")
    save_json(
        {
            "dataset": config.get("dataset", ""),
            "model": config.get("model", ""),
            "columns": TABLE_COLUMNS,
            "rows": rows,
        },
        f"{output_dir}/experiment_table.json",
    )

    with open(f"{output_dir}/experiment_table_latex.txt", "w", encoding="utf-8") as f:
        f.write(latex_table)

    print_table(rows)
    print(f"Saved experiment table CSV to: {output_dir}/experiment_table.csv")
    print(f"Saved experiment table JSON to: {output_dir}/experiment_table.json")
    print(f"Saved LaTeX table to: {output_dir}/experiment_table_latex.txt")


if __name__ == "__main__":
    main()
