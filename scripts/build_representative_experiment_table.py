from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from heal_capo.core import EvaluationResult
from heal_capo.evaluation.mo_metrics import (
    DEFAULT_OBJECTIVE_SPECS,
    ObjectiveSpec,
    fixed_bounds_from_config,
    summarize_mo_metrics,
)

csv.field_size_limit(10 * 1024 * 1024)


OUTPUT_COLUMNS = [
    "method",
    "representative_candidate_id",
    "representative_source_candidate_id",
    "representative_prompt_hash",
    "performance",
    "cost",
    "risk",
    "fairness_risk",
    "best_performance",
    "min_cost",
    "min_fairness_risk",
    "hypervolume",
    "optimistic_hypervolume",
    "pessimistic_hypervolume",
    "approximation_gap",
    "nr2",
    "eligible_portfolio_size",
    "raw_pareto_size",
    "min_performance_threshold",
    "threshold_fallback_used",
    "budget_used",
    "source",
    "representative_prompt",
]


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_csv_rows(path: str) -> list[dict]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json_if_exists(path: str | None) -> dict:
    if not path or not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def objective_specs_from_config(config: dict) -> tuple[ObjectiveSpec, ...]:
    objectives = config.get("objectives")
    if not objectives:
        return DEFAULT_OBJECTIVE_SPECS

    specs: list[ObjectiveSpec] = []
    for item in objectives:
        if isinstance(item, str):
            specs.append(
                ObjectiveSpec(
                    name=item,
                    direction="maximize" if item == "performance" else "minimize",
                )
            )
        else:
            specs.append(
                ObjectiveSpec(
                    name=str(item["name"]),
                    direction=str(item.get("direction", "minimize")),
                )
            )
    return tuple(specs)


def pareto_rows(rows: list[dict], only_pareto: bool) -> list[dict]:
    if not only_pareto or not any("is_pareto" in row for row in rows):
        return rows
    selected = [row for row in rows if parse_bool(row.get("is_pareto"))]
    return selected or rows


def objective_key(row: dict) -> tuple[float, float, float, float]:
    return (
        round(parse_float(row.get("performance")), 9),
        round(parse_float(row.get("cost")), 9),
        round(parse_float(row.get("risk")), 9),
        round(parse_float(row.get("fairness_risk")), 9),
    )


def dedupe_rows(rows: list[dict]) -> list[dict]:
    seen: set[tuple[float, float, float, float]] = set()
    output: list[dict] = []
    for row in rows:
        key = objective_key(row)
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def row_to_result(row: dict, candidate_id: str) -> EvaluationResult:
    return EvaluationResult(
        candidate_id=candidate_id,
        performance=parse_float(row.get("performance")),
        cost=parse_float(row.get("cost")),
        risk=parse_float(row.get("risk")),
        fairness_risk=parse_float(row.get("fairness_risk")),
        drift=parse_float(row.get("drift")),
    )


def choose_representative(rows: list[dict], policy: str) -> dict:
    if not rows:
        raise ValueError("Cannot choose a representative from an empty portfolio.")

    normalized = policy.strip().lower()
    if normalized in {
        "max_performance_then_min_fairness_then_min_cost",
        "accuracy_first",
    }:
        return min(
            rows,
            key=lambda row: (
                -parse_float(row.get("performance")),
                parse_float(row.get("fairness_risk")),
                parse_float(row.get("cost")),
            ),
        )

    if normalized in {
        "min_cost_then_max_performance_then_min_fairness",
        "cost_first",
    }:
        return min(
            rows,
            key=lambda row: (
                parse_float(row.get("cost")),
                -parse_float(row.get("performance")),
                parse_float(row.get("fairness_risk")),
            ),
        )

    # Default: among usable-accuracy candidates, choose the fairest actual prompt.
    return min(
        rows,
        key=lambda row: (
            parse_float(row.get("fairness_risk")),
            -parse_float(row.get("performance")),
            parse_float(row.get("cost")),
        ),
    )


def prompt_hash(row: dict) -> str:
    payload = (
        str(row.get("prompt", ""))
        + "\n"
        + str(row.get("few_shot_examples", row.get("detail_few_shot_examples", "")))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def prepare_method(method_cfg: dict, config: dict) -> dict:
    name = str(method_cfg["name"])
    source = str(method_cfg.get("candidates_csv", ""))
    rows = load_csv_rows(source)
    if not rows:
        raise FileNotFoundError(f"{name}: candidates CSV missing or empty: {source}")

    raw_pareto = pareto_rows(rows, bool(method_cfg.get("only_pareto", True)))
    raw_pareto = dedupe_rows(raw_pareto)

    selection_cfg = config.get("selection", {}) or {}
    method_selection = method_cfg.get("selection", {}) or {}
    threshold = parse_float(
        method_selection.get(
            "min_performance_for_fairness",
            selection_cfg.get("min_performance_for_fairness", 0.0),
        )
    )
    policy = str(
        method_selection.get(
            "representative_policy",
            selection_cfg.get(
                "representative_policy",
                "min_fairness_then_max_performance_then_min_cost",
            ),
        )
    )

    eligible = [
        row
        for row in raw_pareto
        if parse_float(row.get("performance")) >= threshold
    ]
    threshold_fallback_used = False
    if not eligible:
        eligible = list(raw_pareto)
        threshold_fallback_used = True

    representative = choose_representative(eligible, policy)
    results = [
        row_to_result(row, f"{name}::{idx}")
        for idx, row in enumerate(eligible)
    ]

    return {
        "name": name,
        "source": source,
        "method_cfg": method_cfg,
        "threshold": threshold,
        "threshold_fallback_used": threshold_fallback_used,
        "raw_pareto": raw_pareto,
        "eligible": eligible,
        "representative": representative,
        "results": results,
    }


def dedupe_results(results: list[EvaluationResult]) -> list[EvaluationResult]:
    seen: set[tuple[float, float, float, float]] = set()
    output: list[EvaluationResult] = []
    for result in results:
        key = (
            round(result.performance, 9),
            round(result.cost, 9),
            round(result.risk, 9),
            round(result.fairness_risk, 9),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(result)
    return output


def build_rows(config: dict) -> list[dict]:
    methods = [prepare_method(method_cfg, config) for method_cfg in config.get("methods", [])]
    if not methods:
        raise ValueError("Config must list methods.")

    reference_results = dedupe_results(
        [result for method in methods for result in method["results"]]
    )
    objective_specs = objective_specs_from_config(config)
    bounds = fixed_bounds_from_config(config.get("bounds"), objective_specs)
    metric_cfg = config.get("metrics", {}) or {}
    num_preference_vectors = int(metric_cfg.get("num_preference_vectors", 500))
    seed = int(metric_cfg.get("seed", config.get("seed", 0)))

    output: list[dict] = []
    for method in methods:
        representative = method["representative"]
        summary = summarize_mo_metrics(
            candidate_results=method["results"],
            reference_results=reference_results,
            objective_specs=objective_specs,
            num_preference_vectors=num_preference_vectors,
            seed=seed,
            bounds=bounds,
        ).to_dict()

        budget_summary = load_json_if_exists(method["method_cfg"].get("budget_json"))
        budget_used = budget_summary.get(
            "used_budget",
            budget_summary.get("total_tokens"),
        )

        output.append(
            {
                "method": method["name"],
                "representative_candidate_id": representative.get("candidate_id", ""),
                "representative_source_candidate_id": representative.get(
                    "source_candidate_id", ""
                ),
                "representative_prompt_hash": prompt_hash(representative),
                "performance": parse_float(representative.get("performance")),
                "cost": parse_float(representative.get("cost")),
                "risk": parse_float(representative.get("risk")),
                "fairness_risk": parse_float(representative.get("fairness_risk")),
                "best_performance": max(
                    parse_float(row.get("performance")) for row in method["eligible"]
                ),
                "min_cost": min(
                    parse_float(row.get("cost")) for row in method["eligible"]
                ),
                "min_fairness_risk": min(
                    parse_float(row.get("fairness_risk"))
                    for row in method["eligible"]
                ),
                "hypervolume": summary.get("hypervolume"),
                "optimistic_hypervolume": summary.get("optimistic_hypervolume"),
                "pessimistic_hypervolume": summary.get("pessimistic_hypervolume"),
                "approximation_gap": summary.get("approximation_gap"),
                "nr2": summary.get("nr2"),
                "eligible_portfolio_size": len(method["eligible"]),
                "raw_pareto_size": len(method["raw_pareto"]),
                "min_performance_threshold": method["threshold"],
                "threshold_fallback_used": method["threshold_fallback_used"],
                "budget_used": budget_used,
                "source": method["source"],
                "representative_prompt": representative.get("prompt", ""),
            }
        )

    return output


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def save_json(rows: list[dict], config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": config.get("dataset"),
                "model": config.get("model"),
                "selection": config.get("selection", {}),
                "rows": rows,
            },
            handle,
            indent=2,
            default=str,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a cross-method table using one real representative candidate "
            "per method and a shared union reference front for nR2."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    output_dir = Path(args.output_dir or config.get("output_dir", "outputs/experiment_table"))
    rows = build_rows(config)

    csv_path = output_dir / "representative_experiment_table.csv"
    json_path = output_dir / "representative_experiment_table.json"
    save_csv(rows, csv_path)
    save_json(rows, config, json_path)

    print("Representative experiment comparison")
    print("-" * 110)
    print(
        f"{'method':<28}{'acc':>9}{'cost':>12}{'fair':>10}"
        f"{'HV':>10}{'nR2':>10}{'|P|':>7}{'budget':>12}"
    )
    for row in rows:
        print(
            f"{row['method']:<28}"
            f"{row['performance']:>9.3f}"
            f"{row['cost']:>12.2f}"
            f"{row['fairness_risk']:>10.3f}"
            f"{float(row['hypervolume'] or 0.0):>10.3f}"
            f"{float(row['nr2'] or 0.0):>10.3f}"
            f"{row['eligible_portfolio_size']:>7}"
            f"{float(row['budget_used'] or 0.0):>12.1f}"
        )
    print("-" * 110)
    print(f"Saved representative table: {csv_path}")
    print(f"Saved representative JSON: {json_path}")


if __name__ == "__main__":
    main()
