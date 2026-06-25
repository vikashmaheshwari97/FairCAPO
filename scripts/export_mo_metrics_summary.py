from __future__ import annotations

import argparse
import csv
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


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default

    try:
        return float(value)
    except Exception:
        return default


def load_csv_rows(path: str) -> list[dict]:
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def row_to_result(row: dict) -> EvaluationResult:
    candidate_id = str(
        row.get("candidate_id")
        or row.get("id")
        or row.get("method")
        or row.get("prompt_id")
        or ""
    )

    return EvaluationResult(
        candidate_id=candidate_id,
        performance=parse_float(row.get("performance")),
        cost=parse_float(row.get("cost")),
        risk=parse_float(row.get("risk")),
        fairness_risk=parse_float(row.get("fairness_risk")),
        drift=parse_float(row.get("drift")),
        n_examples=int(parse_float(row.get("n_examples"), 0.0)),
        details={
            key: value
            for key, value in row.items()
            if key
            not in {
                "candidate_id",
                "performance",
                "cost",
                "risk",
                "fairness_risk",
                "drift",
                "n_examples",
            }
        },
    )


def rows_to_results(rows: list[dict]) -> list[EvaluationResult]:
    return [row_to_result(row) for row in rows]


def filter_candidate_rows(
    rows: list[dict],
    only_pareto: bool = False,
) -> list[dict]:
    if not only_pareto:
        return rows

    return [
        row
        for row in rows
        if parse_bool(row.get("is_pareto", False))
    ]


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


def make_reference_results(
    config: dict,
    candidate_results: list[EvaluationResult],
) -> list[EvaluationResult]:
    reference_path = config.get("reference_candidates_csv")

    if reference_path:
        reference_rows = load_csv_rows(reference_path)
        reference_only_pareto = bool(config.get("reference_only_pareto", False))
        reference_rows = filter_candidate_rows(
            reference_rows,
            only_pareto=reference_only_pareto,
        )
        return rows_to_results(reference_rows)

    # If no external reference is provided, use self-reference.
    # This makes nR2 and approximation_gap equal to 0.0 by design.
    return candidate_results


def flatten_summary_for_csv(summary: dict, config: dict) -> dict:
    bounds = summary.get("bounds", {})
    minimum = bounds.get("minimum", {})
    maximum = bounds.get("maximum", {})

    row = {
        "experiment_name": config.get("experiment_name", "mo_metrics_summary"),
        "num_points": summary.get("num_points"),
        "num_objectives": summary.get("num_objectives"),
        "hypervolume": summary.get("hypervolume"),
        "optimistic_hypervolume": summary.get("optimistic_hypervolume"),
        "pessimistic_hypervolume": summary.get("pessimistic_hypervolume"),
        "approximation_gap": summary.get("approximation_gap"),
        "nr2": summary.get("nr2"),
        "objective_names": json.dumps(summary.get("objective_names", [])),
    }

    for key, value in minimum.items():
        row[f"min_{key}"] = value

    for key, value in maximum.items():
        row[f"max_{key}"] = value

    return row


def build_mo_metrics_summary(config: dict) -> tuple[dict, dict]:
    candidates_csv = config.get(
        "candidates_csv",
        "outputs/phase2_budgeted_mocapo_subj/phase2_all_candidates.csv",
    )

    rows = load_csv_rows(candidates_csv)
    rows = filter_candidate_rows(
        rows,
        only_pareto=bool(config.get("only_pareto", True)),
    )

    if not rows:
        raise ValueError(
            "No candidate rows available for MO metrics. "
            "Check candidates_csv and only_pareto settings."
        )

    candidate_results = rows_to_results(rows)
    reference_results = make_reference_results(config, candidate_results)

    objective_specs = objective_specs_from_config(config)

    metric_cfg = config.get("metrics", {})
    num_preference_vectors = int(metric_cfg.get("num_preference_vectors", 50))
    seed = int(config.get("seed", metric_cfg.get("seed", 0)))

    bounds = fixed_bounds_from_config(
        bounds_config=config.get("bounds"),
        objective_specs=objective_specs,
    )

    summary = summarize_mo_metrics(
        candidate_results=candidate_results,
        reference_results=reference_results,
        objective_specs=objective_specs,
        num_preference_vectors=num_preference_vectors,
        seed=seed,
        bounds=bounds,
    ).to_dict()

    metadata = {
        "experiment_name": config.get("experiment_name", "mo_metrics_summary"),
        "candidates_csv": candidates_csv,
        "reference_candidates_csv": config.get("reference_candidates_csv", ""),
        "only_pareto": bool(config.get("only_pareto", True)),
        "reference_only_pareto": bool(config.get("reference_only_pareto", False)),
        "num_candidate_results": len(candidate_results),
        "num_reference_results": len(reference_results),
        "num_preference_vectors": num_preference_vectors,
        "seed": seed,
        "uses_fixed_bounds": bounds is not None,
    }

    output = {
        "metadata": metadata,
        "summary": summary,
    }

    flat_row = flatten_summary_for_csv(summary, config)
    flat_row.update(
        {
            "num_candidate_results": len(candidate_results),
            "num_reference_results": len(reference_results),
            "num_preference_vectors": num_preference_vectors,
            "seed": seed,
            "uses_fixed_bounds": bounds is not None,
        }
    )

    return output, flat_row


def print_summary(output: dict):
    print("Multi-objective metrics summary")
    print("-" * 80)
    print(json.dumps(output, indent=2, default=_json_default))
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/mo_metrics_summary.yaml",
        help="MO metrics summary YAML config.",
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
        "outputs/mo_metrics/phase2_budgeted_mocapo_subj",
    )

    output, flat_row = build_mo_metrics_summary(config)

    save_json(output, f"{output_dir}/mo_metrics_summary.json")
    save_csv([flat_row], f"{output_dir}/mo_metrics_summary.csv")

    print_summary(output)

    print(f"Saved MO metrics summary to: {output_dir}/mo_metrics_summary.json")
    print(f"Saved MO metrics CSV to: {output_dir}/mo_metrics_summary.csv")


if __name__ == "__main__":
    main()