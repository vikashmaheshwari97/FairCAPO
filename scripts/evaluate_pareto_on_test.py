from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from heal_capo.core import EvaluationResult, PromptCandidate
from heal_capo.evaluation.mo_metrics import (
    DEFAULT_OBJECTIVE_SPECS,
    ObjectiveSpec,
    fixed_bounds_from_config,
    summarize_mo_metrics,
)
from heal_capo.pareto import non_dominated_ids
from scripts.run_phase2_budgeted_mocapo import build_objective_evaluator


def _json_default(obj: Any):
    try:
        return obj.item()
    except AttributeError:
        return str(obj)


def load_yaml(path: str) -> dict:
    yaml_path = Path(path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}


def save_json(data: Any, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


def save_csv(rows: list[dict], path: str) -> None:
    if not rows:
        raise ValueError(f"No rows to save for {path}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv_rows(path: str) -> list[dict]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def objective_specs_from_config(config: dict) -> tuple[ObjectiveSpec, ...]:
    objectives = config.get("objectives")
    if not objectives:
        return DEFAULT_OBJECTIVE_SPECS
    specs = []
    for item in objectives:
        if isinstance(item, str):
            specs.append(ObjectiveSpec(item, "maximize" if item == "performance" else "minimize"))
        else:
            specs.append(ObjectiveSpec(str(item["name"]), str(item.get("direction", "minimize"))))
    return tuple(specs)


def _parse_few_shot_examples(raw: Any) -> list[dict]:
    if not raw:
        return []
    if isinstance(raw, list):
        parsed = raw
    else:
        try:
            parsed = json.loads(str(raw))
        except (ValueError, TypeError):
            return []
    if not isinstance(parsed, list):
        return []
    return [
        {"input": str(item.get("input", "")), "output": str(item.get("output", ""))}
        for item in parsed
        if isinstance(item, dict) and ("input" in item or "output" in item)
    ]


def deduplicate_portfolio_rows(rows: list[dict]) -> list[dict]:
    """Keep one final row per source candidate or exact rendered prompt."""
    latest: dict[str, dict] = {}
    order: list[str] = []
    for row in rows:
        source_id = str(row.get("source_candidate_id") or row.get("candidate_id") or "").strip()
        key = source_id or (
            str(row.get("prompt") or row.get("instruction") or "")
            + "\n"
            + str(row.get("few_shot_examples") or "")
        )
        if key not in latest:
            order.append(key)
        latest[key] = dict(row)
    return [latest[key] for key in order]


def portfolio_rows_to_candidates(rows: list[dict]) -> list[PromptCandidate]:
    candidates: list[PromptCandidate] = []
    for index, row in enumerate(deduplicate_portfolio_rows(rows)):
        instruction = str(row.get("prompt") or row.get("instruction") or "").strip()
        if not instruction:
            continue
        source_id = str(row.get("source_candidate_id") or row.get("candidate_id") or f"prompt_{index}")
        examples = _parse_few_shot_examples(row.get("few_shot_examples"))
        candidates.append(
            PromptCandidate(
                instruction=instruction,
                examples=examples,
                metadata={
                    "method": str(row.get("method") or row.get("prompt_pool_id") or source_id),
                    "category": row.get("category", "unknown"),
                    "source_candidate_id": source_id,
                    "num_few_shot": len(examples),
                    "row_index": index,
                    "source": "evaluate_pareto_on_test",
                },
            )
        )
    if not candidates:
        raise ValueError("No prompts found in portfolio CSV.")
    return candidates


def example_to_row(example: Any) -> dict:
    if isinstance(example, dict):
        row = {
            "text": str(example.get("text") or example.get("input") or example.get("sentence") or ""),
            "label": str(example.get("label") or example.get("answer") or example.get("output") or ""),
        }
        metadata = example.get("meta") or example.get("metadata")
    else:
        row = {"text": str(getattr(example, "text", "")), "label": str(getattr(example, "label", ""))}
        metadata = getattr(example, "metadata", None)
    if isinstance(metadata, dict):
        row["meta"] = dict(metadata)
    return row


def get_test_data(config: dict) -> list[dict]:
    if config.get("test_data"):
        return [example_to_row(item) for item in config["test_data"]]
    dataset = config.get("dataset")
    if not dataset:
        raise ValueError("Config must provide test_data or dataset.")
    dev = config.get("dev") if isinstance(config.get("dev"), dict) else {}
    from experiments.datasets import load_paper_dataset
    split = load_paper_dataset(
        name=str(dataset),
        dev_size=int(dev.get("dev_size", config.get("dev_size", 10))),
        shots_size=int(dev.get("shots_size", config.get("shots_size", 2))),
        test_size=int(dev.get("test_size", config.get("test_size", 5))),
        seed=int(config.get("seed", 0)),
        allow_smaller=bool(dev.get("allow_smaller", config.get("allow_smaller", False))),
        stratified=bool(dev.get("stratified", config.get("stratified", True))),
        dataset_split=dev.get("dataset_split") or config.get("dataset_split"),
        stratify_group_key=(
            dev.get("stratify_group_key")
            or config.get("stratify_group_key")
            or config.get("group_key")
            or (config.get("fairness") or {}).get("group_key")
        ),
        data_path=dev.get("data_path") or config.get("data_path"),
    )
    return [example_to_row(example) for example in split.test]


def evaluate_portfolio_on_test(config: dict, force_no_llm: bool = False):
    portfolio_csv = config.get("portfolio_csv")
    if not portfolio_csv:
        raise ValueError("Evaluation config is missing portfolio_csv.")
    raw_rows = load_csv_rows(str(portfolio_csv))
    candidates = portfolio_rows_to_candidates(raw_rows)
    test_data = get_test_data(config)
    evaluator = build_objective_evaluator(config, force_no_llm=force_no_llm)
    results = [evaluator.evaluate(candidate, test_data) for candidate in candidates]
    return results, candidates, test_data, len(raw_rows)


def build_candidate_rows(results, candidates, pareto_ids: set[str]) -> list[dict]:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    rows = []
    for result in results:
        candidate = candidate_by_id[result.candidate_id]
        row = {
            "candidate_id": result.candidate_id,
            "method": candidate.metadata.get("method"),
            "category": candidate.metadata.get("category"),
            "is_pareto": result.candidate_id in pareto_ids,
            "performance": result.performance,
            "cost": result.cost,
            "risk": result.risk,
            "fairness_risk": result.fairness_risk,
            "drift": result.drift,
            "n_examples": result.n_examples,
            "objective_vector": str(result.objective_vector),
            "prompt": candidate.instruction,
            "source_candidate_id": candidate.metadata.get("source_candidate_id"),
        }
        for key, value in result.details.items():
            row[f"detail_{key}"] = json.dumps(value, default=_json_default) if isinstance(value, (dict, list, tuple)) else value
        rows.append(row)
    return rows


def run_test_evaluation(config: dict, force_no_llm: bool = False) -> dict:
    results, candidates, test_data, raw_row_count = evaluate_portfolio_on_test(config, force_no_llm)
    pareto_ids = set(non_dominated_ids(results))
    candidate_rows = build_candidate_rows(results, candidates, pareto_ids)
    specs = objective_specs_from_config(config)
    metric_cfg = config.get("metrics", {})
    num_preferences = int(metric_cfg.get("num_preference_vectors", 50))
    seed = int(config.get("seed", metric_cfg.get("seed", 0)))
    bounds = fixed_bounds_from_config(config.get("bounds"), specs)
    pareto_results = [result for result in results if result.candidate_id in pareto_ids]
    summary = summarize_mo_metrics(
        candidate_results=pareto_results,
        reference_results=pareto_results,
        objective_specs=specs,
        num_preference_vectors=num_preferences,
        seed=seed,
        bounds=bounds,
    ).to_dict()
    metadata = {
        "experiment_name": config.get("experiment_name", "evaluate_pareto_on_test"),
        "portfolio_csv": config.get("portfolio_csv"),
        "num_portfolio_rows_raw": raw_row_count,
        "num_prompts": len(results),
        "num_duplicate_rows_removed": raw_row_count - len(results),
        "num_pareto": len(pareto_results),
        "num_test_examples": len(test_data),
        "num_preference_vectors": num_preferences,
        "seed": seed,
        "uses_fixed_bounds": bounds is not None,
        "used_llm": bool((config.get("evaluation") or {}).get("use_llm", False)) and not force_no_llm,
    }
    return {"metadata": metadata, "summary": summary, "candidate_rows": candidate_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a prompt portfolio on a held-out test set.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--portfolio-csv", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.portfolio_csv is not None:
        config["portfolio_csv"] = args.portfolio_csv
    output_dir = args.output_dir or config.get("output_dir", "outputs/evaluation")
    result = run_test_evaluation(config, force_no_llm=args.no_llm)
    save_csv(result["candidate_rows"], f"{output_dir}/test_eval_candidates.csv")
    save_json({"metadata": result["metadata"], "summary": result["summary"]}, f"{output_dir}/test_eval_summary.json")
    print(json.dumps(result["metadata"], indent=2, default=_json_default))
    print(json.dumps(result["summary"], indent=2, default=_json_default))


if __name__ == "__main__":
    main()
