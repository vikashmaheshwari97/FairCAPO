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
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def objective_specs_from_config(config: dict) -> tuple[ObjectiveSpec, ...]:
    """Mirror of export_mo_metrics_summary.objective_specs_from_config."""
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


def _parse_few_shot_examples(raw: Any) -> list[dict]:
    """
    Parse the persisted few-shot demonstrations from a portfolio row.

    The runner serializes ``candidate.examples`` as a JSON list of
    ``{"input": ..., "output": ...}`` dicts in the ``few_shot_examples`` column.
    Missing/empty/malformed values yield an empty list (zero-shot), so older
    portfolios without the column still load.
    """
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

    examples: list[dict] = []
    for item in parsed:
        if isinstance(item, dict) and ("input" in item or "output" in item):
            examples.append(
                {
                    "input": str(item.get("input", "")),
                    "output": str(item.get("output", "")),
                }
            )
    return examples


def portfolio_rows_to_candidates(rows: list[dict]) -> list[PromptCandidate]:
    """
    Rebuild PromptCandidate objects from a portfolio/candidates CSV.

    We only need the prompt text and an identifying method/category to
    re-evaluate the prompt on a fresh test set. A new candidate_id is created
    per row so test-set evaluation results are unambiguous.
    """
    candidates: list[PromptCandidate] = []

    for idx, row in enumerate(rows):
        instruction = str(
            row.get("prompt")
            or row.get("instruction")
            or ""
        ).strip()

        if not instruction:
            continue

        method = str(
            row.get("method")
            or row.get("prompt_pool_id")
            or row.get("candidate_id")
            or f"prompt_{idx}"
        )

        # Restore the few-shot demonstrations the candidate won with so the
        # held-out evaluation tests the EXACT prompt (instruction + shots), not a
        # zero-shot strip of it. Without this a few-shot winner is silently
        # tested without its demos and the staircase collapses.
        examples = _parse_few_shot_examples(row.get("few_shot_examples"))

        candidates.append(
            PromptCandidate(
                instruction=instruction,
                examples=examples,
                metadata={
                    "method": method,
                    "category": row.get("category", "unknown"),
                    "source_candidate_id": row.get("candidate_id", ""),
                    "num_few_shot": len(examples),
                    "row_index": idx,
                    "source": "evaluate_pareto_on_test",
                },
            )
        )

    if not candidates:
        raise ValueError(
            "No prompts found in portfolio CSV. "
            "Expected a 'prompt' or 'instruction' column."
        )

    return candidates


def example_to_row(example: Any) -> dict:
    """Convert a datasets.Example (or dict) to a {text, label} row."""
    if isinstance(example, dict):
        return {
            "text": str(
                example.get("text")
                or example.get("input")
                or example.get("sentence")
                or ""
            ),
            "label": str(
                example.get("label")
                or example.get("answer")
                or example.get("output")
                or ""
            ),
        }

    # datasets.Example dataclass.
    return {
        "text": str(getattr(example, "text", "")),
        "label": str(getattr(example, "label", "")),
    }


def get_test_data(config: dict) -> list[dict]:
    """
    Two modes:
      1. Inline `test_data` list (local default, tiny).
      2. Dataset mode: `dataset` + `test_size` via load_paper_dataset (HPC scale).
    """
    inline = config.get("test_data")

    if inline:
        return [example_to_row(item) for item in inline]

    dataset = config.get("dataset")

    if not dataset:
        raise ValueError(
            "Config must provide either inline 'test_data' or a 'dataset' name."
        )

    from experiments.datasets import load_paper_dataset

    split = load_paper_dataset(
        name=str(dataset),
        dev_size=int(config.get("dev_size", 10)),
        shots_size=int(config.get("shots_size", 2)),
        test_size=int(config.get("test_size", 5)),
        seed=int(config.get("seed", 0)),
        allow_smaller=bool(config.get("allow_smaller", False)),
        stratified=bool(config.get("stratified", True)),
    )

    return [example_to_row(example) for example in split.test]


def evaluate_portfolio_on_test(
    config: dict,
    force_no_llm: bool = False,
) -> tuple[list[EvaluationResult], list[PromptCandidate], list[dict]]:
    portfolio_csv = config.get(
        "portfolio_csv",
        "outputs/phase2_budgeted_mocapo_subj/phase2_prompt_portfolio.csv",
    )

    rows = load_csv_rows(portfolio_csv)
    candidates = portfolio_rows_to_candidates(rows)
    test_data = get_test_data(config)

    evaluator = build_objective_evaluator(config, force_no_llm=force_no_llm)

    results: list[EvaluationResult] = []

    for candidate in candidates:
        result = evaluator.evaluate(candidate, test_data)
        results.append(result)

    return results, candidates, test_data


def build_candidate_rows(
    results: list[EvaluationResult],
    candidates: list[PromptCandidate],
    pareto_ids: set[str],
) -> list[dict]:
    candidate_by_id = {c.candidate_id: c for c in candidates}
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
            if isinstance(value, (dict, list, tuple)):
                row[f"detail_{key}"] = json.dumps(value, default=_json_default)
            else:
                row[f"detail_{key}"] = value

        rows.append(row)

    return rows


def run_test_evaluation(config: dict, force_no_llm: bool = False) -> dict:
    results, candidates, test_data = evaluate_portfolio_on_test(
        config,
        force_no_llm=force_no_llm,
    )

    pareto_ids = set(non_dominated_ids(results))
    candidate_rows = build_candidate_rows(results, candidates, pareto_ids)

    objective_specs = objective_specs_from_config(config)

    metric_cfg = config.get("metrics", {})
    num_preference_vectors = int(metric_cfg.get("num_preference_vectors", 50))
    seed = int(config.get("seed", metric_cfg.get("seed", 0)))

    bounds = fixed_bounds_from_config(
        bounds_config=config.get("bounds"),
        objective_specs=objective_specs,
    )

    pareto_results = [r for r in results if r.candidate_id in pareto_ids]

    summary = summarize_mo_metrics(
        candidate_results=pareto_results,
        reference_results=pareto_results,
        objective_specs=objective_specs,
        num_preference_vectors=num_preference_vectors,
        seed=seed,
        bounds=bounds,
    ).to_dict()

    metadata = {
        "experiment_name": config.get("experiment_name", "evaluate_pareto_on_test"),
        "portfolio_csv": config.get(
            "portfolio_csv",
            "outputs/phase2_budgeted_mocapo_subj/phase2_prompt_portfolio.csv",
        ),
        "num_prompts": len(results),
        "num_pareto": len(pareto_results),
        "num_test_examples": len(test_data),
        "num_preference_vectors": num_preference_vectors,
        "seed": seed,
        "uses_fixed_bounds": bounds is not None,
        "used_llm": bool(config.get("evaluation", {}).get("use_llm", False))
        and not force_no_llm,
    }

    return {
        "metadata": metadata,
        "summary": summary,
        "candidate_rows": candidate_rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a final Pareto prompt portfolio on a held-out test set."
    )
    parser.add_argument(
        "--config",
        default="configs/evaluate_pareto_subj_mistral.yaml",
        help="Test-set evaluation YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Force the deterministic toy evaluator (no LM Studio).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config['seed'] for multi-seed sweeps (seeds the held-out "
        "test-split sampling so it matches the corresponding search seed).",
    )
    parser.add_argument(
        "--portfolio-csv",
        default=None,
        help="Override config['portfolio_csv'] so one eval config can score the "
        "per-seed portfolio of a multi-seed sweep.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.portfolio_csv is not None:
        config["portfolio_csv"] = args.portfolio_csv

    output_dir = args.output_dir or config.get(
        "output_dir",
        "outputs/evaluation/subj_mistral",
    )

    result = run_test_evaluation(config, force_no_llm=args.no_llm)

    candidate_rows = result["candidate_rows"]

    save_csv(candidate_rows, f"{output_dir}/test_eval_candidates.csv")
    save_json(
        {"metadata": result["metadata"], "summary": result["summary"]},
        f"{output_dir}/test_eval_summary.json",
    )

    print("Test-set evaluation of Pareto portfolio")
    print("-" * 80)
    print(json.dumps(result["metadata"], indent=2, default=_json_default))
    print("-" * 80)
    print(json.dumps(result["summary"], indent=2, default=_json_default))
    print("-" * 80)
    print(f"Saved test-set candidates to: {output_dir}/test_eval_candidates.csv")
    print(f"Saved test-set summary to: {output_dir}/test_eval_summary.json")


if __name__ == "__main__":
    main()
