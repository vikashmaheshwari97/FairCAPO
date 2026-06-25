from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
from heal_capo.objectives import ToyObjectiveEvaluator
from heal_capo.pareto import non_dominated_ids, pareto_archive, sort_pareto_results
from heal_capo.components.router import RiskAwareRouter, FairnessAwareRouter


def _json_default(obj: Any):
    try:
        return obj.item()
    except AttributeError:
        return str(obj)


def load_yaml(path: str) -> dict:
    yaml_path = Path(path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Config not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def save_json(data: Any, path: str):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def get_prompt_pool(config: dict) -> list[dict]:
    prompt_pool = config.get("prompt_pool", [])

    if not prompt_pool:
        raise ValueError("Config must contain a non-empty prompt_pool list.")

    normalized = []

    for idx, item in enumerate(prompt_pool):
        if isinstance(item, str):
            normalized.append(
                {
                    "id": f"prompt_{idx}",
                    "category": "unknown",
                    "prompt": item,
                }
            )
            continue

        prompt = str(item.get("prompt", "")).strip()

        if not prompt:
            continue

        normalized.append(
            {
                "id": item.get("id", f"prompt_{idx}"),
                "category": item.get("category", "unknown"),
                "prompt": prompt,
            }
        )

    if not normalized:
        raise ValueError("No valid prompts found in prompt_pool.")

    return normalized


def get_dev_data(config: dict) -> list[dict]:
    """
    The toy evaluator only needs the number of examples,
    but we keep dev_data for a realistic shape.
    """
    dev_data = config.get(
        "dev_data",
        [
            {
                "text": "The movie was released in 1999.",
                "label": "objective",
            },
            {
                "text": "The acting is absolutely wonderful.",
                "label": "subjective",
            },
            {
                "text": "Paris is the capital of France.",
                "label": "objective",
            },
        ],
    )

    return dev_data


def build_all_candidates_portfolio(
    config: dict,
) -> PromptPortfolio:
    prompt_pool = get_prompt_pool(config)
    dev_data = get_dev_data(config)

    dataset = config.get("dataset", "subj")
    task_type = config.get("task_type", "classification")

    evaluator = ToyObjectiveEvaluator()
    portfolio = PromptPortfolio()

    for idx, prompt_item in enumerate(prompt_pool):
        candidate = PromptCandidate(
            instruction=prompt_item["prompt"],
            metadata={
                "prompt_pool_id": prompt_item["id"],
                "category": prompt_item["category"],
                "row_index": idx,
                "dataset": dataset,
                "task_type": task_type,
                "source": "phase2_prompt_pool",
            },
        )

        result = evaluator.evaluate(
            candidate=candidate,
            data=dev_data,
        )

        # The ToyObjectiveEvaluator gives useful toy values, but we enrich
        # details with prompt pool metadata for dashboard/reporting.
        result.details["prompt_pool_id"] = prompt_item["id"]
        result.details["category"] = prompt_item["category"]
        result.details["source"] = "phase2_prompt_pool_demo"

        portfolio.add(candidate, result)

    return portfolio


def make_pareto_portfolio(
    all_portfolio: PromptPortfolio,
) -> PromptPortfolio:
    pareto = PromptPortfolio()

    pareto_evaluations = pareto_archive(all_portfolio.evaluations)
    sorted_results = sort_pareto_results(pareto_evaluations.values())
    sorted_ids = [result.candidate_id for result in sorted_results]

    for candidate_id in sorted_ids:
        candidate = all_portfolio.get(candidate_id)
        result = all_portfolio.get_result(candidate_id)
        pareto.add(candidate, result)

    return pareto


def portfolio_to_rows(
    portfolio: PromptPortfolio,
    pareto_ids: set[str] | None = None,
) -> list[dict]:
    pareto_ids = pareto_ids or set()
    rows = []

    for candidate in portfolio.evaluated_candidates():
        result = portfolio.get_result(candidate.candidate_id)

        row = {
            "candidate_id": candidate.candidate_id,
            "is_pareto": candidate.candidate_id in pareto_ids,
            "method": candidate.metadata.get("prompt_pool_id"),
            "category": candidate.metadata.get("category"),
            "dataset": candidate.metadata.get("dataset"),
            "task_type": candidate.metadata.get("task_type"),
            "performance": result.performance,
            "cost": result.cost,
            "risk": result.risk,
            "fairness_risk": result.fairness_risk,
            "drift": result.drift,
            "n_examples": result.n_examples,
            "objective_vector": str(result.objective_vector),
            "prompt": candidate.instruction,
            "source": candidate.metadata.get("source"),
        }

        for key, value in result.details.items():
            if isinstance(value, (dict, list, tuple)):
                row[f"detail_{key}"] = json.dumps(value, default=_json_default)
            else:
                row[f"detail_{key}"] = value

        rows.append(row)

    return rows


def routing_preferences_from_config(config: dict) -> list[dict]:
    return config.get(
        "routing_preferences",
        [
            {"name": "accuracy_first", "mode": "accuracy_first"},
            {"name": "cost_first", "mode": "cost_first"},
            {"name": "risk_first", "mode": "risk_first"},
            {"name": "fairness_first", "mode": "fairness_first"},
            {
                "name": "balanced",
                "mode": "balanced",
                "performance": 1.0,
                "cost": 0.3,
                "risk": 1.0,
                "fairness": 1.0,
            },
            {
                "name": "fairness_sensitive_balanced",
                "mode": "balanced",
                "performance": 0.8,
                "cost": 0.2,
                "risk": 1.0,
                "fairness": 2.0,
            },
            {
                "name": "cost_sensitive_balanced",
                "mode": "balanced",
                "performance": 0.8,
                "cost": 2.0,
                "risk": 0.8,
                "fairness": 0.8,
            },
        ],
    )


def routing_to_rows(
    portfolio: PromptPortfolio,
    preferences: list[dict],
) -> list[dict]:
    rows = []

    default_router = RiskAwareRouter()
    fairness_router = FairnessAwareRouter()

    for preference in preferences:
        name = preference.get("name", preference.get("mode", "balanced"))

        if preference.get("mode") == "fairness_first":
            router = fairness_router
        else:
            router = default_router

        decision = router.select(
            x="phase2_prompt_pool_demo",
            portfolio=portfolio,
            preference=preference,
        )

        row = decision.to_row()
        row["preference_name"] = name
        rows.append(row)

    return rows


def print_portfolio_summary(
    title: str,
    rows: list[dict],
):
    print(title)
    print("-" * 80)

    for row in rows:
        pareto_mark = "Pareto" if row.get("is_pareto") else "Non-Pareto"

        print(
            f"{pareto_mark} | "
            f"{row.get('method')} | "
            f"category={row.get('category')} | "
            f"performance={row.get('performance')} | "
            f"cost={row.get('cost')} | "
            f"risk={row.get('risk')} | "
            f"fairness_risk={row.get('fairness_risk')} | "
            f"prompt={row.get('prompt')}"
        )

    print("-" * 80)
    print(f"Total rows: {len(rows)}")


def print_routing_summary(rows: list[dict]):
    print("Prompt pool recommendations")
    print("-" * 80)

    for row in rows:
        print(
            f"{row['preference_name']} -> "
            f"{row['instruction']} | "
            f"utility={row['utility']} | "
            f"reason={row['reason']}"
        )

    print("-" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/phase2_prompt_pool_subj.yaml",
        help="Prompt pool YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--recommend-from",
        default=None,
        choices=["pareto", "all"],
        help="Recommend from Pareto portfolio or all candidates.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)

    output_dir = args.output_dir or config.get(
        "output_dir",
        "outputs/phase2_prompt_pool_subj",
    )

    recommend_from = args.recommend_from or config.get(
        "recommend_from",
        "pareto",
    )

    all_portfolio = build_all_candidates_portfolio(config)

    if not all_portfolio.evaluations:
        raise ValueError("No candidates produced from prompt pool.")

    pareto_ids = set(non_dominated_ids(all_portfolio.evaluations.values()))
    pareto_portfolio = make_pareto_portfolio(all_portfolio)

    all_candidate_rows = portfolio_to_rows(
        portfolio=all_portfolio,
        pareto_ids=pareto_ids,
    )
    pareto_rows = portfolio_to_rows(
        portfolio=pareto_portfolio,
        pareto_ids=pareto_ids,
    )

    if not pareto_rows:
        raise ValueError("No Pareto candidates produced from prompt pool.")

    if recommend_from == "all":
        recommendation_portfolio = all_portfolio
    elif recommend_from == "pareto":
        recommendation_portfolio = pareto_portfolio
    else:
        raise ValueError("recommend_from must be either 'pareto' or 'all'.")

    recommendations = routing_to_rows(
        portfolio=recommendation_portfolio,
        preferences=routing_preferences_from_config(config),
    )

    save_csv(
        all_candidate_rows,
        f"{output_dir}/phase2_all_candidates.csv",
    )
    save_json(
        all_candidate_rows,
        f"{output_dir}/phase2_all_candidates.json",
    )
    save_csv(
        pareto_rows,
        f"{output_dir}/phase2_prompt_portfolio.csv",
    )
    save_json(
        pareto_rows,
        f"{output_dir}/phase2_prompt_portfolio.json",
    )
    save_csv(
        recommendations,
        f"{output_dir}/phase2_prompt_recommendations.csv",
    )
    save_json(
        recommendations,
        f"{output_dir}/phase2_prompt_recommendations.json",
    )

    print_portfolio_summary(
        "Phase 2 prompt pool all candidates",
        all_candidate_rows,
    )
    print_portfolio_summary(
        "Phase 2 prompt pool Pareto portfolio",
        pareto_rows,
    )
    print_routing_summary(recommendations)

    print(f"Saved all candidates to: {output_dir}/phase2_all_candidates.csv")
    print(f"Saved Pareto portfolio to: {output_dir}/phase2_prompt_portfolio.csv")
    print(f"Saved recommendations to: {output_dir}/phase2_prompt_recommendations.csv")


if __name__ == "__main__":
    main()