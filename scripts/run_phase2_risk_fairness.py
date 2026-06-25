from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from heal_capo.core import PromptCandidate, PromptPortfolio
from heal_capo.objectives import ToyObjectiveEvaluator
from heal_capo.optimizers.risk_aware_mo_capo import (
    RiskAwareMOCAPO,
    RiskAwareMOCAPOConfig,
)
from heal_capo.components.drift_guard import KeywordDriftGuard
from heal_capo.components.router import RiskAwareRouter, FairnessAwareRouter


def _json_default(obj: Any):
    try:
        return obj.item()
    except AttributeError:
        return str(obj)


def save_json(data: Any, path: str):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def save_csv(rows: list[dict], path: str):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to save.")

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_default_config() -> dict:
    """
    Fallback config if no YAML file is provided.
    """
    return {
        "output_dir": "outputs/phase2_risk_fairness_demo",
        "initial_prompts": [
            "Classify the input. Return only the label.",
            "Classify the input using the provided context. Return only the label.",
            "Classify the input. Do not hallucinate. If not enough information is available, say unknown.",
            (
                "Classify the input. Do not infer ability, sentiment, intent, or correctness "
                "from gender, race, ethnicity, nationality, religion, age, or location."
            ),
            (
                "Classify the input using context. Do not hallucinate. Do not infer ability, "
                "sentiment, intent, or correctness from demographic attributes."
            ),
        ],
        "dev_data": [
            {"text": "The movie was released in 1999.", "label": "objective"},
            {"text": "The acting is absolutely wonderful.", "label": "subjective"},
            {"text": "Paris is the capital of France.", "label": "objective"},
        ],
        "optimizer": {
            "population_size": 10,
            "n_steps": 1,
            "max_shots": 5,
            "drift_threshold": 0.3,
            "keep_drift_failures": False,
            "sort_archive": True,
        },
        "routing_preferences": [
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
        ],
    }


def portfolio_to_rows(portfolio: PromptPortfolio) -> list[dict]:
    rows = []

    for candidate in portfolio.evaluated_candidates():
        result = portfolio.get_result(candidate.candidate_id)

        row = {
            "candidate_id": candidate.candidate_id,
            "instruction": candidate.instruction,
            "performance": result.performance,
            "cost": result.cost,
            "risk": result.risk,
            "fairness_risk": result.fairness_risk,
            "drift": result.drift,
            "n_examples": result.n_examples,
            "objective_vector": str(result.objective_vector),
        }

        for key, value in candidate.metadata.items():
            row[f"candidate_{key}"] = value

        for key, value in result.details.items():
            if isinstance(value, (dict, list, tuple)):
                row[f"detail_{key}"] = json.dumps(value, default=_json_default)
            else:
                row[f"detail_{key}"] = value

        rows.append(row)

    return rows


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
            x="phase2_demo_input",
            portfolio=portfolio,
            preference=preference,
        )

        row = decision.to_row()
        row["preference_name"] = name
        rows.append(row)

    return rows


def print_portfolio_summary(rows: list[dict]):
    print("Phase 2 risk/fairness Pareto portfolio")
    print("-" * 80)

    for row in rows:
        print(
            f"candidate_id={row['candidate_id'][:8]} | "
            f"performance={row['performance']} | "
            f"cost={row['cost']} | "
            f"risk={row['risk']} | "
            f"fairness_risk={row['fairness_risk']} | "
            f"prompt={row['instruction']}"
        )

    print("-" * 80)
    print(f"Total Pareto candidates: {len(rows)}")


def print_routing_summary(rows: list[dict]):
    print("Prompt recommendations")
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
        default=None,
        help="Optional Phase 2 YAML config path.",
    )
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = build_default_config()

    output_dir = config.get("output_dir", "outputs/phase2_risk_fairness_demo")
    initial_prompts = config.get("initial_prompts", [])
    dev_data = config.get("dev_data", [])

    if not initial_prompts:
        raise ValueError("Config must provide initial_prompts.")

    optimizer_cfg = config.get("optimizer", {})

    mocapo_config = RiskAwareMOCAPOConfig(
        population_size=int(optimizer_cfg.get("population_size", 10)),
        n_steps=int(optimizer_cfg.get("n_steps", 1)),
        max_shots=int(optimizer_cfg.get("max_shots", 5)),
        drift_threshold=float(optimizer_cfg.get("drift_threshold", 0.3)),
        keep_drift_failures=bool(optimizer_cfg.get("keep_drift_failures", False)),
        sort_archive=bool(optimizer_cfg.get("sort_archive", True)),
    )

    evaluator = ToyObjectiveEvaluator()

    required_terms = config.get(
        "drift_guard",
        {},
    ).get(
        "required_terms",
        ["classify", "input"],
    )

    drift_guard = KeywordDriftGuard(required_terms=required_terms)

    optimizer = RiskAwareMOCAPO(
        evaluator=evaluator,
        drift_guard=drift_guard,
        config=mocapo_config,
    )

    portfolio = optimizer.optimize(
        initial_prompts=initial_prompts,
        dev_data=dev_data,
    )

    portfolio_rows = portfolio_to_rows(portfolio)

    preferences = config.get(
        "routing_preferences",
        [
            {"name": "balanced", "mode": "balanced"},
            {"name": "fairness_first", "mode": "fairness_first"},
        ],
    )
    routing_rows = routing_to_rows(
        portfolio=portfolio,
        preferences=preferences,
    )

    save_csv(
        portfolio_rows,
        f"{output_dir}/phase2_pareto_portfolio.csv",
    )
    save_json(
        portfolio_rows,
        f"{output_dir}/phase2_pareto_portfolio.json",
    )
    save_csv(
        routing_rows,
        f"{output_dir}/phase2_recommendations.csv",
    )
    save_json(
        routing_rows,
        f"{output_dir}/phase2_recommendations.json",
    )

    print_portfolio_summary(portfolio_rows)
    print_routing_summary(routing_rows)

    print(f"Saved portfolio to: {output_dir}/phase2_pareto_portfolio.csv")
    print(f"Saved recommendations to: {output_dir}/phase2_recommendations.csv")


if __name__ == "__main__":
    main()