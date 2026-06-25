from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

import yaml

from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
from heal_capo.pareto import non_dominated_ids, pareto_archive, sort_pareto_results
from heal_capo.components.router import RiskAwareRouter, FairnessAwareRouter


def _json_default(obj: Any):
    try:
        return obj.item()
    except AttributeError:
        return str(obj)


def _to_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except ValueError:
        return default


def _to_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except ValueError:
        return default


def load_config(path: Optional[str]) -> dict:
    if path is None:
        return {
            "input_table": "outputs/phase1_summary/final_phase1_subj_complete.csv",
            "output_dir": "outputs/phase2_prompt_portfolio",
            "risk_source": "test_error",
            "fairness_source": "prompt_heuristic",
            "default_fairness_risk": 0.10,
            "deduplicate_prompts": True,
            "save_all_candidates": True,
            "recommend_from": "pareto",
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

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv(path: str) -> list[dict]:
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input table not found: {input_path}")

    with open(input_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def compute_risk(row: dict, risk_source: str = "test_error") -> float:
    """
    Derive a risk score from Phase 1 rows.

    Current options:
      test_error = 1 - test_score
      dev_error = 1 - dev_score
      score_gap = abs(test_score - dev_score)

    Later this will be replaced with verifier/hallucination-specific risk.
    """
    test_score = _to_float(row.get("test_score"), default=0.0)
    dev_score = _to_float(row.get("dev_score"), default=0.0)

    if risk_source == "test_error":
        return max(0.0, min(1.0, 1.0 - test_score))

    if risk_source == "dev_error":
        return max(0.0, min(1.0, 1.0 - dev_score))

    if risk_source == "score_gap":
        return max(0.0, min(1.0, abs(test_score - dev_score)))

    return max(0.0, min(1.0, 1.0 - test_score))


def compute_fairness_risk(
    row: dict,
    fairness_source: str = "placeholder",
    default_fairness_risk: float = 0.10,
) -> float:
    """
    Derive fairness risk from Phase 1 rows.

    Phase 1 did not evaluate fairness directly, so we support:
      placeholder      -> same default for all prompts
      prompt_heuristic -> lower risk if prompt contains fairness-related terms
      fairness_risk    -> use fairness_risk column if present
    """
    if fairness_source in row and row.get(fairness_source) not in {None, ""}:
        return _to_float(row.get(fairness_source), default=default_fairness_risk)

    if "fairness_risk" in row and row.get("fairness_risk") not in {None, ""}:
        return _to_float(row.get("fairness_risk"), default=default_fairness_risk)

    if fairness_source == "prompt_heuristic":
        prompt = str(row.get("prompt", "")).lower()

        fairness_terms = [
            "gender",
            "race",
            "ethnicity",
            "nationality",
            "religion",
            "age",
            "location",
            "demographic",
            "do not infer",
            "bias",
            "fair",
            "protected",
        ]

        if any(term in prompt for term in fairness_terms):
            return 0.05

        return default_fairness_risk

    return default_fairness_risk


def make_candidate_from_row(
    row: dict,
    row_index: int,
) -> PromptCandidate:
    prompt = row.get("prompt", "")

    return PromptCandidate(
        instruction=prompt,
        metadata={
            "row_index": row_index,
            "method": row.get("method"),
            "dataset": row.get("dataset"),
            "task_type": row.get("task_type"),
            "llm_backend": row.get("llm_backend"),
            "dev_size": row.get("dev_size"),
            "shots_size": row.get("shots_size"),
            "test_size": row.get("test_size"),
            "seed": row.get("seed"),
            "source_file": row.get("source_file"),
        },
    )


def make_result_from_row(
    row: dict,
    candidate: PromptCandidate,
    risk_source: str,
    fairness_source: str,
    default_fairness_risk: float,
) -> EvaluationResult:
    performance = _to_float(row.get("test_score"), default=0.0)
    cost = _to_float(row.get("test_cost"), default=0.0)
    risk = compute_risk(row, risk_source=risk_source)
    fairness_risk = compute_fairness_risk(
        row=row,
        fairness_source=fairness_source,
        default_fairness_risk=default_fairness_risk,
    )

    n_examples = _to_int(row.get("test_size"), default=0)

    return EvaluationResult(
        candidate_id=candidate.candidate_id,
        performance=performance,
        cost=cost,
        risk=risk,
        fairness_risk=fairness_risk,
        drift=0.0,
        n_examples=n_examples,
        details={
            "source": "phase1_summary_table",
            "method": row.get("method"),
            "dataset": row.get("dataset"),
            "dev_score": _to_float(row.get("dev_score"), default=0.0),
            "test_score": _to_float(row.get("test_score"), default=0.0),
            "dev_cost": _to_float(row.get("dev_cost"), default=0.0),
            "test_cost": _to_float(row.get("test_cost"), default=0.0),
            "dev_input_tokens": _to_float(row.get("dev_input_tokens"), default=0.0),
            "dev_output_tokens": _to_float(row.get("dev_output_tokens"), default=0.0),
            "test_input_tokens": _to_float(row.get("test_input_tokens"), default=0.0),
            "test_output_tokens": _to_float(row.get("test_output_tokens"), default=0.0),
            "risk_source": risk_source,
            "fairness_source": fairness_source,
        },
    )


def build_all_candidates_portfolio(
    rows: list[dict],
    risk_source: str,
    fairness_source: str,
    default_fairness_risk: float,
    deduplicate_prompts: bool = True,
) -> PromptPortfolio:
    """
    Build a portfolio containing all candidate rows from the Phase 1 table.
    """
    portfolio = PromptPortfolio()
    seen_prompt_keys = set()

    for idx, row in enumerate(rows):
        prompt = str(row.get("prompt", "")).strip()

        if not prompt:
            continue

        if deduplicate_prompts:
            key = (
                row.get("dataset"),
                row.get("llm_backend"),
                row.get("method"),
                row.get("dev_size"),
                row.get("test_size"),
                prompt,
            )

            if key in seen_prompt_keys:
                continue

            seen_prompt_keys.add(key)

        candidate = make_candidate_from_row(row, row_index=idx)
        result = make_result_from_row(
            row=row,
            candidate=candidate,
            risk_source=risk_source,
            fairness_source=fairness_source,
            default_fairness_risk=default_fairness_risk,
        )

        portfolio.add(candidate, result)

    return portfolio


def make_pareto_portfolio(all_portfolio: PromptPortfolio) -> PromptPortfolio:
    """
    Convert an all-candidates portfolio into a Pareto-only portfolio.
    """
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
    pareto_ids: Optional[set[str]] = None,
) -> list[dict]:
    rows = []

    pareto_ids = pareto_ids or set()

    for candidate in portfolio.evaluated_candidates():
        result = portfolio.get_result(candidate.candidate_id)

        row = {
            "candidate_id": candidate.candidate_id,
            "is_pareto": candidate.candidate_id in pareto_ids,
            "method": candidate.metadata.get("method"),
            "dataset": candidate.metadata.get("dataset"),
            "task_type": candidate.metadata.get("task_type"),
            "llm_backend": candidate.metadata.get("llm_backend"),
            "dev_size": candidate.metadata.get("dev_size"),
            "shots_size": candidate.metadata.get("shots_size"),
            "test_size": candidate.metadata.get("test_size"),
            "seed": candidate.metadata.get("seed"),
            "performance": result.performance,
            "cost": result.cost,
            "risk": result.risk,
            "fairness_risk": result.fairness_risk,
            "drift": result.drift,
            "n_examples": result.n_examples,
            "objective_vector": str(result.objective_vector),
            "prompt": candidate.instruction,
            "source_file": candidate.metadata.get("source_file"),
        }

        for key, value in result.details.items():
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
            x="phase2_prompt_portfolio",
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
            f"performance={row.get('performance')} | "
            f"cost={row.get('cost')} | "
            f"risk={row.get('risk')} | "
            f"fairness_risk={row.get('fairness_risk')} | "
            f"prompt={row.get('prompt')}"
        )

    print("-" * 80)
    print(f"Total rows: {len(rows)}")


def print_routing_summary(rows: list[dict]):
    print("Prompt portfolio recommendations")
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
        help="Optional YAML config path.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional Phase 1 summary CSV path override.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    input_table = args.input or config.get(
        "input_table",
        "outputs/phase1_summary/final_phase1_subj_complete.csv",
    )
    output_dir = args.output_dir or config.get(
        "output_dir",
        "outputs/phase2_prompt_portfolio",
    )

    risk_source = config.get("risk_source", "test_error")
    fairness_source = config.get("fairness_source", "placeholder")
    default_fairness_risk = float(config.get("default_fairness_risk", 0.10))
    deduplicate_prompts = bool(config.get("deduplicate_prompts", True))
    save_all_candidates = bool(config.get("save_all_candidates", True))
    recommend_from = config.get("recommend_from", "pareto")

    phase1_rows = read_csv(input_table)

    all_portfolio = build_all_candidates_portfolio(
        rows=phase1_rows,
        risk_source=risk_source,
        fairness_source=fairness_source,
        default_fairness_risk=default_fairness_risk,
        deduplicate_prompts=deduplicate_prompts,
    )

    if not all_portfolio.evaluations:
        raise ValueError("No candidates produced from Phase 1 table.")

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
        raise ValueError("No Pareto candidates produced from Phase 1 table.")

    preferences = config.get(
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
        ],
    )

    if recommend_from == "all":
        recommendation_portfolio = all_portfolio
    elif recommend_from == "pareto":
        recommendation_portfolio = pareto_portfolio
    else:
        raise ValueError("recommend_from must be either 'pareto' or 'all'.")

    recommendation_rows = routing_to_rows(
        portfolio=recommendation_portfolio,
        preferences=preferences,
    )

    if save_all_candidates:
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
        recommendation_rows,
        f"{output_dir}/phase2_prompt_recommendations.csv",
    )
    save_json(
        recommendation_rows,
        f"{output_dir}/phase2_prompt_recommendations.json",
    )

    if save_all_candidates:
        print_portfolio_summary(
            "Phase 2 all candidates from Phase 1 table",
            all_candidate_rows,
        )

    print_portfolio_summary(
        "Phase 2 Pareto prompt portfolio from Phase 1 table",
        pareto_rows,
    )
    print_routing_summary(recommendation_rows)

    if save_all_candidates:
        print(f"Saved all candidates to: {output_dir}/phase2_all_candidates.csv")

    print(f"Saved Pareto portfolio to: {output_dir}/phase2_prompt_portfolio.csv")
    print(f"Saved recommendations to: {output_dir}/phase2_prompt_recommendations.csv")


if __name__ == "__main__":
    main()