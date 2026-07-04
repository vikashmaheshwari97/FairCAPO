from __future__ import annotations

import csv
import json
import math
from pathlib import Path

csv.field_size_limit(10 * 1024 * 1024)


METHODS = {
    "FairCAPO": {
        "search": Path("outputs/hpc/adult_faircapo_7p5m_v3/seed_0"),
        "portfolio": "phase2_prompt_portfolio.csv",
        "events": "budgeted_mocapo_events.csv",
        "eval": Path("outputs/hpc/evaluation_large/seed_0/adult_faircapo_7p5m_v3"),
        "event_required": "evolutionary_intensification",
    },
    "MO-CAPO (fairness objective off)": {
        "search": Path("outputs/hpc/adult_ablation_7p5m_v3/seed_0"),
        "portfolio": "phase2_prompt_portfolio.csv",
        "events": "budgeted_mocapo_events.csv",
        "eval": Path("outputs/hpc/evaluation_large/seed_0/adult_ablation_7p5m_v3"),
        "event_required": "evolutionary_intensification",
        "expect_search_fairness_zero": True,
    },
    "NSGA-II-PO + fairness": {
        "search": Path("outputs/hpc/adult_nsga2po_7p5m_v3/seed_0"),
        "portfolio": "nsga2_po_pareto_portfolio.csv",
        "events": "nsga2_po_events.csv",
        "eval": Path("outputs/hpc/evaluation_large/seed_0/adult_nsga2po_7p5m_v3"),
        "event_required": "environmental_selection",
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_csv(path: Path) -> list[dict]:
    require(path.is_file() and path.stat().st_size > 0, f"Missing or empty CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    require(path.is_file() and path.stat().st_size > 0, f"Missing or empty JSON: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def number(value, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"Invalid numeric value for {label}: {value!r}") from exc
    require(math.isfinite(result), f"Non-finite value for {label}: {value!r}")
    return result


def audit_method(name: str, cfg: dict) -> dict:
    search_dir: Path = cfg["search"]
    eval_dir: Path = cfg["eval"]

    portfolio_rows = read_csv(search_dir / cfg["portfolio"])
    require(portfolio_rows, f"{name}: empty search portfolio")
    require("few_shot_examples" in portfolio_rows[0], f"{name}: exact few-shot examples were not persisted")
    require("num_few_shot" in portfolio_rows[0], f"{name}: num_few_shot missing")

    for index, row in enumerate(portfolio_rows):
        try:
            examples = json.loads(row.get("few_shot_examples") or "[]")
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{name}: malformed few_shot_examples row {index}") from exc
        require(isinstance(examples, list), f"{name}: few_shot_examples is not a list")
        require(len(examples) == int(float(row.get("num_few_shot") or 0)), f"{name}: few-shot count mismatch at row {index}")
        require(len(examples) <= 5, f"{name}: candidate exceeds five-shot cap")

        if cfg.get("expect_search_fairness_zero"):
            search_fairness = number(
                row.get("fairness_risk"),
                f"{name}.search_fairness[{index}]",
            )
            require(
                abs(search_fairness) <= 1e-12,
                f"{name}: search fairness objective was not disabled at row {index}",
            )
            require(
                str(row.get("detail_fairness_source", ""))
                == "disabled_for_adult_v3_mocapo_ablation",
                f"{name}: strict ablation provenance is missing at row {index}",
            )
            require(
                str(row.get("detail_fairness_objective_enabled", "")).lower()
                in {"false", "0"},
                f"{name}: fairness objective was not marked disabled at row {index}",
            )

    events = read_csv(search_dir / cfg["events"])
    event_types = {str(row.get("event_type", "")) for row in events}
    require(cfg["event_required"] in event_types, f"{name}: required search event {cfg['event_required']} missing")

    budget = read_json(search_dir / "budget_summary.json")
    used_budget = number(
        budget.get("used_budget", budget.get("total_tokens", 0.0)),
        f"{name}.used_budget",
    )
    require(used_budget > 0.0, f"{name}: no search budget recorded")

    eval_rows = read_csv(eval_dir / "test_eval_candidates.csv")
    summary_doc = read_json(eval_dir / "test_eval_summary.json")
    metadata = summary_doc.get("metadata", {})

    require(len(eval_rows) == len(portfolio_rows), f"{name}: held-out row count does not match portfolio")
    require(int(metadata.get("num_test_examples", 0)) == 2000, f"{name}: held-out test size is not 2000")
    require(int(metadata.get("num_prompts", 0)) == len(eval_rows), f"{name}: summary prompt count mismatch")

    performances: list[float] = []
    eligible: list[tuple[float, float, float]] = []
    for index, row in enumerate(eval_rows):
        performance = number(row.get("performance"), f"{name}.performance[{index}]")
        cost = number(row.get("cost"), f"{name}.cost[{index}]")
        fairness = number(row.get("fairness_risk"), f"{name}.fairness[{index}]")
        require(0.0 <= performance <= 1.0, f"{name}: performance outside [0,1]")
        require(cost > 0.0, f"{name}: non-positive held-out cost")
        require(0.0 <= fairness <= 1.0, f"{name}: fairness outside [0,1]")
        require(str(row.get("source_candidate_id", "")).strip(), f"{name}: source candidate identity missing")
        performances.append(performance)
        if performance >= 0.72:
            eligible.append((performance, cost, fairness))

    require(eligible, f"{name}: no held-out candidate reaches the 0.72 reporting threshold")
    best_accuracy = max(performances)
    fairest_eligible = min(eligible, key=lambda item: (item[2], -item[0], item[1]))

    print(
        f"{name}: prompts={len(portfolio_rows)}, budget={used_budget:.0f}, "
        f"best_accuracy={best_accuracy:.3f}, "
        f"fairest_eligible=(acc={fairest_eligible[0]:.3f}, "
        f"cost={fairest_eligible[1]:.2f}, fair={fairest_eligible[2]:.3f})"
    )
    return {
        "name": name,
        "num_prompts": len(portfolio_rows),
        "used_budget": used_budget,
        "best_accuracy": best_accuracy,
        "fairest_eligible": fairest_eligible,
    }


def main() -> None:
    rows = [audit_method(name, cfg) for name, cfg in METHODS.items()]
    require(len(rows) == 3, "Expected three validated methods")
    print("Adult v3 output validation PASSED.")


if __name__ == "__main__":
    main()
