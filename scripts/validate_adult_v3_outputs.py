from __future__ import annotations

import csv
import json
import math
from pathlib import Path

csv.field_size_limit(10 * 1024 * 1024)

TEST_SIZE = 1000
REPORTING_THRESHOLD = 0.70
MAX_BUDGET = 1_000_000.0

METHODS = {
    "FairCAPO": {
        "search": Path("outputs/hpc/adult_faircapo_1m_v3/seed_0"),
        "portfolio": "phase2_prompt_portfolio.csv",
        "events": "budgeted_mocapo_events.csv",
        "eval": Path("outputs/hpc/evaluation_1m/seed_0/adult_faircapo_1m_v3"),
        "event_required": "evolutionary_intensification",
    },
    "MO-CAPO (fairness objective off)": {
        "search": Path("outputs/hpc/adult_ablation_1m_v3/seed_0"),
        "portfolio": "phase2_prompt_portfolio.csv",
        "events": "budgeted_mocapo_events.csv",
        "eval": Path("outputs/hpc/evaluation_1m/seed_0/adult_ablation_1m_v3"),
        "event_required": "evolutionary_intensification",
        "expect_search_fairness_zero": True,
    },
    "NSGA-II-PO + fairness": {
        "search": Path("outputs/hpc/adult_nsga2po_1m_v3/seed_0"),
        "portfolio": "nsga2_po_pareto_portfolio.csv",
        "events": "nsga2_po_events.csv",
        "eval": Path("outputs/hpc/evaluation_1m/seed_0/adult_nsga2po_1m_v3"),
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
    require("few_shot_examples" in portfolio_rows[0], f"{name}: examples missing")
    require("num_few_shot" in portfolio_rows[0], f"{name}: shot count missing")

    for index, row in enumerate(portfolio_rows):
        examples = json.loads(row.get("few_shot_examples") or "[]")
        require(isinstance(examples, list), f"{name}: malformed examples")
        require(len(examples) == int(float(row.get("num_few_shot") or 0)), f"{name}: shot count mismatch")
        require(len(examples) <= 5, f"{name}: exceeds five-shot cap")

        if cfg.get("expect_search_fairness_zero"):
            require(abs(number(row.get("fairness_risk"), f"{name}.search_fairness[{index}]")) <= 1e-12, f"{name}: fairness influenced search")
            require(str(row.get("detail_fairness_source", "")) == "disabled_for_adult_v3_mocapo_ablation", f"{name}: strict-ablation provenance missing")

    events = read_csv(search_dir / cfg["events"])
    event_types = {str(row.get("event_type", "")) for row in events}
    require(cfg["event_required"] in event_types, f"{name}: required search event missing")

    budget = read_json(search_dir / "budget_summary.json")
    used_budget = number(budget.get("used_budget", budget.get("total_tokens", 0.0)), f"{name}.used_budget")
    require(0.0 < used_budget <= MAX_BUDGET, f"{name}: budget outside 1M cap: {used_budget}")

    eval_rows = read_csv(eval_dir / "test_eval_candidates.csv")
    summary_doc = read_json(eval_dir / "test_eval_summary.json")
    metadata = summary_doc.get("metadata", {})
    require(len(eval_rows) == len(portfolio_rows), f"{name}: held-out row count mismatch")
    require(int(metadata.get("num_test_examples", 0)) == TEST_SIZE, f"{name}: held-out test size is not {TEST_SIZE}")

    performances: list[float] = []
    eligible: list[tuple[float, float, float]] = []
    for index, row in enumerate(eval_rows):
        performance = number(row.get("performance"), f"{name}.performance[{index}]")
        cost = number(row.get("cost"), f"{name}.cost[{index}]")
        fairness = number(row.get("fairness_risk"), f"{name}.fairness[{index}]")
        require(0.0 <= performance <= 1.0, f"{name}: invalid performance")
        require(cost > 0.0, f"{name}: non-positive cost")
        require(0.0 <= fairness <= 1.0, f"{name}: invalid fairness")
        require(str(row.get("source_candidate_id", "")).strip(), f"{name}: source identity missing")
        performances.append(performance)
        if performance >= REPORTING_THRESHOLD:
            eligible.append((performance, cost, fairness))

    require(eligible, f"{name}: no candidate reaches {REPORTING_THRESHOLD:.2f} accuracy")
    best_accuracy = max(performances)
    fairest = min(eligible, key=lambda item: (item[2], -item[0], item[1]))
    print(
        f"{name}: prompts={len(portfolio_rows)}, budget={used_budget:.0f}, "
        f"best_accuracy={best_accuracy:.3f}, "
        f"fairest_eligible=(acc={fairest[0]:.3f}, cost={fairest[1]:.2f}, fair={fairest[2]:.3f})"
    )
    return {"name": name, "best_accuracy": best_accuracy, "fairest_eligible": fairest}


def main() -> None:
    rows = [audit_method(name, cfg) for name, cfg in METHODS.items()]
    require(len(rows) == 3, "Expected three methods")
    print("Adult v3 1M output validation PASSED.")


if __name__ == "__main__":
    main()
