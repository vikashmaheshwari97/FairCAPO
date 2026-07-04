from __future__ import annotations

"""Run the Adult v3 MO-CAPO fairness-objective ablation.

The shared Adult v3 evaluator historically falls back to a prompt-text fairness
heuristic when ``fairness.in_loop`` is false. Because ``fairness_risk`` is part
of every EvaluationResult objective vector, that fallback still influences
Pareto selection and is therefore not a true fairness-objective-off ablation.

This entrypoint preserves the shared dataset, prompts, balanced few-shot pool,
model, budget, evolutionary operators, and intensification settings, but forces
search-time fairness risk to a constant zero. The separate held-out evaluator
still computes real equalized-odds risk for reporting after search.
"""

from typing import Any

from heal_capo.core import EvaluationResult
from scripts import run_phase2_budgeted_mocapo as runner


class FairnessObjectiveDisabledEvaluator:
    """Delegate evaluation while removing fairness from the search objective."""

    def __init__(self, base: Any):
        self.base = base

    def evaluate(self, candidate, data) -> EvaluationResult:
        result = self.base.evaluate(candidate, data)
        details = dict(result.details or {})
        details.update(
            {
                "fairness_source": "disabled_for_adult_v3_mocapo_ablation",
                "fairness_objective_enabled": False,
                "search_fairness_risk_before_disable": float(
                    result.fairness_risk
                ),
            }
        )
        result.fairness_risk = 0.0
        result.details = details
        return result


def build_fairness_disabled_evaluator(
    config: dict,
    force_no_llm: bool = False,
):
    if bool((config.get("fairness", {}) or {}).get("in_loop", False)):
        raise ValueError(
            "Adult v3 MO-CAPO ablation requires fairness.in_loop: false."
        )

    base = _ORIGINAL_BUILD_OBJECTIVE_EVALUATOR(
        config=config,
        force_no_llm=force_no_llm,
    )
    return FairnessObjectiveDisabledEvaluator(base)


_ORIGINAL_BUILD_OBJECTIVE_EVALUATOR = runner.build_objective_evaluator


def main() -> None:
    runner.build_objective_evaluator = build_fairness_disabled_evaluator
    runner.main()


if __name__ == "__main__":
    main()
