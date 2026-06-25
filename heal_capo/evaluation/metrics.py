from __future__ import annotations

from typing import Iterable

from ..core import EvaluationResult


def failure_recovery_rate(before_failures: int, fixed_failures: int) -> float:
    """
    Fraction of previously observed failures that were fixed after repair.

    Higher is better.
    """
    if before_failures <= 0:
        return 0.0

    return fixed_failures / before_failures


def regression_rate(previously_correct: int, newly_broken: int) -> float:
    """
    Fraction of previously correct examples that became incorrect after repair.

    Lower is better.
    """
    if previously_correct <= 0:
        return 0.0

    return newly_broken / previously_correct


def fairness_recovery_rate(before_violations: int, fixed_violations: int) -> float:
    """
    Fraction of fairness violations fixed after repair.

    Higher is better.
    """
    if before_violations <= 0:
        return 0.0

    return fixed_violations / before_violations


def risk_reduction(before_risk: float, after_risk: float) -> float:
    """
    Absolute reduction in risk.

    Positive means risk improved.
    """
    return before_risk - after_risk


def fairness_risk_reduction(
    before_fairness_risk: float,
    after_fairness_risk: float,
) -> float:
    """
    Absolute reduction in fairness risk.

    Positive means fairness improved.
    """
    return before_fairness_risk - after_fairness_risk


def drift_violation_rate(num_drift_violations: int, total_checks: int) -> float:
    """
    Fraction of prompt updates that violated drift constraints.

    Lower is better.
    """
    if total_checks <= 0:
        return 0.0

    return num_drift_violations / total_checks


def mean_objectives(results: Iterable[EvaluationResult]) -> dict:
    """
    Mean objective values across evaluation results.
    """
    vals = list(results)

    if not vals:
        return {
            "performance": 0.0,
            "cost": 0.0,
            "risk": 0.0,
            "fairness_risk": 0.0,
            "drift": 0.0,
        }

    n = len(vals)

    return {
        "performance": sum(r.performance for r in vals) / n,
        "cost": sum(r.cost for r in vals) / n,
        "risk": sum(r.risk for r in vals) / n,
        "fairness_risk": sum(r.fairness_risk for r in vals) / n,
        "drift": sum(r.drift for r in vals) / n,
    }


def portfolio_size(results: Iterable[EvaluationResult]) -> int:
    """
    Number of evaluated prompts in a portfolio.
    """
    return len(list(results))


def best_performance(results: Iterable[EvaluationResult]) -> float:
    vals = list(results)

    if not vals:
        return 0.0

    return max(r.performance for r in vals)


def lowest_cost(results: Iterable[EvaluationResult]) -> float:
    vals = list(results)

    if not vals:
        return 0.0

    return min(r.cost for r in vals)


def lowest_risk(results: Iterable[EvaluationResult]) -> float:
    vals = list(results)

    if not vals:
        return 0.0

    return min(r.risk for r in vals)


def lowest_fairness_risk(results: Iterable[EvaluationResult]) -> float:
    vals = list(results)

    if not vals:
        return 0.0

    return min(r.fairness_risk for r in vals)