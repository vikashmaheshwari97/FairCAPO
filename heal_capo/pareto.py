from __future__ import annotations

from typing import Dict, Iterable, List

from .core import EvaluationResult


def performance_constraint_violation(result: EvaluationResult) -> float:
    """Return the configured minimum-performance shortfall, or zero."""
    details = result.details or {}
    if "performance_feasibility_shortfall" in details:
        return max(0.0, float(details.get("performance_feasibility_shortfall") or 0.0))
    threshold = float(details.get("performance_feasibility_threshold") or 0.0)
    return max(0.0, threshold - float(result.performance))


def dominates(a: EvaluationResult, b: EvaluationResult) -> bool:
    """Constraint-aware Pareto dominance using a minimization convention."""
    a_violation = performance_constraint_violation(a)
    b_violation = performance_constraint_violation(b)

    # Deb-style constraint handling: feasible points dominate infeasible points;
    # among infeasible points the smaller accuracy shortfall is preferred.
    if a_violation <= 0.0 < b_violation:
        return True
    if b_violation <= 0.0 < a_violation:
        return False
    if a_violation > 0.0 and b_violation > 0.0:
        return a_violation < b_violation

    av = a.objective_vector
    bv = b.objective_vector
    if len(av) != len(bv):
        raise ValueError(
            f"Objective vectors must have the same length. Got {len(av)} and {len(bv)}."
        )
    return all(x <= y for x, y in zip(av, bv)) and any(x < y for x, y in zip(av, bv))


def non_dominated_ids(results: Iterable[EvaluationResult]) -> List[str]:
    results = list(results)
    keep: List[str] = []
    for result in results:
        if not any(
            other.candidate_id != result.candidate_id and dominates(other, result)
            for other in results
        ):
            keep.append(result.candidate_id)
    return keep


def pareto_archive(
    results_by_id: Dict[str, EvaluationResult],
) -> Dict[str, EvaluationResult]:
    ids = set(non_dominated_ids(results_by_id.values()))
    return {
        candidate_id: result
        for candidate_id, result in results_by_id.items()
        if candidate_id in ids
    }


def sort_pareto_results(
    results: Iterable[EvaluationResult],
) -> List[EvaluationResult]:
    return sorted(
        list(results),
        key=lambda result: (
            performance_constraint_violation(result),
            -result.performance,
            result.fairness_risk,
            result.cost,
            result.risk,
        ),
    )
