from __future__ import annotations

from typing import Dict, Iterable, List

from .core import EvaluationResult


def dominates(a: EvaluationResult, b: EvaluationResult) -> bool:
    """
    Return True if evaluation result a Pareto-dominates b.

    We use a minimization convention:
      - lower -performance is better, meaning higher performance is better
      - lower cost is better
      - lower risk is better
      - lower fairness_risk is better

    a dominates b if:
      - a is no worse than b on every objective
      - a is strictly better than b on at least one objective
    """
    av = a.objective_vector
    bv = b.objective_vector

    if len(av) != len(bv):
        raise ValueError(
            f"Objective vectors must have the same length. "
            f"Got {len(av)} and {len(bv)}."
        )

    return all(x <= y for x, y in zip(av, bv)) and any(
        x < y for x, y in zip(av, bv)
    )


def non_dominated_ids(results: Iterable[EvaluationResult]) -> List[str]:
    """
    Return candidate IDs that are not dominated by any other result.
    """
    results = list(results)
    keep: List[str] = []

    for result in results:
        dominated = False

        for other in results:
            if other.candidate_id == result.candidate_id:
                continue

            if dominates(other, result):
                dominated = True
                break

        if not dominated:
            keep.append(result.candidate_id)

    return keep


def pareto_archive(
    results_by_id: Dict[str, EvaluationResult],
) -> Dict[str, EvaluationResult]:
    """
    Keep only non-dominated evaluation results.
    """
    ids = set(non_dominated_ids(results_by_id.values()))

    return {
        candidate_id: result
        for candidate_id, result in results_by_id.items()
        if candidate_id in ids
    }


def sort_pareto_results(
    results: Iterable[EvaluationResult],
) -> List[EvaluationResult]:
    """
    Sort Pareto results for readable reporting.

    Priority:
      1. Higher performance
      2. Lower risk
      3. Lower fairness risk
      4. Lower cost
    """
    return sorted(
        list(results),
        key=lambda result: (
            -result.performance,
            result.risk,
            result.fairness_risk,
            result.cost,
        ),
    )