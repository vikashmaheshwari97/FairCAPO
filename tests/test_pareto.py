import pytest

from heal_capo.core import EvaluationResult
from heal_capo.pareto import (
    dominates,
    non_dominated_ids,
    pareto_archive,
    sort_pareto_results,
)


def test_dominates_with_fairness_risk():
    a = EvaluationResult(
        "a",
        performance=0.9,
        cost=1.0,
        risk=0.1,
        fairness_risk=0.05,
    )
    b = EvaluationResult(
        "b",
        performance=0.8,
        cost=1.2,
        risk=0.2,
        fairness_risk=0.10,
    )

    assert dominates(a, b)


def test_does_not_dominate_when_fairness_is_worse():
    a = EvaluationResult(
        "a",
        performance=0.9,
        cost=1.0,
        risk=0.1,
        fairness_risk=0.30,
    )
    b = EvaluationResult(
        "b",
        performance=0.8,
        cost=1.2,
        risk=0.2,
        fairness_risk=0.10,
    )

    assert not dominates(a, b)


def test_non_dominated_ids_with_fairness_tradeoff():
    high_accuracy_unfair = EvaluationResult(
        "high_accuracy_unfair",
        performance=0.9,
        cost=1.0,
        risk=0.1,
        fairness_risk=0.30,
    )
    lower_accuracy_fair = EvaluationResult(
        "lower_accuracy_fair",
        performance=0.8,
        cost=1.0,
        risk=0.1,
        fairness_risk=0.05,
    )

    ids = non_dominated_ids([high_accuracy_unfair, lower_accuracy_fair])

    assert set(ids) == {"high_accuracy_unfair", "lower_accuracy_fair"}


def test_non_dominated_ids_removes_dominated_result():
    a = EvaluationResult(
        "a",
        performance=0.9,
        cost=1.0,
        risk=0.1,
        fairness_risk=0.05,
    )
    b = EvaluationResult(
        "b",
        performance=0.8,
        cost=1.2,
        risk=0.2,
        fairness_risk=0.10,
    )

    ids = non_dominated_ids([a, b])

    assert ids == ["a"]


def test_pareto_archive_keeps_only_non_dominated_results():
    a = EvaluationResult(
        "a",
        performance=0.9,
        cost=1.0,
        risk=0.1,
        fairness_risk=0.05,
    )
    b = EvaluationResult(
        "b",
        performance=0.8,
        cost=1.2,
        risk=0.2,
        fairness_risk=0.10,
    )

    archive = pareto_archive({"a": a, "b": b})

    assert set(archive.keys()) == {"a"}
    assert archive["a"] is a


def test_sort_pareto_results():
    a = EvaluationResult(
        "a",
        performance=0.8,
        cost=1.0,
        risk=0.1,
        fairness_risk=0.05,
    )
    b = EvaluationResult(
        "b",
        performance=0.9,
        cost=1.5,
        risk=0.2,
        fairness_risk=0.10,
    )
    c = EvaluationResult(
        "c",
        performance=0.9,
        cost=1.0,
        risk=0.1,
        fairness_risk=0.05,
    )

    sorted_results = sort_pareto_results([a, b, c])

    assert [result.candidate_id for result in sorted_results] == ["c", "b", "a"]


def test_dominates_raises_on_mismatched_objective_lengths():
    class BadEvaluationResult(EvaluationResult):
        @property
        def objective_vector(self):
            return (-self.performance, self.cost)

    a = EvaluationResult(
        "a",
        performance=0.9,
        cost=1.0,
        risk=0.1,
        fairness_risk=0.05,
    )
    b = BadEvaluationResult(
        "b",
        performance=0.8,
        cost=1.2,
        risk=0.2,
        fairness_risk=0.10,
    )

    with pytest.raises(ValueError):
        dominates(a, b)