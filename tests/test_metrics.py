from heal_capo.core import EvaluationResult
from heal_capo.evaluation.metrics import (
    best_performance,
    drift_violation_rate,
    failure_recovery_rate,
    fairness_recovery_rate,
    fairness_risk_reduction,
    lowest_cost,
    lowest_fairness_risk,
    lowest_risk,
    mean_objectives,
    portfolio_size,
    regression_rate,
    risk_reduction,
)


def _results():
    return [
        EvaluationResult(
            candidate_id="a",
            performance=0.8,
            cost=10.0,
            risk=0.2,
            fairness_risk=0.1,
            drift=0.0,
        ),
        EvaluationResult(
            candidate_id="b",
            performance=0.6,
            cost=5.0,
            risk=0.4,
            fairness_risk=0.3,
            drift=0.2,
        ),
    ]


def test_failure_recovery_rate():
    assert failure_recovery_rate(10, 4) == 0.4
    assert failure_recovery_rate(0, 4) == 0.0


def test_regression_rate():
    assert regression_rate(20, 2) == 0.1
    assert regression_rate(0, 2) == 0.0


def test_fairness_recovery_rate():
    assert fairness_recovery_rate(5, 2) == 0.4
    assert fairness_recovery_rate(0, 2) == 0.0


def test_risk_reduction():
    assert risk_reduction(0.5, 0.2) == 0.3


def test_fairness_risk_reduction():
    assert fairness_risk_reduction(0.4, 0.1) == 0.30000000000000004


def test_drift_violation_rate():
    assert drift_violation_rate(2, 10) == 0.2
    assert drift_violation_rate(2, 0) == 0.0


def test_mean_objectives():
    means = mean_objectives(_results())

    assert means["performance"] == 0.7
    assert means["cost"] == 7.5
    assert means["risk"] == 0.30000000000000004
    assert means["fairness_risk"] == 0.2
    assert means["drift"] == 0.1


def test_mean_objectives_empty():
    means = mean_objectives([])

    assert means["performance"] == 0.0
    assert means["cost"] == 0.0
    assert means["risk"] == 0.0
    assert means["fairness_risk"] == 0.0
    assert means["drift"] == 0.0


def test_portfolio_summary_metrics():
    results = _results()

    assert portfolio_size(results) == 2
    assert best_performance(results) == 0.8
    assert lowest_cost(results) == 5.0
    assert lowest_risk(results) == 0.2
    assert lowest_fairness_risk(results) == 0.1