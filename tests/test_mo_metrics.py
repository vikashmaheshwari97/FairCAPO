from __future__ import annotations

import math

from heal_capo.core import EvaluationResult
from heal_capo.evaluation.mo_metrics import (
    DEFAULT_OBJECTIVE_SPECS,
    NormalizationBounds,
    approximation_gap,
    approximation_gap_with_bounds,
    chebychev_utility,
    dominates_normalized,
    fixed_bounds_from_config,
    get_objective_value,
    hypervolume,
    infer_bounds,
    nondominated_points,
    normalize_result,
    normalize_results,
    nr2,
    nr2_with_bounds,
    optimistic_hypervolume,
    pessimistic_hypervolume,
    results_to_matrix,
    sample_preference_vectors,
    summarize_mo_metrics,
)


def make_result(
    candidate_id: str,
    performance: float,
    cost: float,
    risk: float,
    fairness_risk: float,
    drift: float = 0.0,
) -> EvaluationResult:
    return EvaluationResult(
        candidate_id=candidate_id,
        performance=performance,
        cost=cost,
        risk=risk,
        fairness_risk=fairness_risk,
        drift=drift,
    )


def sample_results() -> list[EvaluationResult]:
    return [
        make_result("a", performance=0.9, cost=10.0, risk=0.1, fairness_risk=0.1),
        make_result("b", performance=0.8, cost=5.0, risk=0.2, fairness_risk=0.2),
        make_result("c", performance=0.7, cost=3.0, risk=0.3, fairness_risk=0.1),
    ]


def test_get_objective_value_reads_result_attribute():
    result = make_result("a", 0.9, 10.0, 0.1, 0.2)

    assert get_objective_value(result, "performance") == 0.9
    assert get_objective_value(result, "cost") == 10.0


def test_get_objective_value_falls_back_to_details():
    result = make_result("a", 0.9, 10.0, 0.1, 0.2)
    result.details["verifier_failure_risk"] = 0.33

    assert get_objective_value(result, "verifier_failure_risk") == 0.33


def test_results_to_matrix():
    matrix = results_to_matrix(sample_results())

    assert len(matrix) == 3
    assert len(matrix[0]) == len(DEFAULT_OBJECTIVE_SPECS)
    assert matrix[0] == [0.9, 10.0, 0.1, 0.1]


def test_infer_bounds():
    bounds = infer_bounds(sample_results())

    assert bounds.minimum["performance"] == 0.7
    assert bounds.maximum["performance"] == 0.9
    assert bounds.minimum["cost"] == 3.0
    assert bounds.maximum["cost"] == 10.0


def test_infer_bounds_empty_uses_default_zero_one():
    bounds = infer_bounds([])

    assert bounds.minimum["performance"] == 0.0
    assert bounds.maximum["performance"] == 1.0


def test_normalize_result_handles_max_and_min_objectives():
    result = make_result("a", performance=0.9, cost=3.0, risk=0.1, fairness_risk=0.1)

    bounds = NormalizationBounds(
        minimum={
            "performance": 0.7,
            "cost": 3.0,
            "risk": 0.1,
            "fairness_risk": 0.1,
        },
        maximum={
            "performance": 0.9,
            "cost": 10.0,
            "risk": 0.3,
            "fairness_risk": 0.3,
        },
    )

    vector = normalize_result(result, bounds)

    assert vector == [1.0, 1.0, 1.0, 1.0]


def test_normalize_results_infers_bounds():
    vectors = normalize_results(sample_results())

    assert len(vectors) == 3
    assert all(len(vector) == 4 for vector in vectors)
    assert all(0.0 <= value <= 1.0 for vector in vectors for value in vector)


def test_dominates_normalized_true_and_false():
    assert dominates_normalized([1.0, 0.8], [0.8, 0.8])
    assert not dominates_normalized([0.8, 1.0], [1.0, 0.8])


def test_nondominated_points_removes_dominated_point():
    points = [
        [1.0, 1.0],
        [0.8, 0.8],
        [0.5, 1.0],
    ]

    nd = nondominated_points(points)

    assert [0.8, 0.8] not in nd
    assert [1.0, 1.0] in nd


def test_hypervolume_empty_is_zero():
    assert hypervolume([]) == 0.0


def test_hypervolume_single_point_with_explicit_bounds():
    result = make_result("a", performance=1.0, cost=0.0, risk=0.0, fairness_risk=0.0)

    bounds = NormalizationBounds(
        minimum={
            "performance": 0.0,
            "cost": 0.0,
            "risk": 0.0,
            "fairness_risk": 0.0,
        },
        maximum={
            "performance": 1.0,
            "cost": 1.0,
            "risk": 1.0,
            "fairness_risk": 1.0,
        },
    )

    assert hypervolume([result], bounds=bounds) == 1.0


def test_hypervolume_in_range_for_sample_results():
    hv = hypervolume(sample_results())

    assert 0.0 <= hv <= 1.0


def test_optimistic_and_pessimistic_hypervolume_in_range():
    candidates = sample_results()
    references = candidates + [
        make_result("ref", 1.0, 1.0, 0.0, 0.0),
    ]

    hv_opt = optimistic_hypervolume(candidates, references)
    hv_pes = pessimistic_hypervolume(candidates)

    assert 0.0 <= hv_opt <= 1.0
    assert 0.0 <= hv_pes <= 1.0


def test_chebychev_utility_perfect_point_is_one():
    utility = chebychev_utility(
        point=[1.0, 1.0, 1.0],
        weights=[0.2, 0.3, 0.5],
    )

    assert utility == 1.0


def test_chebychev_utility_decreases_with_distance():
    good = chebychev_utility(
        point=[0.9, 0.9],
        weights=[0.5, 0.5],
    )
    bad = chebychev_utility(
        point=[0.2, 0.2],
        weights=[0.5, 0.5],
    )

    assert good > bad


def test_sample_preference_vectors_sum_to_one():
    vectors = sample_preference_vectors(
        num_vectors=10,
        num_objectives=4,
        seed=0,
    )

    assert len(vectors) == 10

    for vector in vectors:
        assert len(vector) == 4
        assert math.isclose(sum(vector), 1.0)


def test_sample_preference_vectors_zero_returns_empty():
    assert sample_preference_vectors(0, 4) == []


def test_nr2_self_reference_is_zero():
    results = sample_results()

    value = nr2(
        candidate_results=results,
        reference_results=results,
        num_preference_vectors=25,
        seed=0,
    )

    assert value == 0.0


def test_nr2_worse_candidates_has_non_negative_gap():
    candidates = [
        make_result("bad", 0.5, 10.0, 0.5, 0.5),
    ]
    references = [
        make_result("good", 1.0, 1.0, 0.0, 0.0),
    ]

    value = nr2(
        candidate_results=candidates,
        reference_results=references,
        num_preference_vectors=25,
        seed=0,
    )

    assert value >= 0.0


def test_approximation_gap_self_reference_is_zero():
    results = sample_results()

    assert approximation_gap(results, results) == 0.0


def test_approximation_gap_empty_candidates_is_one():
    references = sample_results()

    assert approximation_gap([], references) == 1.0


def test_summarize_mo_metrics_returns_expected_fields():
    summary = summarize_mo_metrics(
        candidate_results=sample_results(),
        reference_results=sample_results(),
        num_preference_vectors=25,
        seed=0,
    )

    data = summary.to_dict()

    assert data["num_points"] == 3
    assert data["num_objectives"] == 4
    assert "hypervolume" in data
    assert "optimistic_hypervolume" in data
    assert "pessimistic_hypervolume" in data
    assert "approximation_gap" in data
    assert "nr2" in data
    assert data["objective_names"] == [
        "performance",
        "cost",
        "risk",
        "fairness_risk",
    ]

def test_fixed_bounds_from_config_returns_bounds():
    bounds = fixed_bounds_from_config(
        {
            "performance": [0.0, 1.0],
            "cost": [0.0, 50.0],
            "risk": [0.0, 1.0],
            "fairness_risk": [0.0, 1.0],
        }
    )

    assert bounds is not None
    assert bounds.minimum["performance"] == 0.0
    assert bounds.maximum["performance"] == 1.0
    assert bounds.maximum["cost"] == 50.0


def test_fixed_bounds_from_config_empty_returns_none():
    assert fixed_bounds_from_config(None) is None
    assert fixed_bounds_from_config({}) is None


def test_hypervolume_with_fixed_bounds_nonzero_for_perfect_candidate():
    result = make_result(
        "a",
        performance=1.0,
        cost=5.0,
        risk=0.0,
        fairness_risk=0.1,
    )
    bounds = fixed_bounds_from_config(
        {
            "performance": [0.0, 1.0],
            "cost": [0.0, 50.0],
            "risk": [0.0, 1.0],
            "fairness_risk": [0.0, 1.0],
        }
    )

    hv = hypervolume([result], bounds=bounds)

    assert hv > 0.0
    assert hv <= 1.0


def test_nr2_with_bounds_self_reference_is_zero():
    results = sample_results()
    bounds = fixed_bounds_from_config(
        {
            "performance": [0.0, 1.0],
            "cost": [0.0, 50.0],
            "risk": [0.0, 1.0],
            "fairness_risk": [0.0, 1.0],
        }
    )

    value = nr2_with_bounds(
        candidate_results=results,
        reference_results=results,
        bounds=bounds,
        num_preference_vectors=25,
        seed=0,
    )

    assert value == 0.0


def test_approximation_gap_with_bounds_self_reference_is_zero():
    results = sample_results()
    bounds = fixed_bounds_from_config(
        {
            "performance": [0.0, 1.0],
            "cost": [0.0, 50.0],
            "risk": [0.0, 1.0],
            "fairness_risk": [0.0, 1.0],
        }
    )

    assert approximation_gap_with_bounds(results, results, bounds=bounds) == 0.0


def test_summarize_mo_metrics_with_fixed_bounds_marks_bounds_fixed():
    bounds = fixed_bounds_from_config(
        {
            "performance": [0.0, 1.0],
            "cost": [0.0, 50.0],
            "risk": [0.0, 1.0],
            "fairness_risk": [0.0, 1.0],
        }
    )

    summary = summarize_mo_metrics(
        candidate_results=sample_results(),
        reference_results=sample_results(),
        num_preference_vectors=25,
        seed=0,
        bounds=bounds,
    ).to_dict()

    assert summary["bounds"]["fixed"] is True
    assert summary["hypervolume"] > 0.0