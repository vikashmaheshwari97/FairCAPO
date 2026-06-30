from __future__ import annotations

import random

import pytest

from heal_capo.core import EvaluationResult, PromptCandidate
from heal_capo.optimizers.environmental_selection import (
    EnvironmentalSelectionConfig,
    EnvironmentalSelector,
    comparable_by_blocks,
    dominance_fronts,
    environmental_select,
    least_evaluated_ids,
    nondominated_ids,
)
from heal_capo.optimizers.parent_selection import ParentSelectionConfig


def make_candidate(
    name: str,
    blocks: list[int] | None = None,
) -> PromptCandidate:
    metadata = {"method": name}

    if blocks is not None:
        metadata["evaluated_blocks"] = blocks

    candidate = PromptCandidate(
        instruction=f"Prompt {name}",
        metadata=metadata,
    )
    candidate.candidate_id = name
    return candidate


def make_result(
    candidate_id: str,
    performance: float,
    cost: float,
    risk: float,
    fairness_risk: float,
    blocks: list[int] | None = None,
) -> EvaluationResult:
    details = {}

    if blocks is not None:
        details["evaluated_blocks"] = blocks

    return EvaluationResult(
        candidate_id=candidate_id,
        performance=performance,
        cost=cost,
        risk=risk,
        fairness_risk=fairness_risk,
        drift=0.0,
        n_examples=len(blocks or []),
        details=details,
    )


def test_comparable_by_blocks_true_for_same_blocks():
    candidates = {
        "a": make_candidate("a", blocks=[0, 1]),
        "b": make_candidate("b", blocks=[0, 1]),
    }
    evaluations = {
        "a": make_result("a", 0.9, 10.0, 0.1, 0.1, blocks=[0, 1]),
        "b": make_result("b", 0.8, 12.0, 0.2, 0.2, blocks=[0, 1]),
    }

    assert comparable_by_blocks(["a", "b"], candidates, evaluations)


def test_comparable_by_blocks_false_for_different_blocks():
    candidates = {
        "a": make_candidate("a", blocks=[0]),
        "b": make_candidate("b", blocks=[1]),
    }
    evaluations = {
        "a": make_result("a", 0.9, 10.0, 0.1, 0.1, blocks=[0]),
        "b": make_result("b", 0.8, 12.0, 0.2, 0.2, blocks=[1]),
    }

    assert not comparable_by_blocks(["a", "b"], candidates, evaluations)


def test_nondominated_ids_excludes_dominated_candidate():
    evaluations = {
        "a": make_result("a", 0.9, 10.0, 0.1, 0.1, blocks=[0]),
        "b": make_result("b", 0.8, 12.0, 0.2, 0.2, blocks=[0]),
        "c": make_result("c", 0.7, 5.0, 0.3, 0.3, blocks=[0]),
    }

    result = nondominated_ids(["a", "b", "c"], evaluations)

    assert "b" not in result
    assert "a" in result
    assert "c" in result


def test_dominance_fronts_orders_fronts():
    evaluations = {
        "a": make_result("a", 0.9, 10.0, 0.1, 0.1, blocks=[0]),
        "b": make_result("b", 0.8, 12.0, 0.2, 0.2, blocks=[0]),
        "c": make_result("c", 0.7, 5.0, 0.3, 0.3, blocks=[0]),
    }

    fronts = dominance_fronts(["a", "b", "c"], evaluations)

    assert "b" in fronts[-1]
    assert {"a", "c"}.issubset(set(fronts[0]))


def test_least_evaluated_ids_returns_lowest_level():
    candidates = {
        "a": make_candidate("a", blocks=[0, 1]),
        "b": make_candidate("b", blocks=[0]),
        "c": make_candidate("c", blocks=[0, 1, 2]),
    }
    evaluations = {
        "a": make_result("a", 0.9, 10.0, 0.1, 0.1, blocks=[0, 1]),
        "b": make_result("b", 0.8, 12.0, 0.2, 0.2, blocks=[0]),
        "c": make_result("c", 0.7, 5.0, 0.3, 0.3, blocks=[0, 1, 2]),
    }

    assert least_evaluated_ids(["a", "b", "c"], candidates, evaluations) == ["b"]


def test_environmental_selection_noop_when_within_limit():
    population = [make_candidate("a"), make_candidate("b")]
    selector = EnvironmentalSelector(
        config=EnvironmentalSelectionConfig(population_size=2),
    )

    kept, decision = selector.select(
        population=population,
        incumbent_ids=set(),
        evaluations={},
    )

    assert kept == population
    assert decision.removed_ids == []
    assert decision.reason == "population_within_limit"


def test_environmental_selection_requires_positive_population_size():
    population = [make_candidate("a")]

    selector = EnvironmentalSelector(
        config=EnvironmentalSelectionConfig(population_size=0),
    )

    with pytest.raises(ValueError):
        selector.select(
            population=population,
            incumbent_ids=set(),
            evaluations={},
        )


def test_environmental_selection_removes_unevaluated_non_incumbent_first():
    population = [
        make_candidate("inc", blocks=[0]),
        make_candidate("eval", blocks=[0]),
        make_candidate("uneval"),
    ]
    evaluations = {
        "inc": make_result("inc", 0.9, 10.0, 0.1, 0.1, blocks=[0]),
        "eval": make_result("eval", 0.8, 12.0, 0.2, 0.2, blocks=[0]),
    }

    selector = EnvironmentalSelector(
        config=EnvironmentalSelectionConfig(
            population_size=2,
            random_seed=0,
        ),
    )

    kept, decision = selector.select(
        population=population,
        incumbent_ids={"inc"},
        evaluations=evaluations,
    )

    kept_ids = {candidate.candidate_id for candidate in kept}

    assert "uneval" not in kept_ids
    assert decision.removed_ids == ["uneval"]


def test_environmental_selection_preserves_incumbents_when_possible():
    population = [
        make_candidate("inc", blocks=[0]),
        make_candidate("bad", blocks=[0]),
        make_candidate("worse", blocks=[0]),
    ]
    evaluations = {
        "inc": make_result("inc", 0.9, 10.0, 0.1, 0.1, blocks=[0]),
        "bad": make_result("bad", 0.7, 12.0, 0.3, 0.3, blocks=[0]),
        "worse": make_result("worse", 0.6, 20.0, 0.4, 0.4, blocks=[0]),
    }

    selector = EnvironmentalSelector(
        config=EnvironmentalSelectionConfig(
            population_size=2,
            random_seed=0,
        ),
    )

    kept, decision = selector.select(
        population=population,
        incumbent_ids={"inc"},
        evaluations=evaluations,
    )

    kept_ids = {candidate.candidate_id for candidate in kept}

    assert "inc" in kept_ids
    assert len(kept) == 2
    assert len(decision.removed_ids) == 1


def test_environmental_selection_removes_worst_dominated_non_incumbent():
    population = [
        make_candidate("inc", blocks=[0]),
        make_candidate("good", blocks=[0]),
        make_candidate("bad", blocks=[0]),
    ]
    evaluations = {
        "inc": make_result("inc", 0.9, 10.0, 0.1, 0.1, blocks=[0]),
        "good": make_result("good", 0.8, 12.0, 0.2, 0.2, blocks=[0]),
        "bad": make_result("bad", 0.7, 20.0, 0.4, 0.4, blocks=[0]),
    }

    selector = EnvironmentalSelector(
        config=EnvironmentalSelectionConfig(
            population_size=2,
            random_seed=0,
        ),
    )

    kept, decision = selector.select(
        population=population,
        incumbent_ids={"inc"},
        evaluations=evaluations,
    )

    kept_ids = {candidate.candidate_id for candidate in kept}

    assert "bad" not in kept_ids
    assert "bad" in decision.removed_ids


def test_environmental_selection_removes_least_evaluated_when_not_comparable():
    population = [
        make_candidate("a", blocks=[0, 1]),
        make_candidate("b", blocks=[2]),
        make_candidate("c", blocks=[3, 4]),
    ]
    evaluations = {
        "a": make_result("a", 0.9, 20.0, 0.1, 0.1, blocks=[0, 1]),
        "b": make_result("b", 0.8, 10.0, 0.2, 0.2, blocks=[2]),
        "c": make_result("c", 0.7, 5.0, 0.3, 0.3, blocks=[3, 4]),
    }

    selector = EnvironmentalSelector(
        config=EnvironmentalSelectionConfig(
            population_size=2,
            random_seed=0,
        ),
    )

    kept, decision = selector.select(
        population=population,
        incumbent_ids=set(),
        evaluations=evaluations,
    )

    kept_ids = {candidate.candidate_id for candidate in kept}

    assert "b" not in kept_ids
    assert decision.removed_ids == ["b"]


def test_environmental_select_function_wrapper():
    population = [
        make_candidate("a", blocks=[0]),
        make_candidate("b", blocks=[0]),
        make_candidate("c", blocks=[0]),
    ]
    evaluations = {
        "a": make_result("a", 0.9, 10.0, 0.1, 0.1, blocks=[0]),
        "b": make_result("b", 0.8, 12.0, 0.2, 0.2, blocks=[0]),
        "c": make_result("c", 0.7, 20.0, 0.4, 0.4, blocks=[0]),
    }

    kept, decision = environmental_select(
        population=population,
        incumbent_ids={"a"},
        evaluations=evaluations,
        config=EnvironmentalSelectionConfig(
            population_size=2,
            random_seed=0,
        ),
        parent_config=ParentSelectionConfig(),
        rng=random.Random(0),
    )

    kept_ids = {candidate.candidate_id for candidate in kept}

    assert len(kept) == 2
    assert "a" in kept_ids
    assert len(decision.removed_ids) == 1