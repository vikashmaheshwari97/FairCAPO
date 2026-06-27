from __future__ import annotations

import math
import random

import pytest

from heal_capo.core import EvaluationResult, PromptCandidate
from heal_capo.optimizers.parent_selection import (
    ParentSelectionConfig,
    ParentSelector,
    block_subset_relation,
    common_blocks,
    crowding_distance,
    dominates,
    evaluation_level,
    get_evaluated_blocks,
    select_parents,
    weighted_tiebreak_score,
)


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


def test_dominates_when_better_or_equal_all_and_strictly_better_one():
    left = make_result(
        candidate_id="left",
        performance=0.90,
        cost=10.0,
        risk=0.10,
        fairness_risk=0.10,
    )
    right = make_result(
        candidate_id="right",
        performance=0.80,
        cost=10.0,
        risk=0.20,
        fairness_risk=0.10,
    )

    assert dominates(left, right)
    assert not dominates(right, left)


def test_dominates_false_for_tradeoff():
    left = make_result(
        candidate_id="left",
        performance=0.90,
        cost=20.0,
        risk=0.10,
        fairness_risk=0.10,
    )
    right = make_result(
        candidate_id="right",
        performance=0.80,
        cost=10.0,
        risk=0.20,
        fairness_risk=0.10,
    )

    assert not dominates(left, right)
    assert not dominates(right, left)


def test_get_evaluated_blocks_from_candidate_metadata():
    candidate = make_candidate("a", blocks=[0, 2, 3])

    assert get_evaluated_blocks(candidate) == {0, 2, 3}


def test_get_evaluated_blocks_from_result_details():
    candidate = make_candidate("a")
    result = make_result(
        candidate_id="a",
        performance=1.0,
        cost=1.0,
        risk=0.0,
        fairness_risk=0.0,
        blocks=[1, 4],
    )

    assert get_evaluated_blocks(candidate, result) == {1, 4}


def test_common_blocks_and_subset_relation():
    left = make_candidate("left", blocks=[0, 1])
    right = make_candidate("right", blocks=[0, 1, 2])

    assert common_blocks(left, right) == {0, 1}
    assert block_subset_relation(left, right) == "left_subset_right"


def test_evaluation_level_prefers_block_count_over_n_examples():
    candidate = make_candidate("a", blocks=[0, 1, 2])
    result = make_result(
        candidate_id="a",
        performance=1.0,
        cost=1.0,
        risk=0.0,
        fairness_risk=0.0,
        blocks=None,
    )
    result.n_examples = 100

    assert evaluation_level(candidate, result) == 3


def test_crowding_distance_gives_boundary_infinity():
    results = [
        make_result("a", 0.70, 30.0, 0.30, 0.30),
        make_result("b", 0.80, 20.0, 0.20, 0.20),
        make_result("c", 0.90, 10.0, 0.10, 0.10),
    ]

    distances = crowding_distance(results)

    assert math.isinf(distances["a"]) or math.isinf(distances["c"])
    assert set(distances) == {"a", "b", "c"}


def test_selector_prefers_incumbent_over_non_incumbent():
    incumbent = make_candidate("inc", blocks=[0])
    challenger = make_candidate("challenger", blocks=[0])

    evaluations = {
        "inc": make_result("inc", 0.70, 10.0, 0.30, 0.20, blocks=[0]),
        "challenger": make_result(
            "challenger",
            0.95,
            5.0,
            0.05,
            0.05,
            blocks=[0],
        ),
    }

    selector = ParentSelector(
        config=ParentSelectionConfig(random_seed=7),
    )

    winner, decision = selector.compare(
        left=incumbent,
        right=challenger,
        incumbent_ids={"inc"},
        evaluations=evaluations,
        population=[incumbent, challenger],
    )

    assert winner.candidate_id == "inc"
    assert decision.reason == "incumbent_preference"


def test_selector_uses_dominance_at_same_level():
    left = make_candidate("left", blocks=[0, 1])
    right = make_candidate("right", blocks=[0, 1])

    evaluations = {
        "left": make_result("left", 0.90, 10.0, 0.10, 0.10, blocks=[0, 1]),
        "right": make_result("right", 0.80, 12.0, 0.20, 0.20, blocks=[0, 1]),
    }

    selector = ParentSelector(
        config=ParentSelectionConfig(random_seed=1),
    )

    winner, decision = selector.compare(
        left=left,
        right=right,
        incumbent_ids=set(),
        evaluations=evaluations,
        population=[left, right],
    )

    assert winner.candidate_id == "left"
    assert decision.reason == "same_level_dominance"


def test_selector_uses_subset_dominance():
    less_evaluated = make_candidate("less", blocks=[0])
    more_evaluated = make_candidate("more", blocks=[0, 1])

    evaluations = {
        "less": make_result("less", 0.70, 12.0, 0.30, 0.30, blocks=[0]),
        "more": make_result("more", 0.90, 10.0, 0.10, 0.10, blocks=[0, 1]),
    }

    selector = ParentSelector(
        config=ParentSelectionConfig(random_seed=1),
    )

    winner, decision = selector.compare(
        left=less_evaluated,
        right=more_evaluated,
        incumbent_ids=set(),
        evaluations=evaluations,
        population=[less_evaluated, more_evaluated],
    )

    assert winner.candidate_id == "more"
    assert decision.reason == "subset_relation_left_subset_right_dominance"


def test_selector_prefers_more_evaluated_non_incumbent_when_no_dominance():
    less = make_candidate("less", blocks=[0])
    more = make_candidate("more", blocks=[1, 2])

    evaluations = {
        "less": make_result("less", 0.90, 20.0, 0.10, 0.10, blocks=[0]),
        "more": make_result("more", 0.80, 10.0, 0.20, 0.20, blocks=[1, 2]),
    }

    selector = ParentSelector(
        config=ParentSelectionConfig(random_seed=1),
    )

    winner, decision = selector.compare(
        left=less,
        right=more,
        incumbent_ids=set(),
        evaluations=evaluations,
        population=[less, more],
    )

    assert winner.candidate_id == "more"
    assert decision.reason == "more_evaluated_non_incumbent"


def test_weighted_tiebreak_prefers_cheaper_incomparable_candidate():
    expensive = make_candidate("expensive", blocks=[0])
    cheap = make_candidate("cheap", blocks=[0])
    evaluations = {
        "expensive": make_result(
            "expensive",
            performance=0.98,
            cost=3000.0,
            risk=0.0,
            fairness_risk=0.05,
            blocks=[0],
        ),
        "cheap": make_result(
            "cheap",
            performance=0.96,
            cost=1000.0,
            risk=0.0,
            fairness_risk=0.05,
            blocks=[0],
        ),
    }

    selector = ParentSelector(
        config=ParentSelectionConfig(
            random_seed=1,
            use_weighted_tiebreak=True,
            weighted_tiebreak={
                "performance": 1.0,
                "cost": 0.5,
                "risk": 0.0,
                "fairness_risk": 0.0,
                "drift": 0.0,
                "cost_scale": 5000.0,
            },
        ),
    )

    winner, decision = selector.compare(
        left=expensive,
        right=cheap,
        incumbent_ids=set(),
        evaluations=evaluations,
        population=[expensive, cheap],
    )

    assert winner.candidate_id == "cheap"
    assert decision.reason == "weighted_tiebreak"
    assert weighted_tiebreak_score(evaluations["cheap"], selector.config) > (
        weighted_tiebreak_score(evaluations["expensive"], selector.config)
    )


def test_selector_returns_evaluated_candidate_over_unevaluated():
    evaluated = make_candidate("evaluated", blocks=[0])
    unevaluated = make_candidate("unevaluated")

    evaluations = {
        "evaluated": make_result("evaluated", 0.7, 10.0, 0.3, 0.2, blocks=[0]),
    }

    selector = ParentSelector(
        config=ParentSelectionConfig(random_seed=1),
    )

    winner, decision = selector.compare(
        left=unevaluated,
        right=evaluated,
        incumbent_ids=set(),
        evaluations=evaluations,
        population=[unevaluated, evaluated],
    )

    assert winner.candidate_id == "evaluated"
    assert decision.reason == "evaluated_vs_unevaluated"


def test_select_two_parents_returns_distinct_candidates():
    population = [
        make_candidate("a", blocks=[0]),
        make_candidate("b", blocks=[0]),
        make_candidate("c", blocks=[0]),
    ]

    evaluations = {
        "a": make_result("a", 0.9, 10.0, 0.1, 0.1, blocks=[0]),
        "b": make_result("b", 0.8, 8.0, 0.2, 0.2, blocks=[0]),
        "c": make_result("c", 0.7, 6.0, 0.3, 0.3, blocks=[0]),
    }

    selector = ParentSelector(
        config=ParentSelectionConfig(random_seed=4),
    )

    parent1, parent2, decisions = selector.select_two_parents(
        population=population,
        incumbent_ids={"a"},
        evaluations=evaluations,
    )

    assert parent1.candidate_id != parent2.candidate_id
    assert len(decisions) == 2


def test_select_parents_function_wrapper():
    population = [
        make_candidate("a", blocks=[0]),
        make_candidate("b", blocks=[0]),
    ]

    evaluations = {
        "a": make_result("a", 0.9, 10.0, 0.1, 0.1, blocks=[0]),
        "b": make_result("b", 0.8, 12.0, 0.2, 0.2, blocks=[0]),
    }

    parent1, parent2, decisions = select_parents(
        population=population,
        incumbent_ids=set(),
        evaluations=evaluations,
        rng=random.Random(0),
    )

    assert {parent1.candidate_id, parent2.candidate_id} == {"a", "b"}
    assert len(decisions) == 2


def test_single_candidate_population_select_parent_only():
    candidate = make_candidate("only", blocks=[0])
    evaluations = {
        "only": make_result("only", 1.0, 1.0, 0.0, 0.0, blocks=[0]),
    }

    selector = ParentSelector(
        config=ParentSelectionConfig(random_seed=1),
    )

    selected, decision = selector.select_parent(
        population=[candidate],
        incumbent_ids={"only"},
        evaluations=evaluations,
    )

    assert selected.candidate_id == "only"
    assert decision.reason == "single_candidate_population"


def test_select_two_parents_requires_at_least_two_candidates():
    candidate = make_candidate("only", blocks=[0])
    selector = ParentSelector()

    with pytest.raises(ValueError):
        selector.select_two_parents(
            population=[candidate],
            incumbent_ids=set(),
            evaluations={},
        )
