from __future__ import annotations

import random

from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
from heal_capo.objectives import ObjectiveEvaluator
from heal_capo.optimizers.advance_incumbents import (
    AdvanceIncumbentsConfig,
    IncumbentAdvancer,
    advance_incumbents,
    common_incumbent_blocks,
    get_candidate_blocks,
    incumbent_block_map,
    incumbents_are_aligned,
    select_block_for_incumbent,
    select_incumbent_to_advance,
    union_incumbent_blocks,
)
from heal_capo.optimizers.block_evaluator import BlockEvaluator
from heal_capo.optimizers.budget_allocator import BudgetAllocator


class SimpleBlockEvaluator(ObjectiveEvaluator):
    def evaluate(self, candidate: PromptCandidate, data):
        total = len(data)
        performance = 0.5 + 0.1 * total
        cost = float(total)
        risk = max(0.0, 1.0 - performance)

        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            performance=performance,
            cost=cost,
            risk=risk,
            fairness_risk=0.1,
            drift=0.0,
            n_examples=total,
            details={},
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
    blocks: list[int],
    performance: float = 0.8,
    cost: float = 1.0,
) -> EvaluationResult:
    return EvaluationResult(
        candidate_id=candidate_id,
        performance=performance,
        cost=cost,
        risk=1.0 - performance,
        fairness_risk=0.1,
        drift=0.0,
        n_examples=len(blocks),
        details={"evaluated_blocks": blocks},
    )


def make_block_evaluator() -> BlockEvaluator:
    data = [
        {"text": "a"},
        {"text": "b"},
        {"text": "c"},
        {"text": "d"},
        {"text": "e"},
        {"text": "f"},
    ]

    return BlockEvaluator.from_data(
        evaluator=SimpleBlockEvaluator(),
        data=data,
        block_size=2,
        drop_last=False,
    )


def make_portfolio(candidates: list[PromptCandidate]) -> PromptPortfolio:
    portfolio = PromptPortfolio()

    for candidate in candidates:
        blocks = candidate.metadata.get("evaluated_blocks", [])
        result = make_result(
            candidate_id=candidate.candidate_id,
            blocks=blocks,
        )
        portfolio.add(candidate, result)

    return portfolio


def test_get_candidate_blocks_from_metadata_or_portfolio():
    candidate = make_candidate("a", blocks=[0, 1])
    portfolio = make_portfolio([candidate])

    assert get_candidate_blocks(candidate, portfolio) == {0, 1}


def test_incumbent_block_map_common_and_union_blocks():
    a = make_candidate("a", blocks=[0, 1])
    b = make_candidate("b", blocks=[1, 2])
    portfolio = make_portfolio([a, b])

    block_map = incumbent_block_map([a, b], portfolio)

    assert block_map == {"a": {0, 1}, "b": {1, 2}}
    assert common_incumbent_blocks([a, b], portfolio) == {1}
    assert union_incumbent_blocks([a, b], portfolio) == {0, 1, 2}


def test_incumbents_are_aligned_true_when_same_blocks():
    a = make_candidate("a", blocks=[0, 1])
    b = make_candidate("b", blocks=[0, 1])
    portfolio = make_portfolio([a, b])

    assert incumbents_are_aligned([a, b], portfolio)


def test_incumbents_are_aligned_false_when_different_blocks():
    a = make_candidate("a", blocks=[0])
    b = make_candidate("b", blocks=[0, 1])
    portfolio = make_portfolio([a, b])

    assert not incumbents_are_aligned([a, b], portfolio)


def test_select_incumbent_to_advance_fewest_blocks():
    a = make_candidate("a", blocks=[0, 1])
    b = make_candidate("b", blocks=[0])
    c = make_candidate("c", blocks=[0, 1, 2])
    portfolio = make_portfolio([a, b, c])

    selected = select_incumbent_to_advance([a, b, c], portfolio)

    assert selected.candidate_id == "b"


def test_select_block_for_incumbent_prioritizes_missing_union_block():
    a = make_candidate("a", blocks=[0])
    b = make_candidate("b", blocks=[0, 1])
    portfolio = make_portfolio([a, b])
    block_evaluator = make_block_evaluator()

    selected_block = select_block_for_incumbent(
        candidate=a,
        incumbents=[a, b],
        block_evaluator=block_evaluator,
        portfolio=portfolio,
        rng=random.Random(0),
    )

    assert selected_block == 1


def test_select_block_for_incumbent_uses_new_block_when_aligned():
    a = make_candidate("a", blocks=[0])
    b = make_candidate("b", blocks=[0])
    portfolio = make_portfolio([a, b])
    block_evaluator = make_block_evaluator()

    selected_block = select_block_for_incumbent(
        candidate=a,
        incumbents=[a, b],
        block_evaluator=block_evaluator,
        portfolio=portfolio,
        rng=random.Random(0),
    )

    assert selected_block in {1, 2}


def test_advancer_returns_no_incumbents_decision():
    block_evaluator = make_block_evaluator()
    budget_allocator = BudgetAllocator(max_budget=10.0)
    portfolio = PromptPortfolio()

    advancer = IncumbentAdvancer(
        block_evaluator=block_evaluator,
        budget_allocator=budget_allocator,
    )

    decision = advancer.advance(
        incumbents=[],
        portfolio=portfolio,
    )

    assert decision.advanced is False
    assert decision.reason == "no_incumbents"


def test_advancer_returns_budget_exhausted_decision():
    candidate = make_candidate("a", blocks=[0])
    block_evaluator = make_block_evaluator()
    budget_allocator = BudgetAllocator(max_budget=2.0)

    evaluation = block_evaluator.evaluate_block(
        candidate=candidate,
        block_id=0,
        use_cache=False,
    )
    budget_allocator.record_block_evaluation(evaluation)

    portfolio = make_portfolio([candidate])

    advancer = IncumbentAdvancer(
        block_evaluator=block_evaluator,
        budget_allocator=budget_allocator,
    )

    decision = advancer.advance(
        incumbents=[candidate],
        portfolio=portfolio,
    )

    assert decision.advanced is False
    assert decision.reason == "budget_exhausted"


def test_advancer_advances_candidate_and_updates_budget_and_blocks():
    a = make_candidate("a", blocks=[0])
    b = make_candidate("b", blocks=[0, 1])
    block_evaluator = make_block_evaluator()
    budget_allocator = BudgetAllocator(max_budget=100.0)
    portfolio = make_portfolio([a, b])

    advancer = IncumbentAdvancer(
        block_evaluator=block_evaluator,
        budget_allocator=budget_allocator,
        config=AdvanceIncumbentsConfig(
            random_seed=0,
            update_pareto_archive=False,
        ),
        rng=random.Random(0),
    )

    decision = advancer.advance(
        incumbents=[a, b],
        portfolio=portfolio,
    )

    assert decision.advanced is True
    assert decision.candidate_id == "a"
    assert decision.block_id == 1
    assert decision.budget_used > 0
    assert sorted(a.metadata["evaluated_blocks"]) == [0, 1]
    assert "a" in portfolio.evaluations


def test_advancer_returns_no_available_block_when_all_blocks_done():
    a = make_candidate("a", blocks=[0, 1, 2])
    b = make_candidate("b", blocks=[0, 1, 2])
    block_evaluator = make_block_evaluator()
    budget_allocator = BudgetAllocator(max_budget=100.0)
    portfolio = make_portfolio([a, b])

    advancer = IncumbentAdvancer(
        block_evaluator=block_evaluator,
        budget_allocator=budget_allocator,
        config=AdvanceIncumbentsConfig(
            allow_new_block_when_aligned=True,
        ),
    )

    decision = advancer.advance(
        incumbents=[a, b],
        portfolio=portfolio,
    )

    assert decision.advanced is False
    assert decision.reason == "no_available_block"


def test_advance_incumbents_function_wrapper():
    a = make_candidate("a", blocks=[0])
    b = make_candidate("b", blocks=[0, 1])
    block_evaluator = make_block_evaluator()
    budget_allocator = BudgetAllocator(max_budget=100.0)
    portfolio = make_portfolio([a, b])

    decision = advance_incumbents(
        incumbents=[a, b],
        portfolio=portfolio,
        block_evaluator=block_evaluator,
        budget_allocator=budget_allocator,
        config=AdvanceIncumbentsConfig(
            random_seed=0,
            update_pareto_archive=False,
        ),
        rng=random.Random(0),
    )

    assert decision.advanced is True
    assert decision.candidate_id == "a"
    assert decision.block_id == 1