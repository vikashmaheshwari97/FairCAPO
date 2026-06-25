"""
Regression tests for the intensification block-bookkeeping fix and the
front-intensification helper.

Background: intensification-accepted challengers used to carry NO record of the
blocks they were evaluated on (merge_results wrote "merged_from_blocks", which
get_evaluated_blocks does not read, and the Intensifier never stamped candidate
metadata). As a result advance_incumbents saw them as 0-block candidates and
wasted advancement budget re-checking block 0, while the intensifier kept racing
challengers on a single common block -> winners decided on ~10 examples.

These tests pin the two halves of the fix:
  1. block bookkeeping is recoverable (result details + challenger metadata), and
  2. advance_front_one_level deepens the whole incumbent front by one aligned
     block per call (capped, budget-aware).
"""

from __future__ import annotations

import random

from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
from heal_capo.objectives import ObjectiveEvaluator
from heal_capo.optimizers.advance_incumbents import (
    AdvanceIncumbentsConfig,
    advance_front_one_level,
    common_incumbent_blocks,
    get_candidate_blocks,
    incumbents_are_aligned,
)
from heal_capo.optimizers.block_evaluator import (
    BlockEvaluation,
    BlockEvaluator,
    merge_results,
)
from heal_capo.optimizers.budget_allocator import BudgetAllocator
from heal_capo.optimizers.intensification import IntensificationConfig, Intensifier
from heal_capo.optimizers.parent_selection import get_evaluated_blocks


class CountingEvaluator(ObjectiveEvaluator):
    """Deterministic per-example evaluator that counts LLM-style calls."""

    def __init__(self):
        self.calls = 0

    def evaluate(self, candidate, data):
        n = len(data)
        self.calls += n

        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            performance=0.7,
            cost=float(n),
            risk=0.3,
            fairness_risk=0.1,
            drift=0.0,
            n_examples=n,
            details={"input_tokens": 5 * n, "output_tokens": n},
        )


def _block_evaluator(num_blocks=6, block_size=2):
    data = [{"text": f"x{i}", "label": "objective"} for i in range(num_blocks * block_size)]
    return BlockEvaluator.from_data(
        evaluator=CountingEvaluator(),
        data=data,
        block_size=block_size,
    )


def _candidate(name: str, blocks: list[int] | None = None) -> PromptCandidate:
    metadata = {"method": name}
    if blocks is not None:
        metadata["evaluated_blocks"] = blocks
    candidate = PromptCandidate(instruction=f"Prompt {name}", metadata=metadata)
    candidate.candidate_id = name
    return candidate


# --------------------------------------------------------------------------
# 1. Bookkeeping: merge_results + Intensifier stamp the real block set.
# --------------------------------------------------------------------------


def test_merge_results_records_evaluated_blocks_in_details():
    evals = [
        BlockEvaluation(
            candidate_id="c",
            block_id=block_id,
            result=EvaluationResult(
                candidate_id="c",
                performance=0.8,
                cost=1.0,
                risk=0.2,
                fairness_risk=0.1,
                drift=0.0,
                n_examples=2,
                details={},
            ),
        )
        for block_id in (2, 0, 1)
    ]

    merged = merge_results("c", evals)

    # Canonical key is sorted and recoverable by get_evaluated_blocks.
    assert merged.details["evaluated_blocks"] == [0, 1, 2]
    assert get_evaluated_blocks(_candidate("c"), merged) == {0, 1, 2}


def test_intensifier_stamps_evaluated_blocks_on_accepted_challenger():
    block_evaluator = _block_evaluator()
    budget = BudgetAllocator(max_budget=10_000.0)
    intensifier = Intensifier(
        block_evaluator=block_evaluator,
        budget_allocator=budget,
        config=IntensificationConfig(),
    )

    challenger = _candidate("challenger")

    decision = intensifier.intensify(challenger=challenger, incumbents=[])

    assert decision.accepted
    # The challenger object now knows which block(s) it was evaluated on, so a
    # later advance_incumbents / parent tournament no longer mistakes it for a
    # 0-block candidate.
    assert challenger.metadata["evaluated_blocks"] == decision.evaluated_blocks
    assert get_evaluated_blocks(challenger) == set(decision.evaluated_blocks)


# --------------------------------------------------------------------------
# 2. advance_front_one_level deepens the whole front by one aligned block.
# --------------------------------------------------------------------------


def _portfolio_for(candidates, block_evaluator, budget):
    """Seed-evaluate each candidate on block 0 and build the portfolio."""
    portfolio = PromptPortfolio()
    for candidate in candidates:
        evaluation = block_evaluator.evaluate_block(candidate, 0, use_cache=False)
        budget.record_block_evaluation(evaluation)
        candidate.metadata["evaluated_blocks"] = [0]
        portfolio.add(candidate, block_evaluator.aggregate_candidate(candidate.candidate_id, [0]))
    return portfolio


def test_advance_front_one_level_aligns_and_deepens_front():
    block_evaluator = _block_evaluator()
    budget = BudgetAllocator(max_budget=10_000.0)
    incumbents = [_candidate("a"), _candidate("b"), _candidate("c")]
    portfolio = _portfolio_for(incumbents, block_evaluator, budget)

    assert common_incumbent_blocks(incumbents, portfolio) == {0}

    decisions = advance_front_one_level(
        incumbents=incumbents,
        portfolio=portfolio,
        block_evaluator=block_evaluator,
        budget_allocator=budget,
        config=AdvanceIncumbentsConfig(random_seed=0, update_pareto_archive=False),
        rng=random.Random(0),
    )

    assert any(d.advanced for d in decisions)
    # Whole front is aligned and exactly one block deeper.
    assert incumbents_are_aligned(incumbents, portfolio)
    assert len(common_incumbent_blocks(incumbents, portfolio)) == 2
    for candidate in incumbents:
        assert len(get_candidate_blocks(candidate, portfolio)) == 2


def test_advance_front_one_level_respects_block_cap():
    block_evaluator = _block_evaluator()
    budget = BudgetAllocator(max_budget=10_000.0)
    incumbents = [_candidate("a"), _candidate("b")]
    portfolio = _portfolio_for(incumbents, block_evaluator, budget)

    # Cap at the current depth -> nothing should advance.
    decisions = advance_front_one_level(
        incumbents=incumbents,
        portfolio=portfolio,
        block_evaluator=block_evaluator,
        budget_allocator=budget,
        config=AdvanceIncumbentsConfig(random_seed=0, update_pareto_archive=False),
        rng=random.Random(0),
        max_blocks=1,
    )

    assert all(not d.advanced for d in decisions)
    assert common_incumbent_blocks(incumbents, portfolio) == {0}


def test_advance_front_one_level_stops_on_exhausted_budget():
    block_evaluator = _block_evaluator()
    budget = BudgetAllocator(max_budget=10_000.0)
    incumbents = [_candidate("a"), _candidate("b")]
    portfolio = _portfolio_for(incumbents, block_evaluator, budget)

    # Drain the budget so no further advancement can be charged.
    budget.record(candidate_id="drain", cost=budget.remaining_budget)
    assert budget.exhausted

    decisions = advance_front_one_level(
        incumbents=incumbents,
        portfolio=portfolio,
        block_evaluator=block_evaluator,
        budget_allocator=budget,
        config=AdvanceIncumbentsConfig(random_seed=0, update_pareto_archive=False),
        rng=random.Random(0),
    )

    assert decisions == []
    assert common_incumbent_blocks(incumbents, portfolio) == {0}


def test_advance_does_not_double_charge_cached_block():
    block_evaluator = _block_evaluator()
    budget = BudgetAllocator(max_budget=10_000.0)
    incumbents = [_candidate("a"), _candidate("b")]
    portfolio = _portfolio_for(incumbents, block_evaluator, budget)

    used_before = budget.used_budget
    records_before = budget.num_records()

    # Advance one full level, then try again with the front aligned: the helper
    # should add new blocks, never re-charge block 0 (a cache hit).
    advance_front_one_level(
        incumbents=incumbents,
        portfolio=portfolio,
        block_evaluator=block_evaluator,
        budget_allocator=budget,
        config=AdvanceIncumbentsConfig(random_seed=0, update_pareto_archive=False),
        rng=random.Random(0),
    )

    # No budget record should reference a (candidate, block) pair twice.
    seen = set()
    for record in budget.state.records:
        if record.block_id is None:
            continue
        key = (record.candidate_id, record.block_id)
        assert key not in seen, f"double-charged {key}"
        seen.add(key)

    assert budget.used_budget > used_before
    assert budget.num_records() > records_before
