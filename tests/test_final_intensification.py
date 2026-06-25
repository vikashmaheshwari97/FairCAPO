"""
Regression test for the final deepening pass in the budgeted MO-CAPO runner.

Background: the evolutionary loop can accept a challenger on a SINGLE block in the
last iteration (no later advance pass deepens it), and the final reported Pareto
front is plain Pareto dominance over all candidates regardless of block depth. A
lucky 1-block estimate (e.g. 10/10 -> perf 1.0 on ~10 examples) therefore
dominates honestly-deepened candidates and lands on the front -- the "1-block
optimism" caveat that made search perf 1.0 collapse to ~0.80 on held-out data.

The fix races the final non-dominated front up to the full block depth before
recomputing the front, so no reported member rests on a 1-block estimate. This
test pins that behaviour against a toy block evaluator whose block 0 is
artificially easy (perf 1.0) while deeper blocks are hard (perf 0.0).
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
)
from heal_capo.optimizers.block_evaluator import BlockEvaluator
from heal_capo.optimizers.budget_allocator import BudgetAllocator
from heal_capo.pareto import non_dominated_ids
from scripts.run_phase2_budgeted_mocapo import front_candidates_from_portfolio


class BlockBiasedEvaluator(ObjectiveEvaluator):
    """Block 0 is trivially easy (perf 1.0); every later block is hard (perf 0.0).

    Examples are tagged ``x{i}``; the first ``block_size`` indices form block 0.
    A candidate scored only on block 0 looks perfect; scored on the full set its
    honest performance collapses toward ``block_size / total_examples``.
    """

    def __init__(self, block_size: int):
        self.block_size = block_size

    def evaluate(self, candidate, data):
        n = len(data)
        correct = sum(1 for row in data if int(row["text"][1:]) < self.block_size)

        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            performance=correct / n if n else 0.0,
            cost=float(n),
            risk=0.2,
            fairness_risk=0.1,
            drift=0.0,
            n_examples=n,
            details={"input_tokens": 5 * n, "output_tokens": n},
        )


def _block_evaluator(num_blocks=6, block_size=2):
    data = [
        {"text": f"x{i}", "label": "objective"}
        for i in range(num_blocks * block_size)
    ]
    return BlockEvaluator.from_data(
        evaluator=BlockBiasedEvaluator(block_size=block_size),
        data=data,
        block_size=block_size,
    )


def _candidate(name: str) -> PromptCandidate:
    candidate = PromptCandidate(instruction=f"Prompt {name}", metadata={"method": name})
    candidate.candidate_id = name
    return candidate


def _evaluate_on(candidate, blocks, block_evaluator, budget, portfolio):
    for block_id in blocks:
        evaluation = block_evaluator.evaluate_block(candidate, block_id, use_cache=False)
        budget.record_block_evaluation(evaluation)
    candidate.metadata["evaluated_blocks"] = sorted(blocks)
    portfolio.add(
        candidate,
        block_evaluator.aggregate_candidate(candidate.candidate_id, sorted(blocks)),
    )


def _run_final_pass(portfolio, block_evaluator, budget, max_blocks):
    """Replicates the runner's inline final-deepening loop over public helpers."""
    previous_depth = -1
    for _ in range(max_blocks + 2):
        if budget.exhausted:
            break
        front = front_candidates_from_portfolio(portfolio)
        if not front:
            break
        depth = len(common_incumbent_blocks(front, portfolio))
        if depth >= max_blocks or depth == previous_depth:
            break
        previous_depth = depth
        decisions = advance_front_one_level(
            incumbents=front,
            portfolio=portfolio,
            block_evaluator=block_evaluator,
            budget_allocator=budget,
            config=AdvanceIncumbentsConfig(random_seed=0, update_pareto_archive=False),
            rng=random.Random(0),
            max_blocks=max_blocks,
            refresh_incumbents=lambda: front_candidates_from_portfolio(portfolio),
        )
        if not any(getattr(d, "advanced", False) for d in decisions):
            break


def test_front_candidates_includes_one_block_fluke_before_deepening():
    block_evaluator = _block_evaluator()
    budget = BudgetAllocator(max_budget=10_000.0)
    portfolio = PromptPortfolio()

    fluke = _candidate("fluke")
    honest = _candidate("honest")
    _evaluate_on(fluke, [0], block_evaluator, budget, portfolio)
    _evaluate_on(honest, [0, 1, 2, 3, 4, 5], block_evaluator, budget, portfolio)

    # The cheap 1-block fluke (perf 1.0, cost 2) dominates the honest 6-block
    # candidate (perf ~0.17, cost 12), so it is the (degenerate) reported front.
    front_ids = set(non_dominated_ids(portfolio.evaluations.values()))
    assert "fluke" in front_ids
    assert portfolio.evaluations["fluke"].performance == 1.0
    assert len(get_candidate_blocks(fluke, portfolio)) == 1


def test_final_pass_deepens_fluke_and_front_has_no_single_block_member():
    block_evaluator = _block_evaluator(num_blocks=6, block_size=2)
    budget = BudgetAllocator(max_budget=10_000.0)
    portfolio = PromptPortfolio()

    fluke = _candidate("fluke")
    honest = _candidate("honest")
    _evaluate_on(fluke, [0], block_evaluator, budget, portfolio)
    _evaluate_on(honest, [0, 1, 2, 3, 4, 5], block_evaluator, budget, portfolio)

    _run_final_pass(portfolio, block_evaluator, budget, max_blocks=6)

    # The fluke is now raced on the full block set -> honest score, not 10/10.
    assert len(get_candidate_blocks(fluke, portfolio)) == 6
    assert portfolio.evaluations["fluke"].performance < 0.5

    # The recomputed front no longer rests on any single-block estimate.
    front_ids = set(non_dominated_ids(portfolio.evaluations.values()))
    assert front_ids
    for candidate_id in front_ids:
        candidate = portfolio.get(candidate_id)
        assert len(get_candidate_blocks(candidate, portfolio)) > 1


def test_final_pass_stops_when_budget_exhausted():
    block_evaluator = _block_evaluator()
    budget = BudgetAllocator(max_budget=10_000.0)
    portfolio = PromptPortfolio()

    fluke = _candidate("fluke")
    _evaluate_on(fluke, [0], block_evaluator, budget, portfolio)

    # Drain the budget: the final pass must not raise and must not deepen.
    budget.record(candidate_id="drain", cost=budget.remaining_budget)
    assert budget.exhausted

    _run_final_pass(portfolio, block_evaluator, budget, max_blocks=6)

    assert len(get_candidate_blocks(fluke, portfolio)) == 1
