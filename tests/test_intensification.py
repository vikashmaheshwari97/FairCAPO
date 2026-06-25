import pytest

from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
from heal_capo.objectives import ObjectiveEvaluator
from heal_capo.optimizers.block_evaluator import BlockEvaluator
from heal_capo.optimizers.budget_allocator import BudgetAllocator
from heal_capo.optimizers.intensification import (
    IntensificationConfig,
    Intensifier,
    choose_closest_incumbent,
    dominates_result,
)


class NamedEvaluator(ObjectiveEvaluator):
    """
    Deterministic evaluator controlled by candidate instruction text.
    """

    def evaluate(self, candidate, data):
        n = len(data)
        instruction = candidate.instruction.lower()

        if "strong" in instruction:
            performance = 0.9
            cost = 2.0 * n
            risk = 0.1
            fairness_risk = 0.1
        elif "weak" in instruction:
            performance = 0.4
            cost = 3.0 * n
            risk = 0.6
            fairness_risk = 0.5
        elif "cheap" in instruction:
            performance = 0.7
            cost = 0.5 * n
            risk = 0.2
            fairness_risk = 0.2
        else:
            performance = 0.6
            cost = 1.5 * n
            risk = 0.3
            fairness_risk = 0.3

        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            performance=performance,
            cost=cost,
            risk=risk,
            fairness_risk=fairness_risk,
            drift=0.0,
            n_examples=n,
            details={
                "input_tokens": 10 * n,
                "output_tokens": 2 * n,
            },
        )


def _data(n=6):
    return [{"text": f"x{i}", "label": "objective"} for i in range(n)]


def _make_block_evaluator():
    return BlockEvaluator.from_data(
        evaluator=NamedEvaluator(),
        data=_data(6),
        block_size=2,
    )


def test_dominates_result():
    strong = EvaluationResult(
        candidate_id="a",
        performance=0.9,
        cost=1.0,
        risk=0.1,
        fairness_risk=0.1,
    )
    weak = EvaluationResult(
        candidate_id="b",
        performance=0.5,
        cost=2.0,
        risk=0.5,
        fairness_risk=0.5,
    )

    assert dominates_result(strong, weak)
    assert not dominates_result(weak, strong)


def test_choose_closest_incumbent():
    challenger = EvaluationResult(
        candidate_id="c",
        performance=0.75,
        cost=2.0,
        risk=0.2,
        fairness_risk=0.2,
    )
    incumbents = {
        "far": EvaluationResult(
            candidate_id="far",
            performance=0.1,
            cost=10.0,
            risk=0.9,
            fairness_risk=0.9,
        ),
        "near": EvaluationResult(
            candidate_id="near",
            performance=0.76,
            cost=2.1,
            risk=0.21,
            fairness_risk=0.2,
        ),
    }

    assert choose_closest_incumbent(challenger, incumbents) == "near"


def test_intensify_accepts_when_no_incumbents():
    block_evaluator = _make_block_evaluator()
    budget = BudgetAllocator(max_budget=100.0)
    intensifier = Intensifier(block_evaluator, budget)

    challenger = PromptCandidate(instruction="strong prompt")
    portfolio = PromptPortfolio()

    decision = intensifier.intensify(
        challenger=challenger,
        incumbents=[],
        portfolio=portfolio,
    )

    assert decision.accepted
    assert not decision.rejected
    assert decision.evaluated_blocks == [0]
    assert challenger.candidate_id in portfolio.evaluations
    assert budget.used_budget > 0


def test_intensify_rejects_dominated_challenger():
    block_evaluator = _make_block_evaluator()
    budget = BudgetAllocator(max_budget=100.0)
    intensifier = Intensifier(block_evaluator, budget)

    incumbent = PromptCandidate(instruction="strong prompt")
    challenger = PromptCandidate(instruction="weak prompt")

    block_evaluator.evaluate_blocks(incumbent, [0, 1])
    budget.record_block_evaluation(
        block_evaluator.history.get(incumbent.candidate_id, 0)
    )
    budget.record_block_evaluation(
        block_evaluator.history.get(incumbent.candidate_id, 1)
    )

    decision = intensifier.intensify(
        challenger=challenger,
        incumbents=[incumbent],
    )

    assert decision.rejected
    assert not decision.accepted
    assert decision.compared_against == incumbent.candidate_id
    assert "dominates" in decision.reason


def test_intensify_accepts_non_dominated_challenger():
    block_evaluator = _make_block_evaluator()
    budget = BudgetAllocator(max_budget=100.0)
    intensifier = Intensifier(block_evaluator, budget)

    incumbent = PromptCandidate(instruction="strong prompt")
    challenger = PromptCandidate(instruction="cheap prompt")

    block_evaluator.evaluate_blocks(incumbent, [0, 1])
    budget.record_block_evaluation(
        block_evaluator.history.get(incumbent.candidate_id, 0)
    )
    budget.record_block_evaluation(
        block_evaluator.history.get(incumbent.candidate_id, 1)
    )

    portfolio = PromptPortfolio()

    decision = intensifier.intensify(
        challenger=challenger,
        incumbents=[incumbent],
        portfolio=portfolio,
    )

    assert decision.accepted
    assert not decision.rejected
    assert decision.evaluated_blocks == [0, 1]
    assert challenger.candidate_id in portfolio.evaluations


def test_intensify_respects_max_blocks_per_challenger():
    block_evaluator = _make_block_evaluator()
    budget = BudgetAllocator(max_budget=100.0)
    intensifier = Intensifier(
        block_evaluator,
        budget,
        config=IntensificationConfig(max_blocks_per_challenger=1),
    )

    incumbent = PromptCandidate(instruction="strong prompt")
    challenger = PromptCandidate(instruction="cheap prompt")

    block_evaluator.evaluate_blocks(incumbent, [0, 1, 2])
    for block_id in [0, 1, 2]:
        budget.record_block_evaluation(
            block_evaluator.history.get(incumbent.candidate_id, block_id)
        )

    decision = intensifier.intensify(
        challenger=challenger,
        incumbents=[incumbent],
    )

    assert decision.evaluated_blocks == [0]


def test_rejected_candidate_not_added_to_portfolio_by_default():
    block_evaluator = _make_block_evaluator()
    budget = BudgetAllocator(max_budget=100.0)
    intensifier = Intensifier(block_evaluator, budget)

    incumbent = PromptCandidate(instruction="strong prompt")
    challenger = PromptCandidate(instruction="weak prompt")

    block_evaluator.evaluate_blocks(incumbent, [0])
    budget.record_block_evaluation(
        block_evaluator.history.get(incumbent.candidate_id, 0)
    )

    portfolio = PromptPortfolio()

    decision = intensifier.intensify(
        challenger=challenger,
        incumbents=[incumbent],
        portfolio=portfolio,
    )

    assert decision.rejected
    assert challenger.candidate_id not in portfolio.evaluations


def test_rejected_candidate_can_be_added_to_population():
    block_evaluator = _make_block_evaluator()
    budget = BudgetAllocator(max_budget=100.0)
    intensifier = Intensifier(
        block_evaluator,
        budget,
        config=IntensificationConfig(add_rejected_to_population=True),
    )

    incumbent = PromptCandidate(instruction="strong prompt")
    challenger = PromptCandidate(instruction="weak prompt")

    block_evaluator.evaluate_blocks(incumbent, [0])
    budget.record_block_evaluation(
        block_evaluator.history.get(incumbent.candidate_id, 0)
    )

    portfolio = PromptPortfolio()

    decision = intensifier.intensify(
        challenger=challenger,
        incumbents=[incumbent],
        portfolio=portfolio,
    )

    assert decision.rejected
    assert challenger.candidate_id in portfolio.evaluations


def test_budget_exhaustion_raises_runtime_error():
    block_evaluator = _make_block_evaluator()
    budget = BudgetAllocator(max_budget=1.0)
    intensifier = Intensifier(block_evaluator, budget)

    challenger = PromptCandidate(instruction="strong prompt")

    with pytest.raises(RuntimeError):
        intensifier.intensify(
            challenger=challenger,
            incumbents=[],
        )


def test_intensifier_does_not_double_record_cached_budget():
    block_evaluator = _make_block_evaluator()
    budget = BudgetAllocator(max_budget=100.0)
    intensifier = Intensifier(block_evaluator, budget)

    challenger = PromptCandidate(instruction="strong prompt")

    first = intensifier.intensify(challenger=challenger, incumbents=[])
    used_after_first = budget.used_budget

    second = intensifier.intensify(challenger=challenger, incumbents=[])
    used_after_second = budget.used_budget

    assert first.accepted
    assert second.accepted
    assert used_after_first == used_after_second