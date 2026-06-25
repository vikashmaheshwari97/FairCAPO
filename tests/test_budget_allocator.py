import pytest

from heal_capo.core import EvaluationResult
from heal_capo.optimizers.block_evaluator import BlockEvaluation
from heal_capo.optimizers.budget_allocator import BudgetAllocator


def _block_eval(
    candidate_id="p1",
    block_id=0,
    cost=5.0,
    input_tokens=10,
    output_tokens=3,
):
    return BlockEvaluation(
        candidate_id=candidate_id,
        block_id=block_id,
        result=EvaluationResult(
            candidate_id=candidate_id,
            performance=0.8,
            cost=cost,
            risk=0.2,
            fairness_risk=0.1,
            drift=0.0,
            n_examples=2,
            details={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        ),
    )


def test_budget_allocator_initial_state():
    allocator = BudgetAllocator(max_budget=10.0)

    assert allocator.max_budget == 10.0
    assert allocator.used_budget == 0.0
    assert allocator.remaining_budget == 10.0
    assert allocator.utilization == 0.0
    assert not allocator.exhausted


def test_budget_allocator_rejects_non_positive_budget():
    with pytest.raises(ValueError):
        BudgetAllocator(max_budget=0.0)


def test_can_spend():
    allocator = BudgetAllocator(max_budget=10.0)

    assert allocator.can_spend(5.0)
    assert not allocator.can_spend(11.0)


def test_record_updates_budget():
    allocator = BudgetAllocator(max_budget=10.0)

    record = allocator.record(
        candidate_id="p1",
        block_id=0,
        cost=4.0,
        input_tokens=10,
        output_tokens=2,
    )

    assert record.candidate_id == "p1"
    assert allocator.used_budget == 4.0
    assert allocator.remaining_budget == 6.0
    assert allocator.num_records() == 1


def test_record_rejects_overspend_by_default():
    allocator = BudgetAllocator(max_budget=5.0)

    allocator.record(candidate_id="p1", block_id=0, cost=4.0)

    with pytest.raises(RuntimeError):
        allocator.record(candidate_id="p2", block_id=0, cost=2.0)


def test_record_allows_overspend_when_configured():
    allocator = BudgetAllocator(max_budget=5.0, allow_overspend=True)

    allocator.record(candidate_id="p1", block_id=0, cost=4.0)
    allocator.record(candidate_id="p2", block_id=0, cost=2.0)

    assert allocator.used_budget == 6.0
    assert allocator.remaining_budget == 0.0
    assert allocator.exhausted


def test_record_rejects_negative_cost():
    allocator = BudgetAllocator(max_budget=5.0)

    with pytest.raises(ValueError):
        allocator.record(candidate_id="p1", cost=-1.0)


def test_record_block_evaluation():
    allocator = BudgetAllocator(max_budget=20.0)
    evaluation = _block_eval()

    record = allocator.record_block_evaluation(evaluation)

    assert record.candidate_id == "p1"
    assert record.block_id == 0
    assert record.cost == 5.0
    assert record.input_tokens == 10
    assert record.output_tokens == 3


def test_candidate_cost_and_tokens():
    allocator = BudgetAllocator(max_budget=30.0)

    allocator.record_block_evaluation(_block_eval(candidate_id="p1", block_id=0, cost=5.0))
    allocator.record_block_evaluation(_block_eval(candidate_id="p1", block_id=1, cost=4.0))
    allocator.record_block_evaluation(_block_eval(candidate_id="p2", block_id=0, cost=3.0))

    assert allocator.candidate_cost("p1") == 9.0
    assert allocator.candidate_cost("p2") == 3.0

    tokens = allocator.candidate_tokens("p1")

    assert tokens["input_tokens"] == 20
    assert tokens["output_tokens"] == 6
    assert tokens["total_tokens"] == 26


def test_block_cost_and_candidate_block_costs():
    allocator = BudgetAllocator(max_budget=30.0)

    allocator.record_block_evaluation(_block_eval(candidate_id="p1", block_id=0, cost=5.0))
    allocator.record_block_evaluation(_block_eval(candidate_id="p1", block_id=1, cost=4.0))
    allocator.record_block_evaluation(_block_eval(candidate_id="p2", block_id=0, cost=3.0))

    assert allocator.block_cost(0) == 8.0
    assert allocator.block_cost(1) == 4.0
    assert allocator.candidate_block_costs("p1") == {0: 5.0, 1: 4.0}


def test_summary():
    allocator = BudgetAllocator(max_budget=30.0)

    allocator.record_block_evaluation(_block_eval(candidate_id="p1", block_id=0, cost=5.0))
    allocator.record_block_evaluation(_block_eval(candidate_id="p2", block_id=1, cost=3.0))

    summary = allocator.summary()

    assert summary["max_budget"] == 30.0
    assert summary["used_budget"] == 8.0
    assert summary["remaining_budget"] == 22.0
    assert summary["num_records"] == 2
    assert summary["num_candidates"] == 2
    assert summary["num_blocks"] == 2
    assert summary["candidate_costs"]["p1"] == 5.0
    assert summary["block_costs"][1] == 3.0


def test_to_rows():
    allocator = BudgetAllocator(max_budget=20.0)
    allocator.record_block_evaluation(_block_eval())

    rows = allocator.to_rows()

    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "p1"
    assert rows[0]["block_id"] == 0
    assert rows[0]["metadata_performance"] == 0.8


def test_reset():
    allocator = BudgetAllocator(max_budget=20.0)

    allocator.record_block_evaluation(_block_eval())
    allocator.reset()

    assert allocator.used_budget == 0.0
    assert allocator.remaining_budget == 20.0
    assert allocator.num_records() == 0


# --- token-denominated budget mode (MO-CAPO 7.5M raw-token budget) ----------


def test_budget_unit_defaults_to_cost():
    allocator = BudgetAllocator(max_budget=10.0)
    assert allocator.budget_unit == "cost"
    assert allocator.summary()["budget_unit"] == "cost"


def test_budget_unit_rejects_invalid():
    with pytest.raises(ValueError):
        BudgetAllocator(max_budget=10.0, budget_unit="dollars")


def test_tokens_mode_charges_raw_tokens_not_cost():
    # In tokens mode the budget is metered by input+output tokens, while the
    # stored cost (the cost OBJECTIVE) stays the weighted value.
    allocator = BudgetAllocator(max_budget=100.0, budget_unit="tokens")

    record = allocator.record(
        candidate_id="p1",
        block_id=0,
        cost=4.0,            # weighted cost — must NOT be what's charged
        input_tokens=10,
        output_tokens=2,
    )

    assert record.cost == 4.0                 # cost objective preserved
    assert allocator.used_budget == 12.0      # charged 10 + 2 tokens
    assert allocator.remaining_budget == 88.0


def test_tokens_mode_exhaustion_uses_tokens():
    allocator = BudgetAllocator(max_budget=15.0, budget_unit="tokens")
    allocator.record_block_evaluation(
        _block_eval(cost=999.0, input_tokens=10, output_tokens=3)
    )
    assert allocator.used_budget == 13.0      # tokens, not the 999 cost
    with pytest.raises(RuntimeError):
        allocator.record_block_evaluation(
            _block_eval(cost=0.0, input_tokens=5, output_tokens=0)
        )


def test_tokens_mode_summary_used_budget_equals_total_tokens():
    allocator = BudgetAllocator(max_budget=1000.0, budget_unit="tokens")
    allocator.record_block_evaluation(
        _block_eval(candidate_id="p1", block_id=0, cost=5.0, input_tokens=10, output_tokens=3)
    )
    allocator.record_block_evaluation(
        _block_eval(candidate_id="p2", block_id=1, cost=7.0, input_tokens=20, output_tokens=4)
    )

    summary = allocator.summary()
    assert summary["budget_unit"] == "tokens"
    # used_budget (tokens) must equal the reported total token count.
    assert summary["used_budget"] == summary["total_tokens"] == 37.0
    # The cost objective is still the weighted cost, independent of metering.
    assert summary["candidate_costs"]["p1"] == 5.0