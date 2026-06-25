from heal_capo.core import EvaluationResult, PromptCandidate
from heal_capo.optimizers.block_evaluator import (
    BlockEvaluator,
    EvaluationHistory,
    make_blocks,
    merge_results,
)
from heal_capo.objectives import ObjectiveEvaluator


class DummyEvaluator(ObjectiveEvaluator):
    def __init__(self):
        self.calls = 0

    def evaluate(self, candidate, data):
        self.calls += 1
        n = len(data)

        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            performance=0.5 + 0.1 * n,
            cost=2.0 * n,
            risk=max(0.0, 1.0 - 0.1 * n),
            fairness_risk=0.2,
            drift=0.1,
            n_examples=n,
            details={"calls": self.calls},
        )


def _data(n=5):
    return [{"text": f"x{i}", "label": "objective"} for i in range(n)]


def test_make_blocks_even_split():
    blocks = make_blocks(_data(4), block_size=2)

    assert len(blocks) == 2
    assert blocks[0].block_id == 0
    assert blocks[0].size == 2
    assert blocks[1].block_id == 1
    assert blocks[1].size == 2


def test_make_blocks_keeps_last_by_default():
    blocks = make_blocks(_data(5), block_size=2)

    assert len(blocks) == 3
    assert blocks[-1].size == 1


def test_make_blocks_can_drop_last():
    blocks = make_blocks(_data(5), block_size=2, drop_last=True)

    assert len(blocks) == 2
    assert all(block.size == 2 for block in blocks)


def test_make_blocks_rejects_bad_block_size():
    try:
        make_blocks(_data(5), block_size=0)
        assert False
    except ValueError:
        assert True


def test_block_evaluator_evaluates_and_caches():
    evaluator = DummyEvaluator()
    block_evaluator = BlockEvaluator.from_data(
        evaluator=evaluator,
        data=_data(4),
        block_size=2,
    )
    candidate = PromptCandidate(instruction="Classify.")

    first = block_evaluator.evaluate_block(candidate, block_id=0)
    second = block_evaluator.evaluate_block(candidate, block_id=0)

    assert first is second
    assert evaluator.calls == 1
    assert first.result.n_examples == 2


def test_block_evaluator_can_disable_cache():
    evaluator = DummyEvaluator()
    block_evaluator = BlockEvaluator.from_data(
        evaluator=evaluator,
        data=_data(4),
        block_size=2,
    )
    candidate = PromptCandidate(instruction="Classify.")

    block_evaluator.evaluate_block(candidate, block_id=0)
    block_evaluator.evaluate_block(candidate, block_id=0, use_cache=False)

    assert evaluator.calls == 2


def test_evaluate_blocks_and_history():
    evaluator = DummyEvaluator()
    history = EvaluationHistory()
    block_evaluator = BlockEvaluator.from_data(
        evaluator=evaluator,
        data=_data(5),
        block_size=2,
        history=history,
    )
    candidate = PromptCandidate(instruction="Classify.")

    evaluations = block_evaluator.evaluate_blocks(candidate, [0, 1])

    assert len(evaluations) == 2
    assert history.num_blocks(candidate.candidate_id) == 2
    assert block_evaluator.evaluated_blocks(candidate.candidate_id) == [0, 1]
    assert block_evaluator.unevaluated_blocks(candidate.candidate_id) == [2]


def test_merge_results_weighted_average_and_cost_sum():
    candidate = PromptCandidate(instruction="Classify.")
    evaluator = DummyEvaluator()
    block_evaluator = BlockEvaluator.from_data(
        evaluator=evaluator,
        data=_data(3),
        block_size=2,
    )

    evaluations = block_evaluator.evaluate_all_blocks(candidate)
    merged = merge_results(candidate.candidate_id, evaluations)

    assert merged.candidate_id == candidate.candidate_id
    assert merged.cost == 6.0
    assert merged.n_examples == 3
    assert merged.details["num_block_evaluations"] == 2
    assert merged.details["merged_from_blocks"] == [0, 1]


def test_aggregate_candidate():
    evaluator = DummyEvaluator()
    block_evaluator = BlockEvaluator.from_data(
        evaluator=evaluator,
        data=_data(4),
        block_size=2,
    )
    candidate = PromptCandidate(instruction="Classify.")

    block_evaluator.evaluate_blocks(candidate, [0, 1])
    aggregate = block_evaluator.aggregate_candidate(candidate.candidate_id)

    assert aggregate.cost == 8.0
    assert aggregate.n_examples == 4


def test_history_common_blocks():
    evaluator = DummyEvaluator()
    block_evaluator = BlockEvaluator.from_data(
        evaluator=evaluator,
        data=_data(6),
        block_size=2,
    )

    c1 = PromptCandidate(instruction="A")
    c2 = PromptCandidate(instruction="B")

    block_evaluator.evaluate_blocks(c1, [0, 1])
    block_evaluator.evaluate_blocks(c2, [1, 2])

    common = block_evaluator.history.common_blocks(
        [c1.candidate_id, c2.candidate_id]
    )

    assert common == [1]