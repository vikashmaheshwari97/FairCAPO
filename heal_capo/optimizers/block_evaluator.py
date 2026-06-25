from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from heal_capo.core import EvaluationResult, PromptCandidate
from heal_capo.objectives import ObjectiveEvaluator


@dataclass(frozen=True)
class DataBlock:
    """
    A block of development examples used for progressive evaluation.
    """

    block_id: int
    examples: list[dict]

    @property
    def size(self) -> int:
        return len(self.examples)


@dataclass
class BlockEvaluation:
    """
    Evaluation of one prompt candidate on one data block.
    """

    candidate_id: str
    block_id: int
    result: EvaluationResult

    @property
    def objective_vector(self) -> tuple:
        return self.result.objective_vector


@dataclass
class EvaluationHistory:
    """
    Stores block-level evaluations.

    Key:
      candidate_id -> block_id -> BlockEvaluation
    """

    evaluations: Dict[str, Dict[int, BlockEvaluation]] = field(default_factory=dict)

    def add(self, evaluation: BlockEvaluation) -> None:
        self.evaluations.setdefault(evaluation.candidate_id, {})
        self.evaluations[evaluation.candidate_id][evaluation.block_id] = evaluation

    def has(self, candidate_id: str, block_id: int) -> bool:
        return candidate_id in self.evaluations and block_id in self.evaluations[candidate_id]

    def get(self, candidate_id: str, block_id: int) -> BlockEvaluation:
        return self.evaluations[candidate_id][block_id]

    def get_candidate_blocks(self, candidate_id: str) -> list[int]:
        if candidate_id not in self.evaluations:
            return []

        return sorted(self.evaluations[candidate_id].keys())

    def get_candidate_evaluations(self, candidate_id: str) -> list[BlockEvaluation]:
        block_ids = self.get_candidate_blocks(candidate_id)
        return [self.evaluations[candidate_id][block_id] for block_id in block_ids]

    def num_blocks(self, candidate_id: str) -> int:
        return len(self.get_candidate_blocks(candidate_id))

    def total_cost(self, candidate_id: Optional[str] = None) -> float:
        if candidate_id is not None:
            return sum(
                evaluation.result.cost
                for evaluation in self.get_candidate_evaluations(candidate_id)
            )

        total = 0.0
        for candidate_evals in self.evaluations.values():
            for evaluation in candidate_evals.values():
                total += evaluation.result.cost

        return total

    def candidates(self) -> list[str]:
        return sorted(self.evaluations.keys())

    def common_blocks(self, candidate_ids: Sequence[str]) -> list[int]:
        if not candidate_ids:
            return []

        block_sets = []

        for candidate_id in candidate_ids:
            block_sets.append(set(self.get_candidate_blocks(candidate_id)))

        if not block_sets:
            return []

        common = set.intersection(*block_sets)

        return sorted(common)


def make_blocks(
    data: Sequence[dict],
    block_size: int,
    drop_last: bool = False,
) -> list[DataBlock]:
    """
    Split data into ordered blocks.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    blocks = []

    for start in range(0, len(data), block_size):
        examples = list(data[start : start + block_size])

        if drop_last and len(examples) < block_size:
            continue

        blocks.append(
            DataBlock(
                block_id=len(blocks),
                examples=examples,
            )
        )

    return blocks


def merge_results(
    candidate_id: str,
    evaluations: Sequence[BlockEvaluation],
) -> EvaluationResult:
    """
    Merge block-level evaluations into one aggregate result.

    Performance/risk/fairness/drift are weighted by number of examples.
    Cost is summed.
    """
    if not evaluations:
        raise ValueError("Cannot merge empty evaluations.")

    total_examples = sum(max(0, evaluation.result.n_examples) for evaluation in evaluations)

    if total_examples <= 0:
        total_examples = len(evaluations)
        weights = [1.0 for _ in evaluations]
    else:
        weights = [max(0, evaluation.result.n_examples) for evaluation in evaluations]

    def weighted_average(field_name: str) -> float:
        total = 0.0

        for evaluation, weight in zip(evaluations, weights):
            total += getattr(evaluation.result, field_name) * weight

        return total / sum(weights)

    block_ids = sorted(evaluation.block_id for evaluation in evaluations)
    merged_details = {
        "merged_from_blocks": [evaluation.block_id for evaluation in evaluations],
        # Canonical key read by parent_selection.get_evaluated_blocks /
        # advance_incumbents: makes every aggregate result self-describing so a
        # candidate's real evaluated-block set is recoverable even when nobody
        # stamped it onto candidate.metadata. Without this, intensification-
        # accepted challengers look like they have 0 evaluated blocks, which
        # misleads incumbent advancement and block-aware tournaments.
        "evaluated_blocks": block_ids,
        "num_block_evaluations": len(evaluations),
    }

    return EvaluationResult(
        candidate_id=candidate_id,
        performance=weighted_average("performance"),
        cost=sum(evaluation.result.cost for evaluation in evaluations),
        risk=weighted_average("risk"),
        fairness_risk=weighted_average("fairness_risk"),
        drift=weighted_average("drift"),
        n_examples=int(total_examples),
        details=merged_details,
    )


class BlockEvaluator:
    """
    Evaluates PromptCandidate objects on data blocks with caching.

    This is the first building block for MO-CAPO-style intensification:
      - evaluate on one block first
      - reuse cached evaluations
      - aggregate over evaluated blocks
    """

    def __init__(
        self,
        evaluator: ObjectiveEvaluator,
        blocks: Sequence[DataBlock],
        history: Optional[EvaluationHistory] = None,
    ):
        if not blocks:
            raise ValueError("BlockEvaluator requires at least one data block.")

        self.evaluator = evaluator
        self.blocks = list(blocks)
        self.history = history or EvaluationHistory()
        self._block_map = {block.block_id: block for block in self.blocks}

    @classmethod
    def from_data(
        cls,
        evaluator: ObjectiveEvaluator,
        data: Sequence[dict],
        block_size: int,
        drop_last: bool = False,
        history: Optional[EvaluationHistory] = None,
    ) -> "BlockEvaluator":
        blocks = make_blocks(
            data=data,
            block_size=block_size,
            drop_last=drop_last,
        )

        return cls(
            evaluator=evaluator,
            blocks=blocks,
            history=history,
        )

    def block_ids(self) -> list[int]:
        return sorted(self._block_map.keys())

    def get_block(self, block_id: int) -> DataBlock:
        if block_id not in self._block_map:
            raise KeyError(f"Unknown block_id: {block_id}")

        return self._block_map[block_id]

    def evaluate_block(
        self,
        candidate: PromptCandidate,
        block_id: int,
        use_cache: bool = True,
    ) -> BlockEvaluation:
        """
        Evaluate one candidate on one block.
        """
        if use_cache and self.history.has(candidate.candidate_id, block_id):
            return self.history.get(candidate.candidate_id, block_id)

        block = self.get_block(block_id)

        result = self.evaluator.evaluate(
            candidate=candidate,
            data=block.examples,
        )

        # Ensure candidate id and n_examples are aligned with the block.
        result.candidate_id = candidate.candidate_id
        if result.n_examples <= 0:
            result.n_examples = block.size

        evaluation = BlockEvaluation(
            candidate_id=candidate.candidate_id,
            block_id=block_id,
            result=result,
        )

        self.history.add(evaluation)

        return evaluation

    def evaluate_blocks(
        self,
        candidate: PromptCandidate,
        block_ids: Iterable[int],
        use_cache: bool = True,
    ) -> list[BlockEvaluation]:
        """
        Evaluate one candidate on several blocks.
        """
        evaluations = []

        for block_id in block_ids:
            evaluations.append(
                self.evaluate_block(
                    candidate=candidate,
                    block_id=block_id,
                    use_cache=use_cache,
                )
            )

        return evaluations

    def evaluate_all_blocks(
        self,
        candidate: PromptCandidate,
        use_cache: bool = True,
    ) -> list[BlockEvaluation]:
        return self.evaluate_blocks(
            candidate=candidate,
            block_ids=self.block_ids(),
            use_cache=use_cache,
        )

    def aggregate_candidate(
        self,
        candidate_id: str,
        block_ids: Optional[Sequence[int]] = None,
    ) -> EvaluationResult:
        """
        Aggregate cached evaluations for a candidate.
        """
        if block_ids is None:
            evaluations = self.history.get_candidate_evaluations(candidate_id)
        else:
            evaluations = [
                self.history.get(candidate_id, block_id)
                for block_id in block_ids
                if self.history.has(candidate_id, block_id)
            ]

        return merge_results(
            candidate_id=candidate_id,
            evaluations=evaluations,
        )

    def evaluated_blocks(self, candidate_id: str) -> list[int]:
        return self.history.get_candidate_blocks(candidate_id)

    def unevaluated_blocks(self, candidate_id: str) -> list[int]:
        evaluated = set(self.evaluated_blocks(candidate_id))

        return [
            block_id
            for block_id in self.block_ids()
            if block_id not in evaluated
        ]

    def total_cost(self) -> float:
        return self.history.total_cost()