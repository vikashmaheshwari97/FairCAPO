from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
from heal_capo.optimizers.block_evaluator import (
    BlockEvaluation,
    BlockEvaluator,
)
from heal_capo.optimizers.budget_allocator import BudgetAllocator
from heal_capo.pareto import dominates


@dataclass
class IntensificationConfig:
    """
    Controls MO-CAPO-style challenger intensification.
    """

    max_blocks_per_challenger: Optional[int] = None
    reject_when_dominated: bool = True
    accept_when_not_dominated_on_common_blocks: bool = True
    add_rejected_to_population: bool = False
    use_cache: bool = True


@dataclass
class IntensificationDecision:
    """
    Result of intensifying one challenger.
    """

    candidate_id: str
    accepted: bool
    rejected: bool
    reason: str
    evaluated_blocks: list[int] = field(default_factory=list)
    compared_against: Optional[str] = None
    aggregate_result: Optional[EvaluationResult] = None
    budget_used: float = 0.0
    metadata: dict = field(default_factory=dict)


def objective_vector(result: EvaluationResult) -> tuple:
    return result.objective_vector


def dominates_result(
    left: EvaluationResult,
    right: EvaluationResult,
) -> bool:
    return dominates(left, right)


def choose_closest_incumbent(
    challenger_result: EvaluationResult,
    incumbent_results: dict[str, EvaluationResult],
) -> Optional[str]:
    """
    Choose closest incumbent in objective-vector space.

    Objective vectors are used directly. Later we can add min-max normalization.
    """
    if not incumbent_results:
        return None

    challenger_vector = challenger_result.objective_vector

    def distance(candidate_id: str) -> float:
        incumbent_vector = incumbent_results[candidate_id].objective_vector

        return sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(challenger_vector, incumbent_vector)
        )

    return min(incumbent_results.keys(), key=distance)


class Intensifier:
    """
    MO-CAPO-style progressive challenger evaluator.

    Basic behavior:
      1. Find common incumbent blocks.
      2. Evaluate challenger block by block.
      3. After each block, aggregate challenger result.
      4. Compare against closest incumbent on same evaluated blocks.
      5. Reject early if dominated.
      6. Accept if challenger survives required/common blocks.
    """

    def __init__(
        self,
        block_evaluator: BlockEvaluator,
        budget_allocator: BudgetAllocator,
        config: Optional[IntensificationConfig] = None,
    ):
        self.block_evaluator = block_evaluator
        self.budget_allocator = budget_allocator
        self.config = config or IntensificationConfig()

    def intensify(
        self,
        challenger: PromptCandidate,
        incumbents: Sequence[PromptCandidate],
        portfolio: Optional[PromptPortfolio] = None,
    ) -> IntensificationDecision:
        if not incumbents:
            return self._evaluate_without_incumbents(
                challenger=challenger,
                portfolio=portfolio,
            )

        common_blocks = self._common_incumbent_blocks(incumbents)

        if not common_blocks:
            common_blocks = [self.block_evaluator.block_ids()[0]]

        if self.config.max_blocks_per_challenger is not None:
            common_blocks = common_blocks[: self.config.max_blocks_per_challenger]

        evaluated_blocks: list[int] = []
        block_evaluations: list[BlockEvaluation] = []
        last_challenger_result: Optional[EvaluationResult] = None
        compared_against: Optional[str] = None

        for block_id in common_blocks:
            evaluation = self.block_evaluator.evaluate_block(
                candidate=challenger,
                block_id=block_id,
                use_cache=self.config.use_cache,
            )

            if not self._budget_recorded(challenger.candidate_id, block_id):
                self.budget_allocator.record_block_evaluation(evaluation)

            block_evaluations.append(evaluation)
            evaluated_blocks.append(block_id)

            challenger_result = self.block_evaluator.aggregate_candidate(
                challenger.candidate_id,
                block_ids=evaluated_blocks,
            )
            last_challenger_result = challenger_result

            incumbent_results = self._aggregate_incumbents_on_blocks(
                incumbents=incumbents,
                block_ids=evaluated_blocks,
            )

            closest_incumbent_id = choose_closest_incumbent(
                challenger_result=challenger_result,
                incumbent_results=incumbent_results,
            )
            compared_against = closest_incumbent_id

            if (
                closest_incumbent_id is not None
                and self.config.reject_when_dominated
                and dominates_result(
                    incumbent_results[closest_incumbent_id],
                    challenger_result,
                )
            ):
                return self._make_decision(
                    challenger=challenger,
                    accepted=False,
                    rejected=True,
                    reason="Rejected because closest incumbent dominates challenger.",
                    evaluated_blocks=evaluated_blocks,
                    compared_against=closest_incumbent_id,
                    aggregate_result=challenger_result,
                    portfolio=portfolio,
                )

        if last_challenger_result is None:
            return self._make_decision(
                challenger=challenger,
                accepted=False,
                rejected=True,
                reason="No challenger evaluations were completed.",
                evaluated_blocks=evaluated_blocks,
                compared_against=compared_against,
                aggregate_result=None,
                portfolio=portfolio,
            )

        return self._make_decision(
            challenger=challenger,
            accepted=True,
            rejected=False,
            reason="Accepted after surviving intensification.",
            evaluated_blocks=evaluated_blocks,
            compared_against=compared_against,
            aggregate_result=last_challenger_result,
            portfolio=portfolio,
        )

    def _evaluate_without_incumbents(
        self,
        challenger: PromptCandidate,
        portfolio: Optional[PromptPortfolio] = None,
    ) -> IntensificationDecision:
        block_id = self.block_evaluator.block_ids()[0]

        evaluation = self.block_evaluator.evaluate_block(
            candidate=challenger,
            block_id=block_id,
            use_cache=self.config.use_cache,
        )

        if not self._budget_recorded(challenger.candidate_id, block_id):
            self.budget_allocator.record_block_evaluation(evaluation)

        aggregate = self.block_evaluator.aggregate_candidate(
            challenger.candidate_id,
            block_ids=[block_id],
        )

        return self._make_decision(
            challenger=challenger,
            accepted=True,
            rejected=False,
            reason="Accepted because no incumbents exist.",
            evaluated_blocks=[block_id],
            compared_against=None,
            aggregate_result=aggregate,
            portfolio=portfolio,
        )

    def _common_incumbent_blocks(
        self,
        incumbents: Sequence[PromptCandidate],
    ) -> list[int]:
        return self.block_evaluator.history.common_blocks(
            [candidate.candidate_id for candidate in incumbents]
        )

    def _aggregate_incumbents_on_blocks(
        self,
        incumbents: Sequence[PromptCandidate],
        block_ids: Sequence[int],
    ) -> dict[str, EvaluationResult]:
        results = {}

        for incumbent in incumbents:
            if all(
                self.block_evaluator.history.has(incumbent.candidate_id, block_id)
                for block_id in block_ids
            ):
                results[incumbent.candidate_id] = self.block_evaluator.aggregate_candidate(
                    incumbent.candidate_id,
                    block_ids=block_ids,
                )

        return results

    def _budget_recorded(
        self,
        candidate_id: str,
        block_id: int,
    ) -> bool:
        for record in self.budget_allocator.state.records:
            if record.candidate_id == candidate_id and record.block_id == block_id:
                return True

        return False

    def _make_decision(
        self,
        challenger: PromptCandidate,
        accepted: bool,
        rejected: bool,
        reason: str,
        evaluated_blocks: list[int],
        compared_against: Optional[str],
        aggregate_result: Optional[EvaluationResult],
        portfolio: Optional[PromptPortfolio],
    ) -> IntensificationDecision:
        if aggregate_result is not None and evaluated_blocks:
            # Stamp the real evaluated-block set onto the challenger so downstream
            # incumbent advancement and block-aware parent tournaments see the
            # truth (they read candidate.metadata["evaluated_blocks"] first).
            # Without this, an intensification-accepted challenger looks like it
            # has 0 evaluated blocks and wastes advancement budget re-checking
            # block 0.
            challenger.metadata["evaluated_blocks"] = sorted(set(evaluated_blocks))

        if portfolio is not None and aggregate_result is not None:
            if accepted or self.config.add_rejected_to_population:
                portfolio.add(challenger, aggregate_result)

        return IntensificationDecision(
            candidate_id=challenger.candidate_id,
            accepted=accepted,
            rejected=rejected,
            reason=reason,
            evaluated_blocks=list(evaluated_blocks),
            compared_against=compared_against,
            aggregate_result=aggregate_result,
            budget_used=self.budget_allocator.used_budget,
            metadata={
                "remaining_budget": self.budget_allocator.remaining_budget,
                "budget_utilization": self.budget_allocator.utilization,
            },
        )