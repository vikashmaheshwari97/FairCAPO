from __future__ import annotations

from typing import Optional, Sequence

from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
from heal_capo.optimizers import intensification_legacy as _legacy
from heal_capo.optimizers.block_evaluator import BlockEvaluation, BlockEvaluator
from heal_capo.optimizers.budget_allocator import BudgetAllocator

IntensificationDecision = _legacy.IntensificationDecision
objective_vector = _legacy.objective_vector
dominates_result = _legacy.dominates_result
choose_closest_incumbent = _legacy.choose_closest_incumbent

_DEFAULT_MIN_BLOCKS_BEFORE_REJECT = 1


def configure_min_blocks_before_reject(value: int) -> None:
    global _DEFAULT_MIN_BLOCKS_BEFORE_REJECT
    _DEFAULT_MIN_BLOCKS_BEFORE_REJECT = max(1, int(value))


class IntensificationConfig(_legacy.IntensificationConfig):
    def __init__(
        self,
        max_blocks_per_challenger: Optional[int] = None,
        reject_when_dominated: bool = True,
        accept_when_not_dominated_on_common_blocks: bool = True,
        add_rejected_to_population: bool = False,
        use_cache: bool = True,
        min_blocks_before_reject: Optional[int] = None,
    ):
        super().__init__(
            max_blocks_per_challenger=max_blocks_per_challenger,
            reject_when_dominated=reject_when_dominated,
            accept_when_not_dominated_on_common_blocks=accept_when_not_dominated_on_common_blocks,
            add_rejected_to_population=add_rejected_to_population,
            use_cache=use_cache,
        )
        self.min_blocks_before_reject = max(
            1,
            int(
                _DEFAULT_MIN_BLOCKS_BEFORE_REJECT
                if min_blocks_before_reject is None
                else min_blocks_before_reject
            ),
        )


class Intensifier(_legacy.Intensifier):
    """Progressive race that cannot reject before the configured evidence depth."""

    def intensify(
        self,
        challenger: PromptCandidate,
        incumbents: Sequence[PromptCandidate],
        portfolio: Optional[PromptPortfolio] = None,
    ) -> IntensificationDecision:
        if not incumbents:
            return self._evaluate_without_incumbents(challenger=challenger, portfolio=portfolio)

        common_blocks = self._common_incumbent_blocks(incumbents)
        if not common_blocks:
            common_blocks = [self.block_evaluator.block_ids()[0]]
        if self.config.max_blocks_per_challenger is not None:
            common_blocks = common_blocks[: self.config.max_blocks_per_challenger]

        evaluated_blocks: list[int] = []
        last_challenger_result: Optional[EvaluationResult] = None
        compared_against: Optional[str] = None
        prompt_cache_hits = 0

        for block_id in common_blocks:
            evaluation: BlockEvaluation = self.block_evaluator.evaluate_block(
                candidate=challenger,
                block_id=block_id,
                use_cache=self.config.use_cache,
            )
            if evaluation.from_cache:
                prompt_cache_hits += 1
            if (
                not evaluation.from_cache
                and not self._budget_recorded(challenger.candidate_id, block_id)
            ):
                self.budget_allocator.record_block_evaluation(evaluation)

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

            enough_evidence = (
                len(evaluated_blocks)
                >= int(getattr(self.config, "min_blocks_before_reject", 1))
            )
            if (
                enough_evidence
                and closest_incumbent_id is not None
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
                    reason=(
                        "Rejected after minimum evidence depth because the closest "
                        "incumbent dominates the challenger."
                    ),
                    evaluated_blocks=evaluated_blocks,
                    compared_against=closest_incumbent_id,
                    aggregate_result=challenger_result,
                    portfolio=portfolio,
                    extra_metadata={
                        "prompt_cache_hits": prompt_cache_hits,
                        "min_blocks_before_reject": self.config.min_blocks_before_reject,
                    },
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
            reason="Accepted after surviving the configured intensification depth.",
            evaluated_blocks=evaluated_blocks,
            compared_against=compared_against,
            aggregate_result=last_challenger_result,
            portfolio=portfolio,
            extra_metadata={
                "prompt_cache_hits": prompt_cache_hits,
                "min_blocks_before_reject": self.config.min_blocks_before_reject,
            },
        )
