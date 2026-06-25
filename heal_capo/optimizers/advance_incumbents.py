from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
from heal_capo.optimizers.block_evaluator import BlockEvaluator
from heal_capo.optimizers.budget_allocator import BudgetAllocator
from heal_capo.optimizers.parent_selection import get_evaluated_blocks
from heal_capo.pareto import pareto_archive


@dataclass
class AdvanceIncumbentsConfig:
    """
    MO-CAPO-style incumbent advancement.

    Purpose:
      - Keep incumbents evaluated on comparable block sets.
      - Give priority to incumbents with fewer evaluated blocks.
      - If all incumbents share the same block set, evaluate one incumbent on a new block.
      - Update budget and Pareto archive after each advancement.
    """

    random_seed: Optional[int] = None
    use_cache: bool = True
    prefer_fewest_blocks: bool = True
    update_pareto_archive: bool = True
    allow_new_block_when_aligned: bool = True


@dataclass
class IncumbentAdvanceDecision:
    advanced: bool
    candidate_id: Optional[str] = None
    block_id: Optional[int] = None
    reason: str = ""
    budget_used: float = 0.0
    remaining_budget: float = 0.0
    evaluated_blocks_before: list[int] = field(default_factory=list)
    evaluated_blocks_after: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def candidate_id(candidate: PromptCandidate | str) -> str:
    if isinstance(candidate, str):
        return candidate

    return str(candidate.candidate_id)


def get_candidate_blocks(
    candidate: PromptCandidate,
    portfolio: PromptPortfolio | None = None,
) -> set[int]:
    result = None

    if portfolio is not None and candidate.candidate_id in portfolio.evaluations:
        result = portfolio.evaluations[candidate.candidate_id]

    return get_evaluated_blocks(candidate, result)


def all_available_block_ids(block_evaluator: BlockEvaluator) -> list[int]:
    return list(block_evaluator.block_ids())


def incumbent_block_map(
    incumbents: list[PromptCandidate],
    portfolio: PromptPortfolio | None = None,
) -> dict[str, set[int]]:
    return {
        candidate.candidate_id: get_candidate_blocks(candidate, portfolio)
        for candidate in incumbents
    }


def common_incumbent_blocks(
    incumbents: list[PromptCandidate],
    portfolio: PromptPortfolio | None = None,
) -> set[int]:
    if not incumbents:
        return set()

    block_sets = list(incumbent_block_map(incumbents, portfolio).values())

    if not block_sets:
        return set()

    common = set(block_sets[0])

    for blocks in block_sets[1:]:
        common = common.intersection(blocks)

    return common


def union_incumbent_blocks(
    incumbents: list[PromptCandidate],
    portfolio: PromptPortfolio | None = None,
) -> set[int]:
    union: set[int] = set()

    for blocks in incumbent_block_map(incumbents, portfolio).values():
        union.update(blocks)

    return union


def incumbents_are_aligned(
    incumbents: list[PromptCandidate],
    portfolio: PromptPortfolio | None = None,
) -> bool:
    if len(incumbents) <= 1:
        return True

    block_sets = list(incumbent_block_map(incumbents, portfolio).values())

    if not block_sets:
        return True

    first = block_sets[0]

    return all(blocks == first for blocks in block_sets)


def select_incumbent_to_advance(
    incumbents: list[PromptCandidate],
    portfolio: PromptPortfolio | None = None,
    rng: random.Random | None = None,
) -> PromptCandidate | None:
    """
    Select incumbent with fewest evaluated blocks.

    Ties are broken by candidate_id for deterministic behavior unless rng is
    provided, in which case random tie-break is used.
    """
    if not incumbents:
        return None

    block_counts = {
        candidate.candidate_id: len(get_candidate_blocks(candidate, portfolio))
        for candidate in incumbents
    }

    min_count = min(block_counts.values())

    tied = [
        candidate
        for candidate in incumbents
        if block_counts[candidate.candidate_id] == min_count
    ]

    if not tied:
        return None

    if rng is not None:
        return rng.choice(sorted(tied, key=lambda candidate: candidate.candidate_id))

    return sorted(tied, key=lambda candidate: candidate.candidate_id)[0]


def select_block_for_incumbent(
    candidate: PromptCandidate,
    incumbents: list[PromptCandidate],
    block_evaluator: BlockEvaluator,
    portfolio: PromptPortfolio | None = None,
    allow_new_block_when_aligned: bool = True,
    rng: random.Random | None = None,
) -> int | None:
    """
    Select the next block for incumbent advancement.

    Rule:
      1. If candidate is missing blocks that other incumbents already have,
         evaluate one of those missing incumbent-union blocks first.
      2. If all incumbents are aligned and allowed, evaluate a new unseen block.
      3. If no block is available, return None.
    """
    rng = rng or random.Random()

    all_blocks = all_available_block_ids(block_evaluator)
    candidate_blocks = get_candidate_blocks(candidate, portfolio)
    union_blocks = union_incumbent_blocks(incumbents, portfolio)

    missing_from_union = sorted(union_blocks - candidate_blocks)

    if missing_from_union:
        return rng.choice(missing_from_union)

    if allow_new_block_when_aligned:
        unseen_blocks = sorted(set(all_blocks) - candidate_blocks)

        if unseen_blocks:
            return rng.choice(unseen_blocks)

    return None


def update_candidate_block_metadata(
    candidate: PromptCandidate,
    blocks: set[int],
) -> None:
    candidate.metadata["evaluated_blocks"] = sorted(blocks)


class IncumbentAdvancer:
    """
    Advance incumbents by one block while tracking budget.

    This is the final missing MO-CAPO helper before upgrading the runner into a
    true evolutionary loop.
    """

    def __init__(
        self,
        block_evaluator: BlockEvaluator,
        budget_allocator: BudgetAllocator,
        config: AdvanceIncumbentsConfig | None = None,
        rng: random.Random | None = None,
    ):
        self.block_evaluator = block_evaluator
        self.budget_allocator = budget_allocator
        self.config = config or AdvanceIncumbentsConfig()
        self.rng = rng or random.Random(self.config.random_seed)

    def _budget_recorded(self, candidate_id: str, block_id: int) -> bool:
        for record in self.budget_allocator.state.records:
            if record.candidate_id == candidate_id and record.block_id == block_id:
                return True

        return False

    def advance(
        self,
        incumbents: list[PromptCandidate],
        portfolio: PromptPortfolio,
    ) -> IncumbentAdvanceDecision:
        if not incumbents:
            return IncumbentAdvanceDecision(
                advanced=False,
                reason="no_incumbents",
                budget_used=self.budget_allocator.used_budget,
                remaining_budget=self.budget_allocator.remaining_budget,
            )

        if self.budget_allocator.exhausted:
            return IncumbentAdvanceDecision(
                advanced=False,
                reason="budget_exhausted",
                budget_used=self.budget_allocator.used_budget,
                remaining_budget=self.budget_allocator.remaining_budget,
            )

        candidate = select_incumbent_to_advance(
            incumbents=incumbents,
            portfolio=portfolio,
            rng=self.rng if self.config.prefer_fewest_blocks else None,
        )

        if candidate is None:
            return IncumbentAdvanceDecision(
                advanced=False,
                reason="no_candidate_selected",
                budget_used=self.budget_allocator.used_budget,
                remaining_budget=self.budget_allocator.remaining_budget,
            )

        blocks_before = get_candidate_blocks(candidate, portfolio)

        block_id = select_block_for_incumbent(
            candidate=candidate,
            incumbents=incumbents,
            block_evaluator=self.block_evaluator,
            portfolio=portfolio,
            allow_new_block_when_aligned=self.config.allow_new_block_when_aligned,
            rng=self.rng,
        )

        if block_id is None:
            return IncumbentAdvanceDecision(
                advanced=False,
                candidate_id=candidate.candidate_id,
                reason="no_available_block",
                budget_used=self.budget_allocator.used_budget,
                remaining_budget=self.budget_allocator.remaining_budget,
                evaluated_blocks_before=sorted(blocks_before),
                evaluated_blocks_after=sorted(blocks_before),
            )

        evaluation = self.block_evaluator.evaluate_block(
            candidate=candidate,
            block_id=block_id,
            use_cache=self.config.use_cache,
        )

        # Only charge the budget the first time this (candidate, block) pair is
        # evaluated. A cache hit performs no new LLM work, so re-charging it
        # would inflate used_budget for zero information gain.
        if not self._budget_recorded(candidate.candidate_id, block_id):
            self.budget_allocator.record_block_evaluation(evaluation)

        blocks_after = set(blocks_before)
        blocks_after.add(block_id)

        update_candidate_block_metadata(candidate, blocks_after)

        aggregate = self.block_evaluator.aggregate_candidate(
            candidate_id=candidate.candidate_id,
            block_ids=sorted(blocks_after),
        )

        # Preserve candidate in portfolio and replace with aggregate evaluation.
        portfolio.add(candidate, aggregate)

        if self.config.update_pareto_archive:
            portfolio.evaluations = pareto_archive(portfolio.evaluations)
            portfolio.candidates = [
                portfolio_candidate
                for portfolio_candidate in portfolio.candidates
                if portfolio_candidate.candidate_id in portfolio.evaluations
            ]

        block_result = getattr(evaluation, "result", None)

        return IncumbentAdvanceDecision(
            advanced=True,
            candidate_id=candidate.candidate_id,
            block_id=block_id,
            reason="advanced_incumbent",
            budget_used=self.budget_allocator.used_budget,
            remaining_budget=self.budget_allocator.remaining_budget,
            evaluated_blocks_before=sorted(blocks_before),
            evaluated_blocks_after=sorted(blocks_after),
            metadata={
                "evaluation_cost": (
                    block_result.cost
                    if block_result is not None and hasattr(block_result, "cost")
                    else aggregate.cost
                ),
                "performance": aggregate.performance,
                "risk": aggregate.risk,
                "fairness_risk": aggregate.fairness_risk,
                "aligned_before": incumbents_are_aligned(incumbents, portfolio),
            },
        )


def advance_incumbents(
    incumbents: list[PromptCandidate],
    portfolio: PromptPortfolio,
    block_evaluator: BlockEvaluator,
    budget_allocator: BudgetAllocator,
    config: AdvanceIncumbentsConfig | None = None,
    rng: random.Random | None = None,
) -> IncumbentAdvanceDecision:
    advancer = IncumbentAdvancer(
        block_evaluator=block_evaluator,
        budget_allocator=budget_allocator,
        config=config,
        rng=rng,
    )

    return advancer.advance(
        incumbents=incumbents,
        portfolio=portfolio,
    )


def advance_front_one_level(
    incumbents: list[PromptCandidate],
    portfolio: PromptPortfolio,
    block_evaluator: BlockEvaluator,
    budget_allocator: BudgetAllocator,
    config: AdvanceIncumbentsConfig | None = None,
    rng: random.Random | None = None,
    max_blocks: int | None = None,
    refresh_incumbents=None,
) -> list[IncumbentAdvanceDecision]:
    """
    Intensify the WHOLE incumbent front by (at most) one aligned block.

    This is the missing piece that makes MO-CAPO-style intensification actually
    deepen the evaluation. A single :func:`advance_incumbents` call advances one
    incumbent by one block; on its own (once per generation) it can never keep
    the front aligned, so the intensifier keeps racing challengers on the
    intersection of incumbent blocks -- which collapses to a single block and
    makes winners get decided on ~10 examples.

    This helper repeatedly advances the least-evaluated incumbent until either:
      - the front is aligned one block deeper than it started (one full level),
      - every incumbent already covers ``max_blocks`` blocks,
      - the budget is exhausted, or
      - no further block can be advanced.

    ``max_blocks`` caps how deep the common set may grow (mirrors the
    intensifier's ``max_blocks_per_challenger`` so the front and the challengers
    it races against stay comparable). ``refresh_incumbents`` is an optional
    callable returning the current incumbent list (the Pareto archive can change
    as blocks are added); when omitted the passed-in list is reused.

    Returns the list of per-step advancement decisions (possibly empty).
    """
    advancer = IncumbentAdvancer(
        block_evaluator=block_evaluator,
        budget_allocator=budget_allocator,
        config=config,
        rng=rng,
    )

    decisions: list[IncumbentAdvanceDecision] = []

    if not incumbents:
        return decisions

    total_blocks = len(block_evaluator.block_ids())
    block_cap = total_blocks if max_blocks is None else min(max_blocks, total_blocks)

    start_depth = len(common_incumbent_blocks(incumbents, portfolio))

    # Safety bound: alignment converges in O(len(incumbents)) steps; the +2 and
    # the depth guard below are just belt-and-suspenders against pathologies.
    max_steps = max(1, len(incumbents) * (block_cap + 1)) + 2

    for _ in range(max_steps):
        if budget_allocator.exhausted:
            break

        # Already aligned and deep enough -> nothing more to do this level.
        if incumbents_are_aligned(incumbents, portfolio):
            common_depth = len(common_incumbent_blocks(incumbents, portfolio))
            if common_depth >= block_cap or common_depth > start_depth:
                break

        # Don't push any incumbent past the block cap.
        min_blocks = min(
            len(get_candidate_blocks(candidate, portfolio))
            for candidate in incumbents
        )
        if min_blocks >= block_cap:
            break

        decision = advancer.advance(incumbents=incumbents, portfolio=portfolio)
        decisions.append(decision)

        if not decision.advanced:
            break

        if refresh_incumbents is not None:
            refreshed = refresh_incumbents()
            if refreshed:
                incumbents = refreshed

    return decisions