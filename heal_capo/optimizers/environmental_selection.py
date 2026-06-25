from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from heal_capo.core import EvaluationResult, PromptCandidate
from heal_capo.optimizers.parent_selection import (
    ParentSelectionConfig,
    crowding_distance,
    dominates,
    evaluation_level,
    get_evaluated_blocks,
)


@dataclass
class EnvironmentalSelectionConfig:
    """
    MO-CAPO-style environmental selection.

    Goal:
      - keep population size <= mu
      - preserve incumbents/Pareto candidates
      - remove weak non-incumbents first
      - use dominance/crowding when candidates are comparable
      - use evaluation level when candidates are not comparable
    """

    population_size: int = 10
    random_seed: Optional[int] = None
    prefer_keep_incumbents: bool = True
    remove_unevaluated_first: bool = True
    use_crowding_distance: bool = True
    use_dominance_fronts: bool = True


@dataclass
class EnvironmentalSelectionDecision:
    kept_ids: list[str]
    removed_ids: list[str]
    reason: str
    metadata: dict


def candidate_id(candidate: PromptCandidate | str) -> str:
    if isinstance(candidate, str):
        return candidate

    return str(candidate.candidate_id)


def comparable_by_blocks(
    candidate_ids: list[str],
    candidates_by_id: dict[str, PromptCandidate],
    evaluations: dict[str, EvaluationResult],
) -> bool:
    """
    True if all candidates have identical evaluated block sets.

    If block data is missing, fall back to False because comparison is less reliable.
    """
    if not candidate_ids:
        return True

    block_sets = []

    for cid in candidate_ids:
        candidate = candidates_by_id[cid]
        result = evaluations.get(cid)
        blocks = get_evaluated_blocks(candidate, result)

        if not blocks:
            return False

        block_sets.append(blocks)

    first = block_sets[0]
    return all(blocks == first for blocks in block_sets)


def nondominated_ids(
    candidate_ids: list[str],
    evaluations: dict[str, EvaluationResult],
    parent_config: ParentSelectionConfig | None = None,
) -> set[str]:
    """
    Return non-dominated candidate ids among a subset.
    """
    parent_config = parent_config or ParentSelectionConfig()

    result = set(candidate_ids)

    for left_id in candidate_ids:
        left_eval = evaluations.get(left_id)

        if left_eval is None:
            continue

        for right_id in candidate_ids:
            if left_id == right_id:
                continue

            right_eval = evaluations.get(right_id)

            if right_eval is None:
                continue

            if dominates(right_eval, left_eval, parent_config):
                result.discard(left_id)
                break

    return result


def dominance_fronts(
    candidate_ids: list[str],
    evaluations: dict[str, EvaluationResult],
    parent_config: ParentSelectionConfig | None = None,
) -> list[list[str]]:
    """
    Compute simple non-dominated sorting fronts.

    Front 0 = non-dominated.
    Later fronts = progressively worse.
    """
    remaining = list(candidate_ids)
    fronts: list[list[str]] = []

    while remaining:
        front = sorted(nondominated_ids(remaining, evaluations, parent_config))

        if not front:
            # Defensive fallback to avoid infinite loops if data is malformed.
            fronts.append(sorted(remaining))
            break

        fronts.append(front)
        remaining = [cid for cid in remaining if cid not in front]

    return fronts


def least_evaluated_ids(
    candidate_ids: list[str],
    candidates_by_id: dict[str, PromptCandidate],
    evaluations: dict[str, EvaluationResult],
) -> list[str]:
    levels = {}

    for cid in candidate_ids:
        levels[cid] = evaluation_level(
            candidates_by_id[cid],
            evaluations.get(cid),
        )

    if not levels:
        return []

    min_level = min(levels.values())

    return sorted(
        [cid for cid, level in levels.items() if level == min_level]
    )


def most_crowded_id(
    candidate_ids: list[str],
    evaluations: dict[str, EvaluationResult],
    rng: random.Random,
) -> str:
    """
    Return candidate id with smallest crowding distance.

    Smaller crowding distance means more crowded and preferred for removal.
    Boundary points usually have infinity and are preserved.
    """
    results = [
        evaluations[cid]
        for cid in candidate_ids
        if cid in evaluations
    ]

    if not results:
        return rng.choice(candidate_ids)

    distances = crowding_distance(results)

    finite_items = [
        (cid, distances.get(cid, 0.0))
        for cid in candidate_ids
        if not math.isinf(distances.get(cid, 0.0))
    ]

    if not finite_items:
        return rng.choice(sorted(candidate_ids))

    min_distance = min(distance for _, distance in finite_items)
    tied = sorted(
        [cid for cid, distance in finite_items if distance == min_distance]
    )

    return rng.choice(tied)


class EnvironmentalSelector:
    """
    Reduce population to configured size using MO-CAPO-style rules.
    """

    def __init__(
        self,
        config: EnvironmentalSelectionConfig | None = None,
        parent_config: ParentSelectionConfig | None = None,
        rng: random.Random | None = None,
    ):
        self.config = config or EnvironmentalSelectionConfig()
        self.parent_config = parent_config or ParentSelectionConfig()
        self.rng = rng or random.Random(self.config.random_seed)

    def select(
        self,
        population: list[PromptCandidate],
        incumbent_ids: set[str],
        evaluations: dict[str, EvaluationResult],
    ) -> tuple[list[PromptCandidate], EnvironmentalSelectionDecision]:
        if self.config.population_size <= 0:
            raise ValueError("population_size must be positive.")

        if len(population) <= self.config.population_size:
            kept_ids = [candidate.candidate_id for candidate in population]

            return population, EnvironmentalSelectionDecision(
                kept_ids=kept_ids,
                removed_ids=[],
                reason="population_within_limit",
                metadata={
                    "population_size": len(population),
                    "target_population_size": self.config.population_size,
                },
            )

        candidates_by_id = {
            candidate.candidate_id: candidate
            for candidate in population
        }

        current_ids = [candidate.candidate_id for candidate in population]
        removed_ids: list[str] = []
        removal_reasons: list[dict] = []

        while len(current_ids) > self.config.population_size:
            remove_id, reason, metadata = self._choose_removal(
                current_ids=current_ids,
                incumbent_ids=incumbent_ids,
                candidates_by_id=candidates_by_id,
                evaluations=evaluations,
            )

            current_ids.remove(remove_id)
            removed_ids.append(remove_id)
            removal_reasons.append(
                {
                    "candidate_id": remove_id,
                    "reason": reason,
                    **metadata,
                }
            )

        kept_candidates = [
            candidate
            for candidate in population
            if candidate.candidate_id in set(current_ids)
        ]

        return kept_candidates, EnvironmentalSelectionDecision(
            kept_ids=current_ids,
            removed_ids=removed_ids,
            reason="environmental_selection",
            metadata={
                "target_population_size": self.config.population_size,
                "removal_reasons": removal_reasons,
            },
        )

    def _choose_removal(
        self,
        current_ids: list[str],
        incumbent_ids: set[str],
        candidates_by_id: dict[str, PromptCandidate],
        evaluations: dict[str, EvaluationResult],
    ) -> tuple[str, str, dict]:
        non_incumbent_ids = [
            cid for cid in current_ids if cid not in incumbent_ids
        ]

        candidate_pool = non_incumbent_ids

        if not candidate_pool:
            candidate_pool = list(current_ids)

        if self.config.remove_unevaluated_first:
            unevaluated = [
                cid for cid in candidate_pool if cid not in evaluations
            ]

            if unevaluated:
                remove_id = self.rng.choice(sorted(unevaluated))

                return (
                    remove_id,
                    "remove_unevaluated",
                    {"pool": candidate_pool},
                )

        if self.config.use_dominance_fronts and comparable_by_blocks(
            candidate_ids=candidate_pool,
            candidates_by_id=candidates_by_id,
            evaluations=evaluations,
        ):
            fronts = dominance_fronts(
                candidate_ids=candidate_pool,
                evaluations=evaluations,
                parent_config=self.parent_config,
            )

            worst_front = fronts[-1]

            if len(worst_front) == 1:
                return (
                    worst_front[0],
                    "remove_worst_dominance_front",
                    {"fronts": fronts},
                )

            if self.config.use_crowding_distance:
                remove_id = most_crowded_id(
                    candidate_ids=worst_front,
                    evaluations=evaluations,
                    rng=self.rng,
                )

                return (
                    remove_id,
                    "remove_most_crowded_from_worst_front",
                    {"fronts": fronts},
                )

            remove_id = self.rng.choice(sorted(worst_front))

            return (
                remove_id,
                "remove_random_from_worst_front",
                {"fronts": fronts},
            )

        least_evaluated = least_evaluated_ids(
            candidate_ids=candidate_pool,
            candidates_by_id=candidates_by_id,
            evaluations=evaluations,
        )

        if least_evaluated:
            remove_id = self.rng.choice(sorted(least_evaluated))

            return (
                remove_id,
                "remove_least_evaluated",
                {
                    "least_evaluated_ids": least_evaluated,
                    "pool": candidate_pool,
                },
            )

        remove_id = self.rng.choice(sorted(candidate_pool))

        return (
            remove_id,
            "remove_random",
            {"pool": candidate_pool},
        )


def environmental_select(
    population: list[PromptCandidate],
    incumbent_ids: set[str],
    evaluations: dict[str, EvaluationResult],
    config: EnvironmentalSelectionConfig | None = None,
    parent_config: ParentSelectionConfig | None = None,
    rng: random.Random | None = None,
) -> tuple[list[PromptCandidate], EnvironmentalSelectionDecision]:
    selector = EnvironmentalSelector(
        config=config,
        parent_config=parent_config,
        rng=rng,
    )

    return selector.select(
        population=population,
        incumbent_ids=incumbent_ids,
        evaluations=evaluations,
    )