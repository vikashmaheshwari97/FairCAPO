from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from heal_capo.core import EvaluationResult, PromptCandidate


ObjectiveName = str


DEFAULT_OBJECTIVES: tuple[ObjectiveName, ...] = (
    "performance",
    "cost",
    "risk",
    "fairness_risk",
)


@dataclass
class ParentSelectionConfig:
    """
    Configuration for MO-CAPO-style binary tournament parent selection.

    MO-CAPO parent selection should:
      1. Prefer incumbents over non-incumbents.
      2. Compare incumbents using crowding distance to preserve diversity.
      3. Compare candidates with equal evaluation levels using dominance.
      4. Compare candidates with different levels only when a fair common/subset
         block comparison is available.
      5. Fall back to seeded random tie-breaks.
    """

    objectives: tuple[ObjectiveName, ...] = DEFAULT_OBJECTIVES
    maximize_objectives: tuple[ObjectiveName, ...] = ("performance",)
    minimize_objectives: tuple[ObjectiveName, ...] = (
        "cost",
        "risk",
        "fairness_risk",
        "drift",
    )
    prefer_incumbents: bool = True
    use_crowding_distance: bool = True
    use_subset_dominance: bool = True
    random_seed: Optional[int] = None


@dataclass
class TournamentDecision:
    winner_id: str
    loser_id: str
    reason: str
    winner_is_incumbent: bool = False
    loser_is_incumbent: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def candidate_id(candidate: PromptCandidate | str) -> str:
    if isinstance(candidate, str):
        return candidate

    return str(candidate.candidate_id)


def get_metric(result: EvaluationResult, objective: ObjectiveName) -> float:
    value = getattr(result, objective, None)

    if value is None:
        return 0.0

    try:
        if math.isnan(float(value)):
            return 0.0
    except Exception:
        return 0.0

    return float(value)


def objective_better_or_equal(
    left: float,
    right: float,
    objective: ObjectiveName,
    config: ParentSelectionConfig,
) -> bool:
    if objective in config.maximize_objectives:
        return left >= right

    return left <= right


def objective_strictly_better(
    left: float,
    right: float,
    objective: ObjectiveName,
    config: ParentSelectionConfig,
) -> bool:
    if objective in config.maximize_objectives:
        return left > right

    return left < right


def dominates(
    left: EvaluationResult,
    right: EvaluationResult,
    config: ParentSelectionConfig | None = None,
) -> bool:
    """
    True if left Pareto-dominates right.

    performance is maximized.
    cost/risk/fairness_risk/drift are minimized.
    """
    config = config or ParentSelectionConfig()

    better_or_equal_all = True
    strictly_better_any = False

    for objective in config.objectives:
        left_value = get_metric(left, objective)
        right_value = get_metric(right, objective)

        if not objective_better_or_equal(
            left=left_value,
            right=right_value,
            objective=objective,
            config=config,
        ):
            better_or_equal_all = False
            break

        if objective_strictly_better(
            left=left_value,
            right=right_value,
            objective=objective,
            config=config,
        ):
            strictly_better_any = True

    return better_or_equal_all and strictly_better_any


def get_evaluated_blocks(
    candidate: PromptCandidate,
    fallback_result: EvaluationResult | None = None,
) -> set[int]:
    """
    Best-effort extraction of evaluated block ids.

    Different components may store block ids in candidate metadata or result details.
    This function accepts common forms:
      - metadata["evaluated_blocks"]
      - metadata["block_ids"]
      - result.details["evaluated_blocks"]
      - result.details["block_ids"]
    """
    values: Any = None

    for key in ("evaluated_blocks", "block_ids", "blocks"):
        if key in candidate.metadata:
            values = candidate.metadata.get(key)
            break

    if values is None and fallback_result is not None:
        details = fallback_result.details or {}

        for key in ("evaluated_blocks", "block_ids", "blocks"):
            if key in details:
                values = details.get(key)
                break

    if values is None:
        return set()

    if isinstance(values, str):
        cleaned = (
            values.replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
            .replace("{", "")
            .replace("}", "")
        )
        tokens = [token.strip() for token in cleaned.split(",") if token.strip()]
        result = set()

        for token in tokens:
            try:
                result.add(int(token))
            except ValueError:
                continue

        return result

    if isinstance(values, int):
        return {values}

    if isinstance(values, Iterable):
        result = set()

        for value in values:
            try:
                result.add(int(value))
            except Exception:
                continue

        return result

    return set()


def evaluation_level(
    candidate: PromptCandidate,
    result: EvaluationResult | None = None,
) -> int:
    blocks = get_evaluated_blocks(candidate, result)

    if blocks:
        return len(blocks)

    if result is not None and result.n_examples is not None:
        return int(result.n_examples)

    return 0


def common_blocks(
    left: PromptCandidate,
    right: PromptCandidate,
    left_result: EvaluationResult | None = None,
    right_result: EvaluationResult | None = None,
) -> set[int]:
    left_blocks = get_evaluated_blocks(left, left_result)
    right_blocks = get_evaluated_blocks(right, right_result)

    return left_blocks.intersection(right_blocks)


def block_subset_relation(
    left: PromptCandidate,
    right: PromptCandidate,
    left_result: EvaluationResult | None = None,
    right_result: EvaluationResult | None = None,
) -> str:
    """
    Return:
      - "left_subset_right"
      - "right_subset_left"
      - "same"
      - "overlap"
      - "disjoint"
      - "unknown"
    """
    left_blocks = get_evaluated_blocks(left, left_result)
    right_blocks = get_evaluated_blocks(right, right_result)

    if not left_blocks or not right_blocks:
        return "unknown"

    if left_blocks == right_blocks:
        return "same"

    if left_blocks.issubset(right_blocks):
        return "left_subset_right"

    if right_blocks.issubset(left_blocks):
        return "right_subset_left"

    if left_blocks.intersection(right_blocks):
        return "overlap"

    return "disjoint"


def crowding_distance(
    results: Iterable[EvaluationResult],
    config: ParentSelectionConfig | None = None,
) -> dict[str, float]:
    """
    Compute NSGA-II-style crowding distance over available objective vectors.

    Boundary candidates receive infinity.
    Larger distance means better diversity.
    """
    config = config or ParentSelectionConfig()
    result_list = list(results)

    if not result_list:
        return {}

    if len(result_list) <= 2:
        return {result.candidate_id: math.inf for result in result_list}

    distances = {result.candidate_id: 0.0 for result in result_list}

    for objective in config.objectives:
        sorted_results = sorted(
            result_list,
            key=lambda result: get_metric(result, objective),
            reverse=objective in config.maximize_objectives,
        )

        first = sorted_results[0]
        last = sorted_results[-1]

        distances[first.candidate_id] = math.inf
        distances[last.candidate_id] = math.inf

        min_value = min(get_metric(result, objective) for result in sorted_results)
        max_value = max(get_metric(result, objective) for result in sorted_results)
        denom = max_value - min_value

        if denom == 0:
            continue

        for idx in range(1, len(sorted_results) - 1):
            candidate = sorted_results[idx]

            if math.isinf(distances[candidate.candidate_id]):
                continue

            previous_value = get_metric(sorted_results[idx - 1], objective)
            next_value = get_metric(sorted_results[idx + 1], objective)

            distances[candidate.candidate_id] += abs(next_value - previous_value) / denom

    return distances


class ParentSelector:
    """
    MO-CAPO-style parent selector.

    The selector is intentionally independent from the full optimizer loop.
    It only needs:
      - population candidates
      - incumbent ids
      - evaluation results
    """

    def __init__(
        self,
        config: ParentSelectionConfig | None = None,
        rng: random.Random | None = None,
    ):
        self.config = config or ParentSelectionConfig()
        self.rng = rng or random.Random(self.config.random_seed)

    def select_two_parents(
        self,
        population: list[PromptCandidate],
        incumbent_ids: set[str],
        evaluations: dict[str, EvaluationResult],
    ) -> tuple[PromptCandidate, PromptCandidate, list[TournamentDecision]]:
        """
        Run two independent binary tournaments and return distinct parents.
        """
        if len(population) < 2:
            raise ValueError("Need at least two candidates to select parents.")

        first_parent, first_decision = self.select_parent(
            population=population,
            incumbent_ids=incumbent_ids,
            evaluations=evaluations,
        )

        attempts = 0
        second_parent = first_parent
        second_decision = first_decision

        while second_parent.candidate_id == first_parent.candidate_id and attempts < 20:
            second_parent, second_decision = self.select_parent(
                population=population,
                incumbent_ids=incumbent_ids,
                evaluations=evaluations,
            )
            attempts += 1

        if second_parent.candidate_id == first_parent.candidate_id:
            alternatives = [
                candidate
                for candidate in population
                if candidate.candidate_id != first_parent.candidate_id
            ]
            second_parent = self.rng.choice(alternatives)
            second_decision = TournamentDecision(
                winner_id=second_parent.candidate_id,
                loser_id=first_parent.candidate_id,
                reason="forced_distinct_parent",
                winner_is_incumbent=second_parent.candidate_id in incumbent_ids,
                loser_is_incumbent=first_parent.candidate_id in incumbent_ids,
            )

        return first_parent, second_parent, [first_decision, second_decision]

    def select_parent(
        self,
        population: list[PromptCandidate],
        incumbent_ids: set[str],
        evaluations: dict[str, EvaluationResult],
    ) -> tuple[PromptCandidate, TournamentDecision]:
        """
        Run one binary tournament.
        """
        if not population:
            raise ValueError("Population is empty.")

        if len(population) == 1:
            only = population[0]
            return only, TournamentDecision(
                winner_id=only.candidate_id,
                loser_id=only.candidate_id,
                reason="single_candidate_population",
                winner_is_incumbent=only.candidate_id in incumbent_ids,
                loser_is_incumbent=only.candidate_id in incumbent_ids,
            )

        left, right = self.rng.sample(population, 2)

        winner, decision = self.compare(
            left=left,
            right=right,
            incumbent_ids=incumbent_ids,
            evaluations=evaluations,
            population=population,
        )

        return winner, decision

    def compare(
        self,
        left: PromptCandidate,
        right: PromptCandidate,
        incumbent_ids: set[str],
        evaluations: dict[str, EvaluationResult],
        population: list[PromptCandidate] | None = None,
    ) -> tuple[PromptCandidate, TournamentDecision]:
        left_id = left.candidate_id
        right_id = right.candidate_id

        left_is_incumbent = left_id in incumbent_ids
        right_is_incumbent = right_id in incumbent_ids

        left_result = evaluations.get(left_id)
        right_result = evaluations.get(right_id)

        if self.config.prefer_incumbents and left_is_incumbent != right_is_incumbent:
            winner = left if left_is_incumbent else right
            loser = right if left_is_incumbent else left

            return winner, TournamentDecision(
                winner_id=winner.candidate_id,
                loser_id=loser.candidate_id,
                reason="incumbent_preference",
                winner_is_incumbent=True,
                loser_is_incumbent=False,
            )

        if left_result is None and right_result is None:
            return self._random_tie(left, right, "no_evaluations", incumbent_ids)

        if left_result is not None and right_result is None:
            return left, TournamentDecision(
                winner_id=left_id,
                loser_id=right_id,
                reason="evaluated_vs_unevaluated",
                winner_is_incumbent=left_is_incumbent,
                loser_is_incumbent=right_is_incumbent,
            )

        if right_result is not None and left_result is None:
            return right, TournamentDecision(
                winner_id=right_id,
                loser_id=left_id,
                reason="evaluated_vs_unevaluated",
                winner_is_incumbent=right_is_incumbent,
                loser_is_incumbent=left_is_incumbent,
            )

        assert left_result is not None
        assert right_result is not None

        relation = block_subset_relation(left, right, left_result, right_result)

        if relation == "same" or evaluation_level(left, left_result) == evaluation_level(right, right_result):
            dominance_winner = self._dominance_compare(
                left,
                right,
                left_result,
                right_result,
                incumbent_ids,
                reason_prefix="same_level",
            )

            if dominance_winner is not None:
                return dominance_winner

        if (
            self.config.use_subset_dominance
            and relation in {"left_subset_right", "right_subset_left"}
        ):
            dominance_winner = self._dominance_compare(
                left,
                right,
                left_result,
                right_result,
                incumbent_ids,
                reason_prefix=f"subset_relation_{relation}",
            )

            if dominance_winner is not None:
                return dominance_winner

        if self.config.use_crowding_distance and left_is_incumbent and right_is_incumbent:
            cd_winner = self._crowding_compare(
                left=left,
                right=right,
                evaluations=evaluations,
                incumbent_ids=incumbent_ids,
                population=population,
            )

            if cd_winner is not None:
                return cd_winner

        left_level = evaluation_level(left, left_result)
        right_level = evaluation_level(right, right_result)

        if left_level != right_level and not left_is_incumbent and not right_is_incumbent:
            winner = left if left_level > right_level else right
            loser = right if winner is left else left

            return winner, TournamentDecision(
                winner_id=winner.candidate_id,
                loser_id=loser.candidate_id,
                reason="more_evaluated_non_incumbent",
                winner_is_incumbent=winner.candidate_id in incumbent_ids,
                loser_is_incumbent=loser.candidate_id in incumbent_ids,
                metadata={
                    "left_level": left_level,
                    "right_level": right_level,
                    "relation": relation,
                },
            )

        return self._random_tie(
            left,
            right,
            reason=f"random_tie_relation_{relation}",
            incumbent_ids=incumbent_ids,
        )

    def _dominance_compare(
        self,
        left: PromptCandidate,
        right: PromptCandidate,
        left_result: EvaluationResult,
        right_result: EvaluationResult,
        incumbent_ids: set[str],
        reason_prefix: str,
    ) -> tuple[PromptCandidate, TournamentDecision] | None:
        left_dominates = dominates(left_result, right_result, self.config)
        right_dominates = dominates(right_result, left_result, self.config)

        if left_dominates and not right_dominates:
            return left, TournamentDecision(
                winner_id=left.candidate_id,
                loser_id=right.candidate_id,
                reason=f"{reason_prefix}_dominance",
                winner_is_incumbent=left.candidate_id in incumbent_ids,
                loser_is_incumbent=right.candidate_id in incumbent_ids,
            )

        if right_dominates and not left_dominates:
            return right, TournamentDecision(
                winner_id=right.candidate_id,
                loser_id=left.candidate_id,
                reason=f"{reason_prefix}_dominance",
                winner_is_incumbent=right.candidate_id in incumbent_ids,
                loser_is_incumbent=left.candidate_id in incumbent_ids,
            )

        return None

    def _crowding_compare(
        self,
        left: PromptCandidate,
        right: PromptCandidate,
        evaluations: dict[str, EvaluationResult],
        incumbent_ids: set[str],
        population: list[PromptCandidate] | None = None,
    ) -> tuple[PromptCandidate, TournamentDecision] | None:
        if population is None:
            relevant_ids = incumbent_ids
        else:
            relevant_ids = {
                candidate.candidate_id
                for candidate in population
                if candidate.candidate_id in incumbent_ids
            }

        relevant_results = [
            evaluations[candidate_id]
            for candidate_id in relevant_ids
            if candidate_id in evaluations
        ]

        if len(relevant_results) < 2:
            return None

        distances = crowding_distance(relevant_results, self.config)

        left_distance = distances.get(left.candidate_id, 0.0)
        right_distance = distances.get(right.candidate_id, 0.0)

        if left_distance == right_distance:
            return None

        winner = left if left_distance > right_distance else right
        loser = right if winner is left else left

        return winner, TournamentDecision(
            winner_id=winner.candidate_id,
            loser_id=loser.candidate_id,
            reason="incumbent_crowding_distance",
            winner_is_incumbent=True,
            loser_is_incumbent=True,
            metadata={
                "left_crowding_distance": left_distance,
                "right_crowding_distance": right_distance,
            },
        )

    def _random_tie(
        self,
        left: PromptCandidate,
        right: PromptCandidate,
        reason: str,
        incumbent_ids: set[str],
    ) -> tuple[PromptCandidate, TournamentDecision]:
        winner = self.rng.choice([left, right])
        loser = right if winner is left else left

        return winner, TournamentDecision(
            winner_id=winner.candidate_id,
            loser_id=loser.candidate_id,
            reason=reason,
            winner_is_incumbent=winner.candidate_id in incumbent_ids,
            loser_is_incumbent=loser.candidate_id in incumbent_ids,
        )


def select_parents(
    population: list[PromptCandidate],
    incumbent_ids: set[str],
    evaluations: dict[str, EvaluationResult],
    config: ParentSelectionConfig | None = None,
    rng: random.Random | None = None,
) -> tuple[PromptCandidate, PromptCandidate, list[TournamentDecision]]:
    selector = ParentSelector(config=config, rng=rng)

    return selector.select_two_parents(
        population=population,
        incumbent_ids=incumbent_ids,
        evaluations=evaluations,
    )