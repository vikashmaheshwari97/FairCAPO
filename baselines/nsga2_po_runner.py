from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any

from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
from heal_capo.objectives import ObjectiveEvaluator
from heal_capo.optimizers.budget_allocator import BudgetAllocator
from heal_capo.optimizers.evolutionary_ops import (
    EvolutionaryOpsConfig,
    EvolutionaryPromptOps,
)
from heal_capo.optimizers.parent_selection import (
    ParentSelectionConfig,
    crowding_distance,
    dominates,
)
from heal_capo.pareto import non_dominated_ids, sort_pareto_results


@dataclass
class NSGA2PORunnerConfig:
    """
    NSGA-II-PO baseline.

    Paper-aligned simplified structure:
      - initialize population from initial prompts
      - evaluate every candidate on full Ddev
      - select parents by NSGA-II binary tournament:
          lower front rank, then higher crowding distance
      - create offspring using CAPO-style crossover/mutation
      - evaluate all offspring on full Ddev
      - environmental selection by NDS + crowding distance
      - return final population and Pareto portfolio

    Unlike MO-CAPO/HEAL-CAPO budgeted optimization, NSGA-II-PO does not use:
      - block-wise intensification
      - early challenger rejection
      - incumbent advancement
    """

    population_size: int = 10
    max_generations: int = 5
    offspring_per_generation: int = 4
    random_seed: int = 0

    mutation_probability: float = 1.0
    crossover_probability: float = 1.0
    mutate_after_crossover: bool = True
    use_meta_llm: bool = False

    # Optional evaluation budget (for a FAIR comparison with budgeted FairCAPO).
    # When ``max_budget`` is set (> 0) the runner stops generating/evaluating new
    # candidates once the budget is spent. ``budget_unit`` mirrors the budgeted
    # MO-CAPO runner: "tokens" meters raw input+output tokens, "cost" meters
    # weighted cost. Default (None) = no budget, the original full-search NSGA-II.
    max_budget: float | None = None
    allow_overspend: bool = False
    budget_unit: str = "tokens"

    # Keep our trust-aware objective vector by default.
    objectives: tuple[str, ...] = (
        "performance",
        "cost",
        "risk",
        "fairness_risk",
    )

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NSGA2PORunResult:
    all_portfolio: PromptPortfolio
    final_population: list[PromptCandidate]
    pareto_portfolio: PromptPortfolio
    events: list[dict]
    summary: dict
    budget_summary: dict | None = None


def _json_default(obj: Any):
    try:
        return obj.item()
    except AttributeError:
        return str(obj)


def clone_candidate(candidate: PromptCandidate) -> PromptCandidate:
    cloned = PromptCandidate(
        instruction=candidate.instruction,
        examples=list(candidate.examples),
        parent_ids=list(candidate.parent_ids),
        metadata=dict(candidate.metadata),
    )
    cloned.candidate_id = candidate.candidate_id
    return cloned


def make_candidate_from_prompt(
    prompt: str,
    candidate_id: str,
    category: str = "initial",
    dataset: str = "",
    task_type: str = "",
) -> PromptCandidate:
    candidate = PromptCandidate(
        instruction=prompt,
        metadata={
            "method": candidate_id,
            "category": category,
            "dataset": dataset,
            "task_type": task_type,
            "source": "nsga2_po",
        },
    )
    candidate.candidate_id = candidate_id
    return candidate


def evaluate_candidate_full(
    candidate: PromptCandidate,
    evaluator: ObjectiveEvaluator,
    dev_data: list[dict],
) -> EvaluationResult:
    """
    NSGA-II-PO evaluates each candidate on the full development set.
    """
    result = evaluator.evaluate(candidate, dev_data)
    result.candidate_id = candidate.candidate_id

    if result.details is None:
        result.details = {}

    result.details["evaluation_mode"] = "full_dev"
    result.details["method"] = candidate.metadata.get("method")
    result.details["source"] = "nsga2_po"

    return result


def evaluate_population_full(
    population: list[PromptCandidate],
    evaluator: ObjectiveEvaluator,
    dev_data: list[dict],
    portfolio: PromptPortfolio | None = None,
) -> PromptPortfolio:
    portfolio = portfolio or PromptPortfolio()

    for candidate in population:
        if candidate.candidate_id in portfolio.evaluations:
            continue

        result = evaluate_candidate_full(
            candidate=candidate,
            evaluator=evaluator,
            dev_data=dev_data,
        )
        portfolio.add(candidate, result)

    return portfolio


def nondominated_sort(
    candidate_ids: list[str],
    evaluations: dict[str, EvaluationResult],
    parent_config: ParentSelectionConfig | None = None,
) -> list[list[str]]:
    """
    Standard non-dominated sorting.

    Front 0 is best.
    """
    parent_config = parent_config or ParentSelectionConfig()
    remaining = [cid for cid in candidate_ids if cid in evaluations]
    fronts: list[list[str]] = []

    while remaining:
        front = []

        for cid in remaining:
            candidate_result = evaluations[cid]
            is_dominated = False

            for other_id in remaining:
                if cid == other_id:
                    continue

                other_result = evaluations[other_id]

                if dominates(other_result, candidate_result, parent_config):
                    is_dominated = True
                    break

            if not is_dominated:
                front.append(cid)

        if not front:
            fronts.append(sorted(remaining))
            break

        front = sorted(front)
        fronts.append(front)
        remaining = [cid for cid in remaining if cid not in front]

    return fronts


def front_rank_map(fronts: list[list[str]]) -> dict[str, int]:
    ranks = {}

    for rank, front in enumerate(fronts):
        for cid in front:
            ranks[cid] = rank

    return ranks


def crowding_distance_for_front(
    front: list[str],
    evaluations: dict[str, EvaluationResult],
    parent_config: ParentSelectionConfig | None = None,
) -> dict[str, float]:
    parent_config = parent_config or ParentSelectionConfig()

    results = [
        evaluations[cid]
        for cid in front
        if cid in evaluations
    ]

    return crowding_distance(results, parent_config)


def nsga2_rank_and_crowding(
    candidate_ids: list[str],
    evaluations: dict[str, EvaluationResult],
    parent_config: ParentSelectionConfig | None = None,
) -> tuple[dict[str, int], dict[str, float], list[list[str]]]:
    """
    Return NSGA-II front ranks and crowding distances.
    """
    parent_config = parent_config or ParentSelectionConfig()
    fronts = nondominated_sort(candidate_ids, evaluations, parent_config)
    ranks = front_rank_map(fronts)
    distances: dict[str, float] = {}

    for front in fronts:
        distances.update(
            crowding_distance_for_front(
                front=front,
                evaluations=evaluations,
                parent_config=parent_config,
            )
        )

    return ranks, distances, fronts


def compare_by_rank_and_crowding(
    left: PromptCandidate,
    right: PromptCandidate,
    ranks: dict[str, int],
    distances: dict[str, float],
    rng: random.Random,
) -> PromptCandidate:
    left_rank = ranks.get(left.candidate_id, math.inf)
    right_rank = ranks.get(right.candidate_id, math.inf)

    if left_rank < right_rank:
        return left

    if right_rank < left_rank:
        return right

    left_distance = distances.get(left.candidate_id, 0.0)
    right_distance = distances.get(right.candidate_id, 0.0)

    if left_distance > right_distance:
        return left

    if right_distance > left_distance:
        return right

    return rng.choice([left, right])


def tournament_select_parent(
    population: list[PromptCandidate],
    ranks: dict[str, int],
    distances: dict[str, float],
    rng: random.Random,
) -> tuple[PromptCandidate, dict]:
    if len(population) < 2:
        raise ValueError("Need at least two candidates for tournament selection.")

    left, right = rng.sample(population, 2)
    winner = compare_by_rank_and_crowding(
        left=left,
        right=right,
        ranks=ranks,
        distances=distances,
        rng=rng,
    )
    loser = right if winner is left else left

    return winner, {
        "winner_id": winner.candidate_id,
        "loser_id": loser.candidate_id,
        "left_id": left.candidate_id,
        "right_id": right.candidate_id,
        "winner_rank": ranks.get(winner.candidate_id),
        "loser_rank": ranks.get(loser.candidate_id),
        "winner_crowding_distance": distances.get(winner.candidate_id),
        "loser_crowding_distance": distances.get(loser.candidate_id),
    }


def select_two_distinct_parents(
    population: list[PromptCandidate],
    ranks: dict[str, int],
    distances: dict[str, float],
    rng: random.Random,
) -> tuple[PromptCandidate, PromptCandidate, list[dict]]:
    first, first_decision = tournament_select_parent(
        population=population,
        ranks=ranks,
        distances=distances,
        rng=rng,
    )

    second = first
    second_decision = first_decision

    attempts = 0

    while second.candidate_id == first.candidate_id and attempts < 20:
        second, second_decision = tournament_select_parent(
            population=population,
            ranks=ranks,
            distances=distances,
            rng=rng,
        )
        attempts += 1

    if second.candidate_id == first.candidate_id:
        alternatives = [
            candidate
            for candidate in population
            if candidate.candidate_id != first.candidate_id
        ]
        second = rng.choice(alternatives)
        second_decision = {
            "winner_id": second.candidate_id,
            "loser_id": first.candidate_id,
            "reason": "forced_distinct_parent",
        }

    return first, second, [first_decision, second_decision]


def nsga2_environmental_selection(
    candidates: list[PromptCandidate],
    evaluations: dict[str, EvaluationResult],
    population_size: int,
    parent_config: ParentSelectionConfig | None = None,
    rng: random.Random | None = None,
) -> tuple[list[PromptCandidate], dict]:
    """
    Standard NSGA-II environmental selection:
      - add complete fronts until population would overflow
      - for partial front, choose highest crowding distance
    """
    if population_size <= 0:
        raise ValueError("population_size must be positive.")

    rng = rng or random.Random()
    parent_config = parent_config or ParentSelectionConfig()

    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    candidate_ids = [
        candidate.candidate_id
        for candidate in candidates
        if candidate.candidate_id in evaluations
    ]

    fronts = nondominated_sort(candidate_ids, evaluations, parent_config)

    selected_ids: list[str] = []
    partial_front_used = False

    for front in fronts:
        if len(selected_ids) + len(front) <= population_size:
            selected_ids.extend(front)
            continue

        remaining_slots = population_size - len(selected_ids)

        if remaining_slots <= 0:
            break

        distances = crowding_distance_for_front(
            front=front,
            evaluations=evaluations,
            parent_config=parent_config,
        )

        ordered = sorted(
            front,
            key=lambda cid: (
                distances.get(cid, 0.0),
                rng.random(),
            ),
            reverse=True,
        )

        selected_ids.extend(ordered[:remaining_slots])
        partial_front_used = True
        break

    selected = [
        candidates_by_id[cid]
        for cid in selected_ids
        if cid in candidates_by_id
    ]

    removed_ids = [
        cid
        for cid in candidate_ids
        if cid not in set(selected_ids)
    ]

    decision = {
        "reason": "nsga2_environmental_selection",
        "fronts": fronts,
        "selected_ids": selected_ids,
        "removed_ids": removed_ids,
        "partial_front_used": partial_front_used,
        "population_size": population_size,
    }

    return selected, decision


def get_pareto_candidates_from_population(
    population: list[PromptCandidate],
    portfolio: PromptPortfolio,
) -> list[PromptCandidate]:
    population_ids = {candidate.candidate_id for candidate in population}

    results = [
        result
        for candidate_id, result in portfolio.evaluations.items()
        if candidate_id in population_ids
    ]

    if not results:
        return []

    pareto_ids = set(non_dominated_ids(results))

    return [
        candidate
        for candidate in population
        if candidate.candidate_id in pareto_ids
    ]


def build_pareto_portfolio(
    portfolio: PromptPortfolio,
) -> PromptPortfolio:
    pareto_ids = set(non_dominated_ids(portfolio.evaluations.values()))
    pareto_portfolio = PromptPortfolio()

    sorted_results = sort_pareto_results(
        [
            result
            for result in portfolio.evaluations.values()
            if result.candidate_id in pareto_ids
        ]
    )

    for result in sorted_results:
        candidate = portfolio.get(result.candidate_id)
        pareto_portfolio.add(candidate, result)

    return pareto_portfolio


class NSGA2PORunner:
    """
    Direct NSGA-II-PO baseline runner.

    This is intentionally different from MO-CAPO:
      - no block evaluator
      - no budgeted intensification
      - no incumbent advancement
      - full Ddev evaluation for every candidate
      - selection/survival by front rank and crowding distance
    """

    def __init__(
        self,
        config: NSGA2PORunnerConfig,
        evaluator: ObjectiveEvaluator,
        dev_data: list[dict],
        task_description: str = "",
        meta_llm: Any | None = None,
        rng: random.Random | None = None,
    ):
        self.config = config
        self.evaluator = evaluator
        self.dev_data = dev_data
        self.task_description = task_description
        self.meta_llm = meta_llm
        self.rng = rng or random.Random(config.random_seed)

        self.parent_config = ParentSelectionConfig(
            objectives=tuple(config.objectives),
            random_seed=config.random_seed,
        )

        self.evolutionary_ops = EvolutionaryPromptOps(
            config=EvolutionaryOpsConfig(
                random_seed=config.random_seed,
                mutation_probability=config.mutation_probability,
                crossover_probability=config.crossover_probability,
                preserve_output_format=True,
                require_prompt_tags=True,
            ),
            meta_llm=meta_llm if config.use_meta_llm else None,
            rng=self.rng,
        )

        # Optional evaluation budget. The allocator is built with
        # allow_overspend=True so recording never raises mid-search; we enforce
        # the cap ourselves via ``_budget_exhausted`` (which honours the config's
        # allow_overspend). This lets the candidate that crosses the threshold
        # finish (its LLM cost was already spent) and then stops the search,
        # mirroring how the budgeted MO-CAPO runner checks ``exhausted`` before
        # each new evaluation.
        self.budget_allocator: BudgetAllocator | None = None
        if config.max_budget is not None and config.max_budget > 0:
            self.budget_allocator = BudgetAllocator(
                max_budget=config.max_budget,
                allow_overspend=True,
                budget_unit=config.budget_unit,
            )

    def _budget_exhausted(self) -> bool:
        """True when the (optional) budget is set, not overspendable, and spent."""
        if self.budget_allocator is None or self.config.allow_overspend:
            return False

        return self.budget_allocator.used_budget >= self.budget_allocator.max_budget

    def _record_budget(self, result: EvaluationResult) -> None:
        if self.budget_allocator is None:
            return

        details = result.details or {}
        self.budget_allocator.record(
            candidate_id=result.candidate_id,
            cost=result.cost,
            input_tokens=float(details.get("input_tokens", 0.0) or 0.0),
            output_tokens=float(details.get("output_tokens", 0.0) or 0.0),
        )

    def _evaluate_under_budget(
        self,
        candidates: list[PromptCandidate],
        portfolio: PromptPortfolio,
        events: list[dict],
        generation: int,
        phase: str,
    ) -> bool:
        """
        Evaluate ``candidates`` on the full dev set, charging the budget.

        Returns True if all candidates were evaluated, or False if the budget
        was exhausted partway (a ``budget_stop`` event is appended in that case).
        """
        for candidate in candidates:
            if candidate.candidate_id in portfolio.evaluations:
                continue

            if self._budget_exhausted():
                events.append(
                    {
                        "event_type": "budget_stop",
                        "generation": generation,
                        "phase": phase,
                        "candidate_id": candidate.candidate_id,
                        "budget_used": self.budget_allocator.used_budget,
                        "max_budget": self.budget_allocator.max_budget,
                        "budget_unit": self.budget_allocator.budget_unit,
                    }
                )
                return False

            result = evaluate_candidate_full(
                candidate=candidate,
                evaluator=self.evaluator,
                dev_data=self.dev_data,
            )
            portfolio.add(candidate, result)
            self._record_budget(result)

        return True

    def run(
        self,
        initial_population: list[PromptCandidate],
    ) -> NSGA2PORunResult:
        if len(initial_population) < 2:
            raise ValueError("NSGA-II-PO requires at least two initial candidates.")

        population = [
            clone_candidate(candidate)
            for candidate in initial_population[: self.config.population_size]
        ]

        portfolio = PromptPortfolio()
        events: list[dict] = []

        self._evaluate_under_budget(
            candidates=population,
            portfolio=portfolio,
            events=events,
            generation=0,
            phase="initial_population",
        )

        # Under a budget the initial evaluation may stop partway; keep only
        # candidates that were actually evaluated so selection operates on a
        # consistent population.
        population = [
            candidate
            for candidate in population
            if candidate.candidate_id in portfolio.evaluations
        ]

        ranks, distances, fronts = nsga2_rank_and_crowding(
            candidate_ids=[candidate.candidate_id for candidate in population],
            evaluations=portfolio.evaluations,
            parent_config=self.parent_config,
        )

        events.append(
            {
                "event_type": "initial_population_evaluated",
                "generation": 0,
                "num_population": len(population),
                "num_evaluated": len(portfolio.evaluations),
                "fronts": json.dumps(fronts),
            }
        )

        for generation in range(1, self.config.max_generations + 1):
            # Stop generating new candidates once the shared budget is spent.
            if self._budget_exhausted() or len(population) < 2:
                events.append(
                    {
                        "event_type": "budget_stop",
                        "generation": generation,
                        "phase": "generation_start",
                        "budget_used": (
                            self.budget_allocator.used_budget
                            if self.budget_allocator is not None
                            else None
                        ),
                        "max_budget": (
                            self.budget_allocator.max_budget
                            if self.budget_allocator is not None
                            else None
                        ),
                    }
                )
                break

            offspring: list[PromptCandidate] = []

            ranks, distances, fronts = nsga2_rank_and_crowding(
                candidate_ids=[candidate.candidate_id for candidate in population],
                evaluations=portfolio.evaluations,
                parent_config=self.parent_config,
            )

            for offspring_index in range(self.config.offspring_per_generation):
                mother, father, parent_decisions = select_two_distinct_parents(
                    population=population,
                    ranks=ranks,
                    distances=distances,
                    rng=self.rng,
                )

                op_result = self.evolutionary_ops.crossover(
                    mother=mother,
                    father=father,
                    task_description=self.task_description,
                )

                if self.config.mutate_after_crossover:
                    op_result = self.evolutionary_ops.mutate(
                        parent=op_result.candidate,
                        task_description=self.task_description,
                    )

                child = op_result.candidate
                child.metadata["source"] = "nsga2_po"
                child.metadata["method"] = "nsga2_po_offspring"
                child.metadata["category"] = "nsga2_po"
                child.metadata["generation"] = generation
                child.metadata["offspring_index"] = offspring_index

                offspring.append(child)

                events.append(
                    {
                        "event_type": "offspring_created",
                        "generation": generation,
                        "offspring_index": offspring_index,
                        "candidate_id": child.candidate_id,
                        "operator": op_result.operator,
                        "parent_ids": json.dumps(op_result.parent_ids),
                        "used_meta_llm": op_result.used_meta_llm,
                        "parent_selection": json.dumps(
                            parent_decisions,
                            default=_json_default,
                        ),
                    }
                )

            self._evaluate_under_budget(
                candidates=offspring,
                portfolio=portfolio,
                events=events,
                generation=generation,
                phase="offspring",
            )

            # Only offspring that were actually evaluated can compete in
            # environmental selection (the rest were skipped at the budget wall).
            offspring = [
                child
                for child in offspring
                if child.candidate_id in portfolio.evaluations
            ]

            combined = population + offspring

            population, environmental_decision = nsga2_environmental_selection(
                candidates=combined,
                evaluations=portfolio.evaluations,
                population_size=self.config.population_size,
                parent_config=self.parent_config,
                rng=self.rng,
            )

            events.append(
                {
                    "event_type": "environmental_selection",
                    "generation": generation,
                    "selected_ids": json.dumps(
                        environmental_decision["selected_ids"]
                    ),
                    "removed_ids": json.dumps(
                        environmental_decision["removed_ids"]
                    ),
                    "fronts": json.dumps(environmental_decision["fronts"]),
                    "partial_front_used": environmental_decision[
                        "partial_front_used"
                    ],
                    "num_population": len(population),
                    "num_evaluated": len(portfolio.evaluations),
                }
            )

        pareto_portfolio = build_pareto_portfolio(portfolio)
        final_pareto_population = get_pareto_candidates_from_population(
            population=population,
            portfolio=portfolio,
        )

        budget_summary = (
            self.budget_allocator.summary()
            if self.budget_allocator is not None
            else None
        )
        if budget_summary is not None:
            budget_summary["evaluator"] = "lmstudio"
            budget_summary["model_id"] = self.config.metadata.get("model_id")
            budget_summary["algorithm"] = "nsga2_po"

        summary = {
            "method": "nsga2_po",
            "population_size": self.config.population_size,
            "max_generations": self.config.max_generations,
            "offspring_per_generation": self.config.offspring_per_generation,
            "num_population": len(population),
            "num_evaluated_candidates": len(portfolio.evaluations),
            "num_pareto_candidates": len(pareto_portfolio.candidates),
            "num_final_population_pareto_candidates": len(final_pareto_population),
            "random_seed": self.config.random_seed,
            "uses_full_dev_evaluation": True,
            "uses_intensification": False,
            "uses_incumbent_advancement": False,
            "uses_nsga2_rank_cd": True,
            "max_budget": (
                self.budget_allocator.max_budget
                if self.budget_allocator is not None
                else None
            ),
            "budget_unit": (
                self.budget_allocator.budget_unit
                if self.budget_allocator is not None
                else None
            ),
            "used_budget": (
                self.budget_allocator.used_budget
                if self.budget_allocator is not None
                else None
            ),
            **self.config.metadata,
        }

        return NSGA2PORunResult(
            all_portfolio=portfolio,
            final_population=population,
            pareto_portfolio=pareto_portfolio,
            events=events,
            summary=summary,
            budget_summary=budget_summary,
        )


def run_nsga2_po(
    initial_population: list[PromptCandidate],
    evaluator: ObjectiveEvaluator,
    dev_data: list[dict],
    config: NSGA2PORunnerConfig | None = None,
    task_description: str = "",
    meta_llm: Any | None = None,
    rng: random.Random | None = None,
) -> NSGA2PORunResult:
    runner = NSGA2PORunner(
        config=config or NSGA2PORunnerConfig(),
        evaluator=evaluator,
        dev_data=dev_data,
        task_description=task_description,
        meta_llm=meta_llm,
        rng=rng,
    )

    return runner.run(initial_population=initial_population)