from __future__ import annotations

import random

import pytest

from baselines.nsga2_po_runner import (
    NSGA2PORunner,
    NSGA2PORunnerConfig,
    build_pareto_portfolio,
    clone_candidate,
    compare_by_rank_and_crowding,
    crowding_distance_for_front,
    evaluate_candidate_full,
    evaluate_population_full,
    front_rank_map,
    get_pareto_candidates_from_population,
    make_candidate_from_prompt,
    nondominated_sort,
    nsga2_environmental_selection,
    nsga2_rank_and_crowding,
    run_nsga2_po,
    select_two_distinct_parents,
    tournament_select_parent,
)
from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
from heal_capo.objectives import ObjectiveEvaluator


class SimpleObjectiveEvaluator(ObjectiveEvaluator):
    """
    Deterministic evaluator for NSGA-II-PO tests.

    It creates predictable trade-offs:
      - prompts containing "accurate" get higher performance but higher cost
      - prompts containing "cheap" get lower cost
      - prompts containing "safe" get lower risk/fairness risk
    """

    def evaluate(self, candidate: PromptCandidate, data):
        text = candidate.instruction.lower()

        performance = 0.6
        cost = 10.0
        risk = 0.4
        fairness_risk = 0.3

        if "accurate" in text:
            performance += 0.3
            cost += 5.0
            risk -= 0.2

        if "cheap" in text:
            cost -= 5.0

        if "safe" in text:
            risk -= 0.2
            fairness_risk -= 0.2

        performance = min(1.0, max(0.0, performance))
        cost = max(1.0, cost)
        risk = min(1.0, max(0.0, risk))
        fairness_risk = min(1.0, max(0.0, fairness_risk))

        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            performance=performance,
            cost=cost,
            risk=risk,
            fairness_risk=fairness_risk,
            drift=0.0,
            n_examples=len(data),
            details={
                "evaluator": "simple_test",
                "total": len(data),
            },
        )


def make_candidate(
    candidate_id: str,
    instruction: str,
) -> PromptCandidate:
    candidate = PromptCandidate(
        instruction=instruction,
        metadata={
            "method": candidate_id,
            "category": "test",
            "dataset": "subj",
            "task_type": "classification",
        },
    )
    candidate.candidate_id = candidate_id
    return candidate


def make_result(
    candidate_id: str,
    performance: float,
    cost: float,
    risk: float,
    fairness_risk: float,
) -> EvaluationResult:
    return EvaluationResult(
        candidate_id=candidate_id,
        performance=performance,
        cost=cost,
        risk=risk,
        fairness_risk=fairness_risk,
        drift=0.0,
        n_examples=4,
        details={},
    )


def sample_population() -> list[PromptCandidate]:
    return [
        make_candidate("a", "accurate prompt"),
        make_candidate("b", "cheap prompt"),
        make_candidate("c", "safe prompt"),
        make_candidate("d", "accurate safe prompt"),
    ]


def sample_dev_data() -> list[dict]:
    return [
        {"text": "The movie was released in 1999.", "label": "objective"},
        {"text": "The acting is wonderful.", "label": "subjective"},
        {"text": "The book has twelve chapters.", "label": "objective"},
        {"text": "The story feels dull.", "label": "subjective"},
    ]


def test_make_candidate_from_prompt_sets_metadata():
    candidate = make_candidate_from_prompt(
        prompt="Classify.",
        candidate_id="p1",
        category="initial",
        dataset="subj",
        task_type="classification",
    )

    assert candidate.candidate_id == "p1"
    assert candidate.instruction == "Classify."
    assert candidate.metadata["method"] == "p1"
    assert candidate.metadata["source"] == "nsga2_po"
    assert candidate.metadata["dataset"] == "subj"


def test_clone_candidate_preserves_id_and_metadata():
    original = make_candidate("a", "Prompt A")
    original.examples.append(("x", "y"))
    original.parent_ids.append("p0")

    cloned = clone_candidate(original)

    assert cloned.candidate_id == original.candidate_id
    assert cloned.instruction == original.instruction
    assert cloned.metadata == original.metadata
    assert cloned.examples == original.examples
    assert cloned.parent_ids == original.parent_ids
    assert cloned is not original


def test_evaluate_candidate_full_sets_full_dev_details():
    candidate = make_candidate("a", "accurate prompt")
    evaluator = SimpleObjectiveEvaluator()

    result = evaluate_candidate_full(
        candidate=candidate,
        evaluator=evaluator,
        dev_data=sample_dev_data(),
    )

    assert result.candidate_id == "a"
    assert result.details["evaluation_mode"] == "full_dev"
    assert result.details["method"] == "a"
    assert result.details["source"] == "nsga2_po"
    assert result.performance > 0.0


def test_evaluate_population_full_adds_all_candidates():
    population = sample_population()
    evaluator = SimpleObjectiveEvaluator()

    portfolio = evaluate_population_full(
        population=population,
        evaluator=evaluator,
        dev_data=sample_dev_data(),
    )

    assert len(portfolio.candidates) == 4
    assert len(portfolio.evaluations) == 4
    assert set(portfolio.evaluations.keys()) == {"a", "b", "c", "d"}


def test_evaluate_population_full_uses_cache_existing_evaluations():
    population = sample_population()
    evaluator = SimpleObjectiveEvaluator()
    portfolio = PromptPortfolio()

    first = evaluate_population_full(
        population=population,
        evaluator=evaluator,
        dev_data=sample_dev_data(),
        portfolio=portfolio,
    )
    second = evaluate_population_full(
        population=population,
        evaluator=evaluator,
        dev_data=sample_dev_data(),
        portfolio=first,
    )

    assert len(second.evaluations) == 4


def test_nondominated_sort_creates_ordered_fronts():
    evaluations = {
        "a": make_result("a", 0.9, 10.0, 0.1, 0.1),
        "b": make_result("b", 0.8, 12.0, 0.2, 0.2),
        "c": make_result("c", 0.7, 5.0, 0.3, 0.3),
    }

    fronts = nondominated_sort(["a", "b", "c"], evaluations)

    assert "b" in fronts[-1]
    assert "a" in fronts[0]
    assert "c" in fronts[0]


def test_front_rank_map():
    ranks = front_rank_map([["a", "c"], ["b"]])

    assert ranks["a"] == 0
    assert ranks["c"] == 0
    assert ranks["b"] == 1


def test_crowding_distance_for_front_returns_values():
    evaluations = {
        "a": make_result("a", 0.9, 10.0, 0.1, 0.1),
        "b": make_result("b", 0.8, 12.0, 0.2, 0.2),
        "c": make_result("c", 0.7, 5.0, 0.3, 0.3),
    }

    distances = crowding_distance_for_front(["a", "b", "c"], evaluations)

    assert set(distances) == {"a", "b", "c"}


def test_nsga2_rank_and_crowding_returns_ranks_distances_fronts():
    evaluations = {
        "a": make_result("a", 0.9, 10.0, 0.1, 0.1),
        "b": make_result("b", 0.8, 12.0, 0.2, 0.2),
        "c": make_result("c", 0.7, 5.0, 0.3, 0.3),
    }

    ranks, distances, fronts = nsga2_rank_and_crowding(
        candidate_ids=["a", "b", "c"],
        evaluations=evaluations,
    )

    assert ranks["a"] == 0
    assert ranks["c"] == 0
    assert ranks["b"] >= 1
    assert set(distances) == {"a", "b", "c"}
    assert fronts


def test_compare_by_rank_and_crowding_prefers_lower_rank():
    left = make_candidate("a", "Prompt A")
    right = make_candidate("b", "Prompt B")

    winner = compare_by_rank_and_crowding(
        left=left,
        right=right,
        ranks={"a": 0, "b": 1},
        distances={"a": 0.0, "b": 999.0},
        rng=random.Random(0),
    )

    assert winner.candidate_id == "a"


def test_compare_by_rank_and_crowding_prefers_higher_distance_same_rank():
    left = make_candidate("a", "Prompt A")
    right = make_candidate("b", "Prompt B")

    winner = compare_by_rank_and_crowding(
        left=left,
        right=right,
        ranks={"a": 0, "b": 0},
        distances={"a": 0.1, "b": 0.9},
        rng=random.Random(0),
    )

    assert winner.candidate_id == "b"


def test_tournament_select_parent_returns_decision():
    population = sample_population()
    ranks = {"a": 0, "b": 1, "c": 1, "d": 0}
    distances = {"a": 0.5, "b": 0.1, "c": 0.2, "d": 0.7}

    winner, decision = tournament_select_parent(
        population=population,
        ranks=ranks,
        distances=distances,
        rng=random.Random(0),
    )

    assert winner.candidate_id in {"a", "b", "c", "d"}
    assert "winner_id" in decision
    assert "loser_id" in decision


def test_tournament_select_parent_requires_two_candidates():
    with pytest.raises(ValueError):
        tournament_select_parent(
            population=[make_candidate("a", "Prompt A")],
            ranks={"a": 0},
            distances={"a": 0.0},
            rng=random.Random(0),
        )


def test_select_two_distinct_parents_returns_distinct():
    population = sample_population()
    ranks = {"a": 0, "b": 1, "c": 1, "d": 0}
    distances = {"a": 0.5, "b": 0.1, "c": 0.2, "d": 0.7}

    p1, p2, decisions = select_two_distinct_parents(
        population=population,
        ranks=ranks,
        distances=distances,
        rng=random.Random(0),
    )

    assert p1.candidate_id != p2.candidate_id
    assert len(decisions) == 2


def test_nsga2_environmental_selection_keeps_population_size():
    population = sample_population()
    evaluations = {
        "a": make_result("a", 0.9, 10.0, 0.1, 0.1),
        "b": make_result("b", 0.8, 12.0, 0.2, 0.2),
        "c": make_result("c", 0.7, 5.0, 0.3, 0.3),
        "d": make_result("d", 0.95, 15.0, 0.05, 0.05),
    }

    selected, decision = nsga2_environmental_selection(
        candidates=population,
        evaluations=evaluations,
        population_size=2,
        rng=random.Random(0),
    )

    assert len(selected) == 2
    assert len(decision["selected_ids"]) == 2
    assert len(decision["removed_ids"]) == 2
    assert decision["reason"] == "nsga2_environmental_selection"


def test_nsga2_environmental_selection_rejects_non_positive_population_size():
    with pytest.raises(ValueError):
        nsga2_environmental_selection(
            candidates=sample_population(),
            evaluations={},
            population_size=0,
        )


def test_get_pareto_candidates_from_population():
    population = sample_population()
    portfolio = PromptPortfolio()

    evaluations = {
        "a": make_result("a", 0.9, 10.0, 0.1, 0.1),
        "b": make_result("b", 0.8, 12.0, 0.2, 0.2),
        "c": make_result("c", 0.7, 5.0, 0.3, 0.3),
        "d": make_result("d", 0.95, 15.0, 0.05, 0.05),
    }

    for candidate in population:
        portfolio.add(candidate, evaluations[candidate.candidate_id])

    pareto_candidates = get_pareto_candidates_from_population(
        population=population,
        portfolio=portfolio,
    )

    pareto_ids = {candidate.candidate_id for candidate in pareto_candidates}

    assert "b" not in pareto_ids
    assert pareto_ids


def test_build_pareto_portfolio():
    population = sample_population()
    portfolio = PromptPortfolio()

    evaluations = {
        "a": make_result("a", 0.9, 10.0, 0.1, 0.1),
        "b": make_result("b", 0.8, 12.0, 0.2, 0.2),
        "c": make_result("c", 0.7, 5.0, 0.3, 0.3),
        "d": make_result("d", 0.95, 15.0, 0.05, 0.05),
    }

    for candidate in population:
        portfolio.add(candidate, evaluations[candidate.candidate_id])

    pareto_portfolio = build_pareto_portfolio(portfolio)

    assert len(pareto_portfolio.candidates) >= 1
    assert "b" not in pareto_portfolio.evaluations


def test_runner_executes_generations_and_returns_outputs():
    population = sample_population()
    evaluator = SimpleObjectiveEvaluator()

    config = NSGA2PORunnerConfig(
        population_size=4,
        max_generations=2,
        offspring_per_generation=2,
        random_seed=0,
        use_meta_llm=False,
        metadata={
            "dataset": "subj",
            "model_id": "toy",
        },
    )

    runner = NSGA2PORunner(
        config=config,
        evaluator=evaluator,
        dev_data=sample_dev_data(),
        task_description="Classify as subjective or objective.",
        rng=random.Random(0),
    )

    result = runner.run(initial_population=population)

    assert len(result.final_population) == 4
    assert len(result.all_portfolio.evaluations) >= 4
    assert len(result.pareto_portfolio.candidates) >= 1
    assert result.summary["method"] == "nsga2_po"
    assert result.summary["uses_full_dev_evaluation"] is True
    assert result.summary["uses_intensification"] is False
    assert result.summary["uses_nsga2_rank_cd"] is True
    assert any(event["event_type"] == "offspring_created" for event in result.events)
    assert any(
        event["event_type"] == "environmental_selection"
        for event in result.events
    )


def test_run_nsga2_po_function_wrapper():
    population = sample_population()
    evaluator = SimpleObjectiveEvaluator()

    result = run_nsga2_po(
        initial_population=population,
        evaluator=evaluator,
        dev_data=sample_dev_data(),
        config=NSGA2PORunnerConfig(
            population_size=4,
            max_generations=1,
            offspring_per_generation=2,
            random_seed=0,
        ),
        task_description="Classify as subjective or objective.",
        rng=random.Random(0),
    )

    assert result.summary["method"] == "nsga2_po"
    assert len(result.final_population) == 4


def test_runner_requires_at_least_two_initial_candidates():
    evaluator = SimpleObjectiveEvaluator()
    runner = NSGA2PORunner(
        config=NSGA2PORunnerConfig(),
        evaluator=evaluator,
        dev_data=sample_dev_data(),
    )

    with pytest.raises(ValueError):
        runner.run(initial_population=[make_candidate("a", "Prompt A")])