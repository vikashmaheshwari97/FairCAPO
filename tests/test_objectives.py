import pytest

from heal_capo.core import PromptCandidate, EvaluationResult
from heal_capo.objectives import (
    StaticPredictionObjectiveEvaluator,
    ToyObjectiveEvaluator,
)


def test_toy_objective_returns_fairness_risk():
    candidate = PromptCandidate(
        instruction="Classify the input. Do not infer ability from gender or race."
    )
    data = [{"text": "Example", "label": "objective"}]

    evaluator = ToyObjectiveEvaluator()
    result = evaluator.evaluate(candidate, data)

    assert isinstance(result, EvaluationResult)
    assert result.candidate_id == candidate.candidate_id
    assert result.performance > 0
    assert result.cost > 0
    assert result.risk >= 0
    assert result.fairness_risk >= 0
    assert len(result.objective_vector) == 4
    assert result.details["toy"] is True


def test_toy_objective_fairness_instruction_lowers_fairness_risk():
    plain = PromptCandidate(instruction="Classify the input.")
    fair = PromptCandidate(
        instruction="Classify the input. Do not infer ability from gender or race."
    )
    data = [{"text": "Example", "label": "objective"}]

    evaluator = ToyObjectiveEvaluator()

    plain_result = evaluator.evaluate(plain, data)
    fair_result = evaluator.evaluate(fair, data)

    assert fair_result.fairness_risk < plain_result.fairness_risk


def test_toy_objective_safety_instruction_lowers_risk():
    plain = PromptCandidate(instruction="Answer the question.")
    safe = PromptCandidate(
        instruction="Answer the question. Do not hallucinate. Use context."
    )
    data = [{"text": "Example", "label": "objective"}]

    evaluator = ToyObjectiveEvaluator()

    plain_result = evaluator.evaluate(plain, data)
    safe_result = evaluator.evaluate(safe, data)

    assert safe_result.risk < plain_result.risk


def test_static_prediction_objective_classification_performance_and_risk():
    candidate = PromptCandidate(instruction="Classify the input.")
    data = [
        {
            "prediction": "objective",
            "label": "objective",
            "output": "objective",
        },
        {
            "prediction": "objective",
            "label": "subjective",
            "output": "objective",
        },
        {
            "prediction": "subjective",
            "label": "subjective",
            "output": "subjective",
        },
    ]

    evaluator = StaticPredictionObjectiveEvaluator(
        cost_per_candidate=12.5,
        include_output_risk=False,
        include_group_fairness=False,
        include_bias_language=False,
        include_counterfactual_fairness=False,
    )

    result = evaluator.evaluate(candidate, data)

    assert result.performance == pytest.approx(2 / 3)
    assert result.risk == pytest.approx(1 / 3)
    assert result.cost == 12.5
    assert result.fairness_risk == 0.0
    assert result.n_examples == 3


def test_static_prediction_objective_group_fairness():
    candidate = PromptCandidate(instruction="Classify the input.")
    data = [
        {
            "prediction": "yes",
            "label": "yes",
            "group": "A",
            "output": "yes",
        },
        {
            "prediction": "no",
            "label": "no",
            "group": "A",
            "output": "no",
        },
        {
            "prediction": "yes",
            "label": "no",
            "group": "B",
            "output": "yes",
        },
        {
            "prediction": "yes",
            "label": "no",
            "group": "B",
            "output": "yes",
        },
    ]

    evaluator = StaticPredictionObjectiveEvaluator(
        include_output_risk=False,
        include_group_fairness=True,
        include_bias_language=False,
        include_counterfactual_fairness=False,
    )

    result = evaluator.evaluate(candidate, data)

    assert result.performance == 0.5
    assert result.risk == 0.5
    assert result.fairness_risk == 1.0
    assert result.details["group_accuracy_gap"] == 1.0


def test_static_prediction_objective_counterfactual_fairness():
    candidate = PromptCandidate(instruction="Classify the input.")
    data = [
        {
            "base_prediction": "objective",
            "counterfactual_prediction": "subjective",
            "output": "objective",
        },
        {
            "base_prediction": "subjective",
            "counterfactual_prediction": "subjective",
            "output": "subjective",
        },
    ]

    evaluator = StaticPredictionObjectiveEvaluator(
        include_output_risk=False,
        include_group_fairness=False,
        include_bias_language=False,
        include_counterfactual_fairness=True,
    )

    result = evaluator.evaluate(candidate, data)

    assert result.performance == 0.0
    assert result.risk == 0.0
    assert result.fairness_risk == 0.5
    assert result.details["counterfactual_flip_rate"] == 0.5


def test_static_prediction_objective_bias_language_fairness():
    candidate = PromptCandidate(instruction="Classify the input.")
    data = [
        {
            "output": "This answer is neutral.",
        },
        {
            "output": "This person is less capable because of their gender.",
        },
    ]

    evaluator = StaticPredictionObjectiveEvaluator(
        include_output_risk=False,
        include_group_fairness=False,
        include_bias_language=True,
        include_counterfactual_fairness=False,
    )

    result = evaluator.evaluate(candidate, data)

    assert result.fairness_risk == 0.5
    assert result.details["bias_violation_rate"] == 0.5


def test_static_prediction_objective_uses_max_fairness_risk():
    candidate = PromptCandidate(instruction="Classify the input.")
    data = [
        {
            "prediction": "yes",
            "label": "yes",
            "group": "A",
            "output": "neutral",
            "base_prediction": "yes",
            "counterfactual_prediction": "no",
        },
        {
            "prediction": "no",
            "label": "no",
            "group": "A",
            "output": "neutral",
            "base_prediction": "no",
            "counterfactual_prediction": "no",
        },
        {
            "prediction": "yes",
            "label": "no",
            "group": "B",
            "output": "neutral",
            "base_prediction": "yes",
            "counterfactual_prediction": "yes",
        },
        {
            "prediction": "yes",
            "label": "no",
            "group": "B",
            "output": "neutral",
            "base_prediction": "yes",
            "counterfactual_prediction": "yes",
        },
    ]

    evaluator = StaticPredictionObjectiveEvaluator(
        include_output_risk=False,
        include_group_fairness=True,
        include_bias_language=False,
        include_counterfactual_fairness=True,
        fairness_aggregation="max",
    )

    result = evaluator.evaluate(candidate, data)

    assert result.fairness_risk == 1.0
    assert result.details["fairness_details"]["aggregation"] == "max_fairness_risk"
    assert result.details["fairness_details"]["num_fairness_checks"] == 2


def test_static_prediction_objective_combined_fairness_default():
    candidate = PromptCandidate(instruction="Classify.")

    # Group A perfectly accurate, group B fully wrong -> group gap = 1.0.
    # Counterfactual predictions never flip -> flip rate = 0.0.
    data = [
        {
            "prediction": "yes",
            "label": "yes",
            "group": "A",
            "base_prediction": "yes",
            "counterfactual_prediction": "yes",
        },
        {
            "prediction": "no",
            "label": "no",
            "group": "A",
            "base_prediction": "no",
            "counterfactual_prediction": "no",
        },
        {
            "prediction": "yes",
            "label": "no",
            "group": "B",
            "base_prediction": "yes",
            "counterfactual_prediction": "yes",
        },
        {
            "prediction": "yes",
            "label": "no",
            "group": "B",
            "base_prediction": "yes",
            "counterfactual_prediction": "yes",
        },
    ]

    evaluator = StaticPredictionObjectiveEvaluator(
        include_output_risk=False,
        include_group_fairness=True,
        include_bias_language=False,
        include_counterfactual_fairness=True,
        # combined is the default; the blended risk sits strictly between the
        # zero flip rate and the maximal group gap.
    )

    result = evaluator.evaluate(candidate, data)

    assert result.details["fairness_details"]["method"] == "combined_fairness_risk"
    assert 0.0 < result.fairness_risk < 1.0
    assert result.details["fairness_aggregation"] == "combined"


def test_objective_vector_contains_fairness():
    candidate = PromptCandidate(instruction="Classify.")
    data = [{"text": "Example", "label": "objective"}]

    evaluator = ToyObjectiveEvaluator()
    result = evaluator.evaluate(candidate, data)

    objective_vector = result.objective_vector

    assert len(objective_vector) == 4
    assert objective_vector[0] == -result.performance
    assert objective_vector[1] == result.cost
    assert objective_vector[2] == result.risk
    assert objective_vector[3] == result.fairness_risk