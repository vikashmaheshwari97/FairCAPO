from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence

from .core import PromptCandidate, EvaluationResult
from .fairness import (
    CombinedFairnessConfig,
    FairnessResult,
    evaluate_bias_language,
    evaluate_combined_fairness,
    evaluate_counterfactual_fairness,
    evaluate_group_fairness,
)
from .risk import RiskResult, evaluate_classification_risk, evaluate_output_risk


class ObjectiveEvaluator(ABC):
    """
    Evaluates prompt candidates on performance, cost, risk, fairness, and drift.
    """

    @abstractmethod
    def evaluate(
        self,
        candidate: PromptCandidate,
        data: Sequence[Dict[str, Any]],
    ) -> EvaluationResult:
        raise NotImplementedError


class ToyObjectiveEvaluator(ObjectiveEvaluator):
    """
    Deterministic toy evaluator for development tests.

    This is not a real LLM evaluator. It gives predictable values so that
    HEAL-CAPO components can be tested without running a model.
    """

    def evaluate(
        self,
        candidate: PromptCandidate,
        data: Sequence[Dict[str, Any]],
    ) -> EvaluationResult:
        instruction = candidate.instruction.lower()

        length = len(candidate.instruction.split()) + 20 * len(candidate.examples)

        grounding_bonus = 0.06 if "context" in instruction else 0.0
        example_bonus = 0.02 * len(candidate.examples)

        safety_bonus = 0.08 if "do not hallucinate" in instruction else 0.0
        abstain_bonus = 0.04 if "not enough information" in instruction else 0.0

        fairness_bonus = 0.08 if "do not infer" in instruction else 0.0
        fairness_bonus += 0.06 if "gender" in instruction or "race" in instruction else 0.0
        fairness_bonus += 0.04 if "demographic" in instruction else 0.0

        performance = min(
            0.95,
            0.45 + example_bonus + grounding_bonus,
        )

        risk = max(
            0.01,
            0.35 - safety_bonus - grounding_bonus - abstain_bonus,
        )

        fairness_risk = max(
            0.01,
            0.30 - fairness_bonus,
        )

        cost = length / 100.0

        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            performance=performance,
            cost=cost,
            risk=risk,
            fairness_risk=fairness_risk,
            n_examples=len(data),
            details={
                "toy": True,
                "length": length,
                "grounding_bonus": grounding_bonus,
                "example_bonus": example_bonus,
                "safety_bonus": safety_bonus,
                "abstain_bonus": abstain_bonus,
                "fairness_bonus": fairness_bonus,
            },
        )


class StaticPredictionObjectiveEvaluator(ObjectiveEvaluator):
    """
    Objective evaluator for already-computed predictions.

    This is useful for Phase 2 integration because we can evaluate candidate
    prompts using stored outputs/predictions without calling the LLM again.

    Expected data row format:
      {
        "prediction": "...",
        "label": "...",
        "output": "...",          optional
        "context": "...",         optional
        "group": "...",           optional
      }

    Optional counterfactual format:
      {
        "base_prediction": "...",
        "counterfactual_prediction": "..."
      }
    """

    def __init__(
        self,
        cost_per_candidate: float = 0.0,
        include_output_risk: bool = True,
        include_group_fairness: bool = True,
        include_bias_language: bool = True,
        include_counterfactual_fairness: bool = True,
        fairness_aggregation: str = "combined",
        fairness_config: Optional[CombinedFairnessConfig] = None,
        fairness_debt: float = 0.0,
    ):
        valid_aggregations = {"combined", "max", "mean"}
        if fairness_aggregation not in valid_aggregations:
            raise ValueError(
                f"fairness_aggregation must be one of {sorted(valid_aggregations)}."
            )

        self.cost_per_candidate = cost_per_candidate
        self.include_output_risk = include_output_risk
        self.include_group_fairness = include_group_fairness
        self.include_bias_language = include_bias_language
        self.include_counterfactual_fairness = include_counterfactual_fairness
        self.fairness_aggregation = fairness_aggregation
        self.fairness_config = fairness_config or CombinedFairnessConfig()
        self.fairness_debt = float(fairness_debt)

    def evaluate(
        self,
        candidate: PromptCandidate,
        data: Sequence[Dict[str, Any]],
    ) -> EvaluationResult:
        predictions = [
            str(row.get("prediction", ""))
            for row in data
            if "prediction" in row
        ]
        labels = [
            str(row.get("label", ""))
            for row in data
            if "label" in row
        ]

        outputs = [
            str(row.get("output", row.get("prediction", "")))
            for row in data
        ]

        contexts = [
            str(row.get("context", ""))
            for row in data
            if "context" in row
        ]

        groups = [
            str(row.get("group", ""))
            for row in data
            if "group" in row
        ]

        performance = 0.0
        risk_result: Optional[RiskResult] = None

        if predictions and labels and len(predictions) == len(labels):
            classification_risk = evaluate_classification_risk(
                predictions=predictions,
                labels=labels,
            )
            performance = 1.0 - classification_risk.wrong_answer_rate
            risk_result = classification_risk

        if self.include_output_risk:
            output_risk = evaluate_output_risk(
                outputs=outputs,
                predictions=predictions if predictions else None,
                labels=labels if labels else None,
                contexts=contexts if contexts else None,
            )
            risk_result = output_risk

        risk_score = risk_result.risk_score if risk_result is not None else 0.0

        fairness_result = self._evaluate_fairness(
            data=data,
            predictions=predictions,
            labels=labels,
            groups=groups,
            outputs=outputs,
        )

        fairness_risk = fairness_result.fairness_risk if fairness_result else 0.0

        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            performance=performance,
            cost=self.cost_per_candidate,
            risk=risk_score,
            fairness_risk=fairness_risk,
            n_examples=len(data),
            details={
                "evaluator": "StaticPredictionObjectiveEvaluator",
                "risk_details": risk_result.details if risk_result else {},
                "fairness_details": fairness_result.details if fairness_result else {},
                "wrong_answer_rate": (
                    risk_result.wrong_answer_rate if risk_result else None
                ),
                "unsupported_claim_rate": (
                    risk_result.unsupported_claim_rate if risk_result else None
                ),
                "unsafe_output_rate": (
                    risk_result.unsafe_output_rate if risk_result else None
                ),
                "refusal_rate": (
                    risk_result.refusal_rate if risk_result else None
                ),
                "counterfactual_flip_rate": (
                    fairness_result.counterfactual_flip_rate
                    if fairness_result
                    else None
                ),
                "group_accuracy_gap": (
                    fairness_result.group_accuracy_gap
                    if fairness_result
                    else None
                ),
                "bias_violation_rate": (
                    fairness_result.bias_violation_rate
                    if fairness_result
                    else None
                ),
                "fairness_aggregation": self.fairness_aggregation,
                "fairness_breakdown": (
                    fairness_result.details.get("breakdown")
                    if fairness_result
                    else None
                ),
            },
        )

    def _evaluate_fairness(
        self,
        data: Sequence[Dict[str, Any]],
        predictions: Sequence[str],
        labels: Sequence[str],
        groups: Sequence[str],
        outputs: Sequence[str],
    ) -> Optional[FairnessResult]:
        base_predictions = [
            str(row.get("base_prediction", ""))
            for row in data
            if "base_prediction" in row
        ]
        counterfactual_predictions = [
            str(row.get("counterfactual_prediction", ""))
            for row in data
            if "counterfactual_prediction" in row
        ]
        expected_same = [
            bool(row.get("expected_same_prediction", True))
            for row in data
            if "base_prediction" in row
        ]

        have_counterfactual = (
            self.include_counterfactual_fairness
            and base_predictions
            and counterfactual_predictions
            and len(base_predictions) == len(counterfactual_predictions)
        )
        have_group = (
            self.include_group_fairness
            and predictions
            and labels
            and groups
            and len(predictions) == len(labels) == len(groups)
        )
        have_bias = self.include_bias_language and bool(outputs)

        if self.fairness_aggregation == "combined":
            if not (have_counterfactual or have_group or have_bias):
                return None

            return evaluate_combined_fairness(
                base_predictions=base_predictions if have_counterfactual else None,
                counterfactual_predictions=(
                    counterfactual_predictions if have_counterfactual else None
                ),
                expected_same_prediction=(
                    expected_same
                    if have_counterfactual and len(expected_same) == len(base_predictions)
                    else None
                ),
                predictions=predictions if have_group else None,
                labels=labels if have_group else None,
                groups=groups if have_group else None,
                outputs=outputs if have_bias else None,
                fairness_debt=self.fairness_debt,
                config=self.fairness_config,
            )

        # max / mean aggregation over individual single-signal evaluators.
        fairness_results: list[FairnessResult] = []

        if have_counterfactual:
            fairness_results.append(
                evaluate_counterfactual_fairness(
                    base_predictions=base_predictions,
                    counterfactual_predictions=counterfactual_predictions,
                    expected_same_prediction=(
                        expected_same
                        if len(expected_same) == len(base_predictions)
                        else None
                    ),
                )
            )

        if have_group:
            fairness_results.append(
                evaluate_group_fairness(
                    predictions=predictions,
                    labels=labels,
                    groups=groups,
                )
            )

        if have_bias:
            fairness_results.append(
                evaluate_bias_language(outputs=outputs)
            )

        if not fairness_results:
            return None

        all_risks = [result.fairness_risk for result in fairness_results]

        if self.fairness_aggregation == "mean":
            aggregated_risk = sum(all_risks) / len(all_risks)
            base_result = max(fairness_results, key=lambda r: r.fairness_risk)
            base_result.fairness_risk = aggregated_risk
            aggregation_label = "mean_fairness_risk"
        else:  # max
            base_result = max(fairness_results, key=lambda r: r.fairness_risk)
            aggregation_label = "max_fairness_risk"

        base_result.details = {
            **base_result.details,
            "aggregation": aggregation_label,
            "num_fairness_checks": len(fairness_results),
            "all_fairness_risks": all_risks,
        }

        return base_result