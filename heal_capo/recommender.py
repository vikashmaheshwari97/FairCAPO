from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from .core import PromptCandidate, PromptPortfolio, EvaluationResult


RecommendationMode = Literal[
    "accuracy_first",
    "cost_first",
    "risk_first",
    "fairness_first",
    "balanced",
]


@dataclass
class RecommendationWeights:
    """
    Weights used by the balanced recommender.

    Higher accuracy_weight means prefer higher performance.
    Higher cost_weight means penalize cost more.
    Higher risk_weight means penalize risk more.
    Higher fairness_weight means penalize fairness risk more.
    """

    accuracy_weight: float = 1.0
    cost_weight: float = 1.0
    risk_weight: float = 1.0
    fairness_weight: float = 1.0


@dataclass
class PromptRecommendation:
    """
    One prompt recommendation result.
    """

    candidate: PromptCandidate
    result: EvaluationResult
    mode: RecommendationMode
    utility: float
    reason: str
    rank: int = 0

    def to_row(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "mode": self.mode,
            "candidate_id": self.candidate.candidate_id,
            "instruction": self.candidate.instruction,
            "utility": self.utility,
            "performance": self.result.performance,
            "cost": self.result.cost,
            "risk": self.result.risk,
            "fairness_risk": self.result.fairness_risk,
            "drift": self.result.drift,
            "n_examples": self.result.n_examples,
            "reason": self.reason,
        }


def _safe_min_max(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0

    return min(values), max(values)


def _normalize_higher_is_better(value: float, min_value: float, max_value: float) -> float:
    """
    Normalize a metric where higher is better to [0, 1].
    """
    if max_value == min_value:
        return 1.0

    return (value - min_value) / (max_value - min_value)


def _normalize_lower_is_better(value: float, min_value: float, max_value: float) -> float:
    """
    Normalize a metric where lower is better to [0, 1].
    """
    if max_value == min_value:
        return 1.0

    return (max_value - value) / (max_value - min_value)


class PromptRecommender:
    """
    Recommends prompts from a PromptPortfolio.

    Supported modes:
      - accuracy_first: highest performance
      - cost_first: lowest cost
      - risk_first: lowest risk
      - fairness_first: lowest fairness risk
      - balanced: weighted utility across accuracy, cost, risk, fairness
    """

    def __init__(
        self,
        weights: Optional[RecommendationWeights] = None,
    ):
        self.weights = weights or RecommendationWeights()

    def recommend(
        self,
        portfolio: PromptPortfolio,
        mode: RecommendationMode = "balanced",
    ) -> PromptRecommendation:
        """
        Return the single best prompt under a recommendation mode.
        """
        recommendations = self.rank(
            portfolio=portfolio,
            mode=mode,
        )

        if not recommendations:
            raise ValueError("Portfolio has no evaluated candidates.")

        return recommendations[0]

    def rank(
        self,
        portfolio: PromptPortfolio,
        mode: RecommendationMode = "balanced",
    ) -> list[PromptRecommendation]:
        """
        Rank all evaluated prompts under the selected recommendation mode.
        """
        evaluated_candidates = list(portfolio.evaluated_candidates())

        if not evaluated_candidates:
            return []

        scored: list[PromptRecommendation] = []

        for candidate in evaluated_candidates:
            result = portfolio.get_result(candidate.candidate_id)
            utility = self._utility(
                result=result,
                portfolio=portfolio,
                mode=mode,
            )
            reason = self._reason(
                mode=mode,
                result=result,
                utility=utility,
            )

            scored.append(
                PromptRecommendation(
                    candidate=candidate,
                    result=result,
                    mode=mode,
                    utility=utility,
                    reason=reason,
                )
            )

        scored.sort(
            key=lambda rec: (
                -rec.utility,
                -rec.result.performance,
                rec.result.risk,
                rec.result.fairness_risk,
                rec.result.cost,
            )
        )

        for idx, rec in enumerate(scored, start=1):
            rec.rank = idx

        return scored

    def _utility(
        self,
        result: EvaluationResult,
        portfolio: PromptPortfolio,
        mode: RecommendationMode,
    ) -> float:
        if mode == "accuracy_first":
            return result.performance

        if mode == "cost_first":
            return -result.cost

        if mode == "risk_first":
            return -result.risk

        if mode == "fairness_first":
            return -result.fairness_risk

        if mode == "balanced":
            return self._balanced_utility(
                result=result,
                portfolio=portfolio,
            )

        raise ValueError(f"Unknown recommendation mode: {mode}")

    def _balanced_utility(
        self,
        result: EvaluationResult,
        portfolio: PromptPortfolio,
    ) -> float:
        results = list(portfolio.evaluated_results())

        performances = [r.performance for r in results]
        costs = [r.cost for r in results]
        risks = [r.risk for r in results]
        fairness_risks = [r.fairness_risk for r in results]

        min_perf, max_perf = _safe_min_max(performances)
        min_cost, max_cost = _safe_min_max(costs)
        min_risk, max_risk = _safe_min_max(risks)
        min_fairness, max_fairness = _safe_min_max(fairness_risks)

        accuracy_score = _normalize_higher_is_better(
            result.performance,
            min_perf,
            max_perf,
        )
        cost_score = _normalize_lower_is_better(
            result.cost,
            min_cost,
            max_cost,
        )
        risk_score = _normalize_lower_is_better(
            result.risk,
            min_risk,
            max_risk,
        )
        fairness_score = _normalize_lower_is_better(
            result.fairness_risk,
            min_fairness,
            max_fairness,
        )

        total_weight = (
            self.weights.accuracy_weight
            + self.weights.cost_weight
            + self.weights.risk_weight
            + self.weights.fairness_weight
        )

        if total_weight <= 0:
            raise ValueError("Total recommendation weight must be positive.")

        return (
            self.weights.accuracy_weight * accuracy_score
            + self.weights.cost_weight * cost_score
            + self.weights.risk_weight * risk_score
            + self.weights.fairness_weight * fairness_score
        ) / total_weight

    def _reason(
        self,
        mode: RecommendationMode,
        result: EvaluationResult,
        utility: float,
    ) -> str:
        if mode == "accuracy_first":
            return f"Selected because it has high performance ({result.performance:.4f})."

        if mode == "cost_first":
            return f"Selected because it has low cost ({result.cost:.4f})."

        if mode == "risk_first":
            return f"Selected because it has low risk ({result.risk:.4f})."

        if mode == "fairness_first":
            return (
                "Selected because it has low fairness risk "
                f"({result.fairness_risk:.4f})."
            )

        if mode == "balanced":
            return (
                "Selected using balanced utility over performance, cost, risk, "
                f"and fairness risk (utility={utility:.4f})."
            )

        return "Selected by prompt recommender."