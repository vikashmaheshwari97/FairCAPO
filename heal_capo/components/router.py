from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..core import PromptCandidate, PromptPortfolio
from ..recommender import (
    PromptRecommender,
    RecommendationMode,
    RecommendationWeights,
)


@dataclass
class RoutingDecision:
    """
    Result of selecting one prompt from a portfolio.
    """

    candidate_id: str
    reason: str
    mode: str = "balanced"
    utility: float = 0.0
    instruction: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_row(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "mode": self.mode,
            "utility": self.utility,
            "instruction": self.instruction,
            "reason": self.reason,
            "metadata": self.metadata or {},
        }


class RiskAwareRouter:
    """
    Preference-aware router over a HEAL-CAPO prompt portfolio.

    This router supports:
      - accuracy_first
      - cost_first
      - risk_first
      - fairness_first
      - balanced

    Later this can be replaced or extended with:
      - classifier router
      - contextual bandit router
      - risk-triggered escalation
      - fairness-triggered escalation
    """

    def __init__(
        self,
        lambda_perf: float = 1.0,
        lambda_cost: float = 0.3,
        lambda_risk: float = 1.0,
        lambda_fairness: float = 1.0,
    ):
        self.lambda_perf = lambda_perf
        self.lambda_cost = lambda_cost
        self.lambda_risk = lambda_risk
        self.lambda_fairness = lambda_fairness

    def select(
        self,
        x: str,
        portfolio: PromptPortfolio,
        preference: Optional[Dict[str, float | str]] = None,
    ) -> RoutingDecision:
        """
        Select a prompt for input x.

        x is included for future contextual routing.
        Current implementation uses portfolio-level metrics only.
        """
        mode = self._get_mode(preference)
        weights = self._get_weights(preference)

        recommender = PromptRecommender(weights=weights)
        recommendation = recommender.recommend(
            portfolio=portfolio,
            mode=mode,
        )

        return RoutingDecision(
            candidate_id=recommendation.candidate.candidate_id,
            reason=recommendation.reason,
            mode=mode,
            utility=recommendation.utility,
            instruction=recommendation.candidate.instruction,
            metadata={
                "input": x,
                "performance": recommendation.result.performance,
                "cost": recommendation.result.cost,
                "risk": recommendation.result.risk,
                "fairness_risk": recommendation.result.fairness_risk,
                "rank": recommendation.rank,
            },
        )

    def rank(
        self,
        portfolio: PromptPortfolio,
        preference: Optional[Dict[str, float | str]] = None,
    ):
        """
        Return ranked prompt recommendations under a preference.
        """
        mode = self._get_mode(preference)
        weights = self._get_weights(preference)

        recommender = PromptRecommender(weights=weights)

        return recommender.rank(
            portfolio=portfolio,
            mode=mode,
        )

    def _get_mode(
        self,
        preference: Optional[Dict[str, float | str]] = None,
    ) -> RecommendationMode:
        if preference is None:
            return "balanced"

        mode = preference.get("mode", "balanced")

        valid_modes = {
            "accuracy_first",
            "cost_first",
            "risk_first",
            "fairness_first",
            "balanced",
        }

        if mode not in valid_modes:
            raise ValueError(f"Unknown routing mode: {mode}")

        return mode  # type: ignore[return-value]

    def _get_weights(
        self,
        preference: Optional[Dict[str, float | str]] = None,
    ) -> RecommendationWeights:
        if preference is None:
            return RecommendationWeights(
                accuracy_weight=self.lambda_perf,
                cost_weight=self.lambda_cost,
                risk_weight=self.lambda_risk,
                fairness_weight=self.lambda_fairness,
            )

        return RecommendationWeights(
            accuracy_weight=float(preference.get("performance", self.lambda_perf)),
            cost_weight=float(preference.get("cost", self.lambda_cost)),
            risk_weight=float(preference.get("risk", self.lambda_risk)),
            fairness_weight=float(preference.get("fairness", self.lambda_fairness)),
        )


class FairnessAwareRouter(RiskAwareRouter):
    """
    Convenience router that defaults to fairness-sensitive behavior.
    """

    def __init__(self):
        super().__init__(
            lambda_perf=0.5,
            lambda_cost=0.2,
            lambda_risk=1.0,
            lambda_fairness=2.0,
        )

    def select(
        self,
        x: str,
        portfolio: PromptPortfolio,
        preference: Optional[Dict[str, float | str]] = None,
    ) -> RoutingDecision:
        if preference is None:
            preference = {"mode": "fairness_first"}

        return super().select(
            x=x,
            portfolio=portfolio,
            preference=preference,
        )