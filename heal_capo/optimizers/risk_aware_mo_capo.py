from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from ..core import PromptCandidate, PromptPortfolio
from ..objectives import ObjectiveEvaluator
from ..pareto import pareto_archive, sort_pareto_results
from ..components.drift_guard import DriftGuard


@dataclass
class RiskAwareMOCAPOConfig:
    """
    Configuration for the Phase 2 risk/fairness-aware MO-CAPO prototype.

    This is still a lightweight prototype, not the full evolutionary algorithm.
    It evaluates an initial prompt population and keeps the non-dominated
    accuracy-cost-risk-fairness Pareto archive.
    """

    population_size: int = 10
    n_steps: int = 1
    max_shots: int = 5
    drift_threshold: float = 0.3
    keep_drift_failures: bool = False
    sort_archive: bool = True


class RiskAwareMOCAPO:
    """
    Risk- and fairness-aware MO-CAPO prototype.

    Current Phase 2 behavior:
      1. Take initial prompts.
      2. Convert each prompt into a PromptCandidate.
      3. Check semantic drift against the original prompt.
      4. Evaluate performance, cost, risk, and fairness risk.
      5. Keep only non-dominated Pareto candidates.

    Objective vector comes from EvaluationResult:
      (-performance, cost, risk, fairness_risk)

    Later this class can be extended with:
      - CAPO/EvoPrompt-style crossover and mutation
      - risk-aware intensification
      - fairness-aware repair
      - verifier-guided self-healing
    """

    def __init__(
        self,
        evaluator: ObjectiveEvaluator,
        drift_guard: DriftGuard,
        config: Optional[RiskAwareMOCAPOConfig] = None,
    ):
        self.evaluator = evaluator
        self.drift_guard = drift_guard
        self.config = config or RiskAwareMOCAPOConfig()

    def optimize(
        self,
        initial_prompts: Sequence[str],
        dev_data: Sequence[Dict[str, Any]],
    ) -> PromptPortfolio:
        """
        Evaluate initial prompts and return a Pareto prompt portfolio.
        """
        if not initial_prompts:
            raise ValueError("initial_prompts must contain at least one prompt.")

        portfolio = PromptPortfolio()
        original_instruction = initial_prompts[0]

        for prompt_index, prompt in enumerate(initial_prompts[: self.config.population_size]):
            candidate = PromptCandidate(
                instruction=prompt,
                metadata={
                    "source": "initial_prompt",
                    "prompt_index": prompt_index,
                },
            )

            drift = self.drift_guard.check(
                original_instruction,
                prompt,
            )

            if not drift.passed and not self.config.keep_drift_failures:
                continue

            result = self.evaluator.evaluate(
                candidate=candidate,
                data=dev_data,
            )
            result.drift = drift.drift_score
            result.details["drift_passed"] = drift.passed
            result.details["drift_score"] = drift.drift_score
            result.details["prompt_index"] = prompt_index

            portfolio.add(candidate, result)

        portfolio.evaluations = pareto_archive(portfolio.evaluations)
        portfolio.candidates = [
            candidate
            for candidate in portfolio.candidates
            if candidate.candidate_id in portfolio.evaluations
        ]

        if self.config.sort_archive:
            portfolio = self._sort_portfolio(portfolio)

        return portfolio

    def _sort_portfolio(self, portfolio: PromptPortfolio) -> PromptPortfolio:
        """
        Sort portfolio candidates according to readable Pareto ordering.

        Priority:
          1. higher performance
          2. lower risk
          3. lower fairness risk
          4. lower cost
        """
        sorted_results = sort_pareto_results(portfolio.evaluations.values())
        sorted_ids = [result.candidate_id for result in sorted_results]

        id_to_candidate = {
            candidate.candidate_id: candidate
            for candidate in portfolio.candidates
        }

        portfolio.candidates = [
            id_to_candidate[candidate_id]
            for candidate_id in sorted_ids
            if candidate_id in id_to_candidate
        ]

        portfolio.evaluations = {
            result.candidate_id: result
            for result in sorted_results
        }

        return portfolio

    def summarize_portfolio(
        self,
        portfolio: PromptPortfolio,
    ) -> list[dict[str, Any]]:
        """
        Return a readable table-like summary of the current portfolio.
        """
        rows = []

        for candidate in portfolio.evaluated_candidates():
            result = portfolio.get_result(candidate.candidate_id)

            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "instruction": candidate.instruction,
                    "performance": result.performance,
                    "cost": result.cost,
                    "risk": result.risk,
                    "fairness_risk": result.fairness_risk,
                    "drift": result.drift,
                    "n_examples": result.n_examples,
                    "objective_vector": result.objective_vector,
                    "metadata": candidate.metadata,
                    "details": result.details,
                }
            )

        return rows