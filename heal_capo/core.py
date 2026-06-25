from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import uuid


@dataclass
class PromptCandidate:
    """
    A prompt candidate plus optional few-shot examples and metadata.

    This is the main unit optimized by HEAL-CAPO.
    """

    instruction: str
    examples: List[Dict[str, str]] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)
    candidate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render(self, x: str) -> str:
        """
        Render the prompt for one input example.
        """
        shots = "\n".join(
            f"Input: {ex.get('input', '')}\nOutput: {ex.get('output', '')}"
            for ex in self.examples
        )

        if shots:
            return (
                f"{self.instruction}\n\n"
                f"Examples:\n{shots}\n\n"
                f"Input: {x}\n"
                f"Output:"
            )

        return (
            f"{self.instruction}\n\n"
            f"Input: {x}\n"
            f"Output:"
        )


@dataclass
class EvaluationResult:
    """
    Evaluation metrics for one PromptCandidate.

    Objective convention:
      - performance should be maximized
      - cost should be minimized
      - risk should be minimized
      - fairness_risk should be minimized
      - drift should be minimized

    Pareto methods use the minimization vector:
      (-performance, cost, risk, fairness_risk)
    """

    candidate_id: str
    performance: float
    cost: float
    risk: float
    fairness_risk: float = 0.0
    drift: float = 0.0
    n_examples: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def objective_vector(self) -> tuple[float, float, float, float]:
        """
        Main HEAL-CAPO objective vector.

        Minimization convention:
          maximize performance by minimizing -performance
          minimize cost
          minimize risk
          minimize fairness_risk
        """
        return (
            -self.performance,
            self.cost,
            self.risk,
            self.fairness_risk,
        )

    @property
    def objective_vector_with_drift(self) -> tuple[float, float, float, float, float]:
        """
        Optional extended objective vector including drift.

        We usually treat drift as a hard guard/constraint, but this is useful
        for analysis or ablations.
        """
        return (
            -self.performance,
            self.cost,
            self.risk,
            self.fairness_risk,
            self.drift,
        )

    def to_row(self) -> Dict[str, Any]:
        """
        Flatten result into a CSV/logging-friendly row.
        """
        row = {
            "candidate_id": self.candidate_id,
            "performance": self.performance,
            "cost": self.cost,
            "risk": self.risk,
            "fairness_risk": self.fairness_risk,
            "drift": self.drift,
            "n_examples": self.n_examples,
        }

        for key, value in self.details.items():
            row[f"detail_{key}"] = value

        return row


@dataclass
class PromptPortfolio:
    """
    Collection of prompt candidates and their evaluations.
    """

    candidates: List[PromptCandidate] = field(default_factory=list)
    evaluations: Dict[str, EvaluationResult] = field(default_factory=dict)

    def add(
        self,
        candidate: PromptCandidate,
        result: Optional[EvaluationResult] = None,
    ) -> None:
        """
        Add a prompt candidate and optionally its evaluation.
        """
        self.candidates.append(candidate)

        if result is not None:
            self.evaluations[candidate.candidate_id] = result

    def get(self, candidate_id: str) -> PromptCandidate:
        """
        Retrieve a candidate by ID.
        """
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate

        raise KeyError(candidate_id)

    def get_result(self, candidate_id: str) -> EvaluationResult:
        """
        Retrieve an evaluation result by candidate ID.
        """
        if candidate_id not in self.evaluations:
            raise KeyError(candidate_id)

        return self.evaluations[candidate_id]

    def evaluated_candidates(self) -> Sequence[PromptCandidate]:
        """
        Return only candidates that have evaluation results.
        """
        return [
            candidate
            for candidate in self.candidates
            if candidate.candidate_id in self.evaluations
        ]

    def evaluated_results(self) -> Sequence[EvaluationResult]:
        """
        Return all evaluation results.
        """
        return list(self.evaluations.values())

    def best_by_performance(self) -> Optional[PromptCandidate]:
        """
        Return the evaluated candidate with highest performance.
        """
        evaluated = self.evaluated_candidates()

        if not evaluated:
            return None

        return max(
            evaluated,
            key=lambda candidate: self.evaluations[candidate.candidate_id].performance,
        )

    def lowest_risk(self) -> Optional[PromptCandidate]:
        """
        Return the evaluated candidate with lowest risk.
        """
        evaluated = self.evaluated_candidates()

        if not evaluated:
            return None

        return min(
            evaluated,
            key=lambda candidate: self.evaluations[candidate.candidate_id].risk,
        )

    def lowest_fairness_risk(self) -> Optional[PromptCandidate]:
        """
        Return the evaluated candidate with lowest fairness risk.
        """
        evaluated = self.evaluated_candidates()

        if not evaluated:
            return None

        return min(
            evaluated,
            key=lambda candidate: self.evaluations[candidate.candidate_id].fairness_risk,
        )

    def lowest_cost(self) -> Optional[PromptCandidate]:
        """
        Return the evaluated candidate with lowest cost.
        """
        evaluated = self.evaluated_candidates()

        if not evaluated:
            return None

        return min(
            evaluated,
            key=lambda candidate: self.evaluations[candidate.candidate_id].cost,
        )