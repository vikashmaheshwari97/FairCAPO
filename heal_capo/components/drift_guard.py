from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DriftResult:
    """
    Result of checking whether a new prompt preserves the original task intent.

    Lower drift_score is better.
    """

    drift_score: float
    passed: bool
    explanation: str = ""
    missing_required_terms: list[str] = field(default_factory=list)
    missing_fairness_terms: list[str] = field(default_factory=list)
    missing_risk_terms: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class DriftGuard(ABC):
    """
    Checks whether optimized/repaired prompts preserve original task intent.
    """

    @abstractmethod
    def check(
        self,
        original_instruction: str,
        new_instruction: str,
    ) -> DriftResult:
        raise NotImplementedError


def _normalize_terms(terms: list[str] | None) -> list[str]:
    if not terms:
        return []

    return [str(term).lower().strip() for term in terms if str(term).strip()]


def _missing_terms(
    text: str,
    terms: list[str],
) -> list[str]:
    lowered = text.lower()

    return [term for term in terms if term not in lowered]


class KeywordDriftGuard(DriftGuard):
    """
    Cheap first-pass drift guard.

    It checks whether required terms or constraints remain represented
    in the optimized/repaired prompt.

    This upgraded version supports:
      - task required terms
      - fairness required terms
      - risk/safety required terms
    """

    def __init__(
        self,
        required_terms: list[str],
        max_missing_ratio: float = 0.3,
        fairness_terms: list[str] | None = None,
        risk_terms: list[str] | None = None,
        fairness_max_missing_ratio: float = 0.5,
        risk_max_missing_ratio: float = 0.5,
    ):
        if not 0.0 <= max_missing_ratio <= 1.0:
            raise ValueError("max_missing_ratio must be between 0 and 1.")

        if not 0.0 <= fairness_max_missing_ratio <= 1.0:
            raise ValueError("fairness_max_missing_ratio must be between 0 and 1.")

        if not 0.0 <= risk_max_missing_ratio <= 1.0:
            raise ValueError("risk_max_missing_ratio must be between 0 and 1.")

        self.required_terms = _normalize_terms(required_terms)
        self.fairness_terms = _normalize_terms(fairness_terms)
        self.risk_terms = _normalize_terms(risk_terms)

        self.max_missing_ratio = max_missing_ratio
        self.fairness_max_missing_ratio = fairness_max_missing_ratio
        self.risk_max_missing_ratio = risk_max_missing_ratio

    def check(
        self,
        original_instruction: str,
        new_instruction: str,
    ) -> DriftResult:
        text = new_instruction.lower()

        missing_required = _missing_terms(text, self.required_terms)
        missing_fairness = _missing_terms(text, self.fairness_terms)
        missing_risk = _missing_terms(text, self.risk_terms)

        required_ratio = self._missing_ratio(
            missing=missing_required,
            total=len(self.required_terms),
        )
        fairness_ratio = self._missing_ratio(
            missing=missing_fairness,
            total=len(self.fairness_terms),
        )
        risk_ratio = self._missing_ratio(
            missing=missing_risk,
            total=len(self.risk_terms),
        )

        drift_score = max(
            required_ratio,
            fairness_ratio,
            risk_ratio,
        )

        passed_required = required_ratio <= self.max_missing_ratio
        passed_fairness = fairness_ratio <= self.fairness_max_missing_ratio
        passed_risk = risk_ratio <= self.risk_max_missing_ratio

        passed = passed_required and passed_fairness and passed_risk

        explanation_parts = []

        if missing_required:
            explanation_parts.append(
                f"Missing required terms: {missing_required}"
            )

        if missing_fairness:
            explanation_parts.append(
                f"Missing fairness terms: {missing_fairness}"
            )

        if missing_risk:
            explanation_parts.append(
                f"Missing risk terms: {missing_risk}"
            )

        if not explanation_parts:
            explanation = "All required, fairness, and risk terms preserved."
        else:
            explanation = " | ".join(explanation_parts)

        return DriftResult(
            drift_score=drift_score,
            passed=passed,
            explanation=explanation,
            missing_required_terms=missing_required,
            missing_fairness_terms=missing_fairness,
            missing_risk_terms=missing_risk,
            metadata={
                "required_missing_ratio": required_ratio,
                "fairness_missing_ratio": fairness_ratio,
                "risk_missing_ratio": risk_ratio,
                "passed_required": passed_required,
                "passed_fairness": passed_fairness,
                "passed_risk": passed_risk,
                "original_instruction": original_instruction,
                "new_instruction": new_instruction,
            },
        )

    @staticmethod
    def _missing_ratio(
        missing: list[str],
        total: int,
    ) -> float:
        if total <= 0:
            return 0.0

        return len(missing) / total


class FairnessConstraintDriftGuard(KeywordDriftGuard):
    """
    Convenience guard for fairness-sensitive prompts.

    It requires the task terms plus at least some fairness-related constraints
    to remain present after optimization or repair.
    """

    def __init__(
        self,
        required_terms: list[str] | None = None,
        fairness_terms: list[str] | None = None,
        max_missing_ratio: float = 0.3,
        fairness_max_missing_ratio: float = 0.5,
    ):
        super().__init__(
            required_terms=required_terms or ["classify", "input"],
            max_missing_ratio=max_missing_ratio,
            fairness_terms=fairness_terms
            or [
                "gender",
                "race",
                "ethnicity",
                "demographic",
                "do not infer",
            ],
            fairness_max_missing_ratio=fairness_max_missing_ratio,
        )


class RiskConstraintDriftGuard(KeywordDriftGuard):
    """
    Convenience guard for hallucination/risk-sensitive prompts.
    """

    def __init__(
        self,
        required_terms: list[str] | None = None,
        risk_terms: list[str] | None = None,
        max_missing_ratio: float = 0.3,
        risk_max_missing_ratio: float = 0.5,
    ):
        super().__init__(
            required_terms=required_terms or ["classify", "input"],
            max_missing_ratio=max_missing_ratio,
            risk_terms=risk_terms
            or [
                "do not hallucinate",
                "context",
                "not enough information",
            ],
            risk_max_missing_ratio=risk_max_missing_ratio,
        )