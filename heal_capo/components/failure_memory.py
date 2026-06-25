from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FailureCase:
    x: str
    output: str
    candidate_id: str
    failure_type: str
    explanation: str
    reference: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FailureMemory:
    """
    Stores verifier failures observed during deployment/evaluation.

    Supports:
      - clustering by failure type
      - clustering by candidate
      - repeated failure detection
      - fairness debt tracking
      - simple summaries for self-healing reports
    """

    def __init__(self) -> None:
        self.failures: List[FailureCase] = []

    def add(self, failure: FailureCase) -> None:
        self.failures.append(failure)

    def extend(self, failures: List[FailureCase]) -> None:
        for failure in failures:
            self.add(failure)

    def clear(self) -> None:
        self.failures.clear()

    def __len__(self) -> int:
        return len(self.failures)

    def is_empty(self) -> bool:
        return len(self.failures) == 0

    def clusters(self) -> Dict[str, List[FailureCase]]:
        """
        Group failures by failure type.
        """
        grouped: Dict[str, List[FailureCase]] = defaultdict(list)

        for failure in self.failures:
            grouped[failure.failure_type].append(failure)

        return dict(grouped)

    def clusters_by_candidate(self) -> Dict[str, List[FailureCase]]:
        """
        Group failures by candidate/prompt id.
        """
        grouped: Dict[str, List[FailureCase]] = defaultdict(list)

        for failure in self.failures:
            grouped[failure.candidate_id].append(failure)

        return dict(grouped)

    def clusters_by_candidate_and_type(self) -> Dict[tuple[str, str], List[FailureCase]]:
        """
        Group failures by candidate id and failure type.
        """
        grouped: Dict[tuple[str, str], List[FailureCase]] = defaultdict(list)

        for failure in self.failures:
            grouped[(failure.candidate_id, failure.failure_type)].append(failure)

        return dict(grouped)

    def count_by_type(self) -> Dict[str, int]:
        return dict(Counter(failure.failure_type for failure in self.failures))

    def count_by_candidate(self) -> Dict[str, int]:
        return dict(Counter(failure.candidate_id for failure in self.failures))

    def count_for_candidate(self, candidate_id: str) -> int:
        return sum(
            1
            for failure in self.failures
            if failure.candidate_id == candidate_id
        )

    def count_for_type(self, failure_type: str) -> int:
        return sum(
            1
            for failure in self.failures
            if failure.failure_type == failure_type
        )

    def recent(self, n: int = 10) -> List[FailureCase]:
        if n <= 0:
            return []

        return self.failures[-n:]

    def repeated_failures(
        self,
        min_count: int = 2,
    ) -> Dict[tuple[str, str], List[FailureCase]]:
        """
        Return candidate/type clusters with at least min_count failures.
        """
        grouped = self.clusters_by_candidate_and_type()

        return {
            key: failures
            for key, failures in grouped.items()
            if len(failures) >= min_count
        }

    def fairness_debt(self, candidate_id: Optional[str] = None) -> int:
        """
        Count accumulated fairness failures.

        If candidate_id is provided, only count fairness failures for that prompt.
        """
        failures = self.failures

        if candidate_id is not None:
            failures = [
                failure
                for failure in failures
                if failure.candidate_id == candidate_id
            ]

        return sum(
            1
            for failure in failures
            if failure.failure_type == "fairness"
        )

    def risk_debt(self, candidate_id: Optional[str] = None) -> int:
        """
        Count accumulated risk-like failures.
        """
        risk_types = {
            "hallucination",
            "incorrect",
            "unsafe",
            "over_refusal",
            "format",
        }

        failures = self.failures

        if candidate_id is not None:
            failures = [
                failure
                for failure in failures
                if failure.candidate_id == candidate_id
            ]

        return sum(
            1
            for failure in failures
            if failure.failure_type in risk_types
        )

    def summary(self) -> dict:
        """
        Compact summary for logging/reporting.
        """
        return {
            "num_failures": len(self.failures),
            "num_failure_types": len(self.count_by_type()),
            "num_candidates_with_failures": len(self.count_by_candidate()),
            "count_by_type": self.count_by_type(),
            "count_by_candidate": self.count_by_candidate(),
            "fairness_debt": self.fairness_debt(),
            "risk_debt": self.risk_debt(),
        }

    def to_rows(self) -> List[dict]:
        """
        Export failures to CSV/JSON-friendly rows.
        """
        rows = []

        for failure in self.failures:
            row = {
                "x": failure.x,
                "output": failure.output,
                "candidate_id": failure.candidate_id,
                "failure_type": failure.failure_type,
                "explanation": failure.explanation,
                "reference": failure.reference,
                "context": failure.context,
            }

            for key, value in failure.metadata.items():
                row[f"metadata_{key}"] = value

            rows.append(row)

        return rows