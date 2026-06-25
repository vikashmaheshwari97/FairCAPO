from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from .core import EvaluationResult, PromptCandidate, PromptPortfolio
from .components.failure_memory import FailureCase, FailureMemory
from .components.repair import PromptRepairer
from .components.verifier import VerificationResult, Verifier
from .components.drift_guard import DriftGuard
from .objectives import ObjectiveEvaluator
from .pareto import pareto_archive


@dataclass
class HealingEvent:
    """
    One repair attempt made by the continual healer.
    """

    failure_type: str
    source_candidate_id: str
    repaired_candidate_id: Optional[str]
    accepted: bool
    reason: str
    drift_score: float = 0.0
    performance: Optional[float] = None
    cost: Optional[float] = None
    risk: Optional[float] = None
    fairness_risk: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealingReport:
    """
    Summary of one repair_portfolio call.
    """

    num_failures_seen: int = 0
    num_failure_clusters: int = 0
    num_repairs_attempted: int = 0
    num_repairs_accepted: int = 0
    num_repairs_rejected: int = 0
    events: list[HealingEvent] = field(default_factory=list)

    @property
    def repair_acceptance_rate(self) -> float:
        if self.num_repairs_attempted <= 0:
            return 0.0

        return self.num_repairs_accepted / self.num_repairs_attempted


class ContinualHealer:
    """
    Failure-driven prompt repair and Pareto update.

    The loop is:

      1. observe model output
      2. verify output
      3. store failure if detected
      4. cluster failures
      5. repair prompt responsible for each cluster
      6. check drift
      7. evaluate repaired prompt
      8. add accepted repair to portfolio
      9. update Pareto archive

    This is the core "HEAL" loop.
    """

    def __init__(
        self,
        verifier: Verifier,
        repairer: PromptRepairer,
        drift_guard: DriftGuard,
        evaluator: ObjectiveEvaluator,
        keep_rejected_repairs: bool = False,
    ):
        self.verifier = verifier
        self.repairer = repairer
        self.drift_guard = drift_guard
        self.evaluator = evaluator
        self.keep_rejected_repairs = keep_rejected_repairs
        self.memory = FailureMemory()
        self.last_report = HealingReport()

    def observe(
        self,
        x: str,
        output: str,
        candidate_id: str,
        reference: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Optional[VerificationResult]:
        """
        Verify one output and store a failure if detected.

        Returns the verification result when a failure is detected,
        otherwise returns None.
        """
        result = self.verifier.verify(
            x=x,
            output=output,
            reference=reference,
            context=context,
        )

        if result.failure_type:
            self.memory.add(
                FailureCase(
                    x=x,
                    output=output,
                    candidate_id=candidate_id,
                    failure_type=result.failure_type,
                    explanation=result.explanation,
                )
            )
            return result

        return None

    def observe_batch(
        self,
        observations: Sequence[dict],
    ) -> list[VerificationResult]:
        """
        Observe many outputs.

        Each observation may contain:
          - x
          - output
          - candidate_id
          - reference
          - context
        """
        failures = []

        for item in observations:
            result = self.observe(
                x=item.get("x", ""),
                output=item.get("output", ""),
                candidate_id=item.get("candidate_id", ""),
                reference=item.get("reference"),
                context=item.get("context"),
            )

            if result is not None:
                failures.append(result)

        return failures

    def repair_portfolio(
        self,
        portfolio: PromptPortfolio,
        original_instruction: str,
        dev_data,
    ) -> PromptPortfolio:
        """
        Repair prompts based on stored failures and update the Pareto archive.
        """
        report = HealingReport()

        clusters = self.memory.clusters()
        report.num_failure_clusters = len(clusters)
        report.num_failures_seen = sum(len(items) for items in clusters.values())

        for failure_type, failures in clusters.items():
            if not failures:
                continue

            report.num_repairs_attempted += 1

            event = self._repair_one_cluster(
                portfolio=portfolio,
                original_instruction=original_instruction,
                dev_data=dev_data,
                failure_type=failure_type,
                failures=failures,
            )

            report.events.append(event)

            if event.accepted:
                report.num_repairs_accepted += 1
            else:
                report.num_repairs_rejected += 1

        portfolio.evaluations = pareto_archive(portfolio.evaluations)
        portfolio.candidates = [
            candidate
            for candidate in portfolio.candidates
            if candidate.candidate_id in portfolio.evaluations
        ]

        self.last_report = report
        return portfolio

    def _repair_one_cluster(
        self,
        portfolio: PromptPortfolio,
        original_instruction: str,
        dev_data,
        failure_type: str,
        failures: list[FailureCase],
    ) -> HealingEvent:
        """
        Repair one representative failure cluster.
        """
        representative = failures[0]

        try:
            source = portfolio.get(representative.candidate_id)
        except KeyError:
            return HealingEvent(
                failure_type=failure_type,
                source_candidate_id=representative.candidate_id,
                repaired_candidate_id=None,
                accepted=False,
                reason="Source candidate not found in portfolio.",
                metadata={
                    "num_failures_in_cluster": len(failures),
                },
            )

        feedback = self._make_feedback(
            failure_type=failure_type,
            representative=representative,
        )

        repaired = self.repairer.repair(source, feedback)
        repaired.metadata.update(
            {
                "repair_source_candidate_id": source.candidate_id,
                "repair_failure_type": failure_type,
                "repair_num_failures": len(failures),
                "repair_explanation": representative.explanation,
            }
        )

        drift = self.drift_guard.check(
            original_instruction=original_instruction,
            new_instruction=repaired.instruction,
        )

        if not drift.passed:
            if self.keep_rejected_repairs:
                rejected_result = EvaluationResult(
                    candidate_id=repaired.candidate_id,
                    performance=0.0,
                    cost=0.0,
                    risk=1.0,
                    fairness_risk=1.0,
                    drift=drift.drift_score,
                    n_examples=0,
                    details={
                        "accepted": False,
                        "rejection_reason": "drift_guard_failed",
                        "failure_type": failure_type,
                        "drift_explanation": drift.explanation,
                    },
                )
                portfolio.add(repaired, rejected_result)

            return HealingEvent(
                failure_type=failure_type,
                source_candidate_id=source.candidate_id,
                repaired_candidate_id=repaired.candidate_id,
                accepted=False,
                reason="Rejected by drift guard.",
                drift_score=drift.drift_score,
                metadata={
                    "num_failures_in_cluster": len(failures),
                    "drift_explanation": drift.explanation,
                },
            )

        result = self.evaluator.evaluate(repaired, dev_data)
        result.drift = drift.drift_score
        result.details.update(
            {
                "accepted": True,
                "repair_source_candidate_id": source.candidate_id,
                "repair_failure_type": failure_type,
                "repair_num_failures": len(failures),
                "drift_explanation": drift.explanation,
            }
        )

        portfolio.add(repaired, result)

        return HealingEvent(
            failure_type=failure_type,
            source_candidate_id=source.candidate_id,
            repaired_candidate_id=repaired.candidate_id,
            accepted=True,
            reason="Repair accepted and evaluated.",
            drift_score=drift.drift_score,
            performance=result.performance,
            cost=result.cost,
            risk=result.risk,
            fairness_risk=result.fairness_risk,
            metadata={
                "num_failures_in_cluster": len(failures),
                "drift_explanation": drift.explanation,
            },
        )

    def _make_feedback(
        self,
        failure_type: str,
        representative: FailureCase,
    ) -> VerificationResult:
        """
        Build verifier-style feedback from a stored failure.
        """
        return VerificationResult(
            risk_score=self._risk_score_for_failure(failure_type),
            failure_type=failure_type,
            explanation=representative.explanation,
            metadata={
                "x": representative.x,
                "output": representative.output,
                "candidate_id": representative.candidate_id,
            },
        )

    @staticmethod
    def _risk_score_for_failure(failure_type: str) -> float:
        if failure_type == "unsafe":
            return 0.95

        if failure_type == "fairness":
            return 0.90

        if failure_type == "hallucination":
            return 0.75

        if failure_type == "incorrect":
            return 0.65

        if failure_type == "format":
            return 0.50

        if failure_type == "over_refusal":
            return 0.45

        return 0.60

    def clear_memory(self) -> None:
        """
        Clear stored failures after a successful healing cycle if desired.
        """
        self.memory = FailureMemory()

    def get_last_report(self) -> HealingReport:
        """
        Return the latest healing report.
        """
        return self.last_report