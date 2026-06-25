import pytest

from dataclasses import dataclass

from heal_capo.core import EvaluationResult
from heal_capo.objectives import ObjectiveEvaluator
from heal_capo.optimizers.risk_aware_mo_capo import (
    RiskAwareMOCAPO,
    RiskAwareMOCAPOConfig,
)


@dataclass
class DummyDriftResult:
    passed: bool
    drift_score: float


class DummyDriftGuard:
    def check(self, original: str, candidate: str):
        if "bad drift" in candidate.lower():
            return DummyDriftResult(passed=False, drift_score=1.0)

        return DummyDriftResult(passed=True, drift_score=0.0)


class DummyEvaluator(ObjectiveEvaluator):
    def evaluate(self, candidate, data):
        text = candidate.instruction.lower()

        if "accurate fair" in text:
            return EvaluationResult(
                candidate_id=candidate.candidate_id,
                performance=0.9,
                cost=1.0,
                risk=0.1,
                fairness_risk=0.05,
                n_examples=len(data),
            )

        if "cheap risky" in text:
            return EvaluationResult(
                candidate_id=candidate.candidate_id,
                performance=0.7,
                cost=0.5,
                risk=0.4,
                fairness_risk=0.30,
                n_examples=len(data),
            )

        if "dominated" in text:
            return EvaluationResult(
                candidate_id=candidate.candidate_id,
                performance=0.6,
                cost=2.0,
                risk=0.5,
                fairness_risk=0.50,
                n_examples=len(data),
            )

        if "bad drift" in text:
            return EvaluationResult(
                candidate_id=candidate.candidate_id,
                performance=0.7,
                cost=0.2,
                risk=0.4,
                fairness_risk=0.30,
                n_examples=len(data),
            )

        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            performance=0.8,
            cost=1.2,
            risk=0.2,
            fairness_risk=0.10,
            n_examples=len(data),
        )


def test_risk_aware_mocapo_keeps_non_dominated_prompts():
    optimizer = RiskAwareMOCAPO(
        evaluator=DummyEvaluator(),
        drift_guard=DummyDriftGuard(),
        config=RiskAwareMOCAPOConfig(population_size=10),
    )

    prompts = [
        "accurate fair prompt",
        "cheap risky prompt",
        "dominated prompt",
    ]
    dev_data = [{"text": "example", "label": "objective"}]

    portfolio = optimizer.optimize(
        initial_prompts=prompts,
        dev_data=dev_data,
    )

    instructions = [candidate.instruction for candidate in portfolio.candidates]

    assert "accurate fair prompt" in instructions
    assert "cheap risky prompt" in instructions
    assert "dominated prompt" not in instructions


def test_risk_aware_mocapo_filters_drift_failures():
    optimizer = RiskAwareMOCAPO(
        evaluator=DummyEvaluator(),
        drift_guard=DummyDriftGuard(),
        config=RiskAwareMOCAPOConfig(population_size=10),
    )

    prompts = [
        "accurate fair prompt",
        "bad drift prompt",
    ]
    dev_data = [{"text": "example", "label": "objective"}]

    portfolio = optimizer.optimize(
        initial_prompts=prompts,
        dev_data=dev_data,
    )

    instructions = [candidate.instruction for candidate in portfolio.candidates]

    assert "accurate fair prompt" in instructions
    assert "bad drift prompt" not in instructions


def test_risk_aware_mocapo_can_keep_drift_failures_when_configured():
    optimizer = RiskAwareMOCAPO(
        evaluator=DummyEvaluator(),
        drift_guard=DummyDriftGuard(),
        config=RiskAwareMOCAPOConfig(
            population_size=10,
            keep_drift_failures=True,
        ),
    )

    prompts = [
        "accurate fair prompt",
        "bad drift prompt",
    ]
    dev_data = [{"text": "example", "label": "objective"}]

    portfolio = optimizer.optimize(
        initial_prompts=prompts,
        dev_data=dev_data,
    )

    instructions = [candidate.instruction for candidate in portfolio.candidates]

    assert "accurate fair prompt" in instructions
    assert "bad drift prompt" in instructions


def test_risk_aware_mocapo_summary_contains_fairness():
    optimizer = RiskAwareMOCAPO(
        evaluator=DummyEvaluator(),
        drift_guard=DummyDriftGuard(),
        config=RiskAwareMOCAPOConfig(population_size=10),
    )

    prompts = ["accurate fair prompt"]
    dev_data = [{"text": "example", "label": "objective"}]

    portfolio = optimizer.optimize(
        initial_prompts=prompts,
        dev_data=dev_data,
    )

    rows = optimizer.summarize_portfolio(portfolio)

    assert len(rows) == 1
    assert rows[0]["performance"] == 0.9
    assert rows[0]["risk"] == 0.1
    assert rows[0]["fairness_risk"] == 0.05
    assert len(rows[0]["objective_vector"]) == 4


def test_risk_aware_mocapo_rejects_empty_initial_prompts():
    optimizer = RiskAwareMOCAPO(
        evaluator=DummyEvaluator(),
        drift_guard=DummyDriftGuard(),
        config=RiskAwareMOCAPOConfig(),
    )

    with pytest.raises(ValueError):
        optimizer.optimize(
            initial_prompts=[],
            dev_data=[],
        )