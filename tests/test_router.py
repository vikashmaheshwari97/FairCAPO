import pytest

from heal_capo.components.router import FairnessAwareRouter, RiskAwareRouter
from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio


def _make_portfolio() -> PromptPortfolio:
    portfolio = PromptPortfolio()

    accurate = PromptCandidate(instruction="accurate prompt")
    cheap = PromptCandidate(instruction="cheap prompt")
    safe = PromptCandidate(instruction="safe prompt")
    fair = PromptCandidate(instruction="fair prompt")

    portfolio.add(
        accurate,
        EvaluationResult(
            candidate_id=accurate.candidate_id,
            performance=0.95,
            cost=10.0,
            risk=0.20,
            fairness_risk=0.20,
            n_examples=10,
        ),
    )
    portfolio.add(
        cheap,
        EvaluationResult(
            candidate_id=cheap.candidate_id,
            performance=0.75,
            cost=1.0,
            risk=0.30,
            fairness_risk=0.30,
            n_examples=10,
        ),
    )
    portfolio.add(
        safe,
        EvaluationResult(
            candidate_id=safe.candidate_id,
            performance=0.80,
            cost=5.0,
            risk=0.01,
            fairness_risk=0.15,
            n_examples=10,
        ),
    )
    portfolio.add(
        fair,
        EvaluationResult(
            candidate_id=fair.candidate_id,
            performance=0.78,
            cost=6.0,
            risk=0.12,
            fairness_risk=0.01,
            n_examples=10,
        ),
    )

    return portfolio


def test_router_accuracy_first():
    portfolio = _make_portfolio()
    router = RiskAwareRouter()

    decision = router.select(
        x="example",
        portfolio=portfolio,
        preference={"mode": "accuracy_first"},
    )

    assert decision.instruction == "accurate prompt"
    assert decision.mode == "accuracy_first"
    assert decision.metadata["performance"] == 0.95


def test_router_cost_first():
    portfolio = _make_portfolio()
    router = RiskAwareRouter()

    decision = router.select(
        x="example",
        portfolio=portfolio,
        preference={"mode": "cost_first"},
    )

    assert decision.instruction == "cheap prompt"
    assert decision.mode == "cost_first"
    assert decision.metadata["cost"] == 1.0


def test_router_risk_first():
    portfolio = _make_portfolio()
    router = RiskAwareRouter()

    decision = router.select(
        x="example",
        portfolio=portfolio,
        preference={"mode": "risk_first"},
    )

    assert decision.instruction == "safe prompt"
    assert decision.mode == "risk_first"
    assert decision.metadata["risk"] == 0.01


def test_router_fairness_first():
    portfolio = _make_portfolio()
    router = RiskAwareRouter()

    decision = router.select(
        x="example",
        portfolio=portfolio,
        preference={"mode": "fairness_first"},
    )

    assert decision.instruction == "fair prompt"
    assert decision.mode == "fairness_first"
    assert decision.metadata["fairness_risk"] == 0.01


def test_router_balanced_with_fairness_weight():
    portfolio = _make_portfolio()
    router = RiskAwareRouter()

    decision = router.select(
        x="example",
        portfolio=portfolio,
        preference={
            "mode": "balanced",
            "performance": 0.1,
            "cost": 0.1,
            "risk": 0.1,
            "fairness": 10.0,
        },
    )

    assert decision.instruction == "fair prompt"
    assert decision.mode == "balanced"


def test_fairness_aware_router_defaults_to_fairness_first():
    portfolio = _make_portfolio()
    router = FairnessAwareRouter()

    decision = router.select(
        x="example",
        portfolio=portfolio,
    )

    assert decision.instruction == "fair prompt"
    assert decision.mode == "fairness_first"


def test_router_rank_returns_all_recommendations():
    portfolio = _make_portfolio()
    router = RiskAwareRouter()

    ranked = router.rank(
        portfolio=portfolio,
        preference={"mode": "accuracy_first"},
    )

    assert len(ranked) == 4
    assert ranked[0].candidate.instruction == "accurate prompt"


def test_router_rejects_unknown_mode():
    portfolio = _make_portfolio()
    router = RiskAwareRouter()

    with pytest.raises(ValueError):
        router.select(
            x="example",
            portfolio=portfolio,
            preference={"mode": "unknown"},
        )


def test_router_rejects_empty_portfolio():
    portfolio = PromptPortfolio()
    router = RiskAwareRouter()

    with pytest.raises(ValueError):
        router.select(
            x="example",
            portfolio=portfolio,
        )


def test_routing_decision_to_row():
    portfolio = _make_portfolio()
    router = RiskAwareRouter()

    decision = router.select(
        x="example",
        portfolio=portfolio,
        preference={"mode": "accuracy_first"},
    )

    row = decision.to_row()

    assert row["mode"] == "accuracy_first"
    assert row["instruction"] == "accurate prompt"
    assert "reason" in row
    assert "metadata" in row