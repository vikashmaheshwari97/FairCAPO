import pytest

from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
from heal_capo.recommender import (
    PromptRecommender,
    RecommendationWeights,
)


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


def test_accuracy_first_recommends_highest_performance():
    portfolio = _make_portfolio()
    recommender = PromptRecommender()

    rec = recommender.recommend(portfolio, mode="accuracy_first")

    assert rec.candidate.instruction == "accurate prompt"
    assert rec.rank == 1
    assert rec.utility == 0.95


def test_cost_first_recommends_lowest_cost():
    portfolio = _make_portfolio()
    recommender = PromptRecommender()

    rec = recommender.recommend(portfolio, mode="cost_first")

    assert rec.candidate.instruction == "cheap prompt"
    assert rec.rank == 1
    assert rec.utility == -1.0


def test_risk_first_recommends_lowest_risk():
    portfolio = _make_portfolio()
    recommender = PromptRecommender()

    rec = recommender.recommend(portfolio, mode="risk_first")

    assert rec.candidate.instruction == "safe prompt"
    assert rec.rank == 1
    assert rec.utility == -0.01


def test_fairness_first_recommends_lowest_fairness_risk():
    portfolio = _make_portfolio()
    recommender = PromptRecommender()

    rec = recommender.recommend(portfolio, mode="fairness_first")

    assert rec.candidate.instruction == "fair prompt"
    assert rec.rank == 1
    assert rec.utility == -0.01


def test_balanced_recommender_returns_ranked_recommendations():
    portfolio = _make_portfolio()
    recommender = PromptRecommender()

    ranked = recommender.rank(portfolio, mode="balanced")

    assert len(ranked) == 4
    assert [rec.rank for rec in ranked] == [1, 2, 3, 4]
    assert ranked[0].utility >= ranked[-1].utility


def test_balanced_recommender_can_prioritize_accuracy():
    portfolio = _make_portfolio()
    recommender = PromptRecommender(
        weights=RecommendationWeights(
            accuracy_weight=10.0,
            cost_weight=0.1,
            risk_weight=0.1,
            fairness_weight=0.1,
        )
    )

    rec = recommender.recommend(portfolio, mode="balanced")

    assert rec.candidate.instruction == "accurate prompt"


def test_balanced_recommender_can_prioritize_fairness():
    portfolio = _make_portfolio()
    recommender = PromptRecommender(
        weights=RecommendationWeights(
            accuracy_weight=0.1,
            cost_weight=0.1,
            risk_weight=0.1,
            fairness_weight=10.0,
        )
    )

    rec = recommender.recommend(portfolio, mode="balanced")

    assert rec.candidate.instruction == "fair prompt"


def test_recommend_raises_for_empty_portfolio():
    portfolio = PromptPortfolio()
    recommender = PromptRecommender()

    with pytest.raises(ValueError):
        recommender.recommend(portfolio, mode="balanced")


def test_unknown_mode_raises():
    portfolio = _make_portfolio()
    recommender = PromptRecommender()

    with pytest.raises(ValueError):
        recommender.recommend(portfolio, mode="unknown")  # type: ignore[arg-type]


def test_recommendation_to_row():
    portfolio = _make_portfolio()
    recommender = PromptRecommender()

    rec = recommender.recommend(portfolio, mode="accuracy_first")
    row = rec.to_row()

    assert row["rank"] == 1
    assert row["mode"] == "accuracy_first"
    assert row["instruction"] == "accurate prompt"
    assert row["performance"] == 0.95
    assert "reason" in row