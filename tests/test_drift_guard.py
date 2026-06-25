import pytest

from heal_capo.components.drift_guard import (
    FairnessConstraintDriftGuard,
    KeywordDriftGuard,
    RiskConstraintDriftGuard,
)


def test_keyword_drift_guard_passes_when_required_terms_preserved():
    guard = KeywordDriftGuard(
        required_terms=["classify", "input"],
        max_missing_ratio=0.0,
    )

    result = guard.check(
        original_instruction="Classify the input.",
        new_instruction="Classify the input and return the label.",
    )

    assert result.passed is True
    assert result.drift_score == 0.0
    assert result.missing_required_terms == []
    assert result.metadata["passed_required"] is True


def test_keyword_drift_guard_fails_when_required_terms_missing():
    guard = KeywordDriftGuard(
        required_terms=["classify", "input"],
        max_missing_ratio=0.0,
    )

    result = guard.check(
        original_instruction="Classify the input.",
        new_instruction="Return the label.",
    )

    assert result.passed is False
    assert result.drift_score == 1.0
    assert result.missing_required_terms == ["classify", "input"]
    assert result.metadata["passed_required"] is False


def test_keyword_drift_guard_allows_some_missing_terms():
    guard = KeywordDriftGuard(
        required_terms=["classify", "input"],
        max_missing_ratio=0.5,
    )

    result = guard.check(
        original_instruction="Classify the input.",
        new_instruction="Classify the sentence.",
    )

    assert result.passed is True
    assert result.drift_score == 0.5
    assert result.missing_required_terms == ["input"]


def test_keyword_drift_guard_checks_fairness_terms():
    guard = KeywordDriftGuard(
        required_terms=["classify", "input"],
        fairness_terms=["gender", "race"],
        max_missing_ratio=0.0,
        fairness_max_missing_ratio=0.0,
    )

    result = guard.check(
        original_instruction="Classify the input without using gender or race.",
        new_instruction="Classify the input without using gender.",
    )

    assert result.passed is False
    assert result.missing_fairness_terms == ["race"]
    assert result.metadata["passed_fairness"] is False


def test_keyword_drift_guard_checks_risk_terms():
    guard = KeywordDriftGuard(
        required_terms=["classify", "input"],
        risk_terms=["do not hallucinate", "context"],
        max_missing_ratio=0.0,
        risk_max_missing_ratio=0.0,
    )

    result = guard.check(
        original_instruction="Classify the input using context. Do not hallucinate.",
        new_instruction="Classify the input using context.",
    )

    assert result.passed is False
    assert result.missing_risk_terms == ["do not hallucinate"]
    assert result.metadata["passed_risk"] is False


def test_fairness_constraint_drift_guard_passes_with_fairness_terms():
    guard = FairnessConstraintDriftGuard(
        fairness_terms=["gender", "race"],
        fairness_max_missing_ratio=0.0,
    )

    result = guard.check(
        original_instruction="Classify the input without using gender or race.",
        new_instruction="Classify the input without using gender or race.",
    )

    assert result.passed is True
    assert result.missing_fairness_terms == []


def test_fairness_constraint_drift_guard_fails_missing_fairness_terms():
    guard = FairnessConstraintDriftGuard(
        fairness_terms=["gender", "race"],
        fairness_max_missing_ratio=0.0,
    )

    result = guard.check(
        original_instruction="Classify the input without using gender or race.",
        new_instruction="Classify the input.",
    )

    assert result.passed is False
    assert set(result.missing_fairness_terms) == {"gender", "race"}


def test_risk_constraint_drift_guard_passes_with_risk_terms():
    guard = RiskConstraintDriftGuard(
        risk_terms=["context", "do not hallucinate"],
        risk_max_missing_ratio=0.0,
    )

    result = guard.check(
        original_instruction="Classify the input using context. Do not hallucinate.",
        new_instruction="Classify the input using context. Do not hallucinate.",
    )

    assert result.passed is True
    assert result.missing_risk_terms == []


def test_risk_constraint_drift_guard_fails_missing_risk_terms():
    guard = RiskConstraintDriftGuard(
        risk_terms=["context", "do not hallucinate"],
        risk_max_missing_ratio=0.0,
    )

    result = guard.check(
        original_instruction="Classify the input using context. Do not hallucinate.",
        new_instruction="Classify the input.",
    )

    assert result.passed is False
    assert set(result.missing_risk_terms) == {"context", "do not hallucinate"}


def test_invalid_threshold_raises():
    with pytest.raises(ValueError):
        KeywordDriftGuard(
            required_terms=["classify"],
            max_missing_ratio=1.5,
        )