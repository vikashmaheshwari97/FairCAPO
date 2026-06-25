from heal_capo.fairness import (
    CombinedFairnessConfig,
    FairnessDebtTracker,
    bias_violation_rate,
    combine_fairness_risk,
    counterfactual_flip_rate,
    demographic_parity_gap,
    equal_opportunity_gap,
    equalized_odds_gap,
    evaluate_bias_language,
    evaluate_combined_fairness,
    evaluate_counterfactual_fairness,
    evaluate_group_fairness,
    group_accuracy_gap,
    normalize_prediction,
)


def test_normalize_prediction_removes_final_answer_tags():
    assert normalize_prediction("<final_answer>Subjective</final_answer>") == "subjective"


def test_counterfactual_flip_rate_no_flips():
    base_predictions = ["objective", "subjective", "objective"]
    counterfactual_predictions = ["objective", "subjective", "objective"]

    flip_rate, num_flips = counterfactual_flip_rate(
        base_predictions,
        counterfactual_predictions,
    )

    assert flip_rate == 0.0
    assert num_flips == 0


def test_counterfactual_flip_rate_with_flips():
    base_predictions = ["objective", "subjective", "objective"]
    counterfactual_predictions = ["subjective", "subjective", "objective"]

    flip_rate, num_flips = counterfactual_flip_rate(
        base_predictions,
        counterfactual_predictions,
    )

    assert num_flips == 1
    assert flip_rate == 1 / 3


def test_evaluate_counterfactual_fairness():
    base_predictions = ["objective", "subjective"]
    counterfactual_predictions = ["subjective", "subjective"]

    result = evaluate_counterfactual_fairness(
        base_predictions=base_predictions,
        counterfactual_predictions=counterfactual_predictions,
    )

    assert result.num_pairs == 2
    assert result.num_flips == 1
    assert result.counterfactual_flip_rate == 0.5
    assert result.fairness_risk == 0.5
    assert result.details["method"] == "counterfactual_flip_rate"


def test_group_accuracy_gap():
    predictions = ["yes", "no", "yes", "yes"]
    labels = ["yes", "no", "no", "no"]
    groups = ["A", "A", "B", "B"]

    gap = group_accuracy_gap(
        predictions=predictions,
        labels=labels,
        groups=groups,
    )

    # Group A accuracy = 2/2 = 1.0
    # Group B accuracy = 0/2 = 0.0
    assert gap == 1.0


def test_evaluate_group_fairness():
    predictions = ["yes", "no", "yes", "yes"]
    labels = ["yes", "no", "no", "no"]
    groups = ["A", "A", "B", "B"]

    result = evaluate_group_fairness(
        predictions=predictions,
        labels=labels,
        groups=groups,
    )

    assert result.group_accuracy_gap == 1.0
    assert result.fairness_risk == 1.0
    assert result.details["method"] == "group_accuracy_gap"


def test_bias_violation_rate_no_violations():
    outputs = [
        "The answer is based on the text.",
        "There is not enough evidence.",
    ]

    rate, num_violations = bias_violation_rate(outputs)

    assert rate == 0.0
    assert num_violations == 0


def test_bias_violation_rate_with_violation():
    outputs = [
        "The answer is based on the text.",
        "This person is less capable because of their gender.",
    ]

    rate, num_violations = bias_violation_rate(outputs)

    assert rate == 0.5
    assert num_violations == 1


def test_evaluate_bias_language():
    outputs = [
        "The answer is based on the text.",
        "This person is less capable because of their gender.",
    ]

    result = evaluate_bias_language(outputs)

    assert result.bias_violation_rate == 0.5
    assert result.fairness_risk == 0.5
    assert result.details["method"] == "bias_violation_rate"
    assert result.details["num_bias_violations"] == 1


def test_fairness_debt_tracker_update_and_reset():
    tracker = FairnessDebtTracker(decay=0.5, violation_weight=1.0)

    debt_1 = tracker.update(prompt_id="p1", fairness_risk=0.4)
    debt_2 = tracker.update(prompt_id="p1", fairness_risk=0.2)

    assert debt_1 == 0.4
    assert debt_2 == 0.4  # 0.5 * 0.4 + 0.2
    assert tracker.get_debt("p1") == 0.4

    tracker.reset("p1")
    assert tracker.get_debt("p1") == 0.0


# ---------------------------------------------------------------------------
# expected_same_prediction-aware violation rate
# ---------------------------------------------------------------------------


def test_flip_rate_expected_same_treats_flip_as_violation():
    base = ["objective", "subjective"]
    cf = ["subjective", "subjective"]

    # Both pairs expected to stay the same -> the single flip is a violation.
    rate, num = counterfactual_flip_rate(
        base, cf, expected_same_prediction=[True, True]
    )

    assert num == 1
    assert rate == 0.5


def test_flip_rate_expected_different_treats_nonflip_as_violation():
    base = ["objective", "subjective"]
    cf = ["subjective", "subjective"]

    # Pair 0 flips (good, since a change was expected); pair 1 does NOT flip
    # but a change was expected -> that is the violation.
    rate, num = counterfactual_flip_rate(
        base, cf, expected_same_prediction=[False, False]
    )

    assert num == 1
    assert rate == 0.5


def test_flip_rate_expected_length_mismatch_raises():
    try:
        counterfactual_flip_rate(["a"], ["b"], expected_same_prediction=[True, False])
    except ValueError:
        return
    raise AssertionError("expected ValueError on length mismatch")


# ---------------------------------------------------------------------------
# combine_fairness_risk
# ---------------------------------------------------------------------------


def test_combine_fairness_risk_weighted_blend():
    config = CombinedFairnessConfig(
        flip_weight=0.5,
        group_gap_weight=0.25,
        bias_weight=0.15,
        debt_weight=0.10,
    )

    risk, breakdown = combine_fairness_risk(
        flip_rate=0.4,
        group_gap=0.2,
        bias_rate=0.0,
        fairness_debt=0.0,
        config=config,
    )

    # All four signals present (debt/bias are 0.0, still counted).
    expected = 0.5 * 0.4 + 0.25 * 0.2 + 0.15 * 0.0 + 0.10 * 0.0
    assert abs(risk - expected) < 1e-9
    assert set(breakdown["effective_weights"].keys()) == {
        "counterfactual_flip_rate",
        "group_accuracy_gap",
        "bias_violation_rate",
        "fairness_debt",
    }


def test_combine_fairness_risk_renormalizes_missing_signals():
    config = CombinedFairnessConfig(
        flip_weight=0.5,
        group_gap_weight=0.25,
        bias_weight=0.15,
        debt_weight=0.10,
    )

    # Only flip rate present -> its effective weight renormalizes to 1.0.
    risk, breakdown = combine_fairness_risk(flip_rate=0.4, config=config)

    assert abs(risk - 0.4) < 1e-9
    assert breakdown["effective_weights"] == {"counterfactual_flip_rate": 1.0}


def test_combine_fairness_risk_no_signals_returns_zero():
    risk, breakdown = combine_fairness_risk()

    assert risk == 0.0
    assert breakdown["effective_weights"] == {}


def test_combine_fairness_risk_clamps_into_unit_interval():
    config = CombinedFairnessConfig(flip_weight=1.0, group_gap_weight=0.0,
                                    bias_weight=0.0, debt_weight=0.0)

    risk, _ = combine_fairness_risk(flip_rate=5.0, config=config)

    assert risk == 1.0


# ---------------------------------------------------------------------------
# evaluate_combined_fairness
# ---------------------------------------------------------------------------


def test_evaluate_combined_fairness_blends_present_signals():
    result = evaluate_combined_fairness(
        base_predictions=["objective", "subjective"],
        counterfactual_predictions=["subjective", "subjective"],
        outputs=["fine", "this person is less capable because of their gender"],
        config=CombinedFairnessConfig(
            flip_weight=0.5, group_gap_weight=0.25,
            bias_weight=0.15, debt_weight=0.10,
        ),
    )

    assert result.details["method"] == "combined_fairness_risk"
    assert result.counterfactual_flip_rate == 0.5
    assert result.bias_violation_rate == 0.5
    # flip and bias present (no groups, debt=0 -> skipped); renormalize 0.5/0.15.
    assert "counterfactual_flip_rate" in result.details["signals_present"]
    assert "bias_violation_rate" in result.details["signals_present"]
    assert 0.0 < result.fairness_risk <= 1.0


def test_evaluate_combined_fairness_honors_expected_same_prediction():
    # Both pairs flip, but both were expected to flip -> zero violation.
    result = evaluate_combined_fairness(
        base_predictions=["objective", "subjective"],
        counterfactual_predictions=["subjective", "objective"],
        expected_same_prediction=[False, False],
        config=CombinedFairnessConfig(
            flip_weight=1.0, group_gap_weight=0.0,
            bias_weight=0.0, debt_weight=0.0,
        ),
    )

    assert result.counterfactual_flip_rate == 0.0
    assert result.fairness_risk == 0.0


def test_evaluate_combined_fairness_includes_debt_signal():
    result = evaluate_combined_fairness(
        base_predictions=["objective"],
        counterfactual_predictions=["objective"],
        fairness_debt=0.8,
        config=CombinedFairnessConfig(
            flip_weight=0.5, group_gap_weight=0.0,
            bias_weight=0.0, debt_weight=0.5,
        ),
    )

    # flip_rate=0.0 (weight 0.5) + debt=0.8 (weight 0.5) -> 0.4
    assert abs(result.fairness_risk - 0.4) < 1e-9
    assert "fairness_debt" in result.details["signals_present"]

# ---------------------------------------------------------------------------
# Group / allocative fairness metrics (DSP, Equal Opportunity, Equalized Odds)
# ---------------------------------------------------------------------------


def test_demographic_parity_gap():
    # Group A predicts pos 2/2 = 1.0; group B predicts pos 0/2 = 0.0 -> gap 1.0
    preds = ["yes", "yes", "no", "no"]
    groups = ["A", "A", "B", "B"]
    assert demographic_parity_gap(preds, groups, positive_label="yes") == 1.0


def test_demographic_parity_gap_single_group_is_zero():
    assert demographic_parity_gap(["yes", "no"], ["A", "A"], "yes") == 0.0


def test_equal_opportunity_gap():
    # Among label=yes: A has 1/1 TPR, B has 0/1 TPR -> gap 1.0
    preds = ["yes", "no", "no", "no"]
    labels = ["yes", "no", "yes", "no"]
    groups = ["A", "A", "B", "B"]
    assert equal_opportunity_gap(preds, labels, groups, "yes") == 1.0


def test_equalized_odds_gap_takes_max_of_tpr_fpr():
    # TPR equal (both groups 1/1), but FPR differs: A 1/1, B 0/1 -> eqodds = 1.0
    preds = ["yes", "yes", "yes", "no"]
    labels = ["yes", "no", "yes", "no"]
    groups = ["A", "A", "B", "B"]
    assert equalized_odds_gap(preds, labels, groups, "yes") == 1.0


def test_evaluate_combined_fairness_includes_group_metrics_when_configured():
    preds = ["yes", "yes", "no", "no"]
    labels = ["yes", "no", "yes", "no"]
    groups = ["A", "A", "B", "B"]

    result = evaluate_combined_fairness(
        predictions=preds,
        labels=labels,
        groups=groups,
        config=CombinedFairnessConfig(
            flip_weight=0.0, group_gap_weight=0.0, bias_weight=0.0, debt_weight=0.0,
            demographic_parity_weight=1.0,
            equal_opportunity_weight=0.0,
            equalized_odds_weight=0.0,
            positive_label="yes",
        ),
    )
    assert "demographic_parity" in result.details["signals_present"]
    assert result.fairness_risk == 1.0  # DSP gap = 1.0, sole weighted signal


def test_group_metrics_inactive_without_positive_label():
    # Default config: group metrics weight 0 and no positive_label -> not present.
    result = evaluate_combined_fairness(
        predictions=["yes", "no"],
        labels=["yes", "no"],
        groups=["A", "B"],
        config=CombinedFairnessConfig(),
    )
    present = result.details["signals_present"]
    assert "demographic_parity" not in present
    assert "equal_opportunity" not in present
    assert "equalized_odds" not in present
