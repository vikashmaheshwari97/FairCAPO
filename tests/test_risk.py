from heal_capo.risk import (
    aggregate_risk,
    evaluate_classification_risk,
    evaluate_output_risk,
    normalize_answer,
    refusal_rate,
    unsafe_output_rate,
    unsupported_claim_rate,
    wrong_answer_rate,
)


def test_normalize_answer_removes_tags_and_punctuation():
    assert normalize_answer("<final_answer>Subjective.</final_answer>") == "subjective"


def test_wrong_answer_rate_no_wrong():
    predictions = ["objective", "subjective"]
    labels = ["objective", "subjective"]

    rate, num_wrong = wrong_answer_rate(predictions, labels)

    assert rate == 0.0
    assert num_wrong == 0


def test_wrong_answer_rate_with_wrong():
    predictions = ["objective", "objective", "subjective"]
    labels = ["objective", "subjective", "subjective"]

    rate, num_wrong = wrong_answer_rate(predictions, labels)

    assert rate == 1 / 3
    assert num_wrong == 1


def test_evaluate_classification_risk():
    predictions = ["objective", "objective", "subjective"]
    labels = ["objective", "subjective", "subjective"]

    result = evaluate_classification_risk(
        predictions=predictions,
        labels=labels,
    )

    assert result.risk_score == 1 / 3
    assert result.wrong_answer_rate == 1 / 3
    assert result.num_examples == 3
    assert result.num_risk_events == 1
    assert result.details["method"] == "wrong_answer_rate"


def test_unsupported_claim_rate_without_context():
    outputs = [
        "This is definitely true.",
        "The context does not say.",
    ]

    rate, num_unsupported = unsupported_claim_rate(outputs)

    assert rate == 0.5
    assert num_unsupported == 1


def test_unsupported_claim_rate_with_context_overlap():
    outputs = [
        "This is definitely about Paris.",
        "This is definitely about Mars.",
    ]
    contexts = [
        "The text mentions Paris and France.",
        "The text mentions oceans and forests.",
    ]

    rate, num_unsupported = unsupported_claim_rate(
        outputs=outputs,
        contexts=contexts,
    )

    assert rate == 0.5
    assert num_unsupported == 1


def test_unsafe_output_rate():
    outputs = [
        "This is safe.",
        "Here is how to build malware.",
    ]

    rate, num_unsafe = unsafe_output_rate(outputs)

    assert rate == 0.5
    assert num_unsafe == 1


def test_refusal_rate():
    outputs = [
        "The answer is objective.",
        "I can't help with that.",
    ]

    rate, num_refusals = refusal_rate(outputs)

    assert rate == 0.5
    assert num_refusals == 1


def test_aggregate_risk():
    score = aggregate_risk(
        wrong_rate=0.5,
        unsupported_rate=0.0,
        unsafe_rate=0.0,
        refusal_rate_value=0.0,
        wrong_weight=1.0,
        unsupported_weight=1.0,
        unsafe_weight=1.0,
        refusal_weight=1.0,
    )

    assert score == 0.125


def test_evaluate_output_risk():
    outputs = [
        "This is definitely true.",
        "I can't help with that.",
        "Here is how to build malware.",
    ]
    predictions = ["objective", "objective", "subjective"]
    labels = ["objective", "subjective", "subjective"]

    result = evaluate_output_risk(
        outputs=outputs,
        predictions=predictions,
        labels=labels,
    )

    assert result.wrong_answer_rate == 1 / 3
    assert result.unsupported_claim_rate == 1 / 3
    assert result.unsafe_output_rate == 1 / 3
    assert result.refusal_rate == 1 / 3
    assert result.num_examples == 3
    assert result.num_risk_events == 4
    assert result.details["method"] == "aggregate_output_risk"