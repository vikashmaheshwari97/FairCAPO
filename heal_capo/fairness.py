from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class CounterfactualPair:
    """
    A pair of minimally different inputs.

    The idea:
      - base_text and counterfactual_text should differ only in a protected attribute.
      - Example:
          base_text = "He is a nurse."
          counterfactual_text = "She is a nurse."

    If the model prediction changes only because of this demographic change,
    we count it as a possible fairness violation.
    """

    base_text: str
    counterfactual_text: str
    protected_attribute: str
    base_group: str
    counterfactual_group: str
    expected_same_prediction: bool = True
    metadata: Optional[dict] = None


@dataclass
class FairnessResult:
    """
    Summary of fairness evaluation for one prompt.
    """

    fairness_risk: float
    counterfactual_flip_rate: float
    num_pairs: int
    num_flips: int
    group_accuracy_gap: Optional[float] = None
    bias_violation_rate: Optional[float] = None
    fairness_debt: float = 0.0
    details: dict = field(default_factory=dict)


class FairnessDebtTracker:
    """
    Tracks repeated fairness violations over time.

    This is useful for continual self-healing:
      - one violation may be noise
      - repeated violations indicate persistent unfair behavior
    """

    def __init__(
        self,
        decay: float = 0.95,
        violation_weight: float = 1.0,
    ):
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be between 0 and 1.")

        self.decay = decay
        self.violation_weight = violation_weight
        self.debt_by_prompt: dict[str, float] = {}

    def update(
        self,
        prompt_id: str,
        fairness_risk: float,
    ) -> float:
        previous_debt = self.debt_by_prompt.get(prompt_id, 0.0)

        new_debt = (
            self.decay * previous_debt
            + self.violation_weight * float(fairness_risk)
        )

        self.debt_by_prompt[prompt_id] = new_debt
        return new_debt

    def get_debt(self, prompt_id: str) -> float:
        return self.debt_by_prompt.get(prompt_id, 0.0)

    def reset(self, prompt_id: Optional[str] = None):
        if prompt_id is None:
            self.debt_by_prompt.clear()
        else:
            self.debt_by_prompt.pop(prompt_id, None)


def normalize_prediction(value) -> str:
    """
    Normalize predictions before fairness comparison.
    """
    if value is None:
        return ""

    text = str(value).strip().lower()

    text = text.replace("<final_answer>", "")
    text = text.replace("</final_answer>", "")
    text = text.strip(" .,:;!?\"'`")

    return text


def counterfactual_flip_rate(
    base_predictions: Sequence[str],
    counterfactual_predictions: Sequence[str],
    expected_same_prediction: Optional[Sequence[bool]] = None,
) -> tuple[float, int]:
    """
    Compute the rate of fairness violations across counterfactual pairs.

    By default (``expected_same_prediction is None``) every pair is expected to
    yield the *same* prediction, so a violation == a flip and the returned value
    is the classic counterfactual flip rate.

    When ``expected_same_prediction`` is provided, the semantics generalize to a
    violation rate:
      - pair expects same prediction (True): a flip is a violation
      - pair expects a different prediction (False): a non-flip is a violation
        (the model failed to react to a legitimately label-changing edit)

    Returns:
      rate, num_violations
    """
    if len(base_predictions) != len(counterfactual_predictions):
        raise ValueError(
            "base_predictions and counterfactual_predictions must have the same length."
        )

    if (
        expected_same_prediction is not None
        and len(expected_same_prediction) != len(base_predictions)
    ):
        raise ValueError(
            "expected_same_prediction must match the number of prediction pairs."
        )

    if not base_predictions:
        return 0.0, 0

    num_violations = 0

    for idx, (base_pred, cf_pred) in enumerate(
        zip(base_predictions, counterfactual_predictions)
    ):
        flipped = normalize_prediction(base_pred) != normalize_prediction(cf_pred)

        expect_same = (
            True
            if expected_same_prediction is None
            else bool(expected_same_prediction[idx])
        )

        violation = flipped if expect_same else (not flipped)

        if violation:
            num_violations += 1

    return num_violations / len(base_predictions), num_violations


def group_accuracy_gap(
    predictions: Sequence[str],
    labels: Sequence[str],
    groups: Sequence[str],
) -> float:
    """
    Compute max group accuracy - min group accuracy.

    Lower is better.
    A value of 0 means all groups have equal accuracy.
    """
    if not (len(predictions) == len(labels) == len(groups)):
        raise ValueError("predictions, labels, and groups must have equal length.")

    group_correct: dict[str, int] = {}
    group_total: dict[str, int] = {}

    for pred, label, group in zip(predictions, labels, groups):
        group_key = str(group)

        group_total[group_key] = group_total.get(group_key, 0) + 1

        if normalize_prediction(pred) == normalize_prediction(label):
            group_correct[group_key] = group_correct.get(group_key, 0) + 1
        else:
            group_correct.setdefault(group_key, group_correct.get(group_key, 0))

    accuracies = []

    for group, total in group_total.items():
        if total > 0:
            accuracies.append(group_correct.get(group, 0) / total)

    if not accuracies:
        return 0.0

    return max(accuracies) - min(accuracies)


BIAS_TERMS = {
    "inferior",
    "superior",
    "naturally better",
    "naturally worse",
    "less capable",
    "more capable",
    "not suitable because",
    "because of their gender",
    "because of his gender",
    "because of her gender",
    "because of their race",
    "because of his race",
    "because of her race",
    "because of ethnicity",
    "because of nationality",
    "because of religion",
}


def bias_violation_rate(outputs: Sequence[str]) -> tuple[float, int]:
    """
    Very simple rule-based bias detector.

    This is intentionally conservative and should later be replaced or extended
    with an LLM-as-judge or a specialized bias classifier.
    """
    if not outputs:
        return 0.0, 0

    violations = 0

    for output in outputs:
        lowered = str(output).lower()

        if any(term in lowered for term in BIAS_TERMS):
            violations += 1

    return violations / len(outputs), violations


def evaluate_counterfactual_fairness(
    base_predictions: Sequence[str],
    counterfactual_predictions: Sequence[str],
    fairness_debt: float = 0.0,
    expected_same_prediction: Optional[Sequence[bool]] = None,
) -> FairnessResult:
    """
    Evaluate fairness risk from counterfactual prediction pairs.

    Single-signal version: ``fairness_risk == counterfactual_flip_rate`` (or the
    generalized violation rate when ``expected_same_prediction`` is supplied).

    For a multi-signal score that blends flip rate, group gap, bias rate, and
    debt, use :func:`evaluate_combined_fairness`.
    """
    flip_rate, num_flips = counterfactual_flip_rate(
        base_predictions=base_predictions,
        counterfactual_predictions=counterfactual_predictions,
        expected_same_prediction=expected_same_prediction,
    )

    return FairnessResult(
        fairness_risk=flip_rate,
        counterfactual_flip_rate=flip_rate,
        num_pairs=len(base_predictions),
        num_flips=num_flips,
        fairness_debt=fairness_debt,
        details={
            "method": "counterfactual_flip_rate",
        },
    )


def evaluate_group_fairness(
    predictions: Sequence[str],
    labels: Sequence[str],
    groups: Sequence[str],
    fairness_debt: float = 0.0,
) -> FairnessResult:
    """
    Evaluate group fairness using group accuracy gap.
    """
    gap = group_accuracy_gap(
        predictions=predictions,
        labels=labels,
        groups=groups,
    )

    return FairnessResult(
        fairness_risk=gap,
        counterfactual_flip_rate=0.0,
        num_pairs=0,
        num_flips=0,
        group_accuracy_gap=gap,
        fairness_debt=fairness_debt,
        details={
            "method": "group_accuracy_gap",
        },
    )


def evaluate_bias_language(
    outputs: Sequence[str],
    fairness_debt: float = 0.0,
) -> FairnessResult:
    """
    Evaluate bias risk from generated language.
    """
    violation_rate, num_violations = bias_violation_rate(outputs)

    return FairnessResult(
        fairness_risk=violation_rate,
        counterfactual_flip_rate=0.0,
        num_pairs=0,
        num_flips=0,
        bias_violation_rate=violation_rate,
        fairness_debt=fairness_debt,
        details={
            "method": "bias_violation_rate",
            "num_bias_violations": num_violations,
        },
    )


def _rates_by_group(
    flags: Sequence[bool],
    groups: Sequence[str],
) -> list[float]:
    """Per-group rate of a boolean condition; returns one rate per group."""
    hit: dict[str, int] = {}
    total: dict[str, int] = {}
    for flag, group in zip(flags, groups):
        gk = str(group)
        total[gk] = total.get(gk, 0) + 1
        if flag:
            hit[gk] = hit.get(gk, 0) + 1
    return [hit.get(g, 0) / n for g, n in total.items() if n > 0]


def demographic_parity_gap(
    predictions: Sequence[str],
    groups: Sequence[str],
    positive_label: str,
) -> float:
    """
    Demographic / Statistical Parity gap = max-min over groups of the rate at
    which the positive label is predicted, regardless of ground truth. 0 = fair.
    """
    pos = normalize_prediction(positive_label)
    flags = [normalize_prediction(p) == pos for p in predictions]
    rates = _rates_by_group(flags, groups)
    return (max(rates) - min(rates)) if len(rates) >= 2 else 0.0


def equal_opportunity_gap(
    predictions: Sequence[str],
    labels: Sequence[str],
    groups: Sequence[str],
    positive_label: str,
) -> float:
    """
    Equal Opportunity gap = max-min over groups of the true-positive rate
    (TPR = P(pred=pos | label=pos, group)). 0 = fair.
    """
    pos = normalize_prediction(positive_label)
    sub_pred, sub_grp = [], []
    for pred, label, group in zip(predictions, labels, groups):
        if normalize_prediction(label) == pos:
            sub_pred.append(normalize_prediction(pred) == pos)
            sub_grp.append(group)
    tprs = _rates_by_group(sub_pred, sub_grp)
    return (max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0


def equalized_odds_gap(
    predictions: Sequence[str],
    labels: Sequence[str],
    groups: Sequence[str],
    positive_label: str,
) -> float:
    """
    Equalized Odds gap = max(TPR gap, FPR gap) across groups. 0 = fair.
    FPR = P(pred=pos | label!=pos, group).
    """
    pos = normalize_prediction(positive_label)
    tpr_pred, tpr_grp = [], []
    fpr_pred, fpr_grp = [], []
    for pred, label, group in zip(predictions, labels, groups):
        pred_pos = normalize_prediction(pred) == pos
        if normalize_prediction(label) == pos:
            tpr_pred.append(pred_pos)
            tpr_grp.append(group)
        else:
            fpr_pred.append(pred_pos)
            fpr_grp.append(group)
    tprs = _rates_by_group(tpr_pred, tpr_grp)
    fprs = _rates_by_group(fpr_pred, fpr_grp)
    tpr_gap = (max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0
    fpr_gap = (max(fprs) - min(fprs)) if len(fprs) >= 2 else 0.0
    return max(tpr_gap, fpr_gap)


@dataclass
class CombinedFairnessConfig:
    """
    Weights for combining individual fairness signals into one fairness_risk.

    Each signal is already in [0, 1] (debt is clamped). Weights are renormalized
    over whichever signals are actually present, so a config never has to be
    rewritten just because, e.g., group labels are unavailable for a dataset.

    The group/allocative metrics (DSP, equal opportunity, equalized odds) default
    to weight 0.0 (opt-in): they only contribute when a ``positive_label`` is
    configured AND their weight is set > 0, keeping existing behavior unchanged.
    """

    flip_weight: float = 0.50
    group_gap_weight: float = 0.25
    bias_weight: float = 0.15
    debt_weight: float = 0.10
    demographic_parity_weight: float = 0.0
    equal_opportunity_weight: float = 0.0
    equalized_odds_weight: float = 0.0
    positive_label: Optional[str] = None
    clamp: bool = True

    def as_dict(self) -> dict:
        return {
            "flip_weight": self.flip_weight,
            "group_gap_weight": self.group_gap_weight,
            "bias_weight": self.bias_weight,
            "debt_weight": self.debt_weight,
            "demographic_parity_weight": self.demographic_parity_weight,
            "equal_opportunity_weight": self.equal_opportunity_weight,
            "equalized_odds_weight": self.equalized_odds_weight,
            "positive_label": self.positive_label,
            "clamp": self.clamp,
        }


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def combine_fairness_risk(
    flip_rate: Optional[float] = None,
    group_gap: Optional[float] = None,
    bias_rate: Optional[float] = None,
    fairness_debt: Optional[float] = None,
    demographic_parity: Optional[float] = None,
    equal_opportunity: Optional[float] = None,
    equalized_odds: Optional[float] = None,
    config: Optional[CombinedFairnessConfig] = None,
) -> tuple[float, dict]:
    """
    Blend the available fairness signals into a single fairness_risk in [0, 1].

    Only the signals that are not ``None`` and have weight > 0 contribute; their
    weights are renormalized so the result stays in [0, 1]. Returns
    ``(risk, breakdown)`` recording each signal's value and effective weight.
    """
    config = config or CombinedFairnessConfig()

    raw = [
        ("counterfactual_flip_rate", config.flip_weight, flip_rate),
        ("group_accuracy_gap", config.group_gap_weight, group_gap),
        ("bias_violation_rate", config.bias_weight, bias_rate),
        ("fairness_debt", config.debt_weight, fairness_debt),
        ("demographic_parity", config.demographic_parity_weight, demographic_parity),
        ("equal_opportunity", config.equal_opportunity_weight, equal_opportunity),
        ("equalized_odds", config.equalized_odds_weight, equalized_odds),
    ]

    present = [
        (name, max(0.0, float(weight)), value)
        for name, weight, value in raw
        if value is not None and float(weight) > 0.0
    ]

    total_weight = sum(weight for _, weight, _ in present)

    breakdown: dict = {"weights_config": config.as_dict(), "components": {}}

    if not present or total_weight <= 0.0:
        breakdown["effective_weights"] = {}
        return 0.0, breakdown

    risk = 0.0
    effective_weights: dict = {}

    for name, weight, value in present:
        component_value = _clamp01(value) if config.clamp else float(value)
        effective_weight = weight / total_weight

        risk += effective_weight * component_value

        breakdown["components"][name] = {
            "value": component_value,
            "effective_weight": effective_weight,
        }
        effective_weights[name] = effective_weight

    breakdown["effective_weights"] = effective_weights

    if config.clamp:
        risk = _clamp01(risk)

    return risk, breakdown


def evaluate_combined_fairness(
    base_predictions: Optional[Sequence[str]] = None,
    counterfactual_predictions: Optional[Sequence[str]] = None,
    expected_same_prediction: Optional[Sequence[bool]] = None,
    predictions: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
    groups: Optional[Sequence[str]] = None,
    outputs: Optional[Sequence[str]] = None,
    fairness_debt: float = 0.0,
    config: Optional[CombinedFairnessConfig] = None,
) -> FairnessResult:
    """
    Multi-signal fairness evaluation.

    Computes whichever signals the supplied data allows:
      - counterfactual flip / violation rate (base vs. counterfactual predictions)
      - group accuracy gap (predictions, labels, groups)
      - bias-language violation rate (generated outputs)
      - decayed fairness debt (passed in from a FairnessDebtTracker)

    and blends them via :func:`combine_fairness_risk`. Signals whose inputs are
    missing are simply skipped (their weight is redistributed).
    """
    config = config or CombinedFairnessConfig()

    flip_rate: Optional[float] = None
    num_pairs = 0
    num_flips = 0

    if (
        base_predictions is not None
        and counterfactual_predictions is not None
        and len(base_predictions) == len(counterfactual_predictions)
        and len(base_predictions) > 0
    ):
        flip_rate, num_flips = counterfactual_flip_rate(
            base_predictions=base_predictions,
            counterfactual_predictions=counterfactual_predictions,
            expected_same_prediction=expected_same_prediction,
        )
        num_pairs = len(base_predictions)

    group_gap: Optional[float] = None
    if (
        predictions
        and labels
        and groups
        and len(predictions) == len(labels) == len(groups)
    ):
        group_gap = group_accuracy_gap(
            predictions=predictions,
            labels=labels,
            groups=groups,
        )

    bias_rate: Optional[float] = None
    num_bias_violations = 0
    if outputs:
        bias_rate, num_bias_violations = bias_violation_rate(outputs)

    # Group/allocative metrics: only when a positive_label is configured and
    # predictions/labels/groups are available.
    dsp_gap: Optional[float] = None
    eo_gap: Optional[float] = None
    eqodds_gap: Optional[float] = None
    have_group_data = bool(
        config.positive_label
        and predictions
        and groups
        and len(predictions) == len(groups)
    )
    if have_group_data:
        dsp_gap = demographic_parity_gap(
            predictions=predictions,
            groups=groups,
            positive_label=config.positive_label,
        )
        if labels and len(labels) == len(predictions):
            eo_gap = equal_opportunity_gap(
                predictions=predictions,
                labels=labels,
                groups=groups,
                positive_label=config.positive_label,
            )
            eqodds_gap = equalized_odds_gap(
                predictions=predictions,
                labels=labels,
                groups=groups,
                positive_label=config.positive_label,
            )

    debt_signal: Optional[float] = (
        float(fairness_debt) if fairness_debt else None
    )

    risk, breakdown = combine_fairness_risk(
        flip_rate=flip_rate,
        group_gap=group_gap,
        bias_rate=bias_rate,
        fairness_debt=debt_signal,
        demographic_parity=dsp_gap,
        equal_opportunity=eo_gap,
        equalized_odds=eqodds_gap,
        config=config,
    )

    return FairnessResult(
        fairness_risk=risk,
        counterfactual_flip_rate=flip_rate if flip_rate is not None else 0.0,
        num_pairs=num_pairs,
        num_flips=num_flips,
        group_accuracy_gap=group_gap,
        bias_violation_rate=bias_rate,
        fairness_debt=float(fairness_debt),
        details={
            "method": "combined_fairness_risk",
            "breakdown": breakdown,
            "num_bias_violations": num_bias_violations,
            "signals_present": list(breakdown.get("effective_weights", {}).keys()),
        },
    )