import re
from typing import Optional


def normalize_text(value) -> str:
    """
    Normalize model predictions and labels for robust exact-match scoring.
    """
    if value is None:
        return ""

    text = str(value).strip().lower()

    # Remove common answer tags.
    text = re.sub(r"</?final_answer>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?answer>", "", text, flags=re.IGNORECASE)

    # Remove surrounding quotes/backticks.
    text = text.strip(" \n\t\r\"'`")

    # Remove trailing punctuation.
    text = re.sub(r"[.,;:!?]+$", "", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def accuracy(predictions, labels):
    """
    Simple robust exact-match accuracy.

    This is good for:
      - Subj: objective / subjective
      - AG News: World / Sports / Business / Tech
      - SST-5: terrible / bad / okay / good / great
    """
    if not labels:
        return 0.0

    correct = 0

    for pred, label in zip(predictions, labels):
        if normalize_text(pred) == normalize_text(label):
            correct += 1

    return correct / len(labels)


def classification_report_counts(
    predictions,
    labels,
    classes: Optional[list[str]] = None,
):
    """
    Lightweight per-class count summary.
    Useful for debugging label imbalance or systematic model errors.
    """
    rows = {}

    if classes:
        for cls in classes:
            key = normalize_text(cls)
            rows[key] = {
                "label": cls,
                "support": 0,
                "correct": 0,
                "predicted": 0,
            }

    for pred, label in zip(predictions, labels):
        pred_norm = normalize_text(pred)
        label_norm = normalize_text(label)

        if label_norm not in rows:
            rows[label_norm] = {
                "label": label,
                "support": 0,
                "correct": 0,
                "predicted": 0,
            }

        if pred_norm not in rows:
            rows[pred_norm] = {
                "label": pred,
                "support": 0,
                "correct": 0,
                "predicted": 0,
            }

        rows[label_norm]["support"] += 1
        rows[pred_norm]["predicted"] += 1

        if pred_norm == label_norm:
            rows[label_norm]["correct"] += 1

    return list(rows.values())