from __future__ import annotations

from heal_capo.fairness import equalized_odds_gap


def merged_equalized_odds(rows: list[dict], positive_label: str):
    predictions = [str(row.get("prediction", "")) for row in rows]
    labels = [str(row.get("gold", "")) for row in rows]
    groups = [str(row.get("group", "")) for row in rows]
    if not rows or not all(groups) or not (len(predictions) == len(labels) == len(groups)):
        return None
    return equalized_odds_gap(predictions, labels, groups, positive_label)
