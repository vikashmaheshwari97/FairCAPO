"""
Diagnose Bias-in-Bios held-out CSVs without new LLM calls.

The held-out evaluation CSV stores ``detail_predictions`` as JSON. This script
recomputes micro accuracy, macro accuracy, overall group accuracy gap, and the
worst label-conditioned group accuracy gap for each candidate row. It also
writes CPU-only error-analysis tables for profession accuracy, gender accuracy,
gold/predicted confusion pairs, and the top label confusions per candidate.

Example:

PYTHONPATH=. python scripts/diagnose_bios_fairness.py \
  --inputs outputs/hpc/evaluation_large/seed_0/bios_faircapo_500k_v1/test_eval_candidates.csv \
           outputs/hpc/evaluation_large/seed_0/bios_nsga2po_500k_v1/test_eval_candidates.csv \
  --out-dir outputs/diagnostics/bios_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def raise_csv_field_limit() -> None:
    """
    Held-out evaluation rows store large JSON prediction traces in one cell.
    Python's default csv field limit is too small for those rows on Rocket.
    """
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit / 10)

from heal_capo.fairness import (
    group_accuracy_gap,
    label_conditioned_group_accuracy_gap,
    normalize_prediction,
)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_predictions(raw: str) -> list[dict[str, Any]]:
    if not raw:
        return []
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def macro_accuracy(predictions: list[dict[str, Any]]) -> float:
    by_label: dict[str, list[bool]] = {}
    for item in predictions:
        label = normalize_prediction(str(item.get("gold", "")))
        if not label:
            continue
        by_label.setdefault(label, []).append(bool(item.get("correct", False)))
    if not by_label:
        return 0.0
    return sum(sum(vals) / len(vals) for vals in by_label.values()) / len(by_label)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def candidate_prefix(source: Path, row: dict[str, str]) -> dict[str, Any]:
    return {
        "source": source.as_posix(),
        "candidate_id": row.get("candidate_id", ""),
        "method": row.get("method", ""),
        "category": row.get("category", ""),
        "is_pareto": row.get("is_pareto", ""),
        "performance": row.get("performance", ""),
        "cost": row.get("cost", ""),
        "fairness_risk": row.get("fairness_risk", ""),
    }


def profession_accuracy_rows(
    source: Path,
    row: dict[str, str],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prefix = candidate_prefix(source, row)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for item in predictions:
        gold = normalize_prediction(str(item.get("gold", "")))
        if gold:
            by_label.setdefault(gold, []).append(item)

    rows: list[dict[str, Any]] = []
    for label, items in sorted(by_label.items()):
        total = len(items)
        correct = sum(1 for item in items if as_bool(item.get("correct", False)))
        rows.append(
            {
                **prefix,
                "label": label,
                "n_examples": total,
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
            }
        )
    return rows


def gender_accuracy_rows(
    source: Path,
    row: dict[str, str],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prefix = candidate_prefix(source, row)
    by_group: dict[str, list[dict[str, Any]]] = {}
    for item in predictions:
        group = str(item.get("group", "")).strip().lower()
        if group:
            by_group.setdefault(group, []).append(item)

    rows: list[dict[str, Any]] = []
    for group, items in sorted(by_group.items()):
        total = len(items)
        correct = sum(1 for item in items if as_bool(item.get("correct", False)))
        rows.append(
            {
                **prefix,
                "group": group,
                "n_examples": total,
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
            }
        )
    return rows


def confusion_rows(
    source: Path,
    row: dict[str, str],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prefix = candidate_prefix(source, row)
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for item in predictions:
        gold = normalize_prediction(str(item.get("gold", "")))
        pred = normalize_prediction(str(item.get("prediction", "")))
        if not gold or not pred:
            continue
        key = (gold, pred)
        counts.setdefault(key, {"count": 0, "correct": 0})
        counts[key]["count"] += 1
        if gold == pred:
            counts[key]["correct"] += 1

    rows: list[dict[str, Any]] = []
    for (gold, pred), values in sorted(
        counts.items(), key=lambda item: (-item[1]["count"], item[0])
    ):
        rows.append(
            {
                **prefix,
                "gold": gold,
                "prediction": pred,
                "count": values["count"],
                "correct": values["correct"],
                "is_error": gold != pred,
            }
        )
    return rows


def top_confusion_rows(
    confusion: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    by_candidate: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in confusion:
        if not as_bool(row.get("is_error", False)):
            continue
        key = (str(row.get("source", "")), str(row.get("candidate_id", "")))
        by_candidate.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    for items in by_candidate.values():
        rows.extend(
            sorted(
                items,
                key=lambda item: (
                    -int(item.get("count", 0)),
                    str(item.get("gold", "")),
                    str(item.get("prediction", "")),
                ),
            )[:top_k]
        )
    return rows


def multiclass_demographic_parity_gap(
    predictions: list[str],
    groups: list[str],
) -> float:
    """
    Max total-variation distance between predicted-label distributions by group.
    """
    labels = sorted({normalize_prediction(pred) for pred in predictions if pred})
    group_keys = sorted({str(group) for group in groups if str(group)})
    if len(group_keys) < 2 or not labels:
        return 0.0

    distributions: dict[str, dict[str, float]] = {}
    for group in group_keys:
        group_preds = [
            normalize_prediction(pred)
            for pred, pred_group in zip(predictions, groups)
            if str(pred_group) == group
        ]
        total = len(group_preds)
        distributions[group] = {
            label: (group_preds.count(label) / total if total else 0.0)
            for label in labels
        }

    worst = 0.0
    for idx, left in enumerate(group_keys):
        for right in group_keys[idx + 1 :]:
            tv = 0.5 * sum(
                abs(distributions[left][label] - distributions[right][label])
                for label in labels
            )
            worst = max(worst, tv)
    return worst


def summarize_candidate(
    source: Path,
    row: dict[str, str],
    min_count_per_group: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions = parse_predictions(row.get("detail_predictions", ""))
    pred_labels = [str(item.get("prediction", "")) for item in predictions]
    gold_labels = [str(item.get("gold", "")) for item in predictions]
    groups = [str(item.get("group", "")) for item in predictions]
    correct = [bool(item.get("correct", False)) for item in predictions]

    micro_acc = sum(correct) / len(correct) if correct else 0.0
    overall_gap = (
        group_accuracy_gap(pred_labels, gold_labels, groups)
        if pred_labels and len(pred_labels) == len(gold_labels) == len(groups)
        else 0.0
    )
    label_gap, label_details = label_conditioned_group_accuracy_gap(
        pred_labels,
        gold_labels,
        groups,
        min_count_per_group=min_count_per_group,
    )
    dp_gap = multiclass_demographic_parity_gap(pred_labels, groups)

    summary = {
        "source": source.as_posix(),
        "candidate_id": row.get("candidate_id", ""),
        "method": row.get("method", ""),
        "category": row.get("category", ""),
        "is_pareto": row.get("is_pareto", ""),
        "csv_performance": row.get("performance", ""),
        "micro_accuracy": micro_acc,
        "macro_accuracy": macro_accuracy(predictions),
        "csv_fairness_risk": row.get("fairness_risk", ""),
        "overall_group_accuracy_gap": overall_gap,
        "label_conditioned_group_accuracy_gap": label_gap,
        "label_conditioned_mean_gap": label_details.get(
            "mean_label_conditioned_gap", 0.0
        ),
        "worst_label": label_details.get("worst_label"),
        "num_valid_label_gaps": label_details.get("num_valid_labels", 0),
        "multiclass_demographic_parity_gap": dp_gap,
        "cost": row.get("cost", ""),
        "risk": row.get("risk", ""),
        "n_examples": len(predictions),
        "prompt": row.get("prompt", ""),
    }

    per_label_rows: list[dict[str, Any]] = []
    for item in label_details.get("per_label", []):
        per_label_rows.append(
            {
                "source": source.as_posix(),
                "candidate_id": row.get("candidate_id", ""),
                "label": item.get("label"),
                "gap": item.get("gap"),
                "rates": json.dumps(item.get("rates", {}), sort_keys=True),
                "support": json.dumps(item.get("support", {}), sort_keys=True),
            }
        )

    return summary, per_label_rows


def main() -> None:
    raise_csv_field_limit()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-count-per-group", type=int, default=1)
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    per_label: list[dict[str, Any]] = []
    profession_accuracy: list[dict[str, Any]] = []
    gender_accuracy: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []

    for input_path in args.inputs:
        path = Path(input_path)
        for row in load_rows(path):
            summary, label_rows = summarize_candidate(
                source=path,
                row=row,
                min_count_per_group=args.min_count_per_group,
            )
            summaries.append(summary)
            per_label.extend(label_rows)
            predictions = parse_predictions(row.get("detail_predictions", ""))
            profession_accuracy.extend(
                profession_accuracy_rows(path, row, predictions)
            )
            gender_accuracy.extend(gender_accuracy_rows(path, row, predictions))
            confusion.extend(confusion_rows(path, row, predictions))

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "bios_fairness_diagnostics.csv", summaries)
    write_csv(out_dir / "bios_label_group_gaps.csv", per_label)
    write_csv(out_dir / "bios_profession_accuracy.csv", profession_accuracy)
    write_csv(out_dir / "bios_gender_accuracy.csv", gender_accuracy)
    write_csv(out_dir / "bios_confusion_matrix.csv", confusion)
    write_csv(out_dir / "bios_top_confusions.csv", top_confusion_rows(confusion, 5))

    print(f"Saved {out_dir / 'bios_fairness_diagnostics.csv'}")
    print(f"Saved {out_dir / 'bios_label_group_gaps.csv'}")
    print(f"Saved {out_dir / 'bios_profession_accuracy.csv'}")
    print(f"Saved {out_dir / 'bios_gender_accuracy.csv'}")
    print(f"Saved {out_dir / 'bios_confusion_matrix.csv'}")
    print(f"Saved {out_dir / 'bios_top_confusions.csv'}")


if __name__ == "__main__":
    main()
