from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.datasets import (
    BIOS_PROFESSION_LABELS,
    _bios_gender_label,
    _bios_profession_label,
    _load_hf_dataset,
)


def _row_to_item(row: dict[str, Any], source_index: int) -> dict[str, Any] | None:
    text = str(row.get("hard_text", "")).strip()
    if not text:
        return None

    profession = _bios_profession_label(row.get("profession"))
    gender = _bios_gender_label(row.get("gender"))
    if profession not in BIOS_PROFESSION_LABELS or gender not in {"female", "male"}:
        return None

    return {
        "text": f"Biography: {text}",
        "label": profession,
        "meta": {
            "dataset": "bias_in_bios",
            "source_index": source_index,
            "profession": profession,
            "gender": gender,
            "group": gender,
            "sensitive_attribute": "gender",
        },
    }


def build_probe(
    split: str,
    seed: int,
    examples_per_group: int,
    max_labels: int,
) -> list[dict[str, Any]]:
    ds = _load_hf_dataset("LabHC/bias_in_bios", split=split)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for idx, row in enumerate(ds):
        item = _row_to_item(dict(row), source_index=idx)
        if item is None:
            continue
        meta = item["meta"]
        buckets[(meta["profession"], meta["gender"])].append(item)

    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)

    supported_labels = []
    for label in BIOS_PROFESSION_LABELS:
        female_n = len(buckets[(label, "female")])
        male_n = len(buckets[(label, "male")])
        support = min(female_n, male_n)
        if support >= examples_per_group:
            supported_labels.append((label, support))

    # Prefer labels with enough balanced support; keep canonical order for ties.
    supported_labels.sort(
        key=lambda pair: (-pair[1], BIOS_PROFESSION_LABELS.index(pair[0]))
    )
    labels = [label for label, _ in supported_labels[:max_labels]]
    labels.sort(key=BIOS_PROFESSION_LABELS.index)

    items: list[dict[str, Any]] = []
    for label in labels:
        for gender in ("female", "male"):
            items.extend(buckets[(label, gender)][:examples_per_group])

    rng.shuffle(items)
    return items


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a balanced Bias-in-Bios in-loop fairness probe JSONL."
    )
    parser.add_argument(
        "--out",
        default="data/fairness_bios_probe_search_seed0.jsonl",
        help=(
            "Search-only fairness probe path. Do not reuse this file as the "
            "held-out joint test set."
        ),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples-per-group", type=int, default=5)
    parser.add_argument("--max-labels", type=int, default=8)
    args = parser.parse_args()

    items = build_probe(
        split=args.split,
        seed=args.seed,
        examples_per_group=args.examples_per_group,
        max_labels=args.max_labels,
    )
    if not items:
        raise RuntimeError("No Bias-in-Bios fairness probe examples were selected.")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    labels = sorted({item["label"] for item in items})
    print(
        f"Wrote {len(items)} probe examples across {len(labels)} labels to {output}"
    )
    print("Labels:", ", ".join(labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
