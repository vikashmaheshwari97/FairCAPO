from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.datasets import (
    CIVILCOMMENTS_IDENTITY_COLUMNS,
    _CIVILCOMMENTS_TEXT_CAP,
    _civilcomments_label,
    _civilcomments_primary_identity,
    _civilcomments_split_alias,
    _resolve_civilcomments_csv_path,
    load_paper_dataset,
)


def _row_to_item(
    row: dict[str, Any], source_index: int, text_column: str
) -> dict[str, Any] | None:
    text = str(row.get(text_column, "")).strip()
    if not text:
        return None
    if len(text) > _CIVILCOMMENTS_TEXT_CAP:
        text = text[:_CIVILCOMMENTS_TEXT_CAP].rstrip() + " ..."

    label = _civilcomments_label(row.get("toxicity"))
    identity = _civilcomments_primary_identity(row)
    # Probe measures per-identity fairness, so drop comments with no named
    # identity above threshold.
    if identity == "none":
        return None

    return {
        "text": f"Comment: {text}",
        "label": label,
        "meta": {
            "dataset": "civil_comments",
            "source_index": source_index,
            "identity": identity,
            "group": identity,
            "sensitive_attribute": "identity",
            "toxicity_label": label,
        },
    }


def build_probe(
    csv_path: str,
    split: str,
    seed: int,
    examples_per_group: int,
    exclude_source_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Balanced CivilComments in-loop fairness probe.

    Buckets by ``(toxicity_label, identity)`` over the eight WILDS identity
    subgroups and keeps ``examples_per_group`` per cell, so the probe is balanced
    across identity x toxicity -- the same 16-cell basis WILDS uses for
    worst-group accuracy. With the defaults this yields
    8 identities x 2 labels x 5 = 80 items.
    """
    import csv

    wanted_split = _civilcomments_split_alias(split)
    excluded = exclude_source_indices or set()
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        text_column = "comment_text" if "comment_text" in fieldnames else "text"
        has_split_column = "split" in fieldnames

        for idx, row in enumerate(reader):
            if idx in excluded:
                continue
            if (
                has_split_column
                and _civilcomments_split_alias(row.get("split")) != wanted_split
            ):
                continue
            item = _row_to_item(dict(row), source_index=idx, text_column=text_column)
            if item is None:
                continue
            meta = item["meta"]
            buckets[(meta["toxicity_label"], meta["identity"])].append(item)

    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)

    items: list[dict[str, Any]] = []
    thin_cells: list[str] = []
    for identity in CIVILCOMMENTS_IDENTITY_COLUMNS:
        for label in ("non-toxic", "toxic"):
            cell = buckets[(label, identity)][:examples_per_group]
            if len(cell) < examples_per_group:
                thin_cells.append(f"{identity}/{label} ({len(cell)})")
            items.extend(cell)

    if thin_cells:
        print(
            "WARNING: some identity x toxicity cells had fewer than "
            f"{examples_per_group} examples: {', '.join(thin_cells)}"
        )

    rng.shuffle(items)
    return items


def search_split_source_indices(
    csv_path: str,
    split: str,
    seed: int,
    dev_size: int,
    shots_size: int,
    test_size: int,
) -> set[int]:
    dataset_split = load_paper_dataset(
        name="civil_comments",
        dev_size=dev_size,
        shots_size=shots_size,
        test_size=test_size,
        seed=seed,
        allow_smaller=False,
        stratified=True,
        dataset_split=split,
        stratify_group_key="identity",
        data_path=csv_path,
    )
    excluded: set[int] = set()
    for example in [*dataset_split.dev, *dataset_split.shots]:
        source_index = (example.metadata or {}).get("source_index")
        if source_index is not None:
            excluded.add(int(source_index))
    return excluded


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a balanced CivilComments-WILDS in-loop fairness probe JSONL."
    )
    parser.add_argument(
        "--out",
        default="data/fairness_civilcomments_probe_search_seed0.jsonl",
        help=(
            "Search-only fairness probe path. Do not reuse this file as the "
            "held-out joint test set."
        ),
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to the WILDS all_data_with_identities.csv (else auto-resolved).",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples-per-group", type=int, default=5)
    parser.add_argument("--dev-size", type=int, default=300)
    parser.add_argument("--shots-size", type=int, default=112)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument(
        "--allow-probe-overlap",
        action="store_true",
        help="Allow probe examples to overlap the search dev/few-shot split.",
    )
    args = parser.parse_args()

    csv_path = _resolve_civilcomments_csv_path(args.data_path)

    excluded: set[int] = set()
    if not args.allow_probe_overlap:
        excluded = search_split_source_indices(
            csv_path=csv_path,
            split=args.split,
            seed=args.seed,
            dev_size=args.dev_size,
            shots_size=args.shots_size,
            test_size=args.test_size,
        )

    items = build_probe(
        csv_path=csv_path,
        split=args.split,
        seed=args.seed,
        examples_per_group=args.examples_per_group,
        exclude_source_indices=excluded,
    )
    if not items:
        raise RuntimeError("No CivilComments fairness probe examples were selected.")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    identities = sorted({item["meta"]["identity"] for item in items})
    print(
        f"Wrote {len(items)} probe examples across {len(identities)} identities "
        f"to {output}"
    )
    print(f"Excluded {len(excluded)} search dev/few-shot source indices.")
    print("Identities:", ", ".join(identities))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
