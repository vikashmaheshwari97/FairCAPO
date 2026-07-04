from __future__ import annotations

import random
from collections import Counter, defaultdict


EXPECTED_CELLS = {
    ("<=50K", "Female"),
    ("<=50K", "Male"),
    (">50K", "Female"),
    (">50K", "Male"),
}


def source_ids(rows):
    return [int((row.metadata or {})["source_index"]) for row in rows]


def cell(row):
    return row.label, str((row.metadata or {}).get("sex", ""))


def extend_balanced(base, candidates, total, seed):
    """Extend a split to equal label-by-sex cells without reusing source rows."""
    if total <= 0 or total % len(EXPECTED_CELLS) != 0:
        raise ValueError("Balanced Adult split size must be positive and divisible by four.")

    target = total // len(EXPECTED_CELLS)
    selected = list(base)
    selected_ids = set(source_ids(selected))
    if len(selected_ids) != len(selected):
        raise ValueError("Base Adult split contains duplicate source rows.")

    counts = Counter(cell(row) for row in selected)
    unexpected = set(counts).difference(EXPECTED_CELLS)
    if unexpected:
        raise ValueError(f"Unexpected Adult label/sex cells: {sorted(unexpected)}")

    buckets = defaultdict(list)
    for row in candidates:
        source_index = int((row.metadata or {})["source_index"])
        if source_index in selected_ids:
            continue
        key = cell(row)
        if key in EXPECTED_CELLS:
            buckets[key].append(row)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    for key in sorted(EXPECTED_CELLS):
        needed = target - counts.get(key, 0)
        if needed < 0:
            raise ValueError(f"Base split exceeds target for {key}: {counts[key]} > {target}")
        if len(buckets.get(key, [])) < needed:
            raise ValueError(
                f"Not enough unused Adult rows for {key}: need {needed}, "
                f"have {len(buckets.get(key, []))}."
            )
        selected.extend(buckets[key][:needed])

    rng.shuffle(selected)
    final_counts = Counter(cell(row) for row in selected)
    final_ids = source_ids(selected)
    if len(selected) != total:
        raise AssertionError(f"Adult split size mismatch: {len(selected)} != {total}")
    if len(final_ids) != len(set(final_ids)):
        raise AssertionError("Adult split contains duplicate source rows.")
    if any(final_counts.get(key, 0) != target for key in EXPECTED_CELLS):
        raise AssertionError(f"Invalid balanced Adult split: {final_counts}")
    return selected


def build_manifest(data_path, output_path):
    """Compatibility alias for the one canonical manifest builder."""
    from heal_capo.adult_v4_manifest import build_fixed_manifest

    return build_fixed_manifest(data_path, output_path)
