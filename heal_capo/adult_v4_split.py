from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from experiments.datasets import load_paper_dataset
from heal_capo.adult_v4_data import load_all_examples


def source_ids(rows):
    return [int((row.metadata or {})["source_index"]) for row in rows]


def cell(row):
    return row.label, str((row.metadata or {}).get("sex", ""))


def extend_balanced(base, candidates, total, seed):
    target = total // 4
    selected = list(base)
    counts = Counter(cell(row) for row in selected)
    buckets = defaultdict(list)
    for row in candidates:
        buckets[cell(row)].append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    for key in sorted(buckets):
        selected.extend(buckets[key][: target - counts.get(key, 0)])
    rng.shuffle(selected)
    final = Counter(cell(row) for row in selected)
    if len(selected) != total or set(final.values()) != {target}:
        raise AssertionError(f"Invalid balanced split: {final}")
    return selected


def build_manifest(data_path, output_path):
    from heal_capo.adult_v4_manifest_v2 import build_fixed_manifest
    return build_fixed_manifest(data_path, output_path)
