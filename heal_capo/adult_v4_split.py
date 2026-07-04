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
    all_rows = load_all_examples(data_path)
    by_id = {int((row.metadata or {})["source_index"]): row for row in all_rows}
    old = load_paper_dataset("adult", data_path=data_path, dev_size=400, shots_size=100,
                             test_size=1000, seed=0, allow_smaller=False,
                             stratified=True, stratify_group_key="sex")
    test_ids = set(source_ids(old.test))
    shot_blocked = test_ids | set(source_ids(old.shots))
    shots = extend_balanced(list(old.shots), [row for i, row in by_id.items() if i not in shot_blocked], 500, 101)
    dev_blocked = test_ids | set(source_ids(shots)) | set(source_ids(old.dev))
    dev = extend_balanced(list(old.dev), [row for i, row in by_id.items() if i not in dev_blocked], 3000, 202)
    payload = {"version": "adult_v4_fixed_test_retrieval", "seed": 0,
               "dev_source_indices": source_ids(dev), "shots_source_indices": source_ids(shots),
               "test_source_indices": source_ids(old.test),
               "sizes": {"dev": 3000, "shots": 500, "test": 1000}}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
