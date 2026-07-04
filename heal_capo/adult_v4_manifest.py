from __future__ import annotations

import json
from pathlib import Path

from experiments.datasets import load_paper_dataset
from heal_capo.adult_v4_data import load_all_examples
from heal_capo.adult_v4_split import extend_balanced, source_ids


def build_fixed_manifest(data_path, output_path):
    rows = load_all_examples(data_path)
    by_id = {int((row.metadata or {})["source_index"]): row for row in rows}
    old = load_paper_dataset(
        "adult", data_path=data_path, dev_size=400, shots_size=100,
        test_size=1000, seed=0, allow_smaller=False,
        stratified=True, stratify_group_key="sex",
    )
    test_ids = set(source_ids(old.test))
    old_dev_ids = set(source_ids(old.dev))
    old_shot_ids = set(source_ids(old.shots))
    blocked = test_ids | old_dev_ids | old_shot_ids
    shots = extend_balanced(
        list(old.shots),
        [row for index, row in by_id.items() if index not in blocked],
        500, 101,
    )
    shot_ids = set(source_ids(shots))
    blocked = test_ids | shot_ids | old_dev_ids
    dev = extend_balanced(
        list(old.dev),
        [row for index, row in by_id.items() if index not in blocked],
        3000, 202,
    )
    dev_ids = set(source_ids(dev))
    if dev_ids & shot_ids or dev_ids & test_ids or shot_ids & test_ids:
        raise AssertionError("Adult v4 split overlap detected")
    payload = {
        "version": "adult_v4_fixed_test_retrieval",
        "dev_source_indices": source_ids(dev),
        "shots_source_indices": source_ids(shots),
        "test_source_indices": source_ids(old.test),
        "sizes": {"dev": 3000, "shots": 500, "test": 1000},
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
