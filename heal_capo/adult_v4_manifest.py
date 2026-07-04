from __future__ import annotations

import json
from pathlib import Path

from experiments.datasets import load_paper_dataset
from heal_capo.adult_v4_data import dataset_fingerprint, load_all_examples, validate_manifest
from heal_capo.adult_v4_split import extend_balanced, source_ids


def build_fixed_manifest(data_path: str, output_path: str) -> dict:
    rows = load_all_examples(data_path)
    by_id = {int((row.metadata or {})["source_index"]): row for row in rows}
    original = load_paper_dataset(
        "adult", data_path=data_path, dev_size=400, shots_size=100,
        test_size=1000, seed=0, allow_smaller=False,
        stratified=True, stratify_group_key="sex",
    )
    test_ids = set(source_ids(original.test))
    old_dev_ids = set(source_ids(original.dev))
    old_shot_ids = set(source_ids(original.shots))

    blocked = test_ids | old_dev_ids | old_shot_ids
    shots = extend_balanced(
        list(original.shots),
        [row for index, row in by_id.items() if index not in blocked],
        500, 101,
    )
    shot_ids = set(source_ids(shots))

    blocked = test_ids | shot_ids | old_dev_ids
    dev = extend_balanced(
        list(original.dev),
        [row for index, row in by_id.items() if index not in blocked],
        3000, 202,
    )

    payload = {
        "version": "adult_v4_fixed_test_retrieval_v2",
        "seed": 0,
        **dataset_fingerprint(data_path),
        "dev_source_indices": source_ids(dev),
        "shots_source_indices": source_ids(shots),
        "test_source_indices": source_ids(original.test),
        "sizes": {"dev": 3000, "shots": 500, "test": 1000},
    }
    validate_manifest(payload, data_path, require_fingerprint=True)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
