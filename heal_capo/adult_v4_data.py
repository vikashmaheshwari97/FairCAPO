from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.datasets import Example, _adult_clean, _normalize_adult_income, _render_adult_record

DEFAULT_MANIFEST = "data/adult_v4_fixed_split_seed0.json"


def load_all_examples(data_path: str) -> list[Example]:
    examples = []
    with open(data_path, "r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            label = _normalize_adult_income(row.get("income"))
            sex = _adult_clean(row.get("sex"))
            examples.append(Example(
                text=_render_adult_record(row),
                label=label,
                metadata={
                    "dataset": "adult",
                    "source_index": index,
                    "sex": sex,
                    "race": _adult_clean(row.get("race")),
                    "group": sex,
                    "income": label,
                },
            ))
    return examples


def load_manifest(path: str = DEFAULT_MANIFEST) -> dict:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def examples_from_manifest(split_name: str, data_path: str, manifest_path: str) -> list[Example]:
    manifest = load_manifest(manifest_path)
    by_index = {
        int((example.metadata or {})["source_index"]): example
        for example in load_all_examples(data_path)
    }
    key = f"{split_name}_source_indices"
    return [by_index[int(index)] for index in manifest[key]]


def rows_from_manifest(config: dict, split_name: str) -> list[dict]:
    dev = config.get("dev") or {}
    data_path = dev.get("data_path", "data/adult_semantic_v3.csv")
    manifest_path = dev.get("split_manifest", DEFAULT_MANIFEST)
    return [
        {"text": example.text, "label": example.label, "meta": dict(example.metadata or {})}
        for example in examples_from_manifest(split_name, data_path, manifest_path)
    ]
