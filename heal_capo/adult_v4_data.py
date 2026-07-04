from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.datasets import Example, _adult_clean, _normalize_adult_income, _render_adult_record
from heal_capo.dataset_identity import file_digest
from heal_capo.split_checks import checked_sets

DEFAULT_MANIFEST = "data/adult_v4_fixed_split_seed0.json"
SPLITS = ("dev", "shots", "test")


def load_all_examples(data_path: str) -> list[Example]:
    examples = []
    with open(data_path, "r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            label = _normalize_adult_income(row.get("income"))
            group = _adult_clean(row.get("sex"))
            examples.append(Example(
                text=_render_adult_record(row), label=label,
                metadata={"dataset": "adult", "source_index": index,
                          "sex": group, "race": _adult_clean(row.get("race")),
                          "group": group, "income": label},
            ))
    return examples


def dataset_fingerprint(data_path: str) -> dict:
    return {"data_sha256": file_digest(data_path), "row_count": len(load_all_examples(data_path))}


def validate_manifest(manifest: dict, data_path: str | None = None, require_fingerprint: bool = True) -> dict:
    if not isinstance(manifest, dict):
        raise ValueError("Invalid split manifest.")
    sets = checked_sets(manifest, SPLITS)
    if data_path:
        current = dataset_fingerprint(data_path)
        saved_hash, saved_rows = manifest.get("data_sha256"), manifest.get("row_count")
        if require_fingerprint and (not saved_hash or saved_rows is None):
            raise ValueError("Split manifest has no dataset fingerprint; regenerate it.")
        if saved_hash and str(saved_hash) != current["data_sha256"]:
            raise ValueError("Dataset SHA256 does not match the split manifest.")
        if saved_rows is not None and int(saved_rows) != current["row_count"]:
            raise ValueError("Dataset row count does not match the split manifest.")
        if any(values and max(values) >= current["row_count"] for values in sets.values()):
            raise ValueError("Split manifest contains an out-of-range index.")
    return manifest


def load_manifest(path: str = DEFAULT_MANIFEST, data_path: str | None = None, require_fingerprint: bool = True) -> dict:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    return validate_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        data_path,
        require_fingerprint,
    )


def examples_from_manifest(split_name: str, data_path: str, manifest_path: str) -> list[Example]:
    if split_name not in SPLITS:
        raise ValueError(f"Unknown split: {split_name!r}")
    manifest = load_manifest(manifest_path, data_path)
    by_index = {int((row.metadata or {})["source_index"]): row for row in load_all_examples(data_path)}
    indices = [int(value) for value in manifest[f"{split_name}_source_indices"]]
    missing = [value for value in indices if value not in by_index]
    if missing:
        raise ValueError(f"Manifest references missing indices: {missing[:10]}")
    return [by_index[value] for value in indices]


def rows_from_manifest(config: dict, split_name: str) -> list[dict]:
    dev = config.get("dev") or {}
    data_path = dev.get("data_path", "data/adult_semantic_v3.csv")
    manifest_path = dev.get("split_manifest", DEFAULT_MANIFEST)
    return [{"text": row.text, "label": row.label, "meta": dict(row.metadata or {})}
            for row in examples_from_manifest(split_name, data_path, manifest_path)]
