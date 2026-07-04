from __future__ import annotations

import json
from pathlib import Path

from heal_capo.adult_v4_data import dataset_fingerprint, validate_manifest
from heal_capo.adult_v4_manifest import build_fixed_manifest as build_legacy_manifest


def build_fixed_manifest(data_path: str, output_path: str) -> dict:
    payload = build_legacy_manifest(data_path, output_path)
    payload.update(dataset_fingerprint(data_path))
    payload["version"] = "adult_v4_fixed_test_retrieval_v2"
    validate_manifest(payload, data_path)
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
