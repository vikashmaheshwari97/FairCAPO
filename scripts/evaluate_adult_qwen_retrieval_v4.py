from __future__ import annotations

from pathlib import Path

from heal_capo.adult_v4_data import load_manifest, rows_from_manifest
from heal_capo.retrieval_evaluator import build_retrieval_evaluator
from scripts import evaluate_pareto_on_test as evaluator


DEFAULT_CONFIG = "configs/HPC_Config/adult_eval_qwen_retrieval_4m_v4_HPC.yaml"


def get_test_data(config: dict) -> list[dict]:
    dev = config.get("dev") or {}
    data_path = str(dev.get("data_path", "data/adult_semantic_v3.csv"))
    manifest_path = str(dev.get("split_manifest", "data/adult_v4_fixed_split_seed0.json"))
    load_manifest(manifest_path, data_path=data_path, require_fingerprint=True)
    rows = rows_from_manifest(config, "test")
    expected = int(dev.get("test_size", len(rows)))
    if len(rows) != expected:
        raise ValueError(f"Adult held-out test size mismatch: expected {expected}, found {len(rows)}")
    return rows


def install() -> None:
    evaluator.get_test_data = get_test_data
    evaluator.build_objective_evaluator = build_retrieval_evaluator


def main() -> None:
    if not Path(DEFAULT_CONFIG).is_file():
        raise FileNotFoundError(DEFAULT_CONFIG)
    install()
    evaluator.main()


if __name__ == "__main__":
    main()
