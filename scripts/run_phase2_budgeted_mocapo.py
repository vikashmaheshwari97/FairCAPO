from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from scripts import run_phase2_budgeted_mocapo_legacy as _legacy
from scripts.run_phase2_budgeted_mocapo_legacy import *

from heal_capo.adult_v4_tokens import TokenCounter
from heal_capo.optimizers.block_evaluator import BlockEvaluator
from heal_capo.optimizers.evolutionary_ops import EvolutionaryOpsConfig, EvolutionaryPromptOps
from heal_capo.optimizers.intensification import (
    IntensificationConfig,
    Intensifier,
    configure_min_blocks_before_reject,
)

_ORIGINAL_LOAD_YAML = _legacy.load_yaml
_ORIGINAL_GET_TASK_DESCRIPTION = _legacy.get_task_description
_TOKEN_COUNTER: TokenCounter | None = None


def _adult_v4_config(config: dict) -> bool:
    dataset = str(config.get("dataset") or (config.get("dev") or {}).get("dataset") or "").lower()
    manifest = str((config.get("dev") or {}).get("split_manifest") or "").lower()
    return dataset == "adult" and ("v4" in manifest or "v4" in str(config.get("output_dir", "")).lower())


def load_yaml(path: str) -> dict:
    config = _ORIGINAL_LOAD_YAML(path)
    if _adult_v4_config(config):
        if not str(config.get("task_description", "")).strip():
            raise ValueError(
                "Adult v4 configs must define task_description; refusing to use the shared fallback task."
            )
        pool = str(config.get("prompt_pool", ""))
        if pool.endswith("adult_accuracy_v3.yaml"):
            raise ValueError("Adult v4 configs must use the v4 prompt pool, not adult_accuracy_v3.yaml.")
    return config


def get_task_description(config: dict) -> str:
    explicit = str(config.get("task_description", "")).strip()
    if explicit:
        return explicit
    if _adult_v4_config(config):
        raise ValueError("Adult v4 task_description is required.")
    return _ORIGINAL_GET_TASK_DESCRIPTION(config)


def simple_token_count(text: str) -> int:
    if _TOKEN_COUNTER is not None:
        return _TOKEN_COUNTER.count(str(text or ""))
    return len(re.findall(r"\w+|[^\w\s]", str(text or ""), flags=re.UNICODE))


def _config_path_from_argv() -> str | None:
    for index, argument in enumerate(sys.argv):
        if argument == "--config" and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if argument.startswith("--config="):
            return argument.split("=", 1)[1]
    return None


def _install_runtime(config: dict) -> None:
    global _TOKEN_COUNTER
    model_id = str((config.get("llm") or {}).get("model_id") or os.environ.get("FAIRCAPO_MODEL_ID", ""))
    _TOKEN_COUNTER = TokenCounter(model_id)

    configure_min_blocks_before_reject(
        int((config.get("intensification") or {}).get("min_blocks_before_reject", 1))
    )

    # The legacy implementation remains available for reproducibility, while the
    # active module installs the corrected components before executing it.
    _legacy.simple_token_count = simple_token_count
    _legacy.get_task_description = get_task_description
    _legacy.load_yaml = load_yaml
    _legacy.BlockEvaluator = BlockEvaluator
    _legacy.EvolutionaryOpsConfig = EvolutionaryOpsConfig
    _legacy.EvolutionaryPromptOps = EvolutionaryPromptOps
    _legacy.IntensificationConfig = IntensificationConfig
    _legacy.Intensifier = Intensifier

    # Preserve entrypoint monkeypatches used by dataset-specific runners.
    _legacy.get_dev_data = globals().get("get_dev_data", _legacy.get_dev_data)
    _legacy.build_objective_evaluator = globals().get(
        "build_objective_evaluator", _legacy.build_objective_evaluator
    )


def main() -> None:
    config_path = _config_path_from_argv()
    config = load_yaml(config_path) if config_path else {}
    _install_runtime(config)
    _legacy.main()


if __name__ == "__main__":
    main()
