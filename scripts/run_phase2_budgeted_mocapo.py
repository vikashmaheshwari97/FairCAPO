from __future__ import annotations

import os
import re
import sys

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
_ORIGINAL_RUN_BUDGETED_MOCAPO = _legacy.run_budgeted_mocapo
_TOKEN_COUNTER: TokenCounter | None = None


def _adult_v4_config(config: dict) -> bool:
    dev = config.get("dev") if isinstance(config.get("dev"), dict) else {}
    dataset = str(config.get("dataset") or dev.get("dataset") or "").strip().lower()
    manifest = str(dev.get("split_manifest") or "").lower()
    output_dir = str(config.get("output_dir") or "").lower()
    return dataset == "adult" and ("v4" in manifest or "v4" in output_dir)


def validate_runtime_config(config: dict) -> None:
    """Fail before model work when an Adult v4 experiment is miswired."""
    if not _adult_v4_config(config):
        return
    if not str(config.get("task_description", "")).strip():
        raise ValueError(
            "Adult v4 configs must define task_description; refusing to use the shared fallback task."
        )
    prompt_pool = str(config.get("prompt_pool", ""))
    if not prompt_pool:
        raise ValueError("Adult v4 configs must define prompt_pool.")
    if prompt_pool.endswith("adult_accuracy_v3.yaml"):
        raise ValueError(
            "Adult v4 configs must use phase2_prompt_pool_adult_accuracy_v4.yaml, not the v3 pool."
        )
    retrieval = config.get("retrieval") or {}
    if bool(retrieval.get("enabled", False)) and int(retrieval.get("k", 0)) <= 0:
        raise ValueError("Enabled Adult v4 retrieval requires k > 0.")


def load_yaml(path: str) -> dict:
    config = _ORIGINAL_LOAD_YAML(path)
    validate_runtime_config(config)
    return config


def get_task_description(config: dict) -> str:
    explicit = str(config.get("task_description", "")).strip()
    if explicit:
        return explicit
    if _adult_v4_config(config):
        raise ValueError("Adult v4 task_description is required.")
    return _ORIGINAL_GET_TASK_DESCRIPTION(config)


def simple_token_count(text: str) -> int:
    """Count model tokens when available, with punctuation-aware fallback."""
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
    validate_runtime_config(config)
    model_id = str(
        (config.get("llm") or {}).get("model_id")
        or os.environ.get("FAIRCAPO_MODEL_ID", "")
    )
    _TOKEN_COUNTER = TokenCounter(model_id)

    configure_min_blocks_before_reject(
        int((config.get("intensification") or {}).get("min_blocks_before_reject", 1))
    )

    _legacy.simple_token_count = simple_token_count
    _legacy.get_task_description = get_task_description
    _legacy.load_yaml = load_yaml
    _legacy.BlockEvaluator = BlockEvaluator
    _legacy.EvolutionaryOpsConfig = EvolutionaryOpsConfig
    _legacy.EvolutionaryPromptOps = EvolutionaryPromptOps
    _legacy.IntensificationConfig = IntensificationConfig
    _legacy.Intensifier = Intensifier

    # Preserve dataset-specific entrypoint overrides.
    _legacy.get_dev_data = globals().get("get_dev_data", _legacy.get_dev_data)
    _legacy.build_objective_evaluator = globals().get(
        "build_objective_evaluator", _legacy.build_objective_evaluator
    )


def run_budgeted_mocapo(config: dict, force_no_llm: bool = False):
    """Public API that always installs the corrected runtime before execution."""
    _install_runtime(config)
    return _ORIGINAL_RUN_BUDGETED_MOCAPO(config, force_no_llm=force_no_llm)


def main() -> None:
    config_path = _config_path_from_argv()
    config = load_yaml(config_path) if config_path else {}
    _install_runtime(config)
    _legacy.main()


if __name__ == "__main__":
    main()
