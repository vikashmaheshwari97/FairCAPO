from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import yaml


def _config_path(default: str) -> str:
    for index, arg in enumerate(sys.argv):
        if arg == "--config" and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return default


def _parse_record(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in str(text or "").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()
    return fields


def _adult_reasoning(text: str, label: str) -> str:
    fields = _parse_record(text)
    evidence = []
    for key in (
        "education",
        "occupation",
        "work class",
        "hours per week",
        "capital gain",
        "capital loss",
        "age",
    ):
        value = fields.get(key)
        if value:
            evidence.append(f"{key}={value}")

    joined = "; ".join(evidence[:7])
    return (
        "Reasoning: compare the combined economic evidence rather than relying "
        f"on one field. This labeled example has {joined}. Using the same "
        f"decision standard across records, the correct bracket is {label}."
    )


def _patch_reasoning_shots(base_module) -> None:
    original: Callable = base_module._shot_output

    def shot_output(
        row: dict,
        label: str,
        task_type: str,
        use_rationale: bool,
    ) -> str:
        if (
            use_rationale
            and str(task_type).strip().lower() == "classification"
            and "person record:" in str(row.get("text", "")).lower()
        ):
            rationale = _adult_reasoning(str(row.get("text", "")), label)
            return f"{rationale}\n<final_answer>{label}</final_answer>"
        return original(row, label, task_type, use_rationale)

    base_module._shot_output = shot_output


def _patch_adult_evolution(fairness_enabled: bool) -> None:
    import heal_capo.optimizers.evolutionary_ops as evo

    objective_text = (
        "maximize predictive accuracy, minimize inference token cost, and "
        "minimize equalized-odds risk across protected groups"
        if fairness_enabled
        else "maximize predictive accuracy and minimize inference token cost"
    )

    def crossover_prompt(
        mother: str,
        father: str,
        task_description: str = "",
    ) -> str:
        return (
            "You are optimizing a reusable prompt for Adult income "
            "classification. "
            f"The objectives are to {objective_text}. Protected attributes are "
            "not present in the input. Merge the two parent prompts into one "
            "concise, materially useful instruction. Preserve the exact labels "
            "<=50K and >50K and require final-answer tags. Encourage calibrated "
            "combination of education level, occupation, work class, age band, "
            "weekly hours, capital gain, and capital loss. Avoid a constant-label "
            "rule, demographic proxies, interactive questions, and verbose "
            "explanations at inference time.\n\n"
            f"Task:\n{task_description}\n\n"
            f"Parent 1:\n{mother}\n\nParent 2:\n{father}\n\n"
            "Return only <prompt>new prompt</prompt>."
        )

    def mutation_prompt(
        instruction: str,
        task_description: str = "",
    ) -> str:
        return (
            "Improve the following reusable Adult income-classification prompt. "
            f"The objectives are to {objective_text}. Make a substantive change "
            "rather than a superficial paraphrase. Preserve the exact labels "
            "<=50K and >50K and the final-answer format. Prefer a calibrated "
            "decision rule that combines multiple economic signals and handles "
            "conflicting evidence. Do not use protected attributes, demographic "
            "stereotypes, or a constant classifier. Keep the prompt concise "
            "enough to remain competitive on cost.\n\n"
            f"Task:\n{task_description}\n\n"
            f"Current prompt:\n{instruction}\n\n"
            "Return only <prompt>new prompt</prompt>."
        )

    evo.make_crossover_meta_prompt = crossover_prompt
    evo.make_mutation_meta_prompt = mutation_prompt


def configure(default_config: str) -> str:
    config_path = _config_path(default_config)
    config = {}
    path = Path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

    fairness_enabled = bool(
        (config.get("fairness", {}) or {}).get("in_loop", False)
    )

    import scripts.run_phase2_budgeted_mocapo as base

    _patch_reasoning_shots(base)
    _patch_adult_evolution(fairness_enabled=fairness_enabled)
    return config_path
