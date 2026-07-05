from __future__ import annotations

from typing import Any

from heal_capo.core import PromptCandidate
from heal_capo.optimizers import evolutionary_ops_legacy as _legacy

EvolutionaryOpsConfig = _legacy.EvolutionaryOpsConfig
EvolutionaryOpResult = _legacy.EvolutionaryOpResult
extract_prompt = _legacy.extract_prompt
clean_prompt = _legacy.clean_prompt
ensure_output_format = _legacy.ensure_output_format
call_meta_llm = _legacy.call_meta_llm
fallback_crossover = _legacy.fallback_crossover
fallback_mutation = _legacy.fallback_mutation


def _feedback_rows(candidate: PromptCandidate, limit: int = 6) -> list[dict]:
    rows = candidate.metadata.get("error_feedback_rows", []) if candidate.metadata else []
    return [dict(row) for row in list(rows)[-limit:] if isinstance(row, dict)]


def _feedback_text(rows: list[dict]) -> str:
    rendered = []
    for index, row in enumerate(rows, start=1):
        rendered.append(
            f"Error {index}: input={str(row.get('input', ''))[:450]} | "
            f"gold={row.get('gold', '')} | prediction={row.get('prediction', '')}"
        )
    return "\n".join(rendered)


def make_crossover_meta_prompt(
    mother: str,
    father: str,
    task_description: str = "",
    mother_feedback: str = "",
    father_feedback: str = "",
) -> str:
    task_block = f"Task:\n{task_description}\n\n" if task_description else ""
    return (
        "Create one improved classification instruction from two parent prompts. "
        "Preserve useful concrete decision rules, remove contradictions, and repair "
        "the recurring development errors shown below. Do not merely change style. "
        "Do not mention protected attributes, training data, or the error examples. "
        "Keep the required output format.\n\n"
        f"{task_block}"
        f"Parent 1:\n{mother}\n\nParent 1 observed errors:\n{mother_feedback}\n\n"
        f"Parent 2:\n{father}\n\nParent 2 observed errors:\n{father_feedback}\n\n"
        "Return only <prompt>new prompt</prompt>."
    )


def make_mutation_meta_prompt(
    instruction: str,
    task_description: str = "",
    feedback: str = "",
) -> str:
    task_block = f"Task:\n{task_description}\n\n" if task_description else ""
    return (
        "Repair the prompt using the observed development errors. Add or calibrate "
        "decision rules that would correct those mistakes while preserving valid rules "
        "and the exact output format. Do not merely rephrase the wording. Do not quote "
        "the error examples or mention protected attributes.\n\n"
        f"{task_block}"
        f"Current prompt:\n{instruction}\n\nObserved errors:\n{feedback}\n\n"
        "Return only <prompt>repaired prompt</prompt>."
    )


class EvolutionaryPromptOps(_legacy.EvolutionaryPromptOps):
    """Use error-driven operators only after development mistakes are available."""

    def crossover(self, mother, father, task_description: str = ""):
        mother_rows = _feedback_rows(mother)
        father_rows = _feedback_rows(father)
        if not mother_rows and not father_rows:
            result = super().crossover(mother, father, task_description)
            result.candidate.metadata["error_feedback_used"] = False
            return result

        original = _legacy.make_crossover_meta_prompt
        _legacy.make_crossover_meta_prompt = (
            lambda mother, father, task_description="": make_crossover_meta_prompt(
                mother,
                father,
                task_description,
                mother_feedback=_feedback_text(mother_rows),
                father_feedback=_feedback_text(father_rows),
            )
        )
        try:
            result = super().crossover(mother, father, task_description)
        finally:
            _legacy.make_crossover_meta_prompt = original
        result.candidate.metadata["error_feedback_used"] = True
        result.candidate.metadata["error_feedback_parent_ids"] = [
            mother.candidate_id,
            father.candidate_id,
        ]
        result.candidate.metadata["error_feedback_count"] = len(mother_rows) + len(father_rows)
        return result

    def mutate(self, parent, task_description: str = ""):
        rows = _feedback_rows(parent)
        if not rows:
            result = super().mutate(parent, task_description)
            result.candidate.metadata["error_feedback_used"] = False
            return result

        original = _legacy.make_mutation_meta_prompt
        _legacy.make_mutation_meta_prompt = (
            lambda instruction, task_description="": make_mutation_meta_prompt(
                instruction,
                task_description,
                feedback=_feedback_text(rows),
            )
        )
        try:
            result = super().mutate(parent, task_description)
        finally:
            _legacy.make_mutation_meta_prompt = original
        result.candidate.metadata["error_feedback_used"] = True
        result.candidate.metadata["error_feedback_parent_ids"] = [parent.candidate_id]
        result.candidate.metadata["error_feedback_count"] = len(rows)
        return result


def crossover_prompts(
    mother: PromptCandidate,
    father: PromptCandidate,
    task_description: str = "",
    config: EvolutionaryOpsConfig | None = None,
    meta_llm: Any | None = None,
    rng=None,
) -> EvolutionaryOpResult:
    return EvolutionaryPromptOps(config=config, meta_llm=meta_llm, rng=rng).crossover(
        mother, father, task_description
    )


def mutate_prompt(
    parent: PromptCandidate,
    task_description: str = "",
    config: EvolutionaryOpsConfig | None = None,
    meta_llm: Any | None = None,
    rng=None,
) -> EvolutionaryOpResult:
    return EvolutionaryPromptOps(config=config, meta_llm=meta_llm, rng=rng).mutate(
        parent, task_description
    )
