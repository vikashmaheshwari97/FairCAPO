from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .verifier import VerificationResult
from ..core import PromptCandidate


class PromptRepairer(ABC):
    """Repairs prompts using verifier feedback."""

    @abstractmethod
    def repair(
        self,
        candidate: PromptCandidate,
        feedback: VerificationResult,
    ) -> PromptCandidate:
        raise NotImplementedError


class TemplateRepairer(PromptRepairer):
    """
    Rule/template-based prompt repairer.

    This is intentionally lightweight and deterministic.
    Later, we can replace or extend it with meta-LLM guided repair.
    """

    DEFAULT_REPAIR = (
        "Be accurate, concise, faithful to the task, and avoid unsupported assumptions."
    )

    REPAIR_RULES = {
        "hallucination": (
            "Use only the provided input and context. If evidence is insufficient, "
            "state that it cannot be determined. Do not hallucinate or invent facts."
        ),
        "incorrect": (
            "Before giving the final answer, verify that the answer satisfies the task "
            "requirements and matches the allowed labels."
        ),
        "unsafe": (
            "Follow safety constraints. Refuse unsafe requests while remaining helpful "
            "for safe alternatives."
        ),
        "fairness": (
            "Do not infer ability, intent, sentiment, correctness, risk, or subjectivity "
            "from gender, race, ethnicity, nationality, religion, age, location, names, "
            "or other demographic attributes. Base the answer only on task-relevant text."
        ),
        "format": (
            "Return the final answer exactly in the required output format and do not "
            "include extra text."
        ),
        "over_refusal": (
            "Do not refuse when the task is safe and enough information is available. "
            "Answer the task directly using the provided evidence."
        ),
        "drift": (
            "Preserve the original task, labels, constraints, and output format. Do not "
            "change the task meaning while improving reliability."
        ),
    }

    def __init__(
        self,
        max_instruction_chars: Optional[int] = None,
        repair_prefix: str = "Repair instruction:",
    ):
        self.max_instruction_chars = max_instruction_chars
        self.repair_prefix = repair_prefix

    def repair(
        self,
        candidate: PromptCandidate,
        feedback: VerificationResult,
    ) -> PromptCandidate:
        failure_type = feedback.failure_type or "unknown"
        repair_text = self.REPAIR_RULES.get(failure_type, self.DEFAULT_REPAIR)

        new_instruction = self._append_repair(
            instruction=candidate.instruction,
            repair_text=repair_text,
        )

        if self.max_instruction_chars is not None:
            new_instruction = new_instruction[: self.max_instruction_chars].rstrip()

        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "repair_from": candidate.candidate_id,
                "repair_failure_type": failure_type,
                "repair_feedback": feedback.explanation,
                "repair_rule": repair_text,
            }
        )

        return PromptCandidate(
            instruction=new_instruction,
            examples=list(candidate.examples),
            parent_ids=[candidate.candidate_id],
            metadata=metadata,
        )

    def _append_repair(
        self,
        instruction: str,
        repair_text: str,
    ) -> str:
        instruction = instruction.rstrip()

        if self._already_contains_repair(
            instruction=instruction,
            repair_text=repair_text,
        ):
            return instruction

        return f"{instruction}\n{self.repair_prefix} {repair_text}"

    @staticmethod
    def _already_contains_repair(
        instruction: str,
        repair_text: str,
    ) -> bool:
        normalized_instruction = _normalize_for_match(instruction)
        normalized_repair = _normalize_for_match(repair_text)

        return normalized_repair in normalized_instruction


def _normalize_for_match(text: str) -> str:
    return " ".join(str(text).lower().split())