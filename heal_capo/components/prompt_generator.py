from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from heal_capo.core import PromptCandidate
from heal_capo.components.drift_guard import (
    DriftResult,
    FairnessConstraintDriftGuard,
    KeywordDriftGuard,
)
from heal_capo.optimizers.evolutionary_ops import (
    call_meta_llm,
    clean_prompt,
    ensure_output_format,
    extract_prompt,
)


# Deterministic fairness-hardening clauses, appended by the fallback generator
# (and as a backstop when the meta-LLM output omits an explicit constraint).
FAIRNESS_REPAIR_CLAUSES = [
    "Do not infer the answer from gender, race, ethnicity, nationality, religion, "
    "age, or other demographic attributes.",
    "Treat sentences that differ only in a demographic term identically; the "
    "label must not change because of such a change.",
    "Base the classification solely on the content of the input, never on the "
    "identity or demographic group of the people mentioned.",
    "Ignore protected attributes (gender, race, ethnicity, religion, nationality) "
    "when deciding the label.",
]


@dataclass
class FairnessGenerationConfig:
    """Configuration for fairness-aware dynamic prompt generation."""

    max_new_prompts_per_seed: int = 2
    min_flips_to_trigger: int = 1
    random_seed: Optional[int] = None
    max_prompt_chars: int = 4000
    preserve_output_format: bool = True
    default_output_format: str = (
        "Return the final answer inside <final_answer> and </final_answer> tags."
    )
    # Repair candidates that fail the drift guard are dropped by default.
    keep_drift_failures: bool = False


@dataclass
class GeneratedPrompt:
    """A repaired prompt candidate plus its provenance and drift verdict."""

    candidate: PromptCandidate
    parent_prompt_id: str
    used_meta_llm: bool
    drift_result: DriftResult
    accepted: bool
    raw_response: str = ""
    triggering_attributes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def summarize_failed_pairs(
    pair_rows: Sequence[dict],
) -> tuple[int, list[str], list[dict]]:
    """
    Inspect per-pair prediction rows for one prompt and surface its failures.

    A row is a failure when its ``violation`` flag is set (preferred) or, as a
    fallback for older rows, when ``flipped`` is true. Returns the number of
    failing pairs, the distinct protected attributes involved, and the failing
    rows themselves (for embedding examples in the repair meta-prompt).
    """
    failed_rows: list[dict] = []

    for row in pair_rows:
        if "violation" in row:
            is_failure = bool(row.get("violation"))
        else:
            is_failure = bool(row.get("flipped"))

        if is_failure:
            failed_rows.append(row)

    attributes: list[str] = []
    for row in failed_rows:
        attr = str(row.get("protected_attribute", "")).strip()
        if attr and attr not in attributes:
            attributes.append(attr)

    return len(failed_rows), attributes, failed_rows


def make_fairness_repair_meta_prompt(
    instruction: str,
    triggering_attributes: Sequence[str],
    failed_examples: Sequence[dict],
    task_description: str = "",
    max_examples: int = 3,
) -> str:
    """
    Build a meta-prompt asking the meta-LLM to harden a prompt against the
    specific counterfactual failures observed during evaluation.
    """
    task_block = f"Task:\n{task_description}\n\n" if task_description else ""

    attr_text = (
        ", ".join(triggering_attributes)
        if triggering_attributes
        else "demographic attributes"
    )

    example_lines = []
    for row in list(failed_examples)[:max_examples]:
        base_text = str(row.get("base_text", "")).strip()
        cf_text = str(row.get("counterfactual_text", "")).strip()
        base_pred = str(row.get("base_prediction", "")).strip()
        cf_pred = str(row.get("counterfactual_prediction", "")).strip()

        if base_text and cf_text:
            example_lines.append(
                f"- \"{base_text}\" -> {base_pred} | "
                f"\"{cf_text}\" -> {cf_pred}"
            )

    examples_block = ""
    if example_lines:
        examples_block = (
            "The current prompt produced inconsistent predictions on these "
            "minimally different pairs (the prediction should not have changed):\n"
            + "\n".join(example_lines)
            + "\n\n"
        )

    return (
        "You are improving a classification prompt so that it treats people "
        "fairly.\n\n"
        f"{task_block}"
        f"{examples_block}"
        f"The model changed its answer based on {attr_text}, which is unfair. "
        "Rewrite the prompt so that the classification depends only on the "
        "content of the input and never on demographic attributes. Preserve the "
        "original task and the output-format requirement. Add an explicit "
        "fairness instruction.\n\n"
        f"Current prompt:\n{instruction}\n\n"
        "Return the new prompt in the following format:\n"
        "<prompt>new prompt</prompt>."
    )


def fairness_repair_fallback(
    instruction: str,
    triggering_attributes: Sequence[str],
    output_format: str,
    rng: random.Random,
) -> str:
    """
    Deterministic non-LLM fairness repair: append a fairness-hardening clause
    that is not already present.
    """
    prompt = str(instruction).strip()
    lowered = prompt.lower()

    # Prefer a clause not already represented in the prompt.
    candidates = [c for c in FAIRNESS_REPAIR_CLAUSES if c.lower() not in lowered]
    if not candidates:
        candidates = list(FAIRNESS_REPAIR_CLAUSES)

    clause = rng.choice(candidates)

    if triggering_attributes:
        attr_text = ", ".join(triggering_attributes)
        clause = f"{clause} (observed unfair sensitivity to: {attr_text})."

    repaired = f"{prompt.rstrip()} {clause}"

    return ensure_output_format(
        prompt=repaired,
        output_format=output_format,
        preserve_output_format=True,
    )


class FairnessAwarePromptGenerator:
    """
    Generate fairness-hardened prompt candidates from observed counterfactual
    failures, gating each candidate through a fairness drift guard.

    Uses a meta-LLM when available, with a deterministic fallback so the
    generator runs locally without a model.
    """

    def __init__(
        self,
        config: FairnessGenerationConfig | None = None,
        meta_llm: Any | None = None,
        drift_guard: KeywordDriftGuard | None = None,
        rng: random.Random | None = None,
    ):
        self.config = config or FairnessGenerationConfig()
        self.meta_llm = meta_llm
        self.drift_guard = drift_guard or FairnessConstraintDriftGuard()
        self.rng = rng or random.Random(self.config.random_seed)

    def generate_from_failures(
        self,
        prompt_id: str,
        instruction: str,
        pair_rows: Sequence[dict],
        task_description: str = "",
        dataset: str | None = None,
        task_type: str | None = None,
    ) -> list[GeneratedPrompt]:
        """
        Produce up to ``max_new_prompts_per_seed`` repaired candidates for one
        prompt, given its per-pair prediction rows. Returns an empty list when
        the prompt has fewer failures than ``min_flips_to_trigger``.
        """
        num_failures, attributes, failed_rows = summarize_failed_pairs(pair_rows)

        if num_failures < self.config.min_flips_to_trigger:
            return []

        generated: list[GeneratedPrompt] = []

        for index in range(self.config.max_new_prompts_per_seed):
            result = self._generate_one(
                prompt_id=prompt_id,
                instruction=instruction,
                triggering_attributes=attributes,
                failed_rows=failed_rows,
                task_description=task_description,
                dataset=dataset,
                task_type=task_type,
                index=index,
            )
            generated.append(result)

        return generated

    def _generate_one(
        self,
        prompt_id: str,
        instruction: str,
        triggering_attributes: list[str],
        failed_rows: Sequence[dict],
        task_description: str,
        dataset: str | None,
        task_type: str | None,
        index: int,
    ) -> GeneratedPrompt:
        raw_response = ""
        used_meta_llm = False
        new_instruction: Optional[str] = None

        if self.meta_llm is not None:
            meta_prompt = make_fairness_repair_meta_prompt(
                instruction=instruction,
                triggering_attributes=triggering_attributes,
                failed_examples=failed_rows,
                task_description=task_description,
            )
            try:
                raw_response = call_meta_llm(self.meta_llm, meta_prompt)
                new_instruction = extract_prompt(raw_response)
                used_meta_llm = True
            except Exception as exc:
                raw_response = f"meta_llm_error: {type(exc).__name__}: {exc}"
                new_instruction = None

        if not new_instruction:
            new_instruction = fairness_repair_fallback(
                instruction=instruction,
                triggering_attributes=triggering_attributes,
                output_format=self.config.default_output_format,
                rng=self.rng,
            )

        new_instruction = clean_prompt(
            new_instruction,
            max_chars=self.config.max_prompt_chars,
        )
        new_instruction = ensure_output_format(
            prompt=new_instruction,
            output_format=self.config.default_output_format,
            preserve_output_format=self.config.preserve_output_format,
        )

        drift_result = self.drift_guard.check(
            original_instruction=instruction,
            new_instruction=new_instruction,
        )

        accepted = drift_result.passed or self.config.keep_drift_failures

        candidate = PromptCandidate(
            instruction=new_instruction,
            metadata={
                "method": f"fairness_repair_of_{prompt_id}",
                "category": "fairness_repair",
                "source": "fairness_aware_generator",
                "parent_prompt_id": prompt_id,
                "used_meta_llm": used_meta_llm,
                "triggering_attributes": list(triggering_attributes),
                "drift_passed": drift_result.passed,
                "repair_index": index,
            },
        )
        candidate.candidate_id = (
            f"fairrepair_{prompt_id}_{index}_{uuid.uuid4().hex[:8]}"
        )

        if dataset is not None:
            candidate.metadata["dataset"] = dataset
        if task_type is not None:
            candidate.metadata["task_type"] = task_type

        return GeneratedPrompt(
            candidate=candidate,
            parent_prompt_id=prompt_id,
            used_meta_llm=used_meta_llm,
            drift_result=drift_result,
            accepted=accepted,
            raw_response=raw_response,
            triggering_attributes=list(triggering_attributes),
            metadata={
                "repair_index": index,
                "drift_explanation": drift_result.explanation,
            },
        )
