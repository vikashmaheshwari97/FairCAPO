from __future__ import annotations

import random
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from heal_capo.core import PromptCandidate


@dataclass
class EvolutionaryOpsConfig:
    """
    CAPO/MO-CAPO-style evolutionary prompt operators.

    These operators support:
      - crossover: merge two parent instructions
      - mutation: rewrite one instruction
      - optional meta-LLM calls
      - robust <prompt>...</prompt> extraction
      - deterministic fallback operators for local tests
    """

    random_seed: Optional[int] = None
    require_prompt_tags: bool = True
    max_prompt_chars: int = 4000
    mutation_probability: float = 1.0
    crossover_probability: float = 1.0
    preserve_output_format: bool = True
    default_output_format: str = (
        "Return the final answer inside <final_answer> and </final_answer> tags."
    )

    # Few-shot operators (MO-CAPO-style: the accuracy/cost trade-off curve is
    # driven mainly by the NUMBER of few-shot examples carried by a prompt).
    few_shot_probability: float = 0.5
    max_few_shot_examples: int = 4


@dataclass
class EvolutionaryOpResult:
    candidate: PromptCandidate
    operator: str
    parent_ids: list[str]
    used_meta_llm: bool
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def extract_prompt(text: str) -> str:
    """
    Extract prompt text from <prompt>...</prompt> if present.
    Otherwise return cleaned text.

    Handles multiline and case-insensitive tags.
    """
    raw = str(text or "").strip()

    match = re.search(
        r"<prompt>\s*(.*?)\s*</prompt>",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    # Also support common markdown code block responses.
    code_match = re.search(
        r"```(?:text|prompt|markdown)?\s*(.*?)\s*```",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if code_match:
        return code_match.group(1).strip()

    return raw.strip()


def clean_prompt(prompt: str, max_chars: int = 4000) -> str:
    cleaned = str(prompt or "").strip()

    # Remove accidental enclosing quotes.
    if (
        len(cleaned) >= 2
        and cleaned[0] in {"'", '"'}
        and cleaned[-1] == cleaned[0]
    ):
        cleaned = cleaned[1:-1].strip()

    # Avoid huge prompts if a meta model rambles.
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].strip()

    return cleaned


def ensure_output_format(
    prompt: str,
    output_format: str,
    preserve_output_format: bool = True,
) -> str:
    if not preserve_output_format:
        return prompt

    lowered = prompt.lower()

    if "<final_answer>" in lowered and "</final_answer>" in lowered:
        return prompt

    if "final_answer" in lowered:
        return prompt

    return f"{prompt.rstrip()}\n\n{output_format}"


def call_meta_llm(meta_llm: Any, prompt: str) -> str:
    if meta_llm is None:
        raise ValueError("meta_llm is None")

    if hasattr(meta_llm, "get_response"):
        response = meta_llm.get_response(prompt)
    elif hasattr(meta_llm, "generate"):
        response = meta_llm.generate(prompt)
    elif callable(meta_llm):
        response = meta_llm(prompt)
    else:
        raise AttributeError(
            "meta_llm must expose get_response(), generate(), or be callable."
        )

    if isinstance(response, list):
        return str(response[0])

    return str(response)


def make_crossover_meta_prompt(
    mother: str,
    father: str,
    task_description: str = "",
) -> str:
    task_block = f"Task:\n{task_description}\n\n" if task_description else ""

    return (
        "You receive two prompts for the following task.\n\n"
        f"{task_block}"
        "Please merge the two prompts into a single coherent prompt. "
        "Maintain useful linguistic features from both original prompts and keep the output format requirement.\n\n"
        f"Prompt 1:\n{mother}\n\n"
        f"Prompt 2:\n{father}\n\n"
        "Return the new prompt in the following format:\n"
        "<prompt>new prompt</prompt>."
    )


def make_mutation_meta_prompt(
    instruction: str,
    task_description: str = "",
) -> str:
    task_block = f"Task:\n{task_description}\n\n" if task_description else ""

    return (
        "You receive a prompt for the following task.\n\n"
        f"{task_block}"
        "Please rephrase the prompt, preserving its core meaning while substantially varying the linguistic style. "
        "Keep the output format requirement.\n\n"
        f"Prompt:\n{instruction}\n\n"
        "Return the new prompt in the following format:\n"
        "<prompt>new prompt</prompt>."
    )


def fallback_crossover(
    mother: str,
    father: str,
    output_format: str,
    rng: random.Random,
) -> str:
    """
    Deterministic non-LLM crossover.

    Combines the shorter task command with safety/format phrases from both parents.
    """
    mother = str(mother).strip()
    father = str(father).strip()

    parents = [mother, father]
    rng.shuffle(parents)

    first, second = parents

    first_sentence = first.split(".")[0].strip()
    second_sentence = second.split(".")[0].strip()

    if not first_sentence:
        first_sentence = first

    if not second_sentence:
        second_sentence = second

    merged = (
        f"{first_sentence}. {second_sentence}. "
        "Use only the provided input and avoid unsupported assumptions."
    )

    return ensure_output_format(
        prompt=merged,
        output_format=output_format,
        preserve_output_format=True,
    )


def fallback_mutation(
    instruction: str,
    output_format: str,
    rng: random.Random,
) -> str:
    """
    Deterministic non-LLM mutation.

    Adds a small clarification selected from stable mutation clauses.
    """
    clauses = [
        "Base the decision only on the given input.",
        "Do not infer missing context beyond the text.",
        "Prefer the most direct valid label.",
        "Avoid using demographic attributes as evidence unless explicitly relevant to the task.",
        "Keep the answer concise and format-compliant.",
    ]

    clause = rng.choice(clauses)
    prompt = str(instruction).strip()

    if clause.lower() not in prompt.lower():
        prompt = f"{prompt.rstrip()} {clause}"

    return ensure_output_format(
        prompt=prompt,
        output_format=output_format,
        preserve_output_format=True,
    )


class EvolutionaryPromptOps:
    """
    CAPO-style crossover and mutation wrapper.

    Designed to be used by the later full MO-CAPO evolutionary loop.
    """

    def __init__(
        self,
        config: EvolutionaryOpsConfig | None = None,
        meta_llm: Any | None = None,
        rng: random.Random | None = None,
    ):
        self.config = config or EvolutionaryOpsConfig()
        self.meta_llm = meta_llm
        self.rng = rng or random.Random(self.config.random_seed)

    def crossover(
        self,
        mother: PromptCandidate,
        father: PromptCandidate,
        task_description: str = "",
    ) -> EvolutionaryOpResult:
        parent_ids = [mother.candidate_id, father.candidate_id]
        use_meta_llm = (
            self.meta_llm is not None
            and self.rng.random() <= self.config.crossover_probability
        )

        raw_response = ""
        used_meta_llm = False

        if use_meta_llm:
            meta_prompt = make_crossover_meta_prompt(
                mother=mother.instruction,
                father=father.instruction,
                task_description=task_description,
            )

            try:
                raw_response = call_meta_llm(self.meta_llm, meta_prompt)
                instruction = extract_prompt(raw_response)
                used_meta_llm = True
            except Exception as exc:
                raw_response = f"meta_llm_error: {type(exc).__name__}: {exc}"
                instruction = fallback_crossover(
                    mother=mother.instruction,
                    father=father.instruction,
                    output_format=self.config.default_output_format,
                    rng=self.rng,
                )
        else:
            instruction = fallback_crossover(
                mother=mother.instruction,
                father=father.instruction,
                output_format=self.config.default_output_format,
                rng=self.rng,
            )

        instruction = clean_prompt(
            instruction,
            max_chars=self.config.max_prompt_chars,
        )
        instruction = ensure_output_format(
            prompt=instruction,
            output_format=self.config.default_output_format,
            preserve_output_format=self.config.preserve_output_format,
        )

        candidate = self._make_child_candidate(
            instruction=instruction,
            operator="crossover",
            parent_ids=parent_ids,
            mother=mother,
            father=father,
            used_meta_llm=used_meta_llm,
            examples=self._combine_examples(mother.examples, father.examples),
        )

        return EvolutionaryOpResult(
            candidate=candidate,
            operator="crossover",
            parent_ids=parent_ids,
            used_meta_llm=used_meta_llm,
            raw_response=raw_response,
            metadata={
                "mother_id": mother.candidate_id,
                "father_id": father.candidate_id,
            },
        )

    def mutate(
        self,
        parent: PromptCandidate,
        task_description: str = "",
    ) -> EvolutionaryOpResult:
        parent_ids = [parent.candidate_id]
        use_meta_llm = (
            self.meta_llm is not None
            and self.rng.random() <= self.config.mutation_probability
        )

        raw_response = ""
        used_meta_llm = False

        if use_meta_llm:
            meta_prompt = make_mutation_meta_prompt(
                instruction=parent.instruction,
                task_description=task_description,
            )

            try:
                raw_response = call_meta_llm(self.meta_llm, meta_prompt)
                instruction = extract_prompt(raw_response)
                used_meta_llm = True
            except Exception as exc:
                raw_response = f"meta_llm_error: {type(exc).__name__}: {exc}"
                instruction = fallback_mutation(
                    instruction=parent.instruction,
                    output_format=self.config.default_output_format,
                    rng=self.rng,
                )
        else:
            instruction = fallback_mutation(
                instruction=parent.instruction,
                output_format=self.config.default_output_format,
                rng=self.rng,
            )

        instruction = clean_prompt(
            instruction,
            max_chars=self.config.max_prompt_chars,
        )
        instruction = ensure_output_format(
            prompt=instruction,
            output_format=self.config.default_output_format,
            preserve_output_format=self.config.preserve_output_format,
        )

        candidate = self._make_child_candidate(
            instruction=instruction,
            operator="mutation",
            parent_ids=parent_ids,
            mother=parent,
            father=None,
            used_meta_llm=used_meta_llm,
        )

        return EvolutionaryOpResult(
            candidate=candidate,
            operator="mutation",
            parent_ids=parent_ids,
            used_meta_llm=used_meta_llm,
            raw_response=raw_response,
            metadata={
                "parent_id": parent.candidate_id,
            },
        )

    def create_offspring(
        self,
        mother: PromptCandidate,
        father: PromptCandidate,
        task_description: str = "",
        mutate_after_crossover: bool = True,
        shot_pool: list[dict] | None = None,
    ) -> list[EvolutionaryOpResult]:
        """
        Create one crossover child and optionally one mutation child.

        When a ``shot_pool`` is provided, the surviving child is additionally
        passed through ``mutate_few_shot`` with probability
        ``few_shot_probability``, so offspring vary in few-shot count (the
        MO-CAPO accuracy/cost trade-off lever).

        The later MO-CAPO loop can call this repeatedly to produce c offspring.
        """
        offspring = []

        crossover_result = self.crossover(
            mother=mother,
            father=father,
            task_description=task_description,
        )
        offspring.append(crossover_result)

        survivor = crossover_result

        if mutate_after_crossover:
            mutation_result = self.mutate(
                parent=crossover_result.candidate,
                task_description=task_description,
            )
            offspring.append(mutation_result)
            survivor = mutation_result

        if shot_pool and self.rng.random() <= self.config.few_shot_probability:
            few_shot_result = self.mutate_few_shot(
                parent=survivor.candidate,
                shot_pool=shot_pool,
            )
            offspring.append(few_shot_result)

        return offspring

    @staticmethod
    def _example_key(ex: dict) -> tuple[str, str]:
        return (str(ex.get("input", "")), str(ex.get("output", "")))

    def _combine_examples(
        self,
        mother_examples: list[dict],
        father_examples: list[dict],
    ) -> list[dict]:
        """Deduped union of both parents' few-shot examples, capped."""
        combined: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for ex in list(mother_examples) + list(father_examples):
            key = self._example_key(ex)
            if key in seen:
                continue
            seen.add(key)
            combined.append(dict(ex))
        return combined[: self.config.max_few_shot_examples]

    def mutate_few_shot(
        self,
        parent: PromptCandidate,
        shot_pool: list[dict] | None = None,
    ) -> EvolutionaryOpResult:
        """
        Few-shot count mutation: add / remove / swap one example.

        This is the operator that produces the MO-CAPO accuracy/cost staircase.
        ``shot_pool`` supplies candidate examples to draw from (each a
        ``{"input": ..., "output": ...}`` dict). The instruction is unchanged.
        """
        pool = [dict(ex) for ex in (shot_pool or [])]
        current = [dict(ex) for ex in parent.examples]
        current_keys = {self._example_key(ex) for ex in current}
        available = [ex for ex in pool if self._example_key(ex) not in current_keys]

        cap = self.config.max_few_shot_examples

        # Decide the action given what is possible.
        actions: list[str] = []
        if len(current) < cap and available:
            actions.append("add")
        if current:
            actions.append("remove")
        if current and available:
            actions.append("swap")

        if not actions:
            action = "noop"
            new_examples = current
        else:
            action = self.rng.choice(actions)
            if action == "add":
                new_examples = current + [self.rng.choice(available)]
            elif action == "remove":
                idx = self.rng.randrange(len(current))
                new_examples = current[:idx] + current[idx + 1:]
            else:  # swap
                idx = self.rng.randrange(len(current))
                new_examples = list(current)
                new_examples[idx] = self.rng.choice(available)

        candidate = self._make_child_candidate(
            instruction=parent.instruction,
            operator="few_shot",
            parent_ids=[parent.candidate_id],
            mother=parent,
            father=None,
            used_meta_llm=False,
            examples=new_examples,
        )

        return EvolutionaryOpResult(
            candidate=candidate,
            operator="few_shot",
            parent_ids=[parent.candidate_id],
            used_meta_llm=False,
            raw_response="",
            metadata={
                "parent_id": parent.candidate_id,
                "few_shot_action": action,
                "num_few_shot_before": len(current),
                "num_few_shot_after": len(new_examples),
            },
        )

    def _make_child_candidate(
        self,
        instruction: str,
        operator: str,
        parent_ids: list[str],
        mother: PromptCandidate,
        father: PromptCandidate | None,
        used_meta_llm: bool,
        examples: list[dict] | None = None,
    ) -> PromptCandidate:
        metadata = {
            "method": f"evolutionary_{operator}",
            "category": "evolutionary",
            "operator": operator,
            "parent_ids": parent_ids,
            "used_meta_llm": used_meta_llm,
            "source": "evolutionary_ops",
        }

        # Preserve useful dataset/task metadata from the first parent.
        for key in ["dataset", "task_type"]:
            if key in mother.metadata:
                metadata[key] = mother.metadata[key]

        if father is not None:
            metadata["mother_id"] = mother.candidate_id
            metadata["father_id"] = father.candidate_id
        else:
            metadata["parent_id"] = mother.candidate_id

        # Carry few-shot examples forward. Instruction-only operators must NOT
        # silently drop a parent's shots (that flattens the accuracy/cost front).
        if examples is None:
            examples = [dict(ex) for ex in mother.examples]

        metadata["num_few_shot"] = len(examples)

        candidate = PromptCandidate(
            instruction=instruction,
            examples=examples,
            metadata=metadata,
        )
        candidate.candidate_id = f"{operator}_{uuid.uuid4().hex[:12]}"

        return candidate


def crossover_prompts(
    mother: PromptCandidate,
    father: PromptCandidate,
    task_description: str = "",
    config: EvolutionaryOpsConfig | None = None,
    meta_llm: Any | None = None,
    rng: random.Random | None = None,
) -> EvolutionaryOpResult:
    ops = EvolutionaryPromptOps(
        config=config,
        meta_llm=meta_llm,
        rng=rng,
    )

    return ops.crossover(
        mother=mother,
        father=father,
        task_description=task_description,
    )


def mutate_prompt(
    parent: PromptCandidate,
    task_description: str = "",
    config: EvolutionaryOpsConfig | None = None,
    meta_llm: Any | None = None,
    rng: random.Random | None = None,
) -> EvolutionaryOpResult:
    ops = EvolutionaryPromptOps(
        config=config,
        meta_llm=meta_llm,
        rng=rng,
    )

    return ops.mutate(
        parent=parent,
        task_description=task_description,
    )