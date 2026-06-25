from __future__ import annotations

import random

from heal_capo.core import PromptCandidate
from heal_capo.optimizers.evolutionary_ops import (
    EvolutionaryOpsConfig,
    EvolutionaryPromptOps,
    call_meta_llm,
    clean_prompt,
    crossover_prompts,
    ensure_output_format,
    extract_prompt,
    fallback_crossover,
    fallback_mutation,
    make_crossover_meta_prompt,
    make_mutation_meta_prompt,
    mutate_prompt,
)


class FakeMetaLLM:
    def __init__(self, response: str):
        self.response = response
        self.prompts = []

    def get_response(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class FailingMetaLLM:
    def get_response(self, prompt: str) -> str:
        raise RuntimeError("meta failed")


def make_candidate(name: str, instruction: str) -> PromptCandidate:
    candidate = PromptCandidate(
        instruction=instruction,
        metadata={
            "method": name,
            "dataset": "subj",
            "task_type": "classification",
        },
    )
    candidate.candidate_id = name
    return candidate


def test_extract_prompt_from_xml_tags():
    text = "Before <prompt>Classify carefully.</prompt> After"

    assert extract_prompt(text) == "Classify carefully."


def test_extract_prompt_from_markdown_code_block():
    text = "```text\nClassify carefully.\n```"

    assert extract_prompt(text) == "Classify carefully."


def test_extract_prompt_falls_back_to_clean_text():
    assert extract_prompt("Classify carefully.") == "Classify carefully."


def test_clean_prompt_strips_quotes_and_truncates():
    cleaned = clean_prompt('"abcdef"', max_chars=3)

    assert cleaned == "abc"


def test_ensure_output_format_appends_when_missing():
    prompt = ensure_output_format(
        "Classify the sentence.",
        "Return inside <final_answer> tags.",
    )

    assert "Classify the sentence." in prompt
    assert "<final_answer>" in prompt


def test_ensure_output_format_does_not_duplicate():
    original = "Classify. Return inside <final_answer> and </final_answer> tags."

    prompt = ensure_output_format(
        original,
        "Return inside <final_answer> tags.",
    )

    assert prompt == original


def test_make_crossover_meta_prompt_contains_parents_and_tags():
    prompt = make_crossover_meta_prompt(
        mother="Prompt A",
        father="Prompt B",
        task_description="Task description",
    )

    assert "Prompt A" in prompt
    assert "Prompt B" in prompt
    assert "<prompt>new prompt</prompt>" in prompt


def test_make_mutation_meta_prompt_contains_instruction_and_tags():
    prompt = make_mutation_meta_prompt(
        instruction="Prompt A",
        task_description="Task description",
    )

    assert "Prompt A" in prompt
    assert "<prompt>new prompt</prompt>" in prompt


def test_call_meta_llm_uses_get_response():
    llm = FakeMetaLLM("<prompt>New prompt</prompt>")

    assert call_meta_llm(llm, "hello") == "<prompt>New prompt</prompt>"
    assert llm.prompts == ["hello"]


def test_fallback_crossover_adds_final_answer_format():
    rng = random.Random(0)

    prompt = fallback_crossover(
        mother="Classify as subjective or objective.",
        father="Use only the input sentence.",
        output_format="Return inside <final_answer> and </final_answer> tags.",
        rng=rng,
    )

    assert "Classify" in prompt or "Use only" in prompt
    assert "<final_answer>" in prompt


def test_fallback_mutation_adds_clause_and_format():
    rng = random.Random(0)

    prompt = fallback_mutation(
        instruction="Classify as subjective or objective.",
        output_format="Return inside <final_answer> and </final_answer> tags.",
        rng=rng,
    )

    assert "Classify as subjective or objective." in prompt
    assert "<final_answer>" in prompt


def test_crossover_with_meta_llm_extracts_prompt():
    mother = make_candidate("mother", "Prompt A")
    father = make_candidate("father", "Prompt B")
    llm = FakeMetaLLM("<prompt>Merged prompt.</prompt>")

    ops = EvolutionaryPromptOps(
        config=EvolutionaryOpsConfig(random_seed=0),
        meta_llm=llm,
    )

    result = ops.crossover(mother, father)

    assert result.operator == "crossover"
    assert result.used_meta_llm is True
    assert result.parent_ids == ["mother", "father"]
    assert result.candidate.instruction.startswith("Merged prompt.")
    assert result.candidate.metadata["operator"] == "crossover"
    assert result.candidate.metadata["dataset"] == "subj"


def test_mutation_with_meta_llm_extracts_prompt():
    parent = make_candidate("parent", "Prompt A")
    llm = FakeMetaLLM("<prompt>Mutated prompt.</prompt>")

    ops = EvolutionaryPromptOps(
        config=EvolutionaryOpsConfig(random_seed=0),
        meta_llm=llm,
    )

    result = ops.mutate(parent)

    assert result.operator == "mutation"
    assert result.used_meta_llm is True
    assert result.parent_ids == ["parent"]
    assert result.candidate.instruction.startswith("Mutated prompt.")
    assert result.candidate.metadata["operator"] == "mutation"


def test_crossover_falls_back_when_meta_llm_fails():
    mother = make_candidate("mother", "Prompt A.")
    father = make_candidate("father", "Prompt B.")

    ops = EvolutionaryPromptOps(
        config=EvolutionaryOpsConfig(random_seed=0),
        meta_llm=FailingMetaLLM(),
    )

    result = ops.crossover(mother, father)

    assert result.operator == "crossover"
    assert result.used_meta_llm is False
    assert result.raw_response.startswith("meta_llm_error:")
    assert "<final_answer>" in result.candidate.instruction


def test_mutation_falls_back_when_meta_llm_fails():
    parent = make_candidate("parent", "Prompt A.")

    ops = EvolutionaryPromptOps(
        config=EvolutionaryOpsConfig(random_seed=0),
        meta_llm=FailingMetaLLM(),
    )

    result = ops.mutate(parent)

    assert result.operator == "mutation"
    assert result.used_meta_llm is False
    assert result.raw_response.startswith("meta_llm_error:")
    assert "<final_answer>" in result.candidate.instruction


def test_create_offspring_returns_crossover_and_mutation():
    mother = make_candidate("mother", "Prompt A.")
    father = make_candidate("father", "Prompt B.")

    ops = EvolutionaryPromptOps(
        config=EvolutionaryOpsConfig(random_seed=0),
        meta_llm=None,
    )

    offspring = ops.create_offspring(
        mother=mother,
        father=father,
        mutate_after_crossover=True,
    )

    assert len(offspring) == 2
    assert offspring[0].operator == "crossover"
    assert offspring[1].operator == "mutation"


def test_crossover_function_wrapper():
    mother = make_candidate("mother", "Prompt A.")
    father = make_candidate("father", "Prompt B.")

    result = crossover_prompts(
        mother=mother,
        father=father,
        rng=random.Random(0),
    )

    assert result.operator == "crossover"


def test_mutate_function_wrapper():
    parent = make_candidate("parent", "Prompt A.")

    result = mutate_prompt(
        parent=parent,
        rng=random.Random(0),
    )

    assert result.operator == "mutation"


def test_probability_zero_uses_fallback_even_with_meta_llm():
    parent = make_candidate("parent", "Prompt A.")
    llm = FakeMetaLLM("<prompt>Should not be used.</prompt>")

    ops = EvolutionaryPromptOps(
        config=EvolutionaryOpsConfig(
            random_seed=0,
            mutation_probability=0.0,
        ),
        meta_llm=llm,
    )

    result = ops.mutate(parent)

    assert result.used_meta_llm is False
    assert llm.prompts == []