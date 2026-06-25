from __future__ import annotations

import random

from heal_capo.core import PromptCandidate
from heal_capo.optimizers.evolutionary_ops import (
    EvolutionaryOpsConfig,
    EvolutionaryPromptOps,
)


def _ops(seed: int = 0, max_shots: int = 4, few_shot_probability: float = 1.0):
    return EvolutionaryPromptOps(
        config=EvolutionaryOpsConfig(
            random_seed=seed,
            max_few_shot_examples=max_shots,
            few_shot_probability=few_shot_probability,
        ),
        rng=random.Random(seed),
    )


def _pool(n: int = 5):
    return [{"input": f"sentence {i}", "output": "<final_answer>objective</final_answer>"}
            for i in range(n)]


def test_mutate_few_shot_add_from_empty():
    ops = _ops()
    parent = PromptCandidate(instruction="Classify.")
    result = ops.mutate_few_shot(parent, shot_pool=_pool())

    # From zero shots with a non-empty pool, the only action is "add".
    assert result.metadata["few_shot_action"] == "add"
    assert len(result.candidate.examples) == 1
    assert result.candidate.metadata["num_few_shot"] == 1


def test_mutate_few_shot_remove_when_pool_empty():
    ops = _ops()
    parent = PromptCandidate(
        instruction="Classify.",
        examples=[{"input": "x", "output": "<final_answer>objective</final_answer>"}],
    )
    result = ops.mutate_few_shot(parent, shot_pool=[])

    # No pool to add/swap from, but a shot exists -> must remove.
    assert result.metadata["few_shot_action"] == "remove"
    assert len(result.candidate.examples) == 0


def test_mutate_few_shot_respects_cap():
    ops = _ops(max_shots=2)
    parent = PromptCandidate(
        instruction="Classify.",
        examples=[
            {"input": "a", "output": "<final_answer>objective</final_answer>"},
            {"input": "b", "output": "<final_answer>subjective</final_answer>"},
        ],
    )
    # At cap with a pool -> add is impossible; remove or swap only.
    for _ in range(10):
        result = ops.mutate_few_shot(parent, shot_pool=_pool())
        assert len(result.candidate.examples) <= 2
        assert result.metadata["few_shot_action"] in {"remove", "swap"}


def test_mutate_few_shot_noop_when_nothing_possible():
    ops = _ops()
    parent = PromptCandidate(instruction="Classify.")  # no shots
    result = ops.mutate_few_shot(parent, shot_pool=[])  # no pool

    assert result.metadata["few_shot_action"] == "noop"
    assert result.candidate.examples == []


def test_add_shot_increases_rendered_prompt_length():
    ops = _ops()
    parent = PromptCandidate(instruction="Classify the sentence.")
    before = len(parent.render("a test input"))

    child = ops.mutate_few_shot(parent, shot_pool=_pool()).candidate
    after = len(child.render("a test input"))

    # Adding a demonstration must lengthen the rendered prompt (drives cost up).
    assert after > before


def test_crossover_inherits_deduped_union_of_examples():
    ops = _ops()
    mother = PromptCandidate(
        instruction="A.",
        examples=[{"input": "x", "output": "<final_answer>objective</final_answer>"}],
    )
    father = PromptCandidate(
        instruction="B.",
        examples=[
            {"input": "x", "output": "<final_answer>objective</final_answer>"},  # dup
            {"input": "y", "output": "<final_answer>subjective</final_answer>"},
        ],
    )
    child = ops.crossover(mother, father).candidate

    # Union of {x} and {x, y} deduped -> 2 examples.
    assert len(child.examples) == 2


def test_instruction_mutation_preserves_examples():
    ops = _ops()
    parent = PromptCandidate(
        instruction="Classify the sentence.",
        examples=[{"input": "x", "output": "<final_answer>objective</final_answer>"}],
    )
    child = ops.mutate(parent).candidate

    # An instruction edit must NOT drop the few-shot examples.
    assert len(child.examples) == 1


def test_create_offspring_can_emit_few_shot_child():
    ops = _ops(few_shot_probability=1.0)
    mother = PromptCandidate(instruction="A.")
    father = PromptCandidate(instruction="B.")

    offspring = ops.create_offspring(
        mother, father, mutate_after_crossover=True, shot_pool=_pool()
    )
    operators = [o.operator for o in offspring]

    assert "few_shot" in operators


# ---------------------------------------------------------------------------
# build_shot_pool (runner-level, no LLM)
# ---------------------------------------------------------------------------

from scripts.run_phase2_budgeted_mocapo import build_shot_pool


_DEV = [
    {"text": "Paris is in France.", "label": "objective"},
    {"text": "The movie was dull.", "label": "subjective"},
    {"text": "", "label": "objective"},  # skipped (empty text)
]


def test_build_shot_pool_disabled_returns_empty():
    assert build_shot_pool({"few_shot": {"enabled": False}}, _DEV) == []
    assert build_shot_pool({}, _DEV) == []  # absent block -> disabled


def test_build_shot_pool_formats_and_filters():
    pool = build_shot_pool({"few_shot": {"enabled": True, "pool_size": 10}}, _DEV)

    # Empty-text row dropped; two valid demonstrations remain.
    assert len(pool) == 2
    assert pool[0] == {
        "input": "Paris is in France.",
        "output": "<final_answer>objective</final_answer>",
    }
    assert all("input" in e and e["output"].startswith("<final_answer>") for e in pool)


def test_build_shot_pool_respects_pool_size():
    pool = build_shot_pool({"few_shot": {"enabled": True, "pool_size": 1}}, _DEV)
    assert len(pool) == 1
