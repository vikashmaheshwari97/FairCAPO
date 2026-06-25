import random

from heal_capo.components.prompt_generator import (
    FairnessAwarePromptGenerator,
    FairnessGenerationConfig,
    fairness_repair_fallback,
    make_fairness_repair_meta_prompt,
    summarize_failed_pairs,
)


def _pair_row(prompt_id, attr, base_pred, cf_pred, expected_same=True):
    flipped = base_pred != cf_pred
    violation = flipped if expected_same else (not flipped)
    return {
        "prompt_id": prompt_id,
        "protected_attribute": attr,
        "base_text": f"He has {attr}.",
        "counterfactual_text": f"She has {attr}.",
        "base_prediction": base_pred,
        "counterfactual_prediction": cf_pred,
        "expected_same_prediction": expected_same,
        "flipped": flipped,
        "violation": violation,
    }


def test_summarize_failed_pairs_counts_violations_and_attributes():
    rows = [
        _pair_row("p0", "gender", "objective", "subjective"),  # violation
        _pair_row("p0", "race", "objective", "objective"),     # ok
        _pair_row("p0", "age", "subjective", "objective"),     # violation
    ]

    num, attrs, failed = summarize_failed_pairs(rows)

    assert num == 2
    assert attrs == ["gender", "age"]
    assert len(failed) == 2


def test_summarize_failed_pairs_falls_back_to_flipped():
    rows = [{"protected_attribute": "gender", "flipped": True}]

    num, attrs, failed = summarize_failed_pairs(rows)

    assert num == 1
    assert attrs == ["gender"]


def test_fairness_repair_fallback_appends_clause_and_format():
    rng = random.Random(0)
    repaired = fairness_repair_fallback(
        instruction="Classify the input.",
        triggering_attributes=["gender"],
        output_format="Return the final answer inside <final_answer> and </final_answer> tags.",
        rng=rng,
    )

    assert "Classify the input." in repaired
    assert "gender" in repaired
    assert "<final_answer>" in repaired.lower()


def test_make_repair_meta_prompt_embeds_examples_and_attributes():
    rows = [_pair_row("p0", "gender", "objective", "subjective")]

    meta = make_fairness_repair_meta_prompt(
        instruction="Classify the input.",
        triggering_attributes=["gender"],
        failed_examples=rows,
        task_description="Subjectivity classification.",
    )

    assert "gender" in meta
    assert "<prompt>" in meta
    assert "Classify the input." in meta


def test_generator_fallback_produces_drift_passing_candidates():
    # No meta-LLM -> deterministic fallback path.
    generator = FairnessAwarePromptGenerator(
        config=FairnessGenerationConfig(
            max_new_prompts_per_seed=2,
            min_flips_to_trigger=1,
            random_seed=0,
        ),
        meta_llm=None,
    )

    rows = [
        _pair_row("p0", "gender", "objective", "subjective"),
        _pair_row("p0", "race", "objective", "subjective"),
    ]

    generated = generator.generate_from_failures(
        prompt_id="p0",
        instruction="Classify the input. Return only the label.",
        pair_rows=rows,
    )

    assert len(generated) == 2
    for gen in generated:
        assert not gen.used_meta_llm
        assert gen.candidate.metadata["parent_prompt_id"] == "p0"
        assert gen.candidate.metadata["category"] == "fairness_repair"
        # Fallback appends fairness language, so the fairness drift guard passes.
        assert gen.accepted
        assert gen.drift_result.passed


def test_generator_skips_when_below_trigger_threshold():
    generator = FairnessAwarePromptGenerator(
        config=FairnessGenerationConfig(min_flips_to_trigger=3, random_seed=0),
        meta_llm=None,
    )

    rows = [_pair_row("p0", "gender", "objective", "subjective")]

    generated = generator.generate_from_failures(
        prompt_id="p0",
        instruction="Classify the input.",
        pair_rows=rows,
    )

    assert generated == []


class _StubMetaLLM:
    """Returns a fairness-hardened prompt in the expected <prompt> format."""

    def __init__(self):
        self.calls = 0

    def get_response(self, prompt: str) -> str:
        self.calls += 1
        return (
            "<prompt>Classify the input based only on its content. "
            "Do not infer the label from gender, race, ethnicity, or other "
            "demographic attributes. "
            "Return the final answer inside <final_answer> and </final_answer> tags."
            "</prompt>"
        )


def test_generator_uses_meta_llm_when_available():
    stub = _StubMetaLLM()
    generator = FairnessAwarePromptGenerator(
        config=FairnessGenerationConfig(
            max_new_prompts_per_seed=1,
            min_flips_to_trigger=1,
            random_seed=0,
        ),
        meta_llm=stub,
    )

    rows = [_pair_row("p0", "gender", "objective", "subjective")]

    generated = generator.generate_from_failures(
        prompt_id="p0",
        instruction="Classify the input.",
        pair_rows=rows,
    )

    assert len(generated) == 1
    gen = generated[0]
    assert gen.used_meta_llm
    assert stub.calls == 1
    assert "do not infer" in gen.candidate.instruction.lower()
    assert gen.accepted


def test_generator_is_deterministic_under_seed():
    rows = [
        _pair_row("p0", "gender", "objective", "subjective"),
        _pair_row("p0", "race", "objective", "subjective"),
    ]

    def run():
        gen = FairnessAwarePromptGenerator(
            config=FairnessGenerationConfig(
                max_new_prompts_per_seed=2,
                min_flips_to_trigger=1,
                random_seed=123,
            ),
            meta_llm=None,
        )
        return [g.candidate.instruction for g in gen.generate_from_failures(
            prompt_id="p0",
            instruction="Classify the input.",
            pair_rows=rows,
        )]

    assert run() == run()
