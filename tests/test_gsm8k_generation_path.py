"""
Tests for the generation/reasoning (GSM8K) path in the budgeted MO-CAPO runner.

The runner was classification-only (fixed-label match). These tests pin the
additive, task_type-gated generation path: free-form final-answer extraction,
numeric normalization, generation-style prompting, and chain-of-thought few-shot
demonstrations -- while confirming the classification path is unchanged.
"""

from __future__ import annotations

from heal_capo.core import PromptCandidate
from scripts.run_phase2_budgeted_mocapo import (
    GENERATION_TASK_TYPES,
    LLMObjectiveEvaluator,
    _shot_output,
    build_shot_pool,
    extract_final_answer,
    make_llm_prompt,
    normalize_numeric_answer,
)


class EchoLLM:
    """LLM stub that returns a fixed response regardless of prompt."""

    def __init__(self, response: str):
        self.response = response

    def get_response(self, prompt: str) -> str:
        return self.response


# --------------------------------------------------------------------------
# Answer extraction + numeric normalization.
# --------------------------------------------------------------------------


def test_extract_final_answer_from_tags():
    assert extract_final_answer("Work... <final_answer>18</final_answer>") == "18"
    assert extract_final_answer("<FINAL_ANSWER>7</FINAL_ANSWER>") == "7"


def test_extract_final_answer_without_tags_returns_text():
    assert extract_final_answer("the answer is 42") == "the answer is 42"


def test_normalize_numeric_answer_strips_formatting():
    assert normalize_numeric_answer("$1,200.00") == "1200"
    assert normalize_numeric_answer("18 apples") == "18"
    assert normalize_numeric_answer("72") == "72"
    assert normalize_numeric_answer("  -5 ") == "-5"


def test_normalize_numeric_answer_non_numeric_falls_back():
    assert normalize_numeric_answer("yellow") == "yellow"


# --------------------------------------------------------------------------
# Task-type-aware prompting.
# --------------------------------------------------------------------------


def test_make_llm_prompt_generation_has_no_labels():
    candidate = PromptCandidate(instruction="Solve the problem.")
    prompt = make_llm_prompt(candidate, "2+2?", [], task_type="math_reasoning")
    assert "Allowed labels" not in prompt
    assert "<final_answer>" in prompt


def test_make_llm_prompt_classification_unchanged():
    candidate = PromptCandidate(instruction="Classify.")
    prompt = make_llm_prompt(
        candidate, "x", ["subjective", "objective"], task_type="classification"
    )
    assert "Allowed labels: subjective, objective" in prompt


def test_math_reasoning_is_a_generation_task():
    assert "math_reasoning" in GENERATION_TASK_TYPES
    assert "classification" not in GENERATION_TASK_TYPES


# --------------------------------------------------------------------------
# Chain-of-thought few-shot demonstrations.
# --------------------------------------------------------------------------


def test_shot_output_includes_chain_of_thought_for_generation():
    row = {"text": "q", "label": "18", "rationale": "3 * 6 = 18.\n#### 18"}
    out = _shot_output(row, "18", "math_reasoning", use_rationale=True)
    assert "3 * 6 = 18." in out
    assert "<final_answer>18</final_answer>" in out
    assert "####" not in out  # the GSM8K marker is replaced by the tag


def test_shot_output_bare_label_for_classification():
    row = {"text": "q", "label": "objective"}
    out = _shot_output(row, "objective", "classification", use_rationale=True)
    assert out == "<final_answer>objective</final_answer>"


def test_shot_output_bare_when_no_rationale_available():
    row = {"text": "q", "label": "18"}
    out = _shot_output(row, "18", "math_reasoning", use_rationale=True)
    assert out == "<final_answer>18</final_answer>"


def test_build_shot_pool_uses_rationale_from_dev_rows():
    config = {
        "task_type": "math_reasoning",
        "few_shot": {"enabled": True, "use_rationale": True, "pool_size": 5},
    }
    dev_data = [
        {"text": "Janet ...", "label": "18", "rationale": "16 - 3 - 4 = 9; 9 * 2 = 18.\n#### 18"},
    ]
    pool = build_shot_pool(config, dev_data)
    assert len(pool) == 1
    assert pool[0]["input"] == "Janet ..."
    assert "9 * 2 = 18." in pool[0]["output"]
    assert pool[0]["output"].endswith("<final_answer>18</final_answer>")


# --------------------------------------------------------------------------
# End-to-end scoring through the evaluator (LLM stubbed).
# --------------------------------------------------------------------------


def test_evaluator_scores_math_by_numeric_match():
    config = {
        "task_type": "math_reasoning",
        "labels": [],
        "evaluation": {"use_llm": True, "require_final_answer_tags": True},
        "cost": {"input_weight": 0.08, "output_weight": 0.32},
        "fairness": {"in_loop": False},
        "llm": {"model_id": "stub"},
    }
    evaluator = LLMObjectiveEvaluator(config, llm=EchoLLM("Reasoning. <final_answer>1,200</final_answer>"))

    # gold "1200" should match the model's "1,200" after normalization.
    result = evaluator.evaluate(
        PromptCandidate(instruction="Solve."),
        [{"text": "problem", "label": "1200"}],
    )
    assert result.performance == 1.0

    # A different gold should score 0.
    result_wrong = evaluator.evaluate(
        PromptCandidate(instruction="Solve."),
        [{"text": "problem", "label": "99"}],
    )
    assert result_wrong.performance == 0.0
