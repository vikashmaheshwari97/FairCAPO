from __future__ import annotations

import json

from heal_capo.core import PromptCandidate
from scripts.run_phase2_budgeted_mocapo import LLMObjectiveEvaluator


class QueueLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def get_response(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("No queued LLM response left")
        return self.responses.pop(0)


def test_two_stage_label_scoring_selects_highest_supported_label():
    llm = QueueLLM(
        [
            "<final_answer>teacher, software_engineer, professor</final_answer>",
            "<final_answer>1</final_answer>",
            "<final_answer>5</final_answer>",
            "<final_answer>2</final_answer>",
        ]
    )
    evaluator = LLMObjectiveEvaluator(
        {
            "labels": ["teacher", "software_engineer", "professor"],
            "task_type": "classification",
            "evaluation": {
                "require_final_answer_tags": True,
                "classification_mode": "two_stage_label_scoring",
                "label_scoring_top_k": 3,
                "label_scoring_score_scale": 5,
            },
            "fairness": {"in_loop": False},
            "cost": {"input_weight": 0.08, "output_weight": 0.32},
            "llm": {"model_id": "fake"},
        },
        llm=llm,
    )
    candidate = PromptCandidate(instruction="Classify the biography.")
    data = [
        {
            "text": "Biography: Built distributed systems and shipped production software.",
            "label": "software_engineer",
        }
    ]

    result = evaluator.evaluate(candidate, data)

    assert result.performance == 1.0
    assert result.cost > 0
    assert len(llm.prompts) == 4
    row = result.details["predictions"][0]
    assert row["prediction"] == "software_engineer"
    assert row["classification_mode"] == "two_stage_label_scoring"
    assert json.loads(row["raw_response"])["label_scores"]["software_engineer"] == 5.0


def test_two_stage_label_scoring_falls_back_when_topk_is_invalid():
    llm = QueueLLM(
        [
            "<final_answer>not_a_profession, unknown, none</final_answer>",
            "<final_answer>teacher</final_answer>",
        ]
    )
    evaluator = LLMObjectiveEvaluator(
        {
            "labels": ["teacher", "software_engineer", "professor"],
            "task_type": "classification",
            "evaluation": {
                "require_final_answer_tags": True,
                "classification_mode": "two_stage_label_scoring",
                "label_scoring_top_k": 3,
            },
            "fairness": {"in_loop": False},
            "llm": {"model_id": "fake"},
        },
        llm=llm,
    )
    candidate = PromptCandidate(instruction="Classify the biography.")
    data = [{"text": "Biography: Taught mathematics.", "label": "teacher"}]

    result = evaluator.evaluate(candidate, data)

    assert result.performance == 1.0
    assert result.details["predictions"][0]["label_scoring_fallback"] is True
    assert len(llm.prompts) == 2


def test_targeted_disambiguation_corrects_known_confusion_pair():
    llm = QueueLLM(
        [
            "<final_answer>physician</final_answer>",
            "<final_answer>surgeon</final_answer>",
        ]
    )
    evaluator = LLMObjectiveEvaluator(
        {
            "labels": ["physician", "surgeon", "teacher"],
            "task_type": "classification",
            "evaluation": {
                "require_final_answer_tags": True,
                "classification_mode": "targeted_disambiguation",
                "label_confusion_groups": [["physician", "surgeon"]],
            },
            "fairness": {"in_loop": False},
            "llm": {"model_id": "fake"},
        },
        llm=llm,
    )
    candidate = PromptCandidate(instruction="Classify the biography.")
    data = [
        {
            "text": "Biography: Performs operations in hospital operating rooms.",
            "label": "surgeon",
        }
    ]

    result = evaluator.evaluate(candidate, data)

    assert result.performance == 1.0
    assert len(llm.prompts) == 2
    row = result.details["predictions"][0]
    assert row["prediction"] == "surgeon"
    assert row["classification_mode"] == "targeted_disambiguation"
    assert row["disambiguation_applied"] is True
    assert row["initial_prediction"] == "physician"
    assert row["disambiguation_group"] == ["physician", "surgeon"]


def test_targeted_disambiguation_skips_unconfigured_prediction():
    llm = QueueLLM(["<final_answer>teacher</final_answer>"])
    evaluator = LLMObjectiveEvaluator(
        {
            "labels": ["physician", "surgeon", "teacher"],
            "task_type": "classification",
            "evaluation": {
                "require_final_answer_tags": True,
                "classification_mode": "targeted_disambiguation",
                "label_confusion_groups": [["physician", "surgeon"]],
            },
            "fairness": {"in_loop": False},
            "llm": {"model_id": "fake"},
        },
        llm=llm,
    )
    candidate = PromptCandidate(instruction="Classify the biography.")
    data = [{"text": "Biography: Teaches mathematics.", "label": "teacher"}]

    result = evaluator.evaluate(candidate, data)

    assert result.performance == 1.0
    assert len(llm.prompts) == 1
    row = result.details["predictions"][0]
    assert row["prediction"] == "teacher"
    assert row["classification_mode"] == "targeted_disambiguation"
    assert row["disambiguation_applied"] is False
