from __future__ import annotations

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


def _make_evaluator(llm: QueueLLM) -> LLMObjectiveEvaluator:
    return LLMObjectiveEvaluator(
        {
            "labels": ["physician", "surgeon", "software_engineer", "model"],
            "task_type": "classification",
            "evaluation": {
                "require_final_answer_tags": True,
                "classification_mode": "reason_then_label",
            },
            "fairness": {"in_loop": False},
            "cost": {"input_weight": 0.08, "output_weight": 0.32},
            "llm": {"model_id": "fake"},
        },
        llm=llm,
    )


def test_reason_then_label_uses_single_call_and_tagged_answer():
    # The model reasons, then commits inside the tags. One call per example.
    llm = QueueLLM(
        [
            "The biography describes operating-room work and surgical training, "
            "so this is a surgeon rather than a general physician.\n"
            "<final_answer>surgeon</final_answer>"
        ]
    )
    evaluator = _make_evaluator(llm)
    candidate = PromptCandidate(instruction="Classify the biography.")
    data = [{"text": "Biography: Performs operations in the OR.", "label": "surgeon"}]

    result = evaluator.evaluate(candidate, data)

    assert result.performance == 1.0
    assert result.cost > 0
    assert len(llm.prompts) == 1
    # The reason-then-answer prompt should invite step-by-step reasoning.
    assert "step by step" in llm.prompts[0].lower()
    row = result.details["predictions"][0]
    assert row["prediction"] == "surgeon"
    assert row["classification_mode"] == "reason_then_label"


def test_reason_then_label_ignores_labels_mentioned_only_in_reasoning():
    # "physician" and "model" appear in the reasoning but the committed answer is
    # surgeon; only the <final_answer> span must decide the prediction.
    llm = QueueLLM(
        [
            "This is not a physician and not a fashion model. The surgical "
            "credentials point elsewhere.\n<final_answer>surgeon</final_answer>"
        ]
    )
    evaluator = _make_evaluator(llm)
    candidate = PromptCandidate(instruction="Classify the biography.")
    data = [{"text": "Biography: A trauma surgeon.", "label": "surgeon"}]

    result = evaluator.evaluate(candidate, data)

    assert result.performance == 1.0
    assert result.details["predictions"][0]["prediction"] == "surgeon"


def test_reason_then_label_counts_wrong_answer_as_miss():
    llm = QueueLLM(
        ["Looks like general medicine.\n<final_answer>physician</final_answer>"]
    )
    evaluator = _make_evaluator(llm)
    candidate = PromptCandidate(instruction="Classify the biography.")
    data = [{"text": "Biography: A trauma surgeon.", "label": "surgeon"}]

    result = evaluator.evaluate(candidate, data)

    assert result.performance == 0.0
    assert result.details["predictions"][0]["prediction"] == "physician"
