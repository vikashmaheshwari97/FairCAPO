from __future__ import annotations

from heal_capo.core import PromptCandidate
from scripts.run_phase2_budgeted_mocapo import (
    LLMObjectiveEvaluator,
    load_inloop_fairness_pairs,
)


FAIRNESS_DATA = "data/fairness_counterfactual_subj.yaml"


class StubLLM:
    """Deterministic LLM: predicts 'subjective' when the input mentions a female
    cue, else 'objective'. This flips on he/she counterfactual pairs so the
    in-loop fairness signal is non-zero. Counts calls to verify caching."""

    def __init__(self):
        self.calls = 0

    def get_response(self, prompt: str) -> str:
        self.calls += 1
        lowered = prompt.lower()
        if "she " in lowered or "her " in lowered:
            return "subjective"
        return "objective"


def _inloop_config(eval_pairs: int = 2, in_loop: bool = True) -> dict:
    return {
        "labels": ["subjective", "objective"],
        "cost": {"input_weight": 0.08, "output_weight": 0.32},
        "evaluation": {"require_final_answer_tags": False},
        "fairness": {
            "in_loop": in_loop,
            "fairness_data": FAIRNESS_DATA,
            "eval_pairs": eval_pairs,
            "flip_weight": 1.0,
            "group_gap_weight": 0.0,
            "bias_weight": 0.0,
            "debt_weight": 0.0,
            "use_expected_same_prediction": True,
        },
    }


def test_load_inloop_pairs_respects_cap_and_disable():
    cfg = _inloop_config(eval_pairs=3)
    assert len(load_inloop_fairness_pairs(cfg)) == 3
    assert load_inloop_fairness_pairs(_inloop_config(in_loop=False)) == []


def test_inloop_fairness_computed_and_cost_folded():
    stub = StubLLM()
    evaluator = LLMObjectiveEvaluator(_inloop_config(eval_pairs=2), llm=stub)

    candidate = PromptCandidate(instruction="Classify the input.")
    dev = [{"text": "Paris is in France.", "label": "objective"}]

    result = evaluator.evaluate(candidate, dev)

    # 2 counterfactual pairs, both flip on he/she -> flip rate 1.0 (sole signal).
    assert result.fairness_risk == 1.0
    assert result.details["fairness_source"] == "counterfactual_in_loop"
    # The one-time fairness-eval token cost was folded into this block's cost.
    assert result.details["fairness_eval_cost"] > 0.0
    assert result.details["fairness_eval_input_tokens"] > 0


def test_inloop_fairness_is_cached_per_prompt():
    stub = StubLLM()
    evaluator = LLMObjectiveEvaluator(_inloop_config(eval_pairs=2), llm=stub)
    candidate = PromptCandidate(instruction="Classify the input.")
    dev = [{"text": "Paris is in France.", "label": "objective"}]

    risk1, _, cost1 = evaluator._evaluate_candidate_fairness(candidate)
    risk2, _, cost2 = evaluator._evaluate_candidate_fairness(candidate)

    assert risk1 == risk2
    assert cost1 > 0.0          # first computation pays the token cost
    assert cost2 == 0.0         # cache hit: no extra cost
    assert len(evaluator._fairness_cache) == 1


def test_inloop_cache_avoids_repeated_llm_calls_across_blocks():
    stub = StubLLM()
    evaluator = LLMObjectiveEvaluator(_inloop_config(eval_pairs=2), llm=stub)
    candidate = PromptCandidate(instruction="Classify the input.")
    block_a = [{"text": "Paris is in France.", "label": "objective"}]
    block_b = [{"text": "The plot was dull.", "label": "subjective"}]

    evaluator.evaluate(candidate, block_a)
    calls_after_first = stub.calls
    evaluator.evaluate(candidate, block_b)
    calls_after_second = stub.calls

    # First block: 1 dev + 2 pairs*2 = 5 calls. Second block: only 1 dev call,
    # because fairness for this prompt is cached (no 4 extra fairness calls).
    assert calls_after_first == 5
    assert calls_after_second - calls_after_first == 1


def test_inloop_disabled_falls_back_to_heuristic():
    stub = StubLLM()
    evaluator = LLMObjectiveEvaluator(_inloop_config(in_loop=False), llm=stub)
    assert evaluator.fairness_in_loop is False

    candidate = PromptCandidate(instruction="Classify the input fairly.")
    result = evaluator.evaluate(candidate, [{"text": "x", "label": "objective"}])

    assert result.details["fairness_source"] == "prompt_heuristic"
    assert 0.0 <= result.fairness_risk <= 1.0
