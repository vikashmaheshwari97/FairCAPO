from __future__ import annotations

from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
from scripts.run_phase2_budgeted_mocapo import (
    compact_bbq_instruction,
    make_cost_repair_candidates,
)


def test_compact_bbq_instruction_keeps_fairness_and_format():
    prompt = (
        "Choose using evidence. Never use demographic stereotypes. Return the "
        "answer in final tags."
    )

    compact = compact_bbq_instruction(prompt)

    assert "demographic stereotypes" in compact
    assert "<final_answer>A</final_answer>" in compact


def test_make_cost_repair_candidates_strips_expensive_few_shots():
    source = PromptCandidate(
        instruction=(
            "Choose the option supported by the context. Never use demographic "
            "stereotypes. Return the answer in final tags."
        ),
        examples=[
            {"input": "x1", "output": "<final_answer>A</final_answer>"},
            {"input": "x2", "output": "<final_answer>B</final_answer>"},
        ],
        metadata={"method": "source"},
    )
    source.candidate_id = "source"

    portfolio = PromptPortfolio()
    portfolio.add(
        source,
        EvaluationResult(
            candidate_id="source",
            performance=0.98,
            cost=3000.0,
            risk=0.0,
            fairness_risk=0.05,
        ),
    )

    repairs = make_cost_repair_candidates([source], portfolio)

    assert repairs
    assert any(repair.examples == [] for repair in repairs)
    assert any(len(repair.examples) == 1 for repair in repairs)
    assert {repair.metadata["source"] for repair in repairs} == {"cost_repair"}
