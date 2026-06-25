from __future__ import annotations

import json

import pytest

from scripts.evaluate_pareto_on_test import (
    _parse_few_shot_examples,
    build_candidate_rows,
    example_to_row,
    get_test_data,
    objective_specs_from_config,
    portfolio_rows_to_candidates,
    run_test_evaluation,
)


SUMMARY_KEYS = {
    "num_points",
    "num_objectives",
    "hypervolume",
    "optimistic_hypervolume",
    "pessimistic_hypervolume",
    "approximation_gap",
    "nr2",
    "bounds",
    "objective_names",
}


def make_config(tmp_path) -> dict:
    portfolio_csv = tmp_path / "portfolio.csv"
    portfolio_csv.write_text(
        "candidate_id,method,prompt\n"
        "c1,prompt_a,Classify the input. Return only the label.\n"
        "c2,prompt_b,Classify the input using context. Do not hallucinate. "
        "Do not infer from demographic attributes.\n",
        encoding="utf-8",
    )

    return {
        "portfolio_csv": str(portfolio_csv),
        "labels": ["subjective", "objective"],
        "evaluation": {"use_llm": False},
        "test_data": [
            {"text": "The movie was released in 1999.", "label": "objective"},
            {"text": "The acting is wonderful.", "label": "subjective"},
        ],
        "objectives": [
            {"name": "performance", "direction": "maximize"},
            {"name": "cost", "direction": "minimize"},
            {"name": "risk", "direction": "minimize"},
            {"name": "fairness_risk", "direction": "minimize"},
        ],
        "bounds": {
            "performance": [0.0, 1.0],
            "cost": [0.0, 50.0],
            "risk": [0.0, 1.0],
            "fairness_risk": [0.0, 1.0],
        },
        "metrics": {"num_preference_vectors": 8, "seed": 0},
    }


def test_portfolio_rows_to_candidates_reads_prompt_column():
    rows = [
        {"candidate_id": "c1", "method": "m1", "prompt": "Do the task."},
        {"candidate_id": "c2", "method": "m2", "instruction": "Other task."},
    ]

    candidates = portfolio_rows_to_candidates(rows)

    assert len(candidates) == 2
    assert candidates[0].instruction == "Do the task."
    assert candidates[1].instruction == "Other task."
    assert candidates[0].metadata["method"] == "m1"
    # Each candidate gets a fresh id distinct from the source id.
    assert candidates[0].metadata["source_candidate_id"] == "c1"


def test_portfolio_rows_to_candidates_skips_empty_and_raises_when_none():
    assert len(portfolio_rows_to_candidates([{"prompt": ""}, {"prompt": "ok"}])) == 1

    with pytest.raises(ValueError):
        portfolio_rows_to_candidates([{"prompt": "   "}, {"foo": "bar"}])


def test_example_to_row_handles_dict_and_dataclass():
    from experiments.datasets import Example

    assert example_to_row({"text": "hi", "label": "objective"}) == {
        "text": "hi",
        "label": "objective",
    }
    # Alternate keys.
    assert example_to_row({"input": "yo", "answer": "subjective"}) == {
        "text": "yo",
        "label": "subjective",
    }
    # Dataclass.
    assert example_to_row(Example(text="x", label="objective")) == {
        "text": "x",
        "label": "objective",
    }


def test_get_test_data_inline_mode():
    config = {
        "test_data": [
            {"text": "a", "label": "objective"},
            {"input": "b", "answer": "subjective"},
        ]
    }

    rows = get_test_data(config)

    assert rows == [
        {"text": "a", "label": "objective"},
        {"text": "b", "label": "subjective"},
    ]


def test_get_test_data_dataset_mode_converts_examples():
    config = {
        "dataset": "toy",
        "test_size": 5,
        "seed": 0,
    }

    rows = get_test_data(config)

    assert rows, "dataset mode should yield test rows"
    for row in rows:
        assert set(row.keys()) == {"text", "label"}
        assert isinstance(row["text"], str)
        assert isinstance(row["label"], str)


def test_objective_specs_from_config_defaults_and_custom():
    default_specs = objective_specs_from_config({})
    assert [s.name for s in default_specs] == [
        "performance",
        "cost",
        "risk",
        "fairness_risk",
    ]
    assert default_specs[0].direction == "maximize"

    custom = objective_specs_from_config(
        {"objectives": ["performance", "cost", "risk"]}
    )
    assert custom[0].is_maximize()
    assert custom[1].is_minimize()


def test_run_test_evaluation_produces_candidates_and_summary(tmp_path):
    config = make_config(tmp_path)

    result = run_test_evaluation(config, force_no_llm=True)

    rows = result["candidate_rows"]
    assert len(rows) == 2

    # Each row has the required columns.
    for row in rows:
        assert "performance" in row
        assert "is_pareto" in row
        assert "prompt" in row
        assert row["n_examples"] == 2

    # At least one prompt is on the (recomputed) test-set Pareto front.
    assert any(row["is_pareto"] for row in rows)

    # Summary carries all MO metric keys.
    assert SUMMARY_KEYS.issubset(set(result["summary"].keys()))

    meta = result["metadata"]
    assert meta["num_prompts"] == 2
    assert meta["num_test_examples"] == 2
    assert meta["used_llm"] is False
    assert meta["uses_fixed_bounds"] is True


def test_run_test_evaluation_is_json_serializable(tmp_path):
    config = make_config(tmp_path)
    result = run_test_evaluation(config, force_no_llm=True)

    # Should not raise.
    json.dumps({"metadata": result["metadata"], "summary": result["summary"]})


def test_parse_few_shot_examples_handles_variants():
    # JSON string (the on-disk CSV form).
    raw = json.dumps(
        [
            {"input": "q1", "output": "<final_answer>a1</final_answer>"},
            {"input": "q2", "output": "<final_answer>a2</final_answer>"},
        ]
    )
    parsed = _parse_few_shot_examples(raw)
    assert len(parsed) == 2
    assert parsed[0] == {"input": "q1", "output": "<final_answer>a1</final_answer>"}

    # Already-decoded list passes through.
    assert _parse_few_shot_examples([{"input": "x", "output": "y"}]) == [
        {"input": "x", "output": "y"}
    ]

    # Empty / missing / malformed -> [] (backward compatible with old portfolios).
    assert _parse_few_shot_examples("") == []
    assert _parse_few_shot_examples(None) == []
    assert _parse_few_shot_examples("[]") == []
    assert _parse_few_shot_examples("not json") == []
    assert _parse_few_shot_examples(json.dumps({"input": "x"})) == []  # not a list
    # List items without input/output keys are dropped.
    assert _parse_few_shot_examples(json.dumps([{"foo": "bar"}])) == []


def test_portfolio_rows_to_candidates_restores_few_shot_examples():
    rows = [
        {
            "candidate_id": "few_shot_abc",
            "method": "few_shot",
            "prompt": "Classify the input.",
            "num_few_shot": "2",
            "few_shot_examples": json.dumps(
                [
                    {"input": "q1", "output": "<final_answer>objective</final_answer>"},
                    {"input": "q2", "output": "<final_answer>subjective</final_answer>"},
                ]
            ),
        }
    ]

    candidates = portfolio_rows_to_candidates(rows)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert len(candidate.examples) == 2
    assert candidate.metadata["num_few_shot"] == 2
    # The reconstructed prompt is the EXACT few-shot prompt, not a zero-shot strip.
    rendered = candidate.render("a test input")
    assert "q1" in rendered
    assert "q2" in rendered
    assert "Examples:" in rendered


def test_portfolio_rows_to_candidates_zero_shot_when_no_column():
    # An older portfolio CSV without the few_shot_examples column -> zero-shot.
    rows = [{"candidate_id": "c1", "method": "m1", "prompt": "Do the task."}]

    candidates = portfolio_rows_to_candidates(rows)

    assert candidates[0].examples == []
    assert candidates[0].metadata["num_few_shot"] == 0


def test_few_shot_persist_restore_round_trip():
    """
    End-to-end: a few-shot winner written by the runner's portfolio_to_rows must
    survive serialization and rebuild with its demonstrations intact. This is the
    regression guard for the Dtest staircase-collapse bug (few-shot winners were
    silently re-evaluated zero-shot on held-out data).
    """
    from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
    from scripts.run_phase2_budgeted_mocapo import portfolio_to_rows

    examples = [
        {"input": "The plot was thrilling.", "output": "<final_answer>subjective</final_answer>"},
        {"input": "It was filmed in 1999.", "output": "<final_answer>objective</final_answer>"},
    ]
    candidate = PromptCandidate(
        instruction="Classify the input.",
        examples=examples,
        candidate_id="few_shot_winner",
        metadata={"method": "few_shot", "category": "baseline"},
    )
    result = EvaluationResult(
        candidate_id="few_shot_winner",
        performance=0.8,
        cost=120.0,
        risk=0.0,
        fairness_risk=0.0,
    )

    portfolio = PromptPortfolio()
    portfolio.add(candidate, result)

    # Runner-side persistence.
    rows = portfolio_to_rows(portfolio, pareto_ids={"few_shot_winner"})
    assert rows[0]["num_few_shot"] == 2

    # Simulate a CSV round-trip: values become strings on disk.
    csv_row = {key: ("" if value is None else str(value)) for key, value in rows[0].items()}

    # Evaluator-side restoration.
    restored = portfolio_rows_to_candidates([csv_row])
    assert len(restored) == 1
    assert len(restored[0].examples) == 2
    assert restored[0].examples == examples
    # The rebuilt prompt renders identically to the original few-shot prompt.
    assert restored[0].render("held-out example") == candidate.render("held-out example")


def test_build_candidate_rows_marks_pareto():
    from heal_capo.core import EvaluationResult, PromptCandidate

    c1 = PromptCandidate(instruction="a", candidate_id="id1", metadata={"method": "m1"})
    c2 = PromptCandidate(instruction="b", candidate_id="id2", metadata={"method": "m2"})

    r1 = EvaluationResult(candidate_id="id1", performance=0.9, cost=1.0, risk=0.1, fairness_risk=0.1)
    r2 = EvaluationResult(candidate_id="id2", performance=0.5, cost=5.0, risk=0.5, fairness_risk=0.5)

    rows = build_candidate_rows([r1, r2], [c1, c2], pareto_ids={"id1"})

    by_method = {row["method"]: row for row in rows}
    assert by_method["m1"]["is_pareto"] is True
    assert by_method["m2"]["is_pareto"] is False
