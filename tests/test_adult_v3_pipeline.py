from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
from scripts.run_adult_v3_mocapo_ablation import (
    FairnessObjectiveDisabledEvaluator,
)
from scripts.run_adult_v3_nsga2_po import add_exact_prompt_identity
from scripts.run_phase2_budgeted_mocapo import LLMObjectiveEvaluator


ROOT = Path(__file__).resolve().parents[1]


def _yaml(relative: str) -> dict:
    with (ROOT / relative).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def test_adult_v3_search_configs_are_aligned():
    fair = _yaml("configs/HPC_Config/adult_faircapo_7p5m_v3_HPC.yaml")
    mo = _yaml("configs/HPC_Config/adult_ablation_7p5m_v3_HPC.yaml")
    nsga = _yaml("configs/HPC_Config/adult_nsga2po_7p5m_v3_HPC.yaml")

    for cfg in (fair, mo, nsga):
        assert cfg["dataset"] == "adult"
        assert cfg["prompt_pool"] == "configs/phase2_prompt_pool_adult_v3.yaml"
        assert cfg["dev"]["data_path"] == "data/adult_semantic_v3.csv"
        assert cfg["dev"]["dev_size"] == 800
        assert cfg["dev"]["shots_size"] == 200
        assert cfg["dev"]["test_size"] == 2000
        assert cfg["llm"]["num_predict"] >= 96
        assert cfg["budget"]["unit"] == "tokens"
        assert cfg["budget"]["max_budget"] == 7_500_000.0
        assert cfg["few_shot"]["max_few_shot_examples"] == 5
        assert cfg["few_shot"]["pool_size"] == 200

    assert fair["fairness"]["in_loop"] is True
    assert nsga["fairness"]["in_loop"] is True
    assert mo["fairness"]["in_loop"] is False

    for cfg in (fair, mo):
        assert cfg["evolutionary"]["population_size"] == 10
        assert cfg["evolutionary"]["offspring_per_iteration"] == 4
        assert cfg["intensification"]["max_blocks_per_challenger"] == 10
        assert cfg["intensification"]["final_intensification"] is True
        assert cfg["intensification"]["add_rejected_to_population"] is False

    assert nsga["nsga2_po"]["population_size"] == 10
    assert nsga["nsga2_po"]["offspring_per_generation"] == 4
    assert nsga["nsga2_po"]["objectives"] == [
        "performance",
        "cost",
        "fairness_risk",
    ]


def test_adult_v3_shared_prompt_pool_keeps_cheap_direct_and_five_shots():
    pool = _yaml("configs/phase2_prompt_pool_adult_v3.yaml")["prompt_pool"]
    ids = [row["id"] for row in pool]

    assert len(pool) == 10
    assert len(ids) == len(set(ids))
    assert "cheap_direct" in ids
    assert max(int(row.get("few_shot_count", 0)) for row in pool) == 5


def test_nsga_serializer_preserves_exact_few_shot_examples():
    candidate = PromptCandidate(
        instruction="Predict income.",
        examples=[
            {
                "input": "Person record:\nEducation: Masters",
                "output": "<final_answer>>50K</final_answer>",
            }
        ],
    )
    candidate.candidate_id = "candidate-1"
    result = EvaluationResult(
        candidate_id=candidate.candidate_id,
        performance=0.8,
        cost=1.0,
        risk=0.2,
        fairness_risk=0.1,
        n_examples=10,
    )
    portfolio = PromptPortfolio()
    portfolio.add(candidate, result)

    rows = add_exact_prompt_identity(
        [{"candidate_id": candidate.candidate_id, "prompt": candidate.instruction}],
        portfolio,
    )

    assert rows[0]["num_few_shot"] == 1
    assert json.loads(rows[0]["few_shot_examples"]) == candidate.examples


def test_mocapo_ablation_forces_search_fairness_objective_to_zero():
    class DummyEvaluator:
        def evaluate(self, candidate, data):
            return EvaluationResult(
                candidate_id=candidate.candidate_id,
                performance=0.8,
                cost=1.0,
                risk=0.2,
                fairness_risk=0.37,
                n_examples=len(data),
                details={"fairness_source": "prompt_heuristic"},
            )

    candidate = PromptCandidate(instruction="Apply one standard consistently.")
    evaluator = FairnessObjectiveDisabledEvaluator(DummyEvaluator())
    result = evaluator.evaluate(candidate, [{"text": "x", "label": "<=50K"}])

    assert result.fairness_risk == 0.0
    assert result.objective_vector == (-0.8, 1.0, 0.0)
    assert result.details["fairness_objective_enabled"] is False
    assert result.details["fairness_source"] == (
        "disabled_for_adult_v3_mocapo_ablation"
    )
    assert result.details["search_fairness_risk_before_disable"] == 0.37


def test_low_accuracy_equalized_odds_candidate_receives_gate_penalty():
    config = {
        "labels": ["<=50K", ">50K"],
        "task_type": "classification",
        "evaluation": {"require_final_answer_tags": True},
        "fairness": {
            "in_loop": True,
            "mode": "equalized_odds",
            "group_key": "sex",
            "positive_label": ">50K",
            "equalized_odds_weight": 1.0,
        },
        "selection": {
            "min_performance_for_fairness": 0.72,
            "low_performance_fairness_penalty": 1.0,
            "fairness_gate_mode": "continuous",
        },
    }

    evaluator = LLMObjectiveEvaluator(config, llm=object())
    adjusted, details = evaluator._apply_fairness_performance_gate(
        performance=0.0,
        fairness_risk=0.0,
        fairness_details={},
    )

    assert adjusted == 0.72
    assert details["fairness_gate_applied"] is True
    assert details["fairness_gate_original_fairness_risk"] == 0.0


def test_python_csv_field_limit_handles_large_portfolio_fields():
    assert csv.field_size_limit() >= 10 * 1024 * 1024
