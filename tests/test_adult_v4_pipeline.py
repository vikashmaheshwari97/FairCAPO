from __future__ import annotations

from pathlib import Path

import yaml

from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
from heal_capo.optimizers.intensification import IntensificationConfig
from heal_capo.pareto import dominates, non_dominated_ids
from scripts.prepare_adult_semantic_csv import enrich_row
from scripts.run_phase2_budgeted_mocapo import validate_runtime_config

ROOT = Path(__file__).resolve().parents[1]
SEARCH_CONFIG = ROOT / "configs/HPC_Config/adult_faircapo_qwen_retrieval_4m_v4_search.yaml"
EVAL_CONFIG = ROOT / "configs/HPC_Config/adult_eval_qwen_retrieval_4m_v4_HPC.yaml"
V3_POOL = ROOT / "configs/phase2_prompt_pool_adult_accuracy_v3.yaml"
V4_POOL = ROOT / "configs/phase2_prompt_pool_adult_accuracy_v4.yaml"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_active_adult_v4_search_config_is_repaired() -> None:
    config = _yaml(SEARCH_CONFIG)
    assert config["dataset"] == "adult"
    assert config["task_description"].strip()
    assert config["prompt_pool"].endswith("phase2_prompt_pool_adult_accuracy_v4.yaml")
    assert config["retrieval"]["policy"] == "natural_similarity"
    assert config["retrieval"]["k"] == 4
    assert config["cost"]["input_weight"] > 0
    assert config["cost"]["output_weight"] > 0
    assert config["selection"]["low_performance_fairness_penalty"] == 0.0
    assert config["selection"]["min_performance_feasibility"] == 0.75
    assert config["num_seed_blocks"] >= 3
    assert config["intensification"]["min_blocks_before_reject"] >= 3
    assert config["intensification"]["max_blocks_per_challenger"] <= 8
    assert config["intensification"]["final_intensification"] is False
    validate_runtime_config(config)


def test_eval_config_exactly_aligns_search_data_and_retrieval() -> None:
    search = _yaml(SEARCH_CONFIG)
    evaluation = _yaml(EVAL_CONFIG)
    assert evaluation["portfolio_csv"].endswith(
        "adult_faircapo_qwen_retrieval_4m_v4/seed_0/phase2_prompt_portfolio.csv"
    )
    assert evaluation["output_dir"].endswith(
        "adult_faircapo_qwen_retrieval_4m_v4_eval/seed_0"
    )
    assert evaluation["dev"] == search["dev"]
    assert evaluation["retrieval"] == search["retrieval"]
    assert evaluation["fairness"] == search["fairness"]
    assert evaluation["selection"] == search["selection"]
    assert evaluation["cost"] == search["cost"]
    assert "dev_size" not in evaluation
    assert "shots_size" not in evaluation
    assert "test_size" not in evaluation


def test_only_one_active_qwen_search_config_remains() -> None:
    hpc_dir = ROOT / "configs/HPC_Config"
    assert SEARCH_CONFIG.is_file()
    assert not (hpc_dir / "adult_faircapo_qwen_retrieval_4m_v4_HPC.yaml").exists()
    assert not (hpc_dir / "adult_qwen_retrieval_v4_search_500k.yaml").exists()
    assert not (hpc_dir / "adult_qwen_retrieval_v4_eval_500k.yaml").exists()


def test_v3_pool_is_deprecated_and_v4_pool_is_active() -> None:
    v3 = _yaml(V3_POOL)
    v4 = _yaml(V4_POOL)
    assert v3["deprecated"] is True
    assert v3["replacement"].endswith("phase2_prompt_pool_adult_accuracy_v4.yaml")
    assert v4["version"] == "adult_accuracy_v4"
    prompts = v4["prompt_pool"]
    ids = [row["id"] for row in prompts]
    assert len(prompts) == len(set(ids)) == 6
    assert all(int(row.get("few_shot_count", 0)) == 0 for row in prompts)
    assert all("sex" not in row["prompt"].lower() for row in prompts)
    assert all("race" not in row["prompt"].lower() for row in prompts)


def test_adult_semantic_enrichment_preserves_protected_and_target_fields() -> None:
    source = {
        "age": "40",
        "workclass": "Private",
        "fnlwgt": "123",
        "education": "Bachelors",
        "education.num": "13",
        "marital.status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital.gain": "1000",
        "capital.loss": "0",
        "hours.per.week": "45",
        "native.country": "United-States",
        "income": ">50K",
    }
    enriched = enrich_row(source)
    assert enriched["sex"] == source["sex"]
    assert enriched["race"] == source["race"]
    assert enriched["income"] == source["income"]
    assert enriched["fnlwgt"] == source["fnlwgt"]
    assert "ordinal education level 13" in enriched["education"]


def test_prompt_portfolio_replaces_existing_candidate_snapshot() -> None:
    candidate = PromptCandidate(instruction="first", candidate_id="same")
    updated = PromptCandidate(instruction="second", candidate_id="same")
    first_result = EvaluationResult("same", 0.70, 1.0, 0.30, fairness_risk=0.20)
    updated_result = EvaluationResult("same", 0.80, 1.2, 0.20, fairness_risk=0.10)
    portfolio = PromptPortfolio()
    portfolio.add(candidate, first_result)
    portfolio.add(updated, updated_result)
    assert len(portfolio.candidates) == 1
    assert portfolio.get("same").instruction == "second"
    assert portfolio.get_result("same").performance == 0.80


def test_performance_feasibility_precedes_pareto_objectives() -> None:
    feasible = EvaluationResult(
        "feasible",
        performance=0.76,
        cost=10.0,
        risk=0.24,
        fairness_risk=0.20,
        details={"performance_feasibility_threshold": 0.75},
    )
    infeasible = EvaluationResult(
        "infeasible",
        performance=0.74,
        cost=1.0,
        risk=0.26,
        fairness_risk=0.01,
        details={"performance_feasibility_threshold": 0.75},
    )
    assert dominates(feasible, infeasible)
    assert non_dominated_ids([feasible, infeasible]) == ["feasible"]


def test_intensification_config_honours_minimum_evidence_depth() -> None:
    config = IntensificationConfig(
        max_blocks_per_challenger=8,
        min_blocks_before_reject=3,
        reject_when_dominated=True,
    )
    assert config.min_blocks_before_reject == 3
    assert config.max_blocks_per_challenger == 8


def test_removed_placeholders_do_not_exist() -> None:
    assert not (ROOT / "scripts/accuracy_discovery_runner.py").exists()
    assert not (ROOT / "heal_capo/adult_diag_data.py").exists()
    assert not (ROOT / "scripts/hpc/run_adult_qwen_v4.slurm").exists()
    assert not (ROOT / "tests/test_adult_v3_pipeline.py").exists()
