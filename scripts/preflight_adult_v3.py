from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

import yaml

from experiments.datasets import load_paper_dataset
from heal_capo.core import PromptCandidate


FAIR_CONFIG = Path("configs/HPC_Config/adult_faircapo_1m_v3_HPC.yaml")
MO_CONFIG = Path("configs/HPC_Config/adult_ablation_1m_v3_HPC.yaml")
NSGA_CONFIG = Path("configs/HPC_Config/adult_nsga2po_1m_v3_HPC.yaml")
EVAL_CONFIG = Path("configs/HPC_Config/adult_eval_large_1m_v3_HPC.yaml")
TABLE_CONFIG = Path("configs/HPC_Config/adult_experiment_table_7p5m_v3_large_HPC.yaml")
PROMPT_POOL = Path("configs/phase2_prompt_pool_adult_v3.yaml")
SUBMIT_PIPELINE = Path("scripts/hpc/submit_adult_7p5m_v3_pipeline.sh")

DEV_SIZE = 400
SHOTS_SIZE = 100
TEST_SIZE = 1000
TOKEN_BUDGET = 1_000_000.0

REQUIRED_RUNTIME_FILES = [
    Path("scripts/run_phase2_budgeted_mocapo.py"),
    Path("scripts/run_adult_v3_mocapo_ablation.py"),
    Path("scripts/run_baseline_nsga2_po.py"),
    Path("scripts/run_adult_v3_nsga2_po.py"),
    Path("scripts/evaluate_pareto_on_test.py"),
    Path("scripts/evaluate_adult_v3_on_test.py"),
    Path("scripts/build_representative_experiment_table.py"),
    Path("scripts/hpc/run_bios_hpc.slurm"),
    Path("scripts/hpc/run_bios_nsga_hpc.slurm"),
    Path("scripts/hpc/run_bios_eval_hpc.slurm"),
    Path("scripts/hpc/build_adult_7p5m_v3_large_outputs.sh"),
    SUBMIT_PIPELINE,
]


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def nested(config: dict, *keys, default=None):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def audit_config_alignment() -> None:
    fair = load_yaml(FAIR_CONFIG)
    mo = load_yaml(MO_CONFIG)
    nsga = load_yaml(NSGA_CONFIG)
    eval_cfg = load_yaml(EVAL_CONFIG)
    table_cfg = load_yaml(TABLE_CONFIG)
    prompt_cfg = load_yaml(PROMPT_POOL)

    for path in REQUIRED_RUNTIME_FILES:
        require(path.is_file(), f"Missing runtime file: {path}")

    pipeline_text = SUBMIT_PIPELINE.read_text(encoding="utf-8")
    require("adult_faircapo_1m_v3_HPC.yaml" in pipeline_text, "Submission does not use the 1M FairCAPO config")
    require("adult_ablation_1m_v3_HPC.yaml" in pipeline_text, "Submission does not use the 1M MO-CAPO config")
    require("adult_nsga2po_1m_v3_HPC.yaml" in pipeline_text, "Submission does not use the 1M NSGA config")
    require("MOCAPO_RUNNER=scripts/run_adult_v3_mocapo_ablation.py" in pipeline_text, "Strict MO-CAPO ablation runner is not wired")
    require("NSGA_RUNNER=scripts/run_adult_v3_nsga2_po.py" in pipeline_text, "Exact-prompt NSGA runner is not wired")
    require("EVAL_RUNNER=scripts/evaluate_adult_v3_on_test.py" in pipeline_text, "Complete-prompt evaluator is not wired")

    prompts = prompt_cfg.get("prompt_pool", [])
    prompt_ids = [str(item.get("id", "")) for item in prompts]
    require(len(prompts) == 10, f"Expected 10 shared prompts, found {len(prompts)}")
    require(len(set(prompt_ids)) == len(prompt_ids), "Duplicate prompt IDs")
    require("cheap_direct" in prompt_ids, "cheap_direct must remain in the pool")
    require(max(int(item.get("few_shot_count", 0) or 0) for item in prompts) == 5, "Five-shot seed prompt is missing")

    configs = {"FairCAPO": fair, "MO-CAPO": mo, "NSGA-II-PO": nsga}
    for name, cfg in configs.items():
        require(cfg.get("dataset") == "adult", f"{name}: wrong dataset")
        require(cfg.get("prompt_pool") == str(PROMPT_POOL), f"{name}: prompt pool mismatch")
        require(nested(cfg, "dev", "data_path") == "data/adult_semantic_v3.csv", f"{name}: data path mismatch")
        require(nested(cfg, "dev", "dev_size") == DEV_SIZE, f"{name}: dev size mismatch")
        require(nested(cfg, "dev", "shots_size") == SHOTS_SIZE, f"{name}: shots size mismatch")
        require(nested(cfg, "dev", "test_size") == TEST_SIZE, f"{name}: test size mismatch")
        require(nested(cfg, "dev", "stratify_group_key") == "sex", f"{name}: group key mismatch")
        require(int(nested(cfg, "llm", "num_predict")) >= 96, f"{name}: num_predict too small")
        require(float(nested(cfg, "budget", "max_budget")) == TOKEN_BUDGET, f"{name}: budget mismatch")
        require(int(nested(cfg, "few_shot", "max_few_shot_examples")) == 5, f"{name}: max shots mismatch")
        require(int(nested(cfg, "few_shot", "pool_size")) == SHOTS_SIZE, f"{name}: shot pool mismatch")

    require(nested(fair, "fairness", "in_loop") is True, "FairCAPO fairness must be enabled")
    require(nested(nsga, "fairness", "in_loop") is True, "NSGA fairness must be enabled")
    require(nested(mo, "fairness", "in_loop") is False, "MO-CAPO fairness objective must be disabled")

    for name, cfg in (("FairCAPO", fair), ("MO-CAPO", mo)):
        require(int(nested(cfg, "evolutionary", "population_size")) == 10, f"{name}: population mismatch")
        require(int(nested(cfg, "evolutionary", "offspring_per_iteration")) == 3, f"{name}: offspring mismatch")
        require(int(nested(cfg, "evolutionary", "max_iterations")) == 20, f"{name}: iteration cap mismatch")
        require(int(nested(cfg, "intensification", "max_blocks_per_challenger")) == 4, f"{name}: block depth mismatch")
        require(nested(cfg, "intensification", "add_rejected_to_population") is False, f"{name}: rejected candidates can enter archive")

    require(int(nested(nsga, "nsga2_po", "population_size")) == 10, "NSGA population mismatch")
    require(int(nested(nsga, "nsga2_po", "offspring_per_generation")) == 3, "NSGA offspring mismatch")
    require(int(nested(nsga, "nsga2_po", "max_generations")) == 20, "NSGA generation cap mismatch")

    require(eval_cfg.get("dev_size") == DEV_SIZE, "Held-out dev offset mismatch")
    require(eval_cfg.get("shots_size") == SHOTS_SIZE, "Held-out shots offset mismatch")
    require(eval_cfg.get("test_size") == TEST_SIZE, "Held-out test size mismatch")
    require(nested(eval_cfg, "fairness", "mode") == "equalized_odds", "Held-out fairness mismatch")
    require(nested(table_cfg, "selection", "min_performance_for_fairness") == 0.70, "Reporting threshold mismatch")
    require(len(table_cfg.get("methods", [])) == 3, "Comparison table must contain three methods")


def audit_csv_and_split(data_path: Path) -> None:
    require(data_path.is_file() and data_path.stat().st_size > 0, f"Missing semantic CSV: {data_path}")

    allowed_missing = {"workclass", "occupation", "native.country"}
    missing_counts: Counter[str] = Counter()
    with data_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) >= 3000, f"Semantic CSV is unexpectedly small: {len(rows)}")

    for row in rows:
        for column, value in row.items():
            if str(value).strip() == "?":
                missing_counts[column] += 1
    require(not set(missing_counts).difference(allowed_missing), "Unexpected '?' column")

    split = load_paper_dataset(
        "adult",
        data_path=str(data_path),
        dev_size=DEV_SIZE,
        shots_size=SHOTS_SIZE,
        test_size=TEST_SIZE,
        seed=0,
        allow_smaller=False,
        stratified=True,
        stratify_group_key="sex",
    )

    index_sets = []
    for split_name, examples, expected_cell in (
        ("dev", split.dev, 100),
        ("shots", split.shots, 25),
        ("test", split.test, 250),
    ):
        indices = {int((example.metadata or {})["source_index"]) for example in examples}
        require(len(indices) == len(examples), f"{split_name}: duplicate rows")
        index_sets.append(indices)
        cells = Counter((example.label, str((example.metadata or {}).get("sex", ""))) for example in examples)
        require(len(cells) == 4 and all(count == expected_cell for count in cells.values()), f"{split_name}: unbalanced cells {cells}")
        for example in examples:
            lowered = example.text.lower()
            require("ordinal education level" in lowered, f"{split_name}: semantic education missing")
            require("sex:" not in lowered and "race:" not in lowered, f"{split_name}: protected leakage")
            require("income:" not in lowered and "fnlwgt" not in lowered, f"{split_name}: target/weight leakage")
            require("?" not in example.text, f"{split_name}: literal '?' reached prompt")

    require(index_sets[0].isdisjoint(index_sets[1]), "Dev and shots overlap")
    require(index_sets[0].isdisjoint(index_sets[2]), "Dev and test overlap")
    require(index_sets[1].isdisjoint(index_sets[2]), "Shots and test overlap")

    old_env = os.environ.get("FAIRCAPO_ADULT_REASONING_SHOTS")
    os.environ["FAIRCAPO_ADULT_REASONING_SHOTS"] = "1"
    try:
        shot = split.shots[0]
        candidate = PromptCandidate(
            instruction="Predict annual income and return one tagged label.",
            examples=[{"input": shot.text, "output": f"<final_answer>{shot.label}</final_answer>"}],
        )
        rendered = candidate.render(split.dev[0].text)
        require("Reasoning: combine the economic evidence" in rendered, "Reasoning shots are inactive")
        require("Sex:" not in rendered and "Race:" not in rendered, "Protected demonstration leakage")
    finally:
        if old_env is None:
            os.environ.pop("FAIRCAPO_ADULT_REASONING_SHOTS", None)
        else:
            os.environ["FAIRCAPO_ADULT_REASONING_SHOTS"] = old_env

    print(f"Semantic CSV rows: {len(rows)}")
    print(f"Literal '?' counts: {dict(missing_counts)}")
    print("Split cells: dev=100, shots=25, test=250 per (income, sex) cell")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Adult v3 1M pilot before submission.")
    parser.add_argument("--data", default="data/adult_semantic_v3.csv")
    args = parser.parse_args()
    audit_config_alignment()
    audit_csv_and_split(Path(args.data))
    print("Adult v3 1M preflight PASSED.")


if __name__ == "__main__":
    main()
