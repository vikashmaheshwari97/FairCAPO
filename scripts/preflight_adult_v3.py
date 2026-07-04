from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

import yaml

from experiments.datasets import load_paper_dataset
from heal_capo.core import PromptCandidate


FAIR_CONFIG = Path("configs/HPC_Config/adult_faircapo_7p5m_v3_HPC.yaml")
MO_CONFIG = Path("configs/HPC_Config/adult_ablation_7p5m_v3_HPC.yaml")
NSGA_CONFIG = Path("configs/HPC_Config/adult_nsga2po_7p5m_v3_HPC.yaml")
EVAL_CONFIG = Path("configs/HPC_Config/adult_eval_large_7p5m_v3_HPC.yaml")
TABLE_CONFIG = Path("configs/HPC_Config/adult_experiment_table_7p5m_v3_large_HPC.yaml")
PROMPT_POOL = Path("configs/phase2_prompt_pool_adult_v3.yaml")

REQUIRED_RUNTIME_FILES = [
    Path("scripts/run_phase2_budgeted_mocapo.py"),
    Path("scripts/run_baseline_nsga2_po.py"),
    Path("scripts/evaluate_pareto_on_test.py"),
    Path("scripts/build_representative_experiment_table.py"),
    Path("scripts/hpc/run_bios_hpc.slurm"),
    Path("scripts/hpc/run_bios_nsga_hpc.slurm"),
    Path("scripts/hpc/run_bios_eval_hpc.slurm"),
    Path("scripts/hpc/build_adult_7p5m_v3_large_outputs.sh"),
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


def audit_config_alignment() -> tuple[dict, dict, dict, dict]:
    fair = load_yaml(FAIR_CONFIG)
    mo = load_yaml(MO_CONFIG)
    nsga = load_yaml(NSGA_CONFIG)
    eval_cfg = load_yaml(EVAL_CONFIG)
    table_cfg = load_yaml(TABLE_CONFIG)
    prompt_cfg = load_yaml(PROMPT_POOL)

    for path in REQUIRED_RUNTIME_FILES:
        require(path.is_file(), f"Missing runtime file: {path}")

    prompts = prompt_cfg.get("prompt_pool", [])
    prompt_ids = [str(item.get("id", "")) for item in prompts]
    require(len(prompts) == 10, f"Expected 10 shared prompts, found {len(prompts)}")
    require(len(set(prompt_ids)) == len(prompt_ids), "Duplicate prompt IDs in v3 pool")
    require("cheap_direct" in prompt_ids, "cheap_direct must remain in the shared pool")
    require(
        max(int(item.get("few_shot_count", 0) or 0) for item in prompts) == 5,
        "Shared prompt pool must include a five-shot accuracy candidate",
    )

    configs = {"FairCAPO": fair, "MO-CAPO": mo, "NSGA-II-PO": nsga}
    for name, cfg in configs.items():
        require(cfg.get("dataset") == "adult", f"{name}: dataset must be adult")
        require(cfg.get("task_type") == "classification", f"{name}: wrong task type")
        require(cfg.get("prompt_pool") == str(PROMPT_POOL), f"{name}: prompt pool mismatch")
        require(nested(cfg, "dev", "data_path") == "data/adult_semantic_v3.csv", f"{name}: data path mismatch")
        require(nested(cfg, "dev", "dev_size") == 800, f"{name}: dev size mismatch")
        require(nested(cfg, "dev", "shots_size") == 200, f"{name}: shots size mismatch")
        require(nested(cfg, "dev", "test_size") == 2000, f"{name}: test size mismatch")
        require(nested(cfg, "dev", "stratify_group_key") == "sex", f"{name}: group key mismatch")
        require(nested(cfg, "llm", "model_id") == "mistralai/mistral-small-3.2", f"{name}: model mismatch")
        require(float(nested(cfg, "llm", "temperature")) == 0.0, f"{name}: temperature must be 0")
        require(int(nested(cfg, "llm", "num_predict")) >= 96, f"{name}: num_predict too small for meta prompts")
        require(float(nested(cfg, "budget", "max_budget")) == 7_500_000.0, f"{name}: budget mismatch")
        require(nested(cfg, "budget", "unit") == "tokens", f"{name}: budget unit mismatch")
        require(int(nested(cfg, "few_shot", "max_few_shot_examples")) == 5, f"{name}: max shots mismatch")
        require(int(nested(cfg, "few_shot", "pool_size")) == 200, f"{name}: shot pool mismatch")
        require(nested(cfg, "few_shot", "selection_strategy") == "fairness_guided", f"{name}: shot strategy mismatch")

    require(nested(fair, "fairness", "in_loop") is True, "FairCAPO fairness must be enabled")
    require(nested(nsga, "fairness", "in_loop") is True, "NSGA fairness must be enabled")
    require(nested(mo, "fairness", "in_loop") is False, "MO-CAPO fairness objective must be disabled")
    require(nested(fair, "fairness", "mode") == "equalized_odds", "FairCAPO fairness mode mismatch")
    require(nested(nsga, "fairness", "mode") == "equalized_odds", "NSGA fairness mode mismatch")

    for name, cfg in (("FairCAPO", fair), ("MO-CAPO", mo)):
        require(int(nested(cfg, "evolutionary", "population_size")) == 10, f"{name}: population mismatch")
        require(int(nested(cfg, "evolutionary", "offspring_per_iteration")) == 4, f"{name}: offspring mismatch")
        require(int(nested(cfg, "intensification", "max_blocks_per_challenger")) == 10, f"{name}: final deepening is not enabled")
        require(nested(cfg, "intensification", "final_intensification") is True, f"{name}: final intensification disabled")
        require(nested(cfg, "intensification", "add_rejected_to_population") is False, f"{name}: rejected candidates could enter archive")

    require(int(nested(nsga, "nsga2_po", "population_size")) == 10, "NSGA population mismatch")
    require(int(nested(nsga, "nsga2_po", "offspring_per_generation")) == 4, "NSGA offspring mismatch")
    require(tuple(nested(nsga, "nsga2_po", "objectives")) == ("performance", "cost", "fairness_risk"), "NSGA objectives mismatch")

    require(eval_cfg.get("data_path") == "data/adult_semantic_v3.csv", "Held-out eval data mismatch")
    require(eval_cfg.get("dev_size") == 800, "Held-out dev offset mismatch")
    require(eval_cfg.get("shots_size") == 200, "Held-out shots offset mismatch")
    require(eval_cfg.get("test_size") == 2000, "Held-out test size mismatch")
    require(nested(eval_cfg, "fairness", "mode") == "equalized_odds", "Held-out fairness mismatch")
    require(nested(table_cfg, "selection", "min_performance_for_fairness") == 0.72, "Reporting threshold mismatch")
    require(len(table_cfg.get("methods", [])) == 3, "Comparison table must contain three methods")

    return fair, mo, nsga, eval_cfg


def audit_csv_and_split(data_path: Path) -> None:
    require(data_path.is_file() and data_path.stat().st_size > 0, f"Missing semantic CSV: {data_path}")

    allowed_missing = {"workclass", "occupation", "native.country"}
    missing_counts: Counter[str] = Counter()
    with data_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    require(len(rows) >= 3000, f"Semantic CSV is unexpectedly small: {len(rows)} rows")

    for row in rows:
        for column, value in row.items():
            if str(value).strip() == "?":
                missing_counts[column] += 1
    unexpected = set(missing_counts).difference(allowed_missing)
    require(not unexpected, f"Unexpected '?' values in columns: {sorted(unexpected)}")

    split = load_paper_dataset(
        "adult",
        data_path=str(data_path),
        dev_size=800,
        shots_size=200,
        test_size=2000,
        seed=0,
        allow_smaller=False,
        stratified=True,
        stratify_group_key="sex",
    )
    require(len(split.dev) == 800, "Adult dev split size mismatch")
    require(len(split.shots) == 200, "Adult shots split size mismatch")
    require(len(split.test) == 2000, "Adult test split size mismatch")

    index_sets = []
    for split_name, examples, expected_cell in (
        ("dev", split.dev, 200),
        ("shots", split.shots, 50),
        ("test", split.test, 500),
    ):
        indices = {int((example.metadata or {})["source_index"]) for example in examples}
        require(len(indices) == len(examples), f"{split_name}: duplicate source rows")
        index_sets.append(indices)

        cells = Counter(
            (example.label, str((example.metadata or {}).get("sex", "")))
            for example in examples
        )
        require(set(cells) == {("<=50K", "Female"), ("<=50K", "Male"), (">50K", "Female"), (">50K", "Male")}, f"{split_name}: missing label/sex cells: {cells}")
        require(all(count == expected_cell for count in cells.values()), f"{split_name}: unbalanced label/sex cells: {cells}")

        for example in examples:
            text = example.text
            lowered = text.lower()
            require("person record:" in lowered, f"{split_name}: malformed record")
            require("ordinal education level" in lowered, f"{split_name}: semantic education signal missing")
            require("sex:" not in lowered and "race:" not in lowered, f"{split_name}: protected attribute leakage")
            require("income:" not in lowered and "fnlwgt" not in lowered, f"{split_name}: target/weight leakage")
            require("education.num" not in lowered, f"{split_name}: raw duplicate feature leaked")
            require("?" not in text, f"{split_name}: literal '?' reached prompt text")

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
        require("Reasoning: combine the economic evidence" in rendered, "Adult reasoning shots are not active")
        require("Sex:" not in rendered and "Race:" not in rendered, "Protected attribute leaked through demonstrations")
    finally:
        if old_env is None:
            os.environ.pop("FAIRCAPO_ADULT_REASONING_SHOTS", None)
        else:
            os.environ["FAIRCAPO_ADULT_REASONING_SHOTS"] = old_env

    print(f"Semantic CSV rows: {len(rows)}")
    print(f"Literal '?' counts (converted to Unknown at render time): {dict(missing_counts)}")
    print("Split cells: dev=200, shots=50, test=500 per (income, sex) cell")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the complete Adult v3 pipeline before sbatch submission.")
    parser.add_argument("--data", default="data/adult_semantic_v3.csv")
    args = parser.parse_args()

    audit_config_alignment()
    audit_csv_and_split(Path(args.data))
    print("Adult v3 preflight PASSED.")


if __name__ == "__main__":
    main()
