import argparse
import yaml

from experiments.datasets import load_paper_dataset
from experiments.logger import save_json, save_csv_row
from baselines.initial_prompt import run_initial_prompt_baseline_with_test
from baselines.promptolution_runner import build_llm, run_promptolution_optimizer_with_test
from baselines.mocapo_style_runner import run_mocapo_style_baseline_with_test


def _attach_common_metadata(
    result: dict,
    split,
    llm_backend: str,
    dev_size: int,
    shots_size: int,
    test_size: int,
    seed: int,
) -> dict:
    """
    Add common experiment metadata to every result row.
    """
    result["dataset"] = split.name
    result["task_type"] = split.task_type
    result["llm_backend"] = llm_backend
    result["dev_size"] = dev_size
    result["shots_size"] = shots_size
    result["test_size"] = test_size
    result["seed"] = seed
    return result


def _safe_int(value, default: int) -> int:
    """
    Convert YAML values to int safely.
    Useful when optional config values are missing or null.
    """
    if value is None:
        return default
    return int(value)


def _safe_bool(value, default: bool = False) -> bool:
    """
    Convert YAML values to bool safely.
    """
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower().strip() in {"true", "1", "yes", "y"}

    return bool(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1_baseline.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_name = config.get("dataset", "toy_subjectivity")
    dev_size = int(config.get("dev_size", 300))
    shots_size = int(config.get("shots_size", 100))
    test_size = int(config.get("test_size", 500))
    seed = int(config.get("seed", 42))
    allow_smaller = bool(config.get("allow_smaller", False))
    stratified = bool(config.get("stratified", True))

    split = load_paper_dataset(
        dataset_name,
        dev_size=dev_size,
        shots_size=shots_size,
        test_size=test_size,
        seed=seed,
        allow_smaller=allow_smaller,
        stratified=stratified,
    )

    llm_config = config.get("llm", {})
    llm_backend = llm_config.get("backend", "toy")
    output_dir = config["output_dir"]

    results = []

    # ------------------------------------------------------------
    # 1. Initial prompt baseline
    # ------------------------------------------------------------
    # If backend is toy, use the rule-based evaluator.
    # If backend is ollama/api/lmstudio, use the same real LLM backend for fair comparison.
    initial_eval_llm = None
    initial_llm_backend = "rule_based_initial"

    if llm_backend != "toy":
        print(f"Running initial prompt with LLM backend: {llm_backend}")
        initial_eval_llm = build_llm(llm_config)
        initial_llm_backend = llm_backend

    initial_result = run_initial_prompt_baseline_with_test(
        dev_dataset=split.dev,
        test_dataset=split.test,
        prompt=config["initial_prompt"],
        llm=initial_eval_llm,
        classes=split.classes,
    )

    initial_result = _attach_common_metadata(
        result=initial_result,
        split=split,
        llm_backend=initial_llm_backend,
        dev_size=len(split.dev),
        shots_size=len(split.shots),
        test_size=len(split.test),
        seed=seed,
    )
    results.append(initial_result)

    # ------------------------------------------------------------
    # 2. Promptolution baselines:
    #    CAPO / EvoPromptGA / EvoPromptDE / OPRO
    # ------------------------------------------------------------
    promptolution_cfg = config.get("promptolution", {})

    if promptolution_cfg.get("enabled", False):
        initial_prompts = config.get("initial_prompts", [config["initial_prompt"]])
        n_steps = int(promptolution_cfg.get("n_steps", 1))
        optimizers = promptolution_cfg.get("optimizers", [])

        # Shared task/CAPO parameters
        task_n_subsamples = promptolution_cfg.get("task_n_subsamples")
        task_eval_strategy = promptolution_cfg.get("task_eval_strategy")
        capo_max_n_blocks_eval = promptolution_cfg.get("capo_max_n_blocks_eval")
        capo_upper_shots = _safe_int(promptolution_cfg.get("capo_upper_shots"), 0)
        capo_crossovers_per_iter = _safe_int(
            promptolution_cfg.get("capo_crossovers_per_iter"),
            1,
        )

        # EvoPromptDE parameters
        evoprompt_de_donor_random = _safe_bool(
            promptolution_cfg.get("evoprompt_de_donor_random"),
            False,
        )

        # OPRO parameters
        opro_max_num_instructions = _safe_int(
            promptolution_cfg.get("opro_max_num_instructions"),
            10,
        )
        opro_num_instructions_per_step = _safe_int(
            promptolution_cfg.get("opro_num_instructions_per_step"),
            2,
        )
        opro_num_few_shots = _safe_int(
            promptolution_cfg.get("opro_num_few_shots"),
            2,
        )

        for optimizer_name in optimizers:
            print(f"Running Promptolution optimizer: {optimizer_name}")
            print(f"LLM backend: {llm_backend}")

            result = run_promptolution_optimizer_with_test(
                dev_dataset=split.dev,
                test_dataset=split.test,
                initial_prompts=initial_prompts,
                optimizer_name=optimizer_name,
                n_steps=n_steps,
                llm_config=llm_config,
                classes=split.classes,
                task_description=split.task_description,
                task_n_subsamples=task_n_subsamples,
                task_eval_strategy=task_eval_strategy,
                capo_max_n_blocks_eval=capo_max_n_blocks_eval,
                capo_upper_shots=capo_upper_shots,
                capo_crossovers_per_iter=capo_crossovers_per_iter,
                evoprompt_de_donor_random=evoprompt_de_donor_random,
                opro_max_num_instructions=opro_max_num_instructions,
                opro_num_instructions_per_step=opro_num_instructions_per_step,
                opro_num_few_shots=opro_num_few_shots,
            )

            result = _attach_common_metadata(
                result=result,
                split=split,
                llm_backend=llm_backend,
                dev_size=len(split.dev),
                shots_size=len(split.shots),
                test_size=len(split.test),
                seed=seed,
            )
            results.append(result)

    # ------------------------------------------------------------
    # 3. MO-CAPO-style Pareto baseline
    # ------------------------------------------------------------
    mocapo_cfg = config.get("mocapo_style", {})

    if mocapo_cfg.get("enabled", False):
        print("Running MO-CAPO-style Pareto baseline")
        print(f"LLM backend: {llm_backend}")

        candidate_prompts = mocapo_cfg.get(
            "candidate_prompts",
            config.get("initial_prompts", [config["initial_prompt"]]),
        )

        if not candidate_prompts:
            raise ValueError(
                "mocapo_style.enabled=true but no candidate_prompts or initial_prompts were provided."
            )

        pareto_results, all_mocapo_results = run_mocapo_style_baseline_with_test(
            dev_dataset=split.dev,
            test_dataset=split.test,
            candidate_prompts=candidate_prompts,
            llm_config=llm_config,
            classes=split.classes,
        )

        # Save all evaluated candidates separately.
        save_json(
            all_mocapo_results,
            f"{output_dir}/mocapo_style_all_candidates.json",
        )

        # Only Pareto candidates go into the main baseline table.
        for result in pareto_results:
            result = _attach_common_metadata(
                result=result,
                split=split,
                llm_backend=llm_backend,
                dev_size=len(split.dev),
                shots_size=len(split.shots),
                test_size=len(split.test),
                seed=seed,
            )
            results.append(result)

    # ------------------------------------------------------------
    # 4. Save outputs
    # ------------------------------------------------------------
    for result in results:
        method = result["method"]
        save_json(result, f"{output_dir}/{method}_result.json")
        save_csv_row(result, f"{output_dir}/baseline_table.csv")

    # ------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------
    print("Phase 1 baselines complete")
    print(f"Dataset: {split.name}")
    print(f"Task type: {split.task_type}")
    print(f"Dev size: {len(split.dev)}")
    print(f"Shots size: {len(split.shots)}")
    print(f"Test size: {len(split.test)}")
    print(f"LLM backend: {llm_backend}")
    print(f"Saved to: {output_dir}")

    for result in results:
        print(
            f"{result['method']}: "
            f"dev_score={result['dev_score']}, "
            f"test_score={result['test_score']}, "
            f"dev_cost={result['dev_cost']}, "
            f"test_cost={result['test_cost']}"
        )


if __name__ == "__main__":
    main()