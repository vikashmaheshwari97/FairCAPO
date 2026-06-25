from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from baselines.gepa_runner import (
    GEPARunner,
    GEPARunnerConfig,
    evaluate_gepa_candidate,
    gepa_result_to_row,
)


def load_yaml(path: str) -> dict:
    yaml_path = Path(path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_csv(rows: list[dict], path: str):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"No rows to save for {path}")

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(data: Any, path: str):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def build_llm_from_config(config: dict):
    llm_config = config.get("llm", {})

    try:
        from baselines.promptolution_runner import build_llm

        return build_llm(llm_config)
    except Exception as exc:
        raise RuntimeError(
            "Could not build LLM from baselines.promptolution_runner.build_llm. "
            "Check LM Studio is running and llm config is valid."
        ) from exc


class NoLLM:
    """
    Deterministic fallback model for smoke tests.

    It uses simple lexical cues for SUBJ-style classification and returns
    marker-based output so extraction logic can be tested.
    """

    SUBJECTIVE_CUES = {
        "wonderful",
        "boring",
        "beautiful",
        "moving",
        "unforgettable",
        "bad",
        "good",
        "great",
        "terrible",
        "best",
        "worst",
        "love",
        "hate",
        "amazing",
        "awful",
        "excellent",
        "dull",
        "predictable",
        "touching",
    }

    def get_response(self, prompt: str) -> str:
        lowered = str(prompt).lower()

        if "<prompt>" in lowered or "revise the prompt" in lowered:
            return (
                "<prompt>Classify the sentence as subjective or objective. "
                "A sentence is subjective if it expresses opinion, emotion, judgment, "
                "or evaluation. A sentence is objective if it states verifiable factual "
                "information. Return the label inside <final_answer> and </final_answer> tags.</prompt>"
            )

        if any(cue in lowered for cue in self.SUBJECTIVE_CUES):
            return "<final_answer>subjective</final_answer>"

        return "<final_answer>objective</final_answer>"


def get_task_description(config: dict) -> str:
    return str(
        config.get(
            "task_description",
            (
                "The dataset contains sentences labeled as either subjective or objective. "
                "The task is to classify each sentence as either subjective or objective. "
                "The class will be extracted between the markers "
                "<final_answer>answer</final_answer>."
            ),
        )
    )


def get_default_subj_data() -> list[dict]:
    return [
        {"text": "The movie was released in 1999.", "label": "objective"},
        {"text": "The acting is absolutely wonderful.", "label": "subjective"},
        {"text": "Paris is the capital of France.", "label": "objective"},
        {"text": "This boring film wastes its talented cast.", "label": "subjective"},
        {"text": "The book contains twelve chapters.", "label": "objective"},
        {"text": "The speech was moving and unforgettable.", "label": "subjective"},
        {"text": "The meeting started at 9 a.m.", "label": "objective"},
        {"text": "The design looks beautiful and modern.", "label": "subjective"},
        {"text": "The film is 105 minutes long.", "label": "objective"},
        {"text": "The story feels dull and predictable.", "label": "subjective"},
        {"text": "The article was published on Monday.", "label": "objective"},
        {"text": "The ending is touching and beautifully acted.", "label": "subjective"},
    ]

def normalize_gepa_examples(data: list[dict]) -> list[dict]:
    """
    Normalize examples for GEPA default adapter.

    GEPA's default adapter expects at least:
      - input
      - answer

    Our internal format often uses:
      - text
      - label / label_text

    We keep all fields for compatibility.
    """
    normalized = []

    for example in data:
        item = dict(example)

        if "input" not in item:
            item["input"] = (
                item.get("text")
                or item.get("sentence")
                or item.get("x")
                or ""
            )

        if "text" not in item:
            item["text"] = item["input"]

        if "label" not in item and "label_text" in item:
            item["label"] = item["label_text"]

        if "label_text" not in item and "label" in item:
            item["label_text"] = item["label"]

        if "answer" not in item:
            item["answer"] = (
                item.get("label")
                or item.get("label_text")
                or item.get("target")
                or item.get("output")
                or item.get("y")
                or ""
            )

        normalized.append(item)

    return normalized

def split_data(config: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Lightweight data loader for now.

    If config contains train_data/dev_data/test_data, use those.
    Otherwise use a small built-in SUBJ-style dataset.

    Later we can replace this with the HuggingFace Ddev/Dshots/Dtest loader.
    """
    if "train_data" in config or "dev_data" in config or "test_data" in config:
        train_data = list(config.get("train_data", []))
        dev_data = list(config.get("dev_data", []))
        test_data = list(config.get("test_data", []))

        if not train_data:
            train_data = dev_data
        if not dev_data:
            dev_data = train_data
        if not test_data:
            test_data = dev_data

        return (
            normalize_gepa_examples(train_data),
            normalize_gepa_examples(dev_data),
            normalize_gepa_examples(test_data),
        )

    data = get_default_subj_data()

    train_data = data[:4]
    dev_data = data[4:8]
    test_data = data[8:]

    return (
        normalize_gepa_examples(train_data),
        normalize_gepa_examples(dev_data),
        normalize_gepa_examples(test_data),
    )


def get_initial_instruction(config: dict) -> str:
    if "initial_instruction" in config:
        return str(config["initial_instruction"])

    initial_instructions = config.get("initial_instructions", [])

    if initial_instructions:
        first = initial_instructions[0]

        if isinstance(first, dict):
            return str(first.get("prompt", first.get("instruction", "")))

        return str(first)

    return (
        "Classify the sentence as either subjective or objective. "
        "Return the class between <final_answer> and </final_answer> tags."
    )


def build_runner_config(config: dict) -> GEPARunnerConfig:
    labels = list(config.get("labels", ["subjective", "objective"]))
    cost_cfg = config.get("cost", {})
    gepa_cfg = config.get("gepa", {})
    evaluation_cfg = config.get("evaluation", {})

    return GEPARunnerConfig(
        task_description=get_task_description(config),
        labels=labels,
        initial_instruction=get_initial_instruction(config),
        max_iterations=int(gepa_cfg.get("max_iterations", 20)),
        budget_tokens=int(gepa_cfg.get("budget_tokens", 50_000)),
        require_final_answer_tags=bool(
            evaluation_cfg.get("require_final_answer_tags", True)
        ),
        input_weight=float(cost_cfg.get("input_weight", 0.08)),
        output_weight=float(cost_cfg.get("output_weight", 0.32)),
        use_native_gepa=bool(gepa_cfg.get("use_native_gepa", True)),
        allow_fallback=bool(gepa_cfg.get("allow_fallback", True)),
        metadata={
            "dataset": config.get("dataset", "subj"),
            "task_type": config.get("task_type", "classification"),
            "model_id": config.get("llm", {}).get("model_id", ""),
            "seed": int(config.get("seed", 0)),
        },
    )


def parse_metadata(row: dict) -> dict:
    raw = row.get("detail_gepa_metadata", "{}")

    if isinstance(raw, dict):
        return raw

    try:
        return json.loads(raw)
    except Exception:
        return {}


def make_recommendation_row(row: dict) -> dict:
    return {
        "preference_name": "accuracy_first",
        "mode": "accuracy_first",
        "candidate_id": row["candidate_id"],
        "instruction": row["prompt"],
        "utility": row["performance"],
        "reason": (
            "Selected GEPA baseline prompt because it is the single optimized "
            "candidate produced by the GEPA runner."
        ),
    }


def make_summary(row: dict, config: dict) -> dict:
    metadata = parse_metadata(row)

    return {
        "dataset": config.get("dataset", "subj"),
        "task_type": config.get("task_type", "classification"),
        "model_id": config.get("llm", {}).get("model_id", ""),
        "method": "gepa",
        "performance": row["performance"],
        "cost": row["cost"],
        "risk": row["risk"],
        "fairness_risk": row["fairness_risk"],
        "n_examples": row["n_examples"],
        "used_native_gepa": row["used_native_gepa"],
        "num_iterations": row["num_iterations"],
        "input_tokens": row.get("detail_input_tokens", 0),
        "output_tokens": row.get("detail_output_tokens", 0),
        "correct": row.get("detail_correct", 0),
        "total": row.get("detail_total", 0),
        "native_gepa_attempted": metadata.get("native_gepa_attempted", False),
        "native_gepa_success": metadata.get("native_gepa_success", False),
        "native_gepa_error": metadata.get("native_gepa_error", ""),
        "native_gepa_result_type": metadata.get("native_gepa_result_type", ""),
        "gepa_total_metric_calls": metadata.get("gepa_total_metric_calls", 0),
        "gepa_num_candidates": metadata.get("gepa_num_candidates", 0),
        "gepa_num_full_val_evals": metadata.get("gepa_num_full_val_evals", 0),
        "gepa_num_val_instances": metadata.get("gepa_num_val_instances", 0),
        "fallback": metadata.get("fallback", ""),
    }


def print_result(row: dict, summary: dict):
    print("GEPA baseline result")
    print("-" * 80)
    print(
        f"method=gepa | "
        f"performance={row['performance']} | "
        f"cost={row['cost']} | "
        f"risk={row['risk']} | "
        f"fairness_risk={row['fairness_risk']} | "
        f"used_native_gepa={row['used_native_gepa']} | "
        f"iterations={row['num_iterations']}"
    )

    if summary.get("native_gepa_attempted"):
        print(
            f"native_gepa_success={summary.get('native_gepa_success')} | "
            f"metric_calls={summary.get('gepa_total_metric_calls')} | "
            f"num_candidates={summary.get('gepa_num_candidates')}"
        )

    if summary.get("native_gepa_error"):
        print(f"native_gepa_error={summary.get('native_gepa_error')}")

    if summary.get("fallback"):
        print(f"fallback={summary.get('fallback')}")

    print("-" * 80)
    print("Optimized prompt")
    print("-" * 80)
    print(row["prompt"])
    print("-" * 80)
    print("Summary")
    print(json.dumps(summary, indent=2, default=str))


def run_gepa_baseline(
    config: dict,
    force_no_llm: bool = False,
) -> tuple[dict, dict, dict]:
    train_data, dev_data, test_data = split_data(config)

    if force_no_llm:
        llm = NoLLM()
    else:
        llm = build_llm_from_config(config)

    runner_config = build_runner_config(config)

    runner = GEPARunner(
        config=runner_config,
        llm=llm,
    )

    run_result = runner.run(
        train_data=train_data,
        dev_data=dev_data,
    )

    evaluation = evaluate_gepa_candidate(
        candidate=run_result.candidate,
        test_data=test_data,
        llm=llm,
        labels=runner_config.labels,
        input_weight=runner_config.input_weight,
        output_weight=runner_config.output_weight,
        require_final_answer_tags=runner_config.require_final_answer_tags,
    )

    row = gepa_result_to_row(
        run_result=run_result,
        evaluation=evaluation,
    )

    if force_no_llm:
        row["used_native_gepa"] = False
        row["detail_evaluator"] = "no_llm"
    else:
        row["detail_evaluator"] = "lmstudio"

    recommendation = make_recommendation_row(row)
    summary = make_summary(row, config)

    return row, recommendation, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/baselines/gepa_subj_mistral.yaml",
        help="GEPA baseline YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use deterministic fallback model for smoke tests.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)

    output_dir = args.output_dir or config.get(
        "output_dir",
        "outputs/baselines/gepa_subj_mistral",
    )

    row, recommendation, summary = run_gepa_baseline(
        config=config,
        force_no_llm=args.no_llm,
    )

    save_csv([row], f"{output_dir}/gepa_candidate.csv")
    save_json(row, f"{output_dir}/gepa_candidate.json")

    save_csv([recommendation], f"{output_dir}/gepa_recommendation.csv")
    save_json(recommendation, f"{output_dir}/gepa_recommendation.json")

    save_json(summary, f"{output_dir}/gepa_summary.json")

    print_result(row, summary)

    print("-" * 80)
    print(f"Saved GEPA candidate to: {output_dir}/gepa_candidate.csv")
    print(f"Saved GEPA recommendation to: {output_dir}/gepa_recommendation.csv")
    print(f"Saved GEPA summary to: {output_dir}/gepa_summary.json")


if __name__ == "__main__":
    main()