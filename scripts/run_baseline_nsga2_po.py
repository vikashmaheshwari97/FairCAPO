from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from baselines.nsga2_po_runner import (
    NSGA2PORunner,
    NSGA2PORunnerConfig,
    make_candidate_from_prompt,
)
from heal_capo.components.router import FairnessAwareRouter, RiskAwareRouter
from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
from heal_capo.objectives import ObjectiveEvaluator, ToyObjectiveEvaluator
from heal_capo.pareto import non_dominated_ids


def _json_default(obj: Any):
    try:
        return obj.item()
    except AttributeError:
        return str(obj)


def load_yaml(path: str) -> dict:
    yaml_path = Path(path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(data: Any, path: str):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


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


def simple_token_count(text: str) -> int:
    return len(str(text or "").split())


def extract_label(response: str, labels: list[str]) -> str:
    text = str(response or "").strip()
    lowered = text.lower()

    if "<final_answer>" in lowered and "</final_answer>" in lowered:
        start = lowered.find("<final_answer>") + len("<final_answer>")
        end = lowered.find("</final_answer>", start)
        text = text[start:end].strip()
        lowered = text.lower()

    cleaned = lowered.strip(" .,:;!?\"'`")

    for label in labels:
        if cleaned == str(label).lower():
            return str(label)

    for label in labels:
        if str(label).lower() in lowered:
            return str(label)

    return cleaned


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


def get_llm_response(llm, prompt: str) -> str:
    if hasattr(llm, "get_response"):
        response = llm.get_response(prompt)
    elif hasattr(llm, "generate"):
        response = llm.generate(prompt)
    elif callable(llm):
        response = llm(prompt)
    else:
        raise AttributeError(
            "LLM object must expose get_response(), generate(), or be callable."
        )

    if isinstance(response, list):
        return str(response[0])

    return str(response)


def get_example_text(example: dict) -> str:
    return str(
        example.get("text")
        or example.get("input")
        or example.get("sentence")
        or example.get("x")
        or ""
    )


def get_example_label(example: dict) -> str:
    return str(
        example.get("label")
        or example.get("label_text")
        or example.get("answer")
        or example.get("output")
        or example.get("target")
        or example.get("y")
        or ""
    )


def make_llm_prompt(
    candidate: PromptCandidate,
    text: str,
    labels: list[str],
    require_final_answer_tags: bool = False,
) -> str:
    rendered = candidate.render(text)
    label_text = ", ".join(labels)

    if require_final_answer_tags:
        return (
            f"{rendered}\n\n"
            f"Allowed labels: {label_text}\n"
            f"Return only the answer inside <final_answer> and </final_answer> tags."
        )

    return (
        f"{rendered}\n\n"
        f"Allowed labels: {label_text}\n"
        f"Return only one label: {label_text}."
    )


def heuristic_fairness_risk(prompt: str) -> float:
    lowered = str(prompt).lower()

    fairness_terms = [
        "gender",
        "race",
        "ethnicity",
        "nationality",
        "religion",
        "age",
        "location",
        "demographic",
        "stereotype",
        "names",
    ]

    if any(term in lowered for term in fairness_terms):
        return 0.12

    if "do not infer" in lowered or "avoid assumptions" in lowered:
        return 0.18

    return 0.30


def heuristic_risk_adjustment(prompt: str) -> float:
    lowered = str(prompt).lower()

    risk_terms = [
        "do not hallucinate",
        "use only",
        "provided context",
        "given sentence",
        "insufficient",
        "cannot be determined",
        "unsupported",
        "missing context",
    ]

    if any(term in lowered for term in risk_terms):
        return -0.10

    return 0.0


class LLMObjectiveEvaluator(ObjectiveEvaluator):
    """
    LM Studio-backed full-dev evaluator for NSGA-II-PO.

    NSGA-II-PO evaluates every candidate/offspring on the full dev set.
    No block intensification is used here.
    """

    def __init__(self, config: dict):
        self.config = config
        self.labels = config.get("labels", ["subjective", "objective"])
        self.evaluation_cfg = config.get("evaluation", {})
        self.cost_cfg = config.get("cost", {})
        self.llm = build_llm_from_config(config)

        self.input_weight = float(self.cost_cfg.get("input_weight", 0.08))
        self.output_weight = float(self.cost_cfg.get("output_weight", 0.32))
        self.require_final_answer_tags = bool(
            self.evaluation_cfg.get("require_final_answer_tags", False)
        )

    def evaluate(self, candidate: PromptCandidate, data) -> EvaluationResult:
        correct = 0
        total = 0
        input_tokens = 0
        output_tokens = 0
        rows = []

        for example in data:
            text = get_example_text(example)
            gold = get_example_label(example).strip().lower()

            prompt = make_llm_prompt(
                candidate=candidate,
                text=text,
                labels=self.labels,
                require_final_answer_tags=self.require_final_answer_tags,
            )

            raw_response = get_llm_response(self.llm, prompt)
            pred = extract_label(raw_response, self.labels).strip().lower()

            is_correct = pred == gold

            correct += int(is_correct)
            total += 1

            prompt_tokens = simple_token_count(prompt)
            response_tokens = simple_token_count(raw_response)

            input_tokens += prompt_tokens
            output_tokens += response_tokens

            rows.append(
                {
                    "text": text,
                    "gold": gold,
                    "prediction": pred,
                    "raw_response": raw_response,
                    "correct": is_correct,
                    "input_tokens": prompt_tokens,
                    "output_tokens": response_tokens,
                }
            )

        performance = correct / total if total else 0.0

        cost = (
            self.input_weight * input_tokens
            + self.output_weight * output_tokens
        )

        base_risk = 1.0 - performance
        risk = min(
            1.0,
            max(
                0.0,
                base_risk + heuristic_risk_adjustment(candidate.instruction),
            ),
        )

        fairness_risk = heuristic_fairness_risk(candidate.instruction)

        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            performance=performance,
            cost=cost,
            risk=risk,
            fairness_risk=fairness_risk,
            drift=0.0,
            n_examples=total,
            details={
                "evaluator": "lmstudio",
                "model_id": self.config.get("llm", {}).get("model_id"),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "correct": correct,
                "total": total,
                "predictions": rows,
            },
        )


def build_objective_evaluator(
    config: dict,
    force_no_llm: bool = False,
) -> ObjectiveEvaluator:
    evaluation_cfg = config.get("evaluation", {})
    use_llm = bool(evaluation_cfg.get("use_llm", False))

    if force_no_llm:
        use_llm = False

    if use_llm:
        dataset = str(config.get("dataset", "")).strip().lower()
        task_type = str(config.get("task_type", "")).strip().lower()
        if dataset == "bbq" or task_type == "multiple_choice":
            # Share FairCAPO's evaluator so NSGA-II-PO optimizes the SAME real
            # objectives (multiple-choice accuracy + canonical BBQ bias score),
            # differing from FairCAPO only in the search algorithm. Without this,
            # NSGA-II-PO would use a prompt-keyword fairness heuristic and would
            # not be comparable.
            from scripts.run_phase2_budgeted_mocapo import (
                LLMObjectiveEvaluator as BudgetedLLMEvaluator,
            )

            return BudgetedLLMEvaluator(config)
        return LLMObjectiveEvaluator(config)

    return ToyObjectiveEvaluator()


def get_prompt_pool(config: dict) -> list[dict]:
    prompt_pool_path = config.get("prompt_pool")

    if prompt_pool_path:
        prompt_pool_config = load_yaml(prompt_pool_path)
        prompt_pool = prompt_pool_config.get("prompt_pool", [])
    else:
        prompt_pool = config.get("prompt_pool_inline", [])

    if not prompt_pool:
        raise ValueError(
            "Config must contain prompt_pool path or prompt_pool_inline list."
        )

    normalized = []

    for idx, item in enumerate(prompt_pool):
        if isinstance(item, str):
            normalized.append(
                {
                    "id": f"prompt_{idx}",
                    "category": "unknown",
                    "prompt": item,
                }
            )
            continue

        prompt = str(item.get("prompt", "")).strip()

        if not prompt:
            continue

        normalized.append(
            {
                "id": str(item.get("id", f"prompt_{idx}")),
                "category": str(item.get("category", "unknown")),
                "prompt": prompt,
            }
        )

    if not normalized:
        raise ValueError("No valid prompts found.")

    return normalized


def get_dev_data(config: dict) -> list[dict]:
    # BBQ loads a real dataset-backed dev split (via the budgeted runner's loader),
    # so NSGA-II-PO sees the exact same Ddev as FairCAPO. Other datasets keep the
    # existing inline/default behavior.
    if str(config.get("dataset", "")).strip().lower() == "bbq":
        from scripts.run_phase2_budgeted_mocapo import get_dev_data as budgeted_get_dev_data

        return budgeted_get_dev_data(config)

    return config.get(
        "dev_data",
        [
            {"text": "The movie was released in 1999.", "label": "objective"},
            {"text": "The acting is absolutely wonderful.", "label": "subjective"},
            {"text": "Paris is the capital of France.", "label": "objective"},
            {
                "text": "This boring film wastes its talented cast.",
                "label": "subjective",
            },
            {"text": "The book contains twelve chapters.", "label": "objective"},
            {
                "text": "The speech was moving and unforgettable.",
                "label": "subjective",
            },
            {"text": "The meeting started at 9 a.m.", "label": "objective"},
            {
                "text": "The design looks beautiful and modern.",
                "label": "subjective",
            },
        ],
    )


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


def make_initial_population(config: dict) -> list[PromptCandidate]:
    dataset = config.get("dataset", "subj")
    task_type = config.get("task_type", "classification")

    population = []

    for item in get_prompt_pool(config):
        population.append(
            make_candidate_from_prompt(
                prompt=item["prompt"],
                candidate_id=item["id"],
                category=item["category"],
                dataset=dataset,
                task_type=task_type,
            )
        )

    return population


def make_meta_llm(config: dict, force_no_llm: bool = False):
    nsga_cfg = config.get("nsga2_po", {})
    use_meta_llm = bool(nsga_cfg.get("use_meta_llm", False))

    if force_no_llm or not use_meta_llm:
        return None

    return build_llm_from_config(config)


def nsga_config_from_yaml(config: dict) -> NSGA2PORunnerConfig:
    nsga_cfg = config.get("nsga2_po", {})

    # Optional shared evaluation budget (same block the budgeted MO-CAPO runner
    # reads). When present, NSGA-II-PO is compared under an identical token cap.
    budget_cfg = config.get("budget", {})
    raw_max_budget = budget_cfg.get("max_budget", nsga_cfg.get("max_budget"))
    max_budget = float(raw_max_budget) if raw_max_budget is not None else None
    budget_unit = str(budget_cfg.get("unit", budget_cfg.get("budget_unit", "tokens")))
    allow_overspend = bool(budget_cfg.get("allow_overspend", False))

    return NSGA2PORunnerConfig(
        population_size=int(nsga_cfg.get("population_size", 10)),
        max_generations=int(nsga_cfg.get("max_generations", 5)),
        offspring_per_generation=int(
            nsga_cfg.get("offspring_per_generation", 4)
        ),
        random_seed=int(config.get("seed", nsga_cfg.get("random_seed", 0))),
        mutation_probability=float(nsga_cfg.get("mutation_probability", 1.0)),
        crossover_probability=float(nsga_cfg.get("crossover_probability", 1.0)),
        mutate_after_crossover=bool(nsga_cfg.get("mutate_after_crossover", True)),
        use_meta_llm=bool(nsga_cfg.get("use_meta_llm", False)),
        objectives=tuple(
            nsga_cfg.get(
                "objectives",
                ["performance", "cost", "risk", "fairness_risk"],
            )
        ),
        max_budget=max_budget,
        allow_overspend=allow_overspend,
        budget_unit=budget_unit,
        metadata={
            "dataset": config.get("dataset", ""),
            "task_type": config.get("task_type", ""),
            "model_id": config.get("llm", {}).get("model_id"),
            "backend": config.get("llm", {}).get("backend"),
        },
    )


def portfolio_to_rows(
    portfolio: PromptPortfolio,
    pareto_ids: set[str] | None = None,
) -> list[dict]:
    pareto_ids = pareto_ids or set()
    rows = []

    for candidate in portfolio.evaluated_candidates():
        result = portfolio.get_result(candidate.candidate_id)

        row = {
            "candidate_id": candidate.candidate_id,
            "is_pareto": candidate.candidate_id in pareto_ids,
            "method": candidate.metadata.get("method")
            or candidate.metadata.get("prompt_pool_id"),
            "category": candidate.metadata.get("category"),
            "dataset": candidate.metadata.get("dataset"),
            "task_type": candidate.metadata.get("task_type"),
            "performance": result.performance,
            "cost": result.cost,
            "risk": result.risk,
            "fairness_risk": result.fairness_risk,
            "drift": result.drift,
            "n_examples": result.n_examples,
            "objective_vector": str(result.objective_vector),
            "prompt": candidate.instruction,
            "source": candidate.metadata.get("source"),
            "operator": candidate.metadata.get("operator"),
            "parent_ids": candidate.metadata.get("parent_ids"),
            "used_meta_llm": candidate.metadata.get("used_meta_llm"),
            "generation": candidate.metadata.get("generation"),
            "offspring_index": candidate.metadata.get("offspring_index"),
        }

        for key, value in result.details.items():
            if key == "predictions":
                row[f"detail_{key}"] = json.dumps(value, default=_json_default)
            elif isinstance(value, (dict, list, tuple)):
                row[f"detail_{key}"] = json.dumps(value, default=_json_default)
            else:
                row[f"detail_{key}"] = value

        rows.append(row)

    return rows


def routing_preferences_from_config(config: dict) -> list[dict]:
    return config.get(
        "routing_preferences",
        [
            {"name": "accuracy_first", "mode": "accuracy_first"},
            {"name": "cost_first", "mode": "cost_first"},
            {"name": "risk_first", "mode": "risk_first"},
            {"name": "fairness_first", "mode": "fairness_first"},
            {
                "name": "balanced",
                "mode": "balanced",
                "performance": 1.0,
                "cost": 0.3,
                "risk": 1.0,
                "fairness": 1.0,
            },
        ],
    )


def routing_to_rows(
    portfolio: PromptPortfolio,
    preferences: list[dict],
) -> list[dict]:
    rows = []

    default_router = RiskAwareRouter()
    fairness_router = FairnessAwareRouter()

    for preference in preferences:
        name = preference.get("name", preference.get("mode", "balanced"))

        if preference.get("mode") == "fairness_first":
            router = fairness_router
        else:
            router = default_router

        decision = router.select(
            x="baseline_nsga2_po",
            portfolio=portfolio,
            preference=preference,
        )

        row = decision.to_row()
        row["preference_name"] = name
        rows.append(row)

    return rows


def print_events(events: list[dict]):
    print("NSGA-II-PO events")
    print("-" * 80)

    for row in events:
        print(
            f"{row.get('event_type')} | "
            f"gen={row.get('generation', '')} | "
            f"candidate={row.get('candidate_id', '')} | "
            f"operator={row.get('operator', '')}"
        )

    print("-" * 80)


def print_portfolio(title: str, rows: list[dict]):
    print(title)
    print("-" * 80)

    for row in rows:
        pareto_mark = "Pareto" if row.get("is_pareto") else "Non-Pareto"

        print(
            f"{pareto_mark} | "
            f"{row.get('method')} | "
            f"category={row.get('category')} | "
            f"performance={row.get('performance')} | "
            f"cost={row.get('cost')} | "
            f"risk={row.get('risk')} | "
            f"fairness_risk={row.get('fairness_risk')} | "
            f"prompt={row.get('prompt')}"
        )

    print("-" * 80)
    print(f"Total rows: {len(rows)}")


def print_recommendations(rows: list[dict]):
    print("NSGA-II-PO recommendations")
    print("-" * 80)

    for row in rows:
        print(
            f"{row['preference_name']} -> "
            f"{row['instruction']} | "
            f"utility={row['utility']} | "
            f"reason={row['reason']}"
        )

    print("-" * 80)


def run_baseline_nsga2_po(
    config: dict,
    force_no_llm: bool = False,
):
    evaluator = build_objective_evaluator(
        config=config,
        force_no_llm=force_no_llm,
    )

    initial_population = make_initial_population(config)
    dev_data = get_dev_data(config)
    task_description = get_task_description(config)
    nsga_config = nsga_config_from_yaml(config)

    meta_llm = make_meta_llm(
        config=config,
        force_no_llm=force_no_llm,
    )

    runner = NSGA2PORunner(
        config=nsga_config,
        evaluator=evaluator,
        dev_data=dev_data,
        task_description=task_description,
        meta_llm=meta_llm,
    )

    return runner.run(initial_population=initial_population)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/baselines/nsga2_po_subj_mistral.yaml",
        help="NSGA-II-PO baseline YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Force ToyObjectiveEvaluator even if evaluation.use_llm is true.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config['seed'] for multi-seed sweeps (parity with the "
        "budgeted MO-CAPO runner). Seeds data sampling + the NSGA-II RNG.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.seed is not None:
        config["seed"] = args.seed

    output_dir = args.output_dir or config.get(
        "output_dir",
        "outputs/baselines/nsga2_po_subj_mistral",
    )

    result = run_baseline_nsga2_po(
        config=config,
        force_no_llm=args.no_llm,
    )

    pareto_ids = set(non_dominated_ids(result.all_portfolio.evaluations.values()))

    all_rows = portfolio_to_rows(
        portfolio=result.all_portfolio,
        pareto_ids=pareto_ids,
    )
    pareto_rows = portfolio_to_rows(
        portfolio=result.pareto_portfolio,
        pareto_ids=pareto_ids,
    )

    recommendations = routing_to_rows(
        portfolio=result.pareto_portfolio,
        preferences=routing_preferences_from_config(config),
    )

    save_csv(all_rows, f"{output_dir}/nsga2_po_all_candidates.csv")
    save_json(all_rows, f"{output_dir}/nsga2_po_all_candidates.json")

    save_csv(pareto_rows, f"{output_dir}/nsga2_po_pareto_portfolio.csv")
    save_json(pareto_rows, f"{output_dir}/nsga2_po_pareto_portfolio.json")

    save_csv(result.events, f"{output_dir}/nsga2_po_events.csv")
    save_json(result.events, f"{output_dir}/nsga2_po_events.json")

    save_csv(recommendations, f"{output_dir}/nsga2_po_recommendations.csv")
    save_json(recommendations, f"{output_dir}/nsga2_po_recommendations.json")

    save_json(result.summary, f"{output_dir}/nsga2_po_summary.json")

    # Written only when a budget is configured, so the experiment table can read
    # `budget_used` for NSGA-II-PO on the same basis as FairCAPO/ablation.
    if result.budget_summary is not None:
        save_json(result.budget_summary, f"{output_dir}/budget_summary.json")

    print_events(result.events)
    print_portfolio("NSGA-II-PO Pareto portfolio", pareto_rows)
    print_recommendations(recommendations)

    print("NSGA-II-PO summary")
    print("-" * 80)
    print(json.dumps(result.summary, indent=2, default=_json_default))
    print("-" * 80)

    print(f"Saved all candidates to: {output_dir}/nsga2_po_all_candidates.csv")
    print(f"Saved Pareto portfolio to: {output_dir}/nsga2_po_pareto_portfolio.csv")
    print(f"Saved events to: {output_dir}/nsga2_po_events.csv")
    print(f"Saved recommendations to: {output_dir}/nsga2_po_recommendations.csv")
    print(f"Saved summary to: {output_dir}/nsga2_po_summary.json")
    if result.budget_summary is not None:
        print(f"Saved budget summary to: {output_dir}/budget_summary.json")


if __name__ == "__main__":
    main()