from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

import yaml

from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
from heal_capo.fairness import (
    CombinedFairnessConfig,
    FairnessDebtTracker,
    evaluate_combined_fairness,
)
from heal_capo.objectives import ToyObjectiveEvaluator
from heal_capo.pareto import non_dominated_ids, pareto_archive, sort_pareto_results
from heal_capo.components.router import RiskAwareRouter, FairnessAwareRouter
from heal_capo.components.drift_guard import FairnessConstraintDriftGuard
from heal_capo.components.prompt_generator import (
    FairnessAwarePromptGenerator,
    FairnessGenerationConfig,
)


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
        return yaml.safe_load(f)


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
        json.dump(data, f, indent=2, default=_json_default)


def simple_token_count(text: str) -> int:
    if text is None:
        return 0

    return len(str(text).split())


def estimate_pair_eval_cost(
    prompts: list[str],
    outputs: list[str],
    input_weight: float = 0.08,
    output_weight: float = 0.32,
) -> dict:
    input_tokens = sum(simple_token_count(prompt) for prompt in prompts)
    output_tokens = sum(simple_token_count(output) for output in outputs)

    cost = input_weight * input_tokens + output_weight * output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
    }


def get_prompt_pool(config: dict) -> list[dict]:
    prompt_pool_path = config.get("prompt_pool")

    if not prompt_pool_path:
        raise ValueError("Config must contain prompt_pool path.")

    prompt_pool_config = load_yaml(prompt_pool_path)
    prompt_pool = prompt_pool_config.get("prompt_pool", [])

    if not prompt_pool:
        raise ValueError("Prompt pool config must contain prompt_pool list.")

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
        raise ValueError("No valid prompts found in prompt pool.")

    return normalized


def get_fairness_pairs(config: dict) -> list[dict]:
    fairness_data_path = config.get("fairness_data")

    if not fairness_data_path:
        raise ValueError("Config must contain fairness_data path.")

    fairness_config = load_yaml(fairness_data_path)
    pairs = fairness_config.get("pairs", [])

    if not pairs:
        raise ValueError("Fairness data must contain a non-empty pairs list.")

    return pairs


def build_fairness_config(config: dict) -> CombinedFairnessConfig:
    """Build a CombinedFairnessConfig from the optional ``fairness:`` block."""
    fairness_cfg = config.get("fairness", {}) or {}

    return CombinedFairnessConfig(
        flip_weight=float(fairness_cfg.get("flip_weight", 0.50)),
        group_gap_weight=float(fairness_cfg.get("group_gap_weight", 0.25)),
        bias_weight=float(fairness_cfg.get("bias_weight", 0.15)),
        debt_weight=float(fairness_cfg.get("debt_weight", 0.10)),
        clamp=bool(fairness_cfg.get("clamp", True)),
    )


def build_eval_prompt(
    instruction: str,
    text: str,
    labels: list[str],
    require_final_answer_tags: bool = False,
) -> str:
    label_text = ", ".join(labels)

    if require_final_answer_tags:
        return (
            f"{instruction}\n\n"
            f"Allowed labels: {label_text}\n"
            f"Input: {text}\n\n"
            f"Return only the answer inside <final_answer> and </final_answer> tags."
        )

    return (
        f"{instruction}\n\n"
        f"Allowed labels: {label_text}\n"
        f"Input: {text}\n\n"
        f"Return only one label: {label_text}."
    )


def extract_label(response: str, labels: list[str]) -> str:
    text = str(response).strip()
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


class RuleBasedSubjModel:
    """
    Lightweight fallback for debugging without LM Studio.

    This is not the final evaluation model.
    """

    SUBJECTIVE_WORDS = {
        "boring",
        "predictable",
        "best",
        "wonderful",
        "beautiful",
        "excellent",
        "touching",
        "amazing",
        "great",
        "felt",
        "believes",
        "thought",
        "said",
    }

    def predict(self, text: str) -> str:
        lowered = str(text).lower()

        if any(word in lowered for word in self.SUBJECTIVE_WORDS):
            return "subjective"

        return "objective"


def build_llm_from_config(config: dict):
    """
    Uses existing project LM Studio builder if available.

    Expected available function from earlier Phase 1:
      baselines.promptolution_runner.build_llm
    """
    llm_config = config.get("llm", {})

    try:
        from baselines.promptolution_runner import build_llm

        return build_llm(llm_config)
    except Exception as exc:
        raise RuntimeError(
            "Could not build LLM from baselines.promptolution_runner.build_llm. "
            "Set evaluation.use_llm=false for rule-based debugging, or check your "
            "LM Studio/backend setup."
        ) from exc


def get_llm_response(llm, prompt: str) -> str:
    response = llm.get_response(prompt)

    if isinstance(response, list):
        return str(response[0])

    return str(response)


def evaluate_prompt_on_counterfactual_pairs(
    prompt_item: dict,
    pairs: list[dict],
    labels: list[str],
    config: dict,
    fairness_config: Optional[CombinedFairnessConfig] = None,
    debt_tracker: Optional[FairnessDebtTracker] = None,
) -> tuple[list[dict], dict]:
    evaluation_cfg = config.get("evaluation", {})
    cost_cfg = config.get("cost", {})
    fairness_cfg = config.get("fairness", {}) or {}

    use_llm = bool(evaluation_cfg.get("use_llm", True))
    require_final_answer_tags = bool(
        evaluation_cfg.get("require_final_answer_tags", False)
    )
    use_expected_same = bool(
        fairness_cfg.get("use_expected_same_prediction", True)
    )

    fairness_config = fairness_config or build_fairness_config(config)

    input_weight = float(cost_cfg.get("input_weight", 0.08))
    output_weight = float(cost_cfg.get("output_weight", 0.32))

    instruction = prompt_item["prompt"]

    rows = []
    eval_prompts = []
    raw_outputs = []

    base_predictions = []
    counterfactual_predictions = []
    expected_same_flags = []

    if use_llm:
        llm = build_llm_from_config(config)
        rule_model = None
    else:
        llm = None
        rule_model = RuleBasedSubjModel()

    for pair in pairs:
        base_text = pair["base_text"]
        counterfactual_text = pair["counterfactual_text"]

        base_prompt = build_eval_prompt(
            instruction=instruction,
            text=base_text,
            labels=labels,
            require_final_answer_tags=require_final_answer_tags,
        )
        cf_prompt = build_eval_prompt(
            instruction=instruction,
            text=counterfactual_text,
            labels=labels,
            require_final_answer_tags=require_final_answer_tags,
        )

        if use_llm:
            base_response = get_llm_response(llm, base_prompt)
            cf_response = get_llm_response(llm, cf_prompt)
        else:
            base_response = rule_model.predict(base_text)
            cf_response = rule_model.predict(counterfactual_text)

        base_pred = extract_label(base_response, labels)
        cf_pred = extract_label(cf_response, labels)

        base_predictions.append(base_pred)
        counterfactual_predictions.append(cf_pred)

        expect_same = bool(pair.get("expected_same_prediction", True))
        expected_same_flags.append(expect_same)

        eval_prompts.extend([base_prompt, cf_prompt])
        raw_outputs.extend([base_response, cf_response])

        flipped = base_pred != cf_pred
        # A flip is a violation only when the pair was expected to stay the same;
        # a non-flip on a label-changing pair is also a violation.
        violation = flipped if expect_same else (not flipped)

        rows.append(
            {
                "prompt_id": prompt_item["id"],
                "category": prompt_item["category"],
                "prompt": instruction,
                "pair_id": pair.get("id"),
                "protected_attribute": pair.get("protected_attribute"),
                "base_group": pair.get("base_group"),
                "counterfactual_group": pair.get("counterfactual_group"),
                "base_text": base_text,
                "counterfactual_text": counterfactual_text,
                "base_prediction": base_pred,
                "counterfactual_prediction": cf_pred,
                "base_response": base_response,
                "counterfactual_response": cf_response,
                "expected_same_prediction": expect_same,
                "flipped": flipped,
                "violation": violation,
            }
        )

    prompt_id = prompt_item["id"]

    # Pull any accumulated fairness debt for this prompt before this round's
    # update, so the combined score reflects persistent past violations.
    prior_debt = (
        debt_tracker.get_debt(prompt_id) if debt_tracker is not None else 0.0
    )

    fairness_result = evaluate_combined_fairness(
        base_predictions=base_predictions,
        counterfactual_predictions=counterfactual_predictions,
        expected_same_prediction=(
            expected_same_flags if use_expected_same else None
        ),
        outputs=raw_outputs,
        fairness_debt=prior_debt,
        config=fairness_config,
    )

    # Update the debt tracker with this round's counterfactual violation rate.
    if debt_tracker is not None:
        debt_tracker.update(
            prompt_id=prompt_id,
            fairness_risk=fairness_result.counterfactual_flip_rate,
        )

    cost_info = estimate_pair_eval_cost(
        prompts=eval_prompts,
        outputs=raw_outputs,
        input_weight=input_weight,
        output_weight=output_weight,
    )

    breakdown = fairness_result.details.get("breakdown", {})
    components = breakdown.get("components", {})

    summary = {
        "prompt_id": prompt_id,
        "category": prompt_item["category"],
        "prompt": instruction,
        "fairness_risk": fairness_result.fairness_risk,
        "counterfactual_flip_rate": fairness_result.counterfactual_flip_rate,
        "group_accuracy_gap": fairness_result.group_accuracy_gap,
        "bias_violation_rate": fairness_result.bias_violation_rate,
        "num_pairs": fairness_result.num_pairs,
        "num_flips": fairness_result.num_flips,
        "fairness_debt": fairness_result.fairness_debt,
        "fairness_method": fairness_result.details.get("method"),
        "fairness_signals_present": ",".join(
            fairness_result.details.get("signals_present", [])
        ),
        "component_flip": components.get("counterfactual_flip_rate", {}).get("value"),
        "component_bias": components.get("bias_violation_rate", {}).get("value"),
        "component_debt": components.get("fairness_debt", {}).get("value"),
        "fairness_eval_input_tokens": cost_info["input_tokens"],
        "fairness_eval_output_tokens": cost_info["output_tokens"],
        "fairness_eval_cost": cost_info["cost"],
    }

    return rows, summary


def build_portfolios_from_fairness_summary(
    summaries: list[dict],
    config: dict,
) -> tuple[PromptPortfolio, PromptPortfolio]:
    """
    Build dashboard-compatible all-candidates and Pareto portfolios.

    We use ToyObjectiveEvaluator for temporary performance/risk/cost estimates,
    then replace fairness_risk with the real counterfactual fairness result.
    """
    prompt_pool = [
        {
            "id": row["prompt_id"],
            "category": row["category"],
            "prompt": row["prompt"],
        }
        for row in summaries
    ]

    dev_data = [
        {"text": "The movie was released in 1999.", "label": "objective"},
        {"text": "The acting is absolutely wonderful.", "label": "subjective"},
        {"text": "Paris is the capital of France.", "label": "objective"},
    ]

    evaluator = ToyObjectiveEvaluator()
    all_portfolio = PromptPortfolio()

    for idx, item in enumerate(prompt_pool):
        candidate = PromptCandidate(
            instruction=item["prompt"],
            metadata={
                "prompt_pool_id": item["id"],
                "method": item["id"],
                "category": item["category"],
                "row_index": idx,
                "dataset": config.get("dataset", "subj"),
                "task_type": config.get("task_type", "classification"),
                "source": "counterfactual_fairness",
            },
        )

        toy_result = evaluator.evaluate(candidate, dev_data)

        fairness_row = summaries[idx]
        fairness_eval_cost = float(fairness_row.get("fairness_eval_cost", 0.0))

        result = EvaluationResult(
            candidate_id=candidate.candidate_id,
            performance=toy_result.performance,
            cost=toy_result.cost + fairness_eval_cost,
            risk=toy_result.risk,
            fairness_risk=float(fairness_row["fairness_risk"]),
            drift=toy_result.drift,
            n_examples=int(fairness_row["num_pairs"]),
            details={
                **toy_result.details,
                "source": "counterfactual_fairness",
                "prompt_id": item["id"],
                "category": item["category"],
                "counterfactual_flip_rate": fairness_row[
                    "counterfactual_flip_rate"
                ],
                "num_pairs": fairness_row["num_pairs"],
                "num_flips": fairness_row["num_flips"],
                "fairness_eval_cost": fairness_eval_cost,
                "fairness_eval_input_tokens": fairness_row[
                    "fairness_eval_input_tokens"
                ],
                "fairness_eval_output_tokens": fairness_row[
                    "fairness_eval_output_tokens"
                ],
            },
        )

        all_portfolio.add(candidate, result)

    pareto_portfolio = PromptPortfolio()
    pareto_evaluations = pareto_archive(all_portfolio.evaluations)
    sorted_results = sort_pareto_results(pareto_evaluations.values())

    for result in sorted_results:
        candidate = all_portfolio.get(result.candidate_id)
        pareto_portfolio.add(candidate, result)

    return all_portfolio, pareto_portfolio


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
            "method": candidate.metadata.get("method"),
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
        }

        for key, value in result.details.items():
            if isinstance(value, (dict, list, tuple)):
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
            x="phase2_counterfactual_fairness",
            portfolio=portfolio,
            preference=preference,
        )

        row = decision.to_row()
        row["preference_name"] = name
        rows.append(row)

    return rows


def print_fairness_summary(summaries: list[dict]):
    print("Counterfactual fairness summary")
    print("-" * 80)

    for row in summaries:
        print(
            f"{row['prompt_id']} | "
            f"category={row['category']} | "
            f"flip_rate={row['counterfactual_flip_rate']} | "
            f"num_flips={row['num_flips']}/{row['num_pairs']} | "
            f"cost={row['fairness_eval_cost']}"
        )

    print("-" * 80)
    print(f"Total prompts evaluated: {len(summaries)}")


def build_prompt_generator(config: dict) -> FairnessAwarePromptGenerator:
    """Build the fairness-aware generator from the ``dynamic_generation:`` block."""
    dyn_cfg = config.get("dynamic_generation", {}) or {}
    guard_cfg = dyn_cfg.get("drift_guard", {}) or {}

    drift_guard = FairnessConstraintDriftGuard(
        required_terms=guard_cfg.get("required_terms"),
        fairness_terms=guard_cfg.get("fairness_terms"),
        max_missing_ratio=float(guard_cfg.get("max_missing_ratio", 0.3)),
        fairness_max_missing_ratio=float(
            guard_cfg.get("fairness_max_missing_ratio", 0.5)
        ),
    )

    gen_config = FairnessGenerationConfig(
        max_new_prompts_per_seed=int(dyn_cfg.get("max_new_prompts_per_seed", 2)),
        min_flips_to_trigger=int(dyn_cfg.get("min_flips_to_trigger", 1)),
        random_seed=config.get("seed"),
        keep_drift_failures=bool(dyn_cfg.get("keep_drift_failures", False)),
    )

    meta_llm = None
    if bool(config.get("evaluation", {}).get("use_llm", True)):
        try:
            meta_llm = build_llm_from_config(config)
        except Exception as exc:  # pragma: no cover - depends on backend
            print(f"Dynamic generation: meta-LLM unavailable, using fallback ({exc}).")
            meta_llm = None

    return FairnessAwarePromptGenerator(
        config=gen_config,
        meta_llm=meta_llm,
        drift_guard=drift_guard,
    )


def run_dynamic_generation(
    config: dict,
    base_summaries: list[dict],
    prediction_rows_by_prompt: list[dict],
    evaluate_item,
) -> list[dict]:
    """
    Repair flipping prompts into fairness-hardened candidates and evaluate them.

    Returns the summaries of the accepted, newly evaluated repair prompts. A
    no-op (returns ``[]``) unless ``dynamic_generation.enabled`` is true.
    """
    dyn_cfg = config.get("dynamic_generation", {}) or {}

    if not bool(dyn_cfg.get("enabled", False)):
        return []

    # Group the per-pair prediction rows by their originating prompt.
    rows_by_prompt: dict[str, list[dict]] = {}
    for row in prediction_rows_by_prompt:
        rows_by_prompt.setdefault(str(row.get("prompt_id")), []).append(row)

    generator = build_prompt_generator(config)
    task_description = config.get("task_description", "")
    dataset = config.get("dataset")
    task_type = config.get("task_type")

    new_summaries: list[dict] = []
    seen_instructions: set[str] = set()

    for summary in base_summaries:
        prompt_id = str(summary.get("prompt_id"))
        instruction = str(summary.get("prompt", ""))
        pair_rows = rows_by_prompt.get(prompt_id, [])

        generated = generator.generate_from_failures(
            prompt_id=prompt_id,
            instruction=instruction,
            pair_rows=pair_rows,
            task_description=task_description,
            dataset=dataset,
            task_type=task_type,
        )

        for gen in generated:
            if not gen.accepted:
                print(
                    f"  drift-rejected repair of {prompt_id}: "
                    f"{gen.drift_result.explanation}"
                )
                continue

            new_instruction = gen.candidate.instruction
            if new_instruction in seen_instructions:
                continue
            seen_instructions.add(new_instruction)

            repair_item = {
                "id": gen.candidate.candidate_id,
                "category": "fairness_repair",
                "prompt": new_instruction,
            }

            print(f"  evaluating fairness repair of {prompt_id}: {gen.candidate.candidate_id}")
            new_summaries.append(evaluate_item(repair_item))

    return new_summaries



def print_recommendations(rows: list[dict]):
    print("Counterfactual fairness recommendations")
    print("-" * 80)

    for row in rows:
        print(
            f"{row['preference_name']} -> "
            f"{row['instruction']} | "
            f"utility={row['utility']} | "
            f"reason={row['reason']}"
        )

    print("-" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/phase2_counterfactual_fairness.yaml",
        help="Counterfactual fairness YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use rule-based debug model instead of LM Studio.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)

    if args.no_llm:
        config.setdefault("evaluation", {})
        config["evaluation"]["use_llm"] = False

    output_dir = args.output_dir or config.get(
        "output_dir",
        "outputs/phase2_counterfactual_fairness_subj",
    )

    labels = config.get("labels", ["subjective", "objective"])
    pairs = get_fairness_pairs(config)
    prompt_pool = get_prompt_pool(config)

    fairness_config = build_fairness_config(config)
    debt_tracker = FairnessDebtTracker()

    prediction_rows = []
    fairness_summaries = []

    def evaluate_item(item: dict) -> dict:
        rows, summary = evaluate_prompt_on_counterfactual_pairs(
            prompt_item=item,
            pairs=pairs,
            labels=labels,
            config=config,
            fairness_config=fairness_config,
            debt_tracker=debt_tracker,
        )
        prediction_rows.extend(rows)
        fairness_summaries.append(summary)
        return summary

    for prompt_item in prompt_pool:
        print(f"Evaluating fairness for prompt: {prompt_item['id']}")
        evaluate_item(prompt_item)

    # Part B: fairness-aware dynamic prompt generation.
    # Repair prompts whose counterfactual pairs flipped into new candidates,
    # each gated by a fairness drift guard, then evaluate them too.
    dynamic_summaries = run_dynamic_generation(
        config=config,
        base_summaries=list(fairness_summaries),
        prediction_rows_by_prompt=prediction_rows,
        evaluate_item=evaluate_item,
    )
    if dynamic_summaries:
        print(f"Dynamic generation added {len(dynamic_summaries)} repaired prompts.")

    all_portfolio, pareto_portfolio = build_portfolios_from_fairness_summary(
        summaries=fairness_summaries,
        config=config,
    )

    pareto_ids = set(non_dominated_ids(all_portfolio.evaluations.values()))

    all_candidate_rows = portfolio_to_rows(
        portfolio=all_portfolio,
        pareto_ids=pareto_ids,
    )
    pareto_rows = portfolio_to_rows(
        portfolio=pareto_portfolio,
        pareto_ids=pareto_ids,
    )

    recommendations = routing_to_rows(
        portfolio=pareto_portfolio,
        preferences=routing_preferences_from_config(config),
    )

    save_csv(
        prediction_rows,
        f"{output_dir}/counterfactual_predictions.csv",
    )
    save_json(
        prediction_rows,
        f"{output_dir}/counterfactual_predictions.json",
    )
    save_csv(
        fairness_summaries,
        f"{output_dir}/counterfactual_fairness_summary.csv",
    )
    save_json(
        fairness_summaries,
        f"{output_dir}/counterfactual_fairness_summary.json",
    )
    save_csv(
        all_candidate_rows,
        f"{output_dir}/phase2_all_candidates.csv",
    )
    save_json(
        all_candidate_rows,
        f"{output_dir}/phase2_all_candidates.json",
    )
    save_csv(
        pareto_rows,
        f"{output_dir}/phase2_prompt_portfolio.csv",
    )
    save_json(
        pareto_rows,
        f"{output_dir}/phase2_prompt_portfolio.json",
    )
    save_csv(
        recommendations,
        f"{output_dir}/phase2_prompt_recommendations.csv",
    )
    save_json(
        recommendations,
        f"{output_dir}/phase2_prompt_recommendations.json",
    )

    print_fairness_summary(fairness_summaries)
    print_recommendations(recommendations)

    print(f"Saved predictions to: {output_dir}/counterfactual_predictions.csv")
    print(f"Saved fairness summary to: {output_dir}/counterfactual_fairness_summary.csv")
    print(f"Saved dashboard candidates to: {output_dir}/phase2_all_candidates.csv")
    print(f"Saved dashboard recommendations to: {output_dir}/phase2_prompt_recommendations.csv")


if __name__ == "__main__":
    main()