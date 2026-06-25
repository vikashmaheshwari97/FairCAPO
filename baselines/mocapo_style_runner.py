from __future__ import annotations

from typing import Optional

from baselines.promptolution_runner import build_llm, evaluate_prompt_with_llm


def _dominates(a: dict, b: dict) -> bool:
    """
    We maximize score and minimize cost.
    a dominates b if:
      a score >= b score
      a cost <= b cost
      and at least one is strictly better.
    """
    return (
        a["dev_score"] >= b["dev_score"]
        and a["dev_cost"] <= b["dev_cost"]
        and (
            a["dev_score"] > b["dev_score"]
            or a["dev_cost"] < b["dev_cost"]
        )
    )


def select_pareto_front(results: list[dict]) -> list[dict]:
    pareto = []

    for candidate in results:
        dominated = False

        for other in results:
            if other is candidate:
                continue

            if _dominates(other, candidate):
                dominated = True
                break

        if not dominated:
            pareto.append(candidate)

    pareto.sort(key=lambda x: (-x["dev_score"], x["dev_cost"]))
    return pareto


def run_mocapo_style_baseline_with_test(
    dev_dataset,
    test_dataset,
    candidate_prompts: list[str],
    llm_config: Optional[dict] = None,
    classes: Optional[list[str]] = None,
):
    """
    Simple MO-CAPO-style baseline for Phase 1.

    This is not the full MO-CAPO optimizer.
    It evaluates a portfolio of candidate prompts and returns the
    non-dominated performance-cost Pareto set.
    """

    all_results = []

    for idx, prompt in enumerate(candidate_prompts):
        dev_llm = build_llm(llm_config)
        test_llm = build_llm(llm_config)

        dev_score, dev_predictions, dev_labels, dev_cost_info, dev_token_info = evaluate_prompt_with_llm(
            dataset=dev_dataset,
            prompt=prompt,
            llm=dev_llm,
            classes=classes,
        )

        test_score, test_predictions, test_labels, test_cost_info, test_token_info = evaluate_prompt_with_llm(
            dataset=test_dataset,
            prompt=prompt,
            llm=test_llm,
            classes=classes,
        )

        all_results.append(
            {
                "method": f"mocapo_style_candidate_{idx}",
                "candidate_id": idx,
                "prompt": prompt,
                "dev_score": dev_score,
                "test_score": test_score,
                "dev_predictions": dev_predictions,
                "test_predictions": test_predictions,
                "dev_labels": dev_labels,
                "test_labels": test_labels,
                "dev_input_tokens": dev_cost_info["input_tokens"],
                "dev_output_tokens": dev_cost_info["output_tokens"],
                "dev_cost": dev_cost_info["cost"],
                "test_input_tokens": test_cost_info["input_tokens"],
                "test_output_tokens": test_cost_info["output_tokens"],
                "test_cost": test_cost_info["cost"],
                "dev_token_info": dev_token_info,
                "test_token_info": test_token_info,
            }
        )

    pareto_results = select_pareto_front(all_results)

    for rank, result in enumerate(pareto_results):
        result["method"] = f"mocapo_style_pareto_{rank}"
        result["pareto_rank"] = rank
        result["num_candidates_evaluated"] = len(all_results)

    return pareto_results, all_results