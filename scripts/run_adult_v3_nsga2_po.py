from __future__ import annotations

import argparse
import json
from typing import Any

from heal_capo.core import PromptPortfolio
from heal_capo.pareto import non_dominated_ids
from scripts.run_baseline_nsga2_po import (
    load_yaml,
    portfolio_to_rows,
    print_events,
    print_portfolio,
    print_recommendations,
    routing_preferences_from_config,
    routing_to_rows,
    run_baseline_nsga2_po,
    save_csv,
    save_json,
)


def _json_default(obj: Any):
    try:
        return obj.item()
    except AttributeError:
        return str(obj)


def add_exact_prompt_identity(
    rows: list[dict],
    portfolio: PromptPortfolio,
) -> list[dict]:
    """Persist the demonstrations carried by every NSGA candidate.

    The generic NSGA serializer historically saved only the instruction. That
    makes held-out evaluation rebuild every portfolio member as zero-shot. Adult
    v3 deliberately searches over 0--5 demonstrations, so the exact ordered
    examples are part of the candidate identity and must be serialized.
    """
    candidates = {
        candidate.candidate_id: candidate
        for candidate in portfolio.evaluated_candidates()
    }

    output: list[dict] = []
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        candidate = candidates.get(candidate_id)
        enriched = dict(row)
        examples = [dict(example) for example in (candidate.examples if candidate else [])]
        enriched["num_few_shot"] = len(examples)
        enriched["few_shot_examples"] = json.dumps(
            examples,
            default=_json_default,
            ensure_ascii=False,
        )
        output.append(enriched)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run NSGA-II-PO while preserving exact Adult v3 few-shot candidate "
            "identity for held-out evaluation."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/HPC_Config/adult_nsga2po_7p5m_v3_HPC.yaml",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.seed is not None:
        config["seed"] = args.seed

    output_dir = args.output_dir or config.get(
        "output_dir",
        "outputs/hpc/adult_nsga2po_7p5m_v3/seed_0",
    )

    result = run_baseline_nsga2_po(
        config=config,
        force_no_llm=args.no_llm,
    )

    pareto_ids = set(
        non_dominated_ids(result.all_portfolio.evaluations.values())
    )
    all_rows = add_exact_prompt_identity(
        portfolio_to_rows(
            portfolio=result.all_portfolio,
            pareto_ids=pareto_ids,
        ),
        result.all_portfolio,
    )
    pareto_rows = add_exact_prompt_identity(
        portfolio_to_rows(
            portfolio=result.pareto_portfolio,
            pareto_ids=pareto_ids,
        ),
        result.pareto_portfolio,
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
    if result.budget_summary is not None:
        save_json(result.budget_summary, f"{output_dir}/budget_summary.json")

    print_events(result.events)
    print_portfolio("NSGA-II-PO Pareto portfolio", pareto_rows)
    print_recommendations(recommendations)
    print(json.dumps(result.summary, indent=2, default=_json_default))
    print(f"Saved exact NSGA portfolio to: {output_dir}/nsga2_po_pareto_portfolio.csv")
    print(f"Saved all NSGA candidates to: {output_dir}/nsga2_po_all_candidates.csv")


if __name__ == "__main__":
    main()
