from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_CONFIG = {
    "output_dir": "outputs/paper_summary",
    "budgeted_portfolio_csv": "outputs/phase2_budgeted_mocapo_subj/phase2_all_candidates.csv",
    "budgeted_pareto_csv": "outputs/phase2_budgeted_mocapo_subj/phase2_prompt_portfolio.csv",
    "budgeted_recommendations_csv": "outputs/phase2_budgeted_mocapo_subj/phase2_prompt_recommendations.csv",
    "budgeted_events_csv": "outputs/phase2_budgeted_mocapo_subj/budgeted_mocapo_events.csv",
    "budget_summary_json": "outputs/phase2_budgeted_mocapo_subj/budget_summary.json",
    "counterfactual_fairness_summary_csv": "outputs/phase2_counterfactual_fairness_subj/counterfactual_fairness_summary.csv",
    "counterfactual_predictions_csv": "outputs/phase2_counterfactual_fairness_subj/counterfactual_predictions.csv",
    "dataset": "SUBJ",
    "model": "mistralai/mistral-small-3.2",
    "method": "HEAL-CAPO",
}


def load_yaml(path: str | None) -> dict:
    if path is None:
        return {}

    yaml_path = Path(path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Config not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_config(config: dict) -> dict:
    merged = dict(DEFAULT_CONFIG)
    merged.update(config)
    return merged


def load_csv_if_exists(path: str) -> pd.DataFrame:
    csv_path = Path(path)

    if not csv_path.exists():
        return pd.DataFrame()

    return pd.read_csv(csv_path)


def load_json_if_exists(path: str) -> dict:
    json_path = Path(path)

    if not json_path.exists():
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        json.dump(data, f, indent=2)


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def summarize_portfolio(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "num_candidates": 0,
            "num_pareto": 0,
            "best_performance": 0.0,
            "lowest_cost": 0.0,
            "lowest_risk": 0.0,
            "lowest_fairness_risk": 0.0,
            "best_prompt_method": "",
            "lowest_cost_method": "",
            "lowest_risk_method": "",
            "lowest_fairness_method": "",
        }

    df = df.copy()

    for column in ["performance", "cost", "risk", "fairness_risk"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    if "is_pareto" in df.columns:
        is_pareto = df["is_pareto"].astype(str).str.lower().isin(["true", "1", "yes"])
        num_pareto = int(is_pareto.sum())
    else:
        num_pareto = 0

    best_perf_row = df.sort_values(
        by=["performance", "risk", "fairness_risk", "cost"],
        ascending=[False, True, True, True],
    ).iloc[0]

    lowest_cost_row = df.sort_values(
        by=["cost", "risk", "fairness_risk", "performance"],
        ascending=[True, True, True, False],
    ).iloc[0]

    lowest_risk_row = df.sort_values(
        by=["risk", "fairness_risk", "cost", "performance"],
        ascending=[True, True, True, False],
    ).iloc[0]

    lowest_fairness_row = df.sort_values(
        by=["fairness_risk", "risk", "cost", "performance"],
        ascending=[True, True, True, False],
    ).iloc[0]

    return {
        "num_candidates": int(len(df)),
        "num_pareto": num_pareto,
        "best_performance": safe_float(df["performance"].max()),
        "lowest_cost": safe_float(df["cost"].min()),
        "lowest_risk": safe_float(df["risk"].min()),
        "lowest_fairness_risk": safe_float(df["fairness_risk"].min()),
        "best_prompt_method": str(best_perf_row.get("method", "")),
        "lowest_cost_method": str(lowest_cost_row.get("method", "")),
        "lowest_risk_method": str(lowest_risk_row.get("method", "")),
        "lowest_fairness_method": str(lowest_fairness_row.get("method", "")),
    }


def summarize_budget(budget_summary: dict, events_df: pd.DataFrame) -> dict:
    if events_df.empty:
        num_events = 0
        num_accepted = 0
        num_rejected = 0
        num_seed = 0
        num_initial_population_events = 0
        num_evolutionary_events = 0
        num_environmental_selection_events = 0
        num_advance_incumbent_events = 0
        num_budget_stop_events = 0
        num_budget_error_events = 0
    else:
        events = events_df.copy()

        if "accepted" in events.columns:
            accepted = events["accepted"].astype(str).str.lower().isin(["true", "1", "yes"])
        else:
            accepted = pd.Series([False] * len(events))

        if "rejected" in events.columns:
            rejected = events["rejected"].astype(str).str.lower().isin(["true", "1", "yes"])
        else:
            rejected = pd.Series([False] * len(events))

        event_type = (
            events["event_type"].astype(str)
            if "event_type" in events.columns
            else pd.Series([""] * len(events))
        )

        num_events = int(len(events))
        num_accepted = int(accepted.sum())
        num_rejected = int(rejected.sum())

        # Backward-compatible old name.
        num_seed = int((event_type == "seed_incumbent").sum())

        # New evolutionary MO-CAPO event names.
        num_initial_population_events = int((event_type == "initial_population").sum())
        num_evolutionary_events = int((event_type == "evolutionary_intensification").sum())
        num_environmental_selection_events = int((event_type == "environmental_selection").sum())
        num_advance_incumbent_events = int((event_type == "advance_incumbent").sum())

        num_budget_stop_events = int(event_type.astype(str).str.contains("budget_stop", na=False).sum())
        num_budget_error_events = int(event_type.astype(str).str.contains("budget_error", na=False).sum())

    return {
        "algorithm": str(budget_summary.get("algorithm", "")),
        "max_budget": safe_float(budget_summary.get("max_budget", 0.0)),
        "used_budget": safe_float(budget_summary.get("used_budget", 0.0)),
        "remaining_budget": safe_float(budget_summary.get("remaining_budget", 0.0)),
        "budget_utilization": safe_float(budget_summary.get("utilization", 0.0)),
        "num_budget_records": safe_int(budget_summary.get("num_records", 0)),
        "num_budget_candidates": safe_int(budget_summary.get("num_candidates", 0)),
        "num_budget_blocks": safe_int(budget_summary.get("num_blocks", 0)),
        "input_tokens": safe_float(budget_summary.get("input_tokens", 0.0)),
        "output_tokens": safe_float(budget_summary.get("output_tokens", 0.0)),
        "total_tokens": safe_float(budget_summary.get("total_tokens", 0.0)),
        "budget_evaluator": str(budget_summary.get("evaluator", "")),
        "budget_model_id": str(budget_summary.get("model_id", "")),

        # New evolutionary summary fields.
        "population_size": safe_int(budget_summary.get("population_size", 0)),
        "max_iterations": safe_int(budget_summary.get("max_iterations", 0)),
        "offspring_per_iteration": safe_int(budget_summary.get("offspring_per_iteration", 0)),
        "num_population_candidates": safe_int(budget_summary.get("num_population_candidates", 0)),
        "num_incumbents": safe_int(budget_summary.get("num_incumbents", 0)),
        "num_evaluated_candidates": safe_int(budget_summary.get("num_evaluated_candidates", 0)),

        # Event counts.
        "num_events": num_events,
        "num_accepted_events": num_accepted,
        "num_rejected_events": num_rejected,
        "num_seed_events": num_seed,
        "num_initial_population_events": num_initial_population_events,
        "num_evolutionary_intensification_events": num_evolutionary_events,
        "num_environmental_selection_events": num_environmental_selection_events,
        "num_advance_incumbent_events": num_advance_incumbent_events,
        "num_budget_stop_events": num_budget_stop_events,
        "num_budget_error_events": num_budget_error_events,

        # Backward-compatible alias.
        "num_intensification_events": num_evolutionary_events or num_events,
    }


def summarize_fairness(summary_df: pd.DataFrame, predictions_df: pd.DataFrame) -> dict:
    if summary_df.empty:
        return {
            "fairness_prompts_evaluated": 0,
            "fairness_pairs": 0,
            "max_counterfactual_flip_rate": 0.0,
            "mean_counterfactual_flip_rate": 0.0,
            "total_fairness_flips": 0,
            "worst_fairness_prompt": "",
            "num_flipped_examples": 0,
        }

    df = summary_df.copy()

    for column in ["counterfactual_flip_rate", "num_pairs", "num_flips", "fairness_risk"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    worst_row = df.sort_values(
        by=["counterfactual_flip_rate", "num_flips"],
        ascending=[False, False],
    ).iloc[0]

    if predictions_df.empty or "flipped" not in predictions_df.columns:
        num_flipped_examples = safe_int(df["num_flips"].sum())
    else:
        flipped = predictions_df["flipped"].astype(str).str.lower().isin(["true", "1", "yes"])
        num_flipped_examples = int(flipped.sum())

    return {
        "fairness_prompts_evaluated": int(len(df)),
        "fairness_pairs": safe_int(df["num_pairs"].max()) if "num_pairs" in df.columns else 0,
        "max_counterfactual_flip_rate": safe_float(df["counterfactual_flip_rate"].max())
        if "counterfactual_flip_rate" in df.columns
        else 0.0,
        "mean_counterfactual_flip_rate": safe_float(df["counterfactual_flip_rate"].mean())
        if "counterfactual_flip_rate" in df.columns
        else 0.0,
        "total_fairness_flips": safe_int(df["num_flips"].sum()) if "num_flips" in df.columns else 0,
        "worst_fairness_prompt": str(worst_row.get("prompt_id", "")),
        "num_flipped_examples": num_flipped_examples,
    }


def summarize_recommendations(recommendations_df: pd.DataFrame) -> dict:
    if recommendations_df.empty:
        return {}

    rows = {}

    for _, row in recommendations_df.iterrows():
        preference = str(row.get("preference_name", "unknown"))
        prefix = f"recommendation_{preference}"

        rows[f"{prefix}_candidate_id"] = row.get("candidate_id", "")
        rows[f"{prefix}_utility"] = safe_float(row.get("utility", 0.0))
        rows[f"{prefix}_instruction"] = row.get("instruction", "")

    return rows


def make_experiment_summary(config: dict) -> dict:
    portfolio_df = load_csv_if_exists(config["budgeted_portfolio_csv"])
    pareto_df = load_csv_if_exists(config["budgeted_pareto_csv"])
    recommendations_df = load_csv_if_exists(config["budgeted_recommendations_csv"])
    events_df = load_csv_if_exists(config["budgeted_events_csv"])
    budget_summary = load_json_if_exists(config["budget_summary_json"])
    fairness_summary_df = load_csv_if_exists(config["counterfactual_fairness_summary_csv"])
    fairness_predictions_df = load_csv_if_exists(config["counterfactual_predictions_csv"])

    portfolio_summary = summarize_portfolio(portfolio_df)
    pareto_summary = summarize_portfolio(pareto_df)
    budget_stats = summarize_budget(budget_summary, events_df)
    fairness_stats = summarize_fairness(fairness_summary_df, fairness_predictions_df)
    recommendation_stats = summarize_recommendations(recommendations_df)

    row = {
        "dataset": config.get("dataset", "SUBJ"),
        "model": config.get("model", ""),
        "method": config.get("method", "HEAL-CAPO"),
        **portfolio_summary,
        "pareto_num_candidates": pareto_summary["num_candidates"],
        "pareto_best_performance": pareto_summary["best_performance"],
        "pareto_lowest_cost": pareto_summary["lowest_cost"],
        "pareto_lowest_risk": pareto_summary["lowest_risk"],
        "pareto_lowest_fairness_risk": pareto_summary["lowest_fairness_risk"],
        **budget_stats,
        **fairness_stats,
        **recommendation_stats,
    }

    return row


def make_latex_table(row: dict) -> str:
    """
    Create a compact LaTeX table row/table for paper draft.
    """
    headers = [
        "Dataset",
        "Model",
        "Method",
        "Candidates",
        "Pareto",
        "Best Perf.",
        "Lowest Cost",
        "Lowest Risk",
        "Lowest Fair.",
        "Budget Used",
        "Util.",
        "Tokens",
        "Fair. Flips",
    ]

    values = [
        row.get("dataset", ""),
        row.get("model", ""),
        row.get("method", ""),
        row.get("num_candidates", 0),
        row.get("pareto_num_candidates", 0),
        f"{safe_float(row.get('best_performance', 0.0)):.3f}",
        f"{safe_float(row.get('lowest_cost', 0.0)):.2f}",
        f"{safe_float(row.get('lowest_risk', 0.0)):.3f}",
        f"{safe_float(row.get('lowest_fairness_risk', 0.0)):.3f}",
        f"{safe_float(row.get('used_budget', 0.0)):.2f}",
        f"{100.0 * safe_float(row.get('budget_utilization', 0.0)):.1f}\\%",
        f"{safe_int(row.get('total_tokens', 0))}",
        f"{safe_int(row.get('total_fairness_flips', 0))}",
    ]

    header_line = " & ".join(headers) + r" \\"
    value_line = " & ".join(str(value) for value in values) + r" \\"

    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lllrrrrrrrrrr}",
            r"\toprule",
            header_line,
            r"\midrule",
            value_line,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Summary of HEAL-CAPO budgeted prompt optimization and fairness evaluation.}",
            r"\label{tab:heal_capo_summary}",
            r"\end{table}",
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config for summary export.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    args = parser.parse_args()

    user_config = load_yaml(args.config)
    config = merge_config(user_config)

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    row = make_experiment_summary(config)
    latex_table = make_latex_table(row)

    save_csv([row], str(output_dir / "experiment_summary.csv"))
    save_json(row, str(output_dir / "experiment_summary.json"))

    with open(output_dir / "latex_table.txt", "w", encoding="utf-8") as f:
        f.write(latex_table)

    print("Experiment summary")
    print("-" * 80)
    for key, value in row.items():
        if key.startswith("recommendation_") and key.endswith("_instruction"):
            continue
        print(f"{key}: {value}")

    print("-" * 80)
    print(f"Saved summary CSV to: {output_dir / 'experiment_summary.csv'}")
    print(f"Saved summary JSON to: {output_dir / 'experiment_summary.json'}")
    print(f"Saved LaTeX table to: {output_dir / 'latex_table.txt'}")


if __name__ == "__main__":
    main()