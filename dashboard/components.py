from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


REQUIRED_PORTFOLIO_COLUMNS = {
    "candidate_id",
    "method",
    "performance",
    "cost",
    "risk",
    "fairness_risk",
    "prompt",
}


REQUIRED_RECOMMENDATION_COLUMNS = {
    "preference_name",
    "candidate_id",
    "instruction",
    "utility",
    "reason",
}


COUNTERFACTUAL_DETAIL_COLUMNS = [
    "detail_counterfactual_flip_rate",
    "detail_num_pairs",
    "detail_num_flips",
    "detail_fairness_eval_cost",
    "detail_fairness_eval_input_tokens",
    "detail_fairness_eval_output_tokens",
]


def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    """
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    return pd.read_csv(csv_path)


def load_portfolio(
    path: str = "outputs/phase2_counterfactual_fairness_subj/phase2_all_candidates.csv",
) -> pd.DataFrame:
    """
    Load all candidate prompts or Pareto portfolio prompts.
    """
    df = load_csv(path)
    return clean_portfolio_dataframe(df)


def load_recommendations(
    path: str = "outputs/phase2_counterfactual_fairness_subj/phase2_prompt_recommendations.csv",
) -> pd.DataFrame:
    """
    Load prompt recommendation table.
    """
    df = load_csv(path)
    return clean_recommendation_dataframe(df)


def clean_portfolio_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize portfolio dataframe types and fill missing columns.
    """
    df = df.copy()

    for column in REQUIRED_PORTFOLIO_COLUMNS:
        if column not in df.columns:
            df[column] = None

    if "category" not in df.columns:
        df["category"] = "unknown"

    if "source" not in df.columns:
        df["source"] = "unknown"

    if "is_pareto" not in df.columns:
        df["is_pareto"] = False

    numeric_columns = [
        "performance",
        "cost",
        "risk",
        "fairness_risk",
        "drift",
        "n_examples",
        "dev_size",
        "test_size",
        *COUNTERFACTUAL_DETAIL_COLUMNS,
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df["is_pareto"] = df["is_pareto"].apply(_to_bool)

    df["prompt"] = df["prompt"].fillna("").astype(str)
    df["method"] = df["method"].fillna("unknown").astype(str)
    df["category"] = df["category"].fillna("unknown").astype(str)

    df["prompt_short"] = df["prompt"].apply(
        lambda text: shorten_text(text, max_chars=90)
    )

    df["display_name"] = df.apply(make_display_name, axis=1)

    df["low_risk_score"] = 1.0 - df["risk"].clip(lower=0.0, upper=1.0)
    df["fairness_score"] = 1.0 - df["fairness_risk"].clip(lower=0.0, upper=1.0)

    if "detail_counterfactual_flip_rate" in df.columns:
        df["counterfactual_flip_rate"] = df[
            "detail_counterfactual_flip_rate"
        ].fillna(0.0)
    else:
        df["counterfactual_flip_rate"] = df["fairness_risk"].fillna(0.0)

    if "detail_num_flips" in df.columns:
        df["num_fairness_flips"] = df["detail_num_flips"].fillna(0.0).astype(int)
    else:
        df["num_fairness_flips"] = 0

    if "detail_num_pairs" in df.columns:
        df["num_fairness_pairs"] = df["detail_num_pairs"].fillna(0.0).astype(int)
    elif "n_examples" in df.columns:
        df["num_fairness_pairs"] = df["n_examples"].fillna(0.0).astype(int)
    else:
        df["num_fairness_pairs"] = 0

    return df


def clean_recommendation_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize recommendation dataframe types.
    """
    df = df.copy()

    for column in REQUIRED_RECOMMENDATION_COLUMNS:
        if column not in df.columns:
            df[column] = None

    if "mode" not in df.columns:
        df["mode"] = df["preference_name"]

    df["preference_name"] = df["preference_name"].fillna("unknown").astype(str)
    df["instruction"] = df["instruction"].fillna("").astype(str)
    df["reason"] = df["reason"].fillna("").astype(str)
    df["utility"] = pd.to_numeric(df["utility"], errors="coerce").fillna(0.0)

    df["instruction_short"] = df["instruction"].apply(
        lambda text: shorten_text(text, max_chars=100)
    )

    return df


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    text = str(value).strip().lower()

    return text in {"true", "1", "yes", "y"}


def shorten_text(text: str, max_chars: int = 120) -> str:
    """
    Shorten long prompt text for display.
    """
    text = str(text)

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 3] + "..."


def make_display_name(row) -> str:
    """
    Make a compact name for a prompt row.
    """
    method = row.get("method", "unknown")
    candidate_id = str(row.get("candidate_id", ""))[:8]

    if candidate_id:
        return f"{method} | {candidate_id}"

    return str(method)


def filter_portfolio(
    df: pd.DataFrame,
    show_only_pareto: bool = False,
    method_filter: Optional[list[str]] = None,
    min_performance: float = 0.0,
    max_cost: Optional[float] = None,
    max_risk: Optional[float] = None,
    max_fairness_risk: Optional[float] = None,
) -> pd.DataFrame:
    """
    Filter portfolio dataframe for dashboard controls.
    """
    filtered = df.copy()

    if show_only_pareto:
        filtered = filtered[filtered["is_pareto"] == True]

    if method_filter:
        filtered = filtered[filtered["method"].isin(method_filter)]

    filtered = filtered[filtered["performance"] >= min_performance]

    if max_cost is not None:
        filtered = filtered[filtered["cost"] <= max_cost]

    if max_risk is not None:
        filtered = filtered[filtered["risk"] <= max_risk]

    if max_fairness_risk is not None:
        filtered = filtered[filtered["fairness_risk"] <= max_fairness_risk]

    return filtered.reset_index(drop=True)


def compute_weighted_utility(
    df: pd.DataFrame,
    accuracy_weight: float = 1.0,
    cost_weight: float = 0.3,
    risk_weight: float = 1.0,
    fairness_weight: float = 1.0,
) -> pd.DataFrame:
    """
    Compute dashboard-side weighted utility.

    Higher performance is better.
    Lower cost, risk, and fairness risk are better.
    """
    df = df.copy()

    if df.empty:
        df["dashboard_utility"] = pd.Series(dtype=float)
        return df

    performance_score = _normalize_higher_is_better(df["performance"])
    cost_score = _normalize_lower_is_better(df["cost"])
    risk_score = _normalize_lower_is_better(df["risk"])
    fairness_score = _normalize_lower_is_better(df["fairness_risk"])

    total_weight = accuracy_weight + cost_weight + risk_weight + fairness_weight

    if total_weight <= 0:
        total_weight = 1.0

    df["dashboard_utility"] = (
        accuracy_weight * performance_score
        + cost_weight * cost_score
        + risk_weight * risk_score
        + fairness_weight * fairness_score
    ) / total_weight

    return df.sort_values(
        by=[
            "dashboard_utility",
            "performance",
            "risk",
            "fairness_risk",
            "cost",
        ],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)


def _normalize_higher_is_better(series: pd.Series) -> pd.Series:
    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series([1.0] * len(series), index=series.index)

    return (series - min_value) / (max_value - min_value)


def _normalize_lower_is_better(series: pd.Series) -> pd.Series:
    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series([1.0] * len(series), index=series.index)

    return (max_value - series) / (max_value - min_value)


def portfolio_summary(df: pd.DataFrame) -> dict:
    """
    Summary stats for dashboard metric cards.
    """
    if df.empty:
        return {
            "num_prompts": 0,
            "num_pareto": 0,
            "best_performance": 0.0,
            "lowest_cost": 0.0,
            "lowest_risk": 0.0,
            "lowest_fairness_risk": 0.0,
            "highest_flip_rate": 0.0,
            "total_fairness_flips": 0,
        }

    highest_flip_rate = (
        float(df["counterfactual_flip_rate"].max())
        if "counterfactual_flip_rate" in df.columns
        else 0.0
    )
    total_fairness_flips = (
        int(df["num_fairness_flips"].sum())
        if "num_fairness_flips" in df.columns
        else 0
    )

    return {
        "num_prompts": int(len(df)),
        "num_pareto": int(df["is_pareto"].sum()) if "is_pareto" in df.columns else 0,
        "best_performance": float(df["performance"].max()),
        "lowest_cost": float(df["cost"].min()),
        "lowest_risk": float(df["risk"].min()),
        "lowest_fairness_risk": float(df["fairness_risk"].min()),
        "highest_flip_rate": highest_flip_rate,
        "total_fairness_flips": total_fairness_flips,
    }


def recommendation_summary(recommendations: pd.DataFrame) -> list[dict]:
    """
    Convert recommendations table into a list of display dictionaries.
    """
    rows = []

    for _, row in recommendations.iterrows():
        rows.append(
            {
                "preference_name": row.get("preference_name"),
                "mode": row.get("mode"),
                "instruction": row.get("instruction"),
                "instruction_short": row.get("instruction_short"),
                "utility": row.get("utility"),
                "reason": row.get("reason"),
                "candidate_id": row.get("candidate_id"),
            }
        )

    return rows


def make_scatter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for accuracy-cost scatter chart.
    """
    columns = [
        "display_name",
        "method",
        "category",
        "performance",
        "cost",
        "risk",
        "fairness_risk",
        "counterfactual_flip_rate",
        "num_fairness_flips",
        "is_pareto",
        "prompt_short",
    ]

    available = [column for column in columns if column in df.columns]

    return df[available].copy()


def make_risk_fairness_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for risk-fairness chart/table.
    """
    columns = [
        "display_name",
        "method",
        "category",
        "risk",
        "fairness_risk",
        "counterfactual_flip_rate",
        "num_fairness_flips",
        "performance",
        "cost",
        "is_pareto",
        "prompt_short",
    ]

    available = [column for column in columns if column in df.columns]

    return df[available].copy()


def get_top_prompt(df: pd.DataFrame) -> Optional[dict]:
    """
    Return top row after utility ranking.
    """
    if df.empty:
        return None

    row = df.iloc[0]

    return {
        "candidate_id": row.get("candidate_id"),
        "method": row.get("method"),
        "category": row.get("category"),
        "prompt": row.get("prompt"),
        "performance": row.get("performance"),
        "cost": row.get("cost"),
        "risk": row.get("risk"),
        "fairness_risk": row.get("fairness_risk"),
        "counterfactual_flip_rate": row.get("counterfactual_flip_rate", 0.0),
        "num_fairness_flips": row.get("num_fairness_flips", 0),
        "utility": row.get("dashboard_utility"),
        "is_pareto": row.get("is_pareto"),
    }


def fairness_failure_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return prompts with at least one counterfactual fairness flip.
    """
    if "num_fairness_flips" not in df.columns:
        return pd.DataFrame()

    return df[df["num_fairness_flips"] > 0].copy().reset_index(drop=True)


def pareto_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only Pareto candidates.
    """
    if "is_pareto" not in df.columns:
        return pd.DataFrame()

    return df[df["is_pareto"] == True].copy().reset_index(drop=True)

def load_budget_events(path: str) -> pd.DataFrame:
    """
    Load budgeted MO-CAPO intensification events.
    """
    df = load_csv(path)
    return clean_budget_events_dataframe(df)


def load_budget_summary(path: str) -> dict:
    """
    Load budget summary JSON.
    """
    json_path = Path(path)

    if not json_path.exists():
        raise FileNotFoundError(f"Budget summary JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_budget_events_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize budget/evolutionary MO-CAPO event dataframe.
    """
    df = df.copy()

    required_columns = {
        "candidate_id",
        "method",
        "event_type",
        "iteration",
        "offspring_index",
        "operator",
        "parent_ids",
        "accepted",
        "rejected",
        "reason",
        "evaluated_blocks",
        "block_id",
        "budget_used",
        "remaining_budget",
        "budget_utilization",
        "kept_ids",
        "removed_ids",
        "metadata",
    }

    for column in required_columns:
        if column not in df.columns:
            df[column] = None

    bool_columns = ["accepted", "rejected"]
    for column in bool_columns:
        df[column] = df[column].apply(_to_bool)

    numeric_columns = [
        "iteration",
        "offspring_index",
        "block_id",
        "budget_used",
        "remaining_budget",
        "budget_utilization",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df["method"] = df["method"].fillna("unknown").astype(str)
    df["event_type"] = df["event_type"].fillna("unknown").astype(str)
    df["operator"] = df["operator"].fillna("").astype(str)
    df["parent_ids"] = df["parent_ids"].fillna("").astype(str)
    df["reason"] = df["reason"].fillna("").astype(str)
    df["evaluated_blocks"] = df["evaluated_blocks"].fillna("[]").astype(str)
    df["kept_ids"] = df["kept_ids"].fillna("").astype(str)
    df["removed_ids"] = df["removed_ids"].fillna("").astype(str)
    df["metadata"] = df["metadata"].fillna("").astype(str)

    df["is_initial_population"] = df["event_type"] == "initial_population"
    df["is_evolutionary_intensification"] = (
        df["event_type"] == "evolutionary_intensification"
    )
    df["is_environmental_selection"] = df["event_type"] == "environmental_selection"
    df["is_advance_incumbent"] = df["event_type"] == "advance_incumbent"

    return df


def budget_events_summary(events_df: pd.DataFrame) -> dict:
    """
    Compact stats for budget/evolutionary MO-CAPO events.
    """
    if events_df.empty:
        return {
            "num_events": 0,
            "num_accepted": 0,
            "num_rejected": 0,
            "num_seed": 0,
            "num_initial_population": 0,
            "num_evolutionary_intensification": 0,
            "num_environmental_selection": 0,
            "num_advance_incumbent": 0,
            "num_budget_stops": 0,
            "num_budget_errors": 0,
        }

    event_type = events_df["event_type"].astype(str)

    return {
        "num_events": int(len(events_df)),
        "num_accepted": int(events_df["accepted"].sum()),
        "num_rejected": int(events_df["rejected"].sum()),

        # Backward-compatible old event name.
        "num_seed": int((event_type == "seed_incumbent").sum()),

        # New evolutionary MO-CAPO event names.
        "num_initial_population": int((event_type == "initial_population").sum()),
        "num_evolutionary_intensification": int(
            (event_type == "evolutionary_intensification").sum()
        ),
        "num_environmental_selection": int(
            (event_type == "environmental_selection").sum()
        ),
        "num_advance_incumbent": int((event_type == "advance_incumbent").sum()),

        "num_budget_stops": int(
            event_type.str.contains("budget_stop", na=False).sum()
        ),
        "num_budget_errors": int(
            event_type.str.contains("budget_error", na=False).sum()
        ),
    }