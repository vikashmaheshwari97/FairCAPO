from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import yaml

from dashboard.components import (
    budget_events_summary,
    compute_weighted_utility,
    filter_portfolio,
    get_top_prompt,
    load_budget_events,
    load_budget_summary,
    load_portfolio,
    load_recommendations,
    make_risk_fairness_dataframe,
    make_scatter_dataframe,
    portfolio_summary,
    recommendation_summary,
)


DEFAULT_CONFIG_PATH = "configs/dashboard.yaml"

FALLBACK_CONFIG = {
    "title": "HEAL-CAPO Prompt Portfolio Dashboard",
    "subtitle": "Interactive prompt selection over performance, cost, risk, and fairness risk.",
    "data": {
        "portfolio_csv": "outputs/phase2_counterfactual_fairness_subj/phase2_all_candidates.csv",
        "recommendations_csv": "outputs/phase2_counterfactual_fairness_subj/phase2_prompt_recommendations.csv",
        "budgeted_mocapo_events_csv": "outputs/phase2_budgeted_mocapo_subj/budgeted_mocapo_events.csv",
        "budgeted_mocapo_budget_summary_json": "outputs/phase2_budgeted_mocapo_subj/budget_summary.json",
    },
    "defaults": {
        "show_only_pareto": False,
        "min_performance": 0.0,
        "max_risk": 1.0,
        "max_fairness_risk": 1.0,
        "max_cost": None,
    },
    "weights": {
        "accuracy": 1.0,
        "cost": 0.3,
        "risk": 1.0,
        "fairness": 1.0,
    },
    "display": {
        "show_downloads": True,
        "show_recommendation_cards": True,
        "show_prompt_comparison": True,
        "show_fairness_analysis": True,
        "show_warning_badges": True,
        "show_budget_analysis": True,
        "show_intensification_events": True,
    },
    "thresholds": {
        "low_risk": 0.20,
        "medium_risk": 0.40,
        "low_fairness_risk": 0.15,
        "medium_fairness_risk": 0.30,
    },
    "tabs": [
        "Overview",
        "All Candidates",
        "Pareto Portfolio",
        "Recommendations",
        "Prompt Comparison",
        "Fairness Analysis",
        "Budget Analysis",
    ],
}


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_dashboard_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    config_path = Path(path)

    if not config_path.exists():
        return FALLBACK_CONFIG

    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    return deep_merge(FALLBACK_CONFIG, loaded)


def cfg_get(config: dict, section: str, key: str, default: Any = None) -> Any:
    return config.get(section, {}).get(key, default)


def render_header(config: dict):
    st.set_page_config(
        page_title=config.get("title", "HEAL-CAPO Prompt Portfolio"),
        page_icon="🧭",
        layout="wide",
    )

    st.title(config.get("title", "HEAL-CAPO Prompt Portfolio Dashboard"))
    st.caption(
        config.get(
            "subtitle",
            "Interactive prompt selection over performance, cost, risk, and fairness risk.",
        )
    )


def render_sidebar(config: dict):
    data_cfg = config.get("data", {})
    defaults_cfg = config.get("defaults", {})
    weights_cfg = config.get("weights", {})

    st.sidebar.header("Config")

    config_path = st.sidebar.text_input(
        "Dashboard config",
        value=DEFAULT_CONFIG_PATH,
    )

    st.sidebar.header("Data")

    portfolio_path = st.sidebar.text_input(
        "Portfolio CSV",
        value=data_cfg.get(
            "portfolio_csv",
            "outputs/phase2_counterfactual_fairness_subj/phase2_all_candidates.csv",
        ),
    )

    recommendations_path = st.sidebar.text_input(
        "Recommendations CSV",
        value=data_cfg.get(
            "recommendations_csv",
            "outputs/phase2_counterfactual_fairness_subj/phase2_prompt_recommendations.csv",
        ),
    )

    quick_source = st.sidebar.selectbox(
        "Quick source",
        options=[
            "budgeted_mocapo",
            "counterfactual_fairness",
            "prompt_pool_demo",
            "phase1_portfolio",
            "custom_paths_above",
        ],
        index=0,
    )

    if quick_source == "budgeted_mocapo":
        portfolio_path = data_cfg.get(
            "budgeted_mocapo_portfolio_csv",
            "outputs/phase2_budgeted_mocapo_subj/phase2_all_candidates.csv",
        )
        recommendations_path = data_cfg.get(
            "budgeted_mocapo_recommendations_csv",
            "outputs/phase2_budgeted_mocapo_subj/phase2_prompt_recommendations.csv",
        )

    elif quick_source == "counterfactual_fairness":
        portfolio_path = data_cfg.get(
            "counterfactual_fairness_portfolio_csv",
            "outputs/phase2_counterfactual_fairness_subj/phase2_all_candidates.csv",
        )
        recommendations_path = data_cfg.get(
            "counterfactual_fairness_recommendations_csv",
            "outputs/phase2_counterfactual_fairness_subj/phase2_prompt_recommendations.csv",
        )

    elif quick_source == "prompt_pool_demo":
        portfolio_path = data_cfg.get(
            "prompt_pool_portfolio_csv",
            "outputs/phase2_prompt_pool_subj/phase2_all_candidates.csv",
        )
        recommendations_path = data_cfg.get(
            "prompt_pool_recommendations_csv",
            "outputs/phase2_prompt_pool_subj/phase2_prompt_recommendations.csv",
        )

    elif quick_source == "phase1_portfolio":
        portfolio_path = data_cfg.get(
            "phase1_portfolio_csv",
            "outputs/phase2_prompt_portfolio/phase2_all_candidates.csv",
        )
        recommendations_path = data_cfg.get(
            "phase1_recommendations_csv",
            "outputs/phase2_prompt_portfolio/phase2_prompt_recommendations.csv",
        )

    st.sidebar.header("Filters")

    show_only_pareto = st.sidebar.checkbox(
        "Show only Pareto candidates",
        value=bool(defaults_cfg.get("show_only_pareto", False)),
    )

    min_performance = st.sidebar.slider(
        "Minimum performance",
        min_value=0.0,
        max_value=1.0,
        value=float(defaults_cfg.get("min_performance", 0.0)),
        step=0.01,
    )

    max_risk = st.sidebar.slider(
        "Maximum risk",
        min_value=0.0,
        max_value=1.0,
        value=float(defaults_cfg.get("max_risk", 1.0)),
        step=0.01,
    )

    max_fairness_risk = st.sidebar.slider(
        "Maximum fairness risk",
        min_value=0.0,
        max_value=1.0,
        value=float(defaults_cfg.get("max_fairness_risk", 1.0)),
        step=0.01,
    )

    st.sidebar.header("Preference weights")

    accuracy_weight = st.sidebar.slider(
        "Accuracy weight",
        min_value=0.0,
        max_value=5.0,
        value=float(weights_cfg.get("accuracy", 1.0)),
        step=0.1,
    )

    cost_weight = st.sidebar.slider(
        "Cost weight",
        min_value=0.0,
        max_value=5.0,
        value=float(weights_cfg.get("cost", 0.3)),
        step=0.1,
    )

    risk_weight = st.sidebar.slider(
        "Risk weight",
        min_value=0.0,
        max_value=5.0,
        value=float(weights_cfg.get("risk", 1.0)),
        step=0.1,
    )

    fairness_weight = st.sidebar.slider(
        "Fairness weight",
        min_value=0.0,
        max_value=5.0,
        value=float(weights_cfg.get("fairness", 1.0)),
        step=0.1,
    )

    return {
        "config_path": config_path,
        "portfolio_path": portfolio_path,
        "recommendations_path": recommendations_path,
        "show_only_pareto": show_only_pareto,
        "min_performance": min_performance,
        "max_risk": max_risk,
        "max_fairness_risk": max_fairness_risk,
        "accuracy_weight": accuracy_weight,
        "cost_weight": cost_weight,
        "risk_weight": risk_weight,
        "fairness_weight": fairness_weight,
    }


def render_summary_cards(df: pd.DataFrame):
    summary = portfolio_summary(df)

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Prompts", summary["num_prompts"])
    col2.metric("Pareto", summary["num_pareto"])
    col3.metric("Best performance", f"{summary['best_performance']:.3f}")
    col4.metric("Lowest cost", f"{summary['lowest_cost']:.3f}")
    col5.metric("Lowest risk", f"{summary['lowest_risk']:.3f}")
    col6.metric("Lowest fairness risk", f"{summary['lowest_fairness_risk']:.3f}")


def render_warning_badges(
    risk: float,
    fairness_risk: float,
    is_pareto: bool,
    config: dict,
):
    thresholds = config.get("thresholds", {})

    low_risk = float(thresholds.get("low_risk", 0.20))
    medium_risk = float(thresholds.get("medium_risk", 0.40))
    low_fairness = float(thresholds.get("low_fairness_risk", 0.15))
    medium_fairness = float(thresholds.get("medium_fairness_risk", 0.30))

    badge_cols = st.columns(3)

    if is_pareto:
        badge_cols[0].success("Pareto-optimal")
    else:
        badge_cols[0].warning("Non-Pareto")

    if risk <= low_risk:
        badge_cols[1].success("Low risk")
    elif risk <= medium_risk:
        badge_cols[1].warning("Medium risk")
    else:
        badge_cols[1].error("High risk")

    if fairness_risk <= low_fairness:
        badge_cols[2].success("Low fairness risk")
    elif fairness_risk <= medium_fairness:
        badge_cols[2].warning("Medium fairness risk")
    else:
        badge_cols[2].error("High fairness risk")


def render_top_recommendation(ranked_df: pd.DataFrame, config: dict):
    top = get_top_prompt(ranked_df)

    st.subheader("Dashboard recommendation")

    if top is None:
        st.warning("No prompt satisfies the current filters.")
        return

    pareto_label = "Pareto candidate" if top["is_pareto"] else "Non-Pareto candidate"

    st.success(
        f"Recommended prompt: {top['method']} "
        f"({pareto_label}, utility={top['utility']:.4f})"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Performance", f"{top['performance']:.3f}")
    col2.metric("Cost", f"{top['cost']:.3f}")
    col3.metric("Risk", f"{top['risk']:.3f}")
    col4.metric("Fairness risk", f"{top['fairness_risk']:.3f}")

    if cfg_get(config, "display", "show_warning_badges", True):
        render_warning_badges(
            risk=float(top["risk"]),
            fairness_risk=float(top["fairness_risk"]),
            is_pareto=bool(top["is_pareto"]),
            config=config,
        )

    with st.expander("Recommended prompt text", expanded=True):
        st.write(top["prompt"])


def render_accuracy_cost_chart(df: pd.DataFrame):
    st.subheader("Accuracy vs cost")

    if df.empty:
        st.info("No data available for chart.")
        return

    chart_df = make_scatter_dataframe(df)

    color_field = "method:N"
    if "category" in df.columns:
        chart_df["category"] = df["category"]
        color_field = "category:N"

    chart = (
        alt.Chart(chart_df)
        .mark_circle(size=130)
        .encode(
            x=alt.X("cost:Q", title="Cost"),
            y=alt.Y("performance:Q", title="Performance"),
            color=alt.Color(color_field, title="Group"),
            shape=alt.Shape("is_pareto:N", title="Pareto"),
            size=alt.Size("risk:Q", title="Risk"),
            tooltip=[
                "display_name:N",
                "method:N",
                "performance:Q",
                "cost:Q",
                "risk:Q",
                "fairness_risk:Q",
                "is_pareto:N",
                "prompt_short:N",
            ],
        )
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


def render_risk_fairness_chart(df: pd.DataFrame):
    st.subheader("Risk vs fairness risk")

    if df.empty:
        st.info("No data available for chart.")
        return

    chart_df = make_risk_fairness_dataframe(df)

    color_field = "method:N"
    if "category" in df.columns:
        chart_df["category"] = df["category"]
        color_field = "category:N"

    chart = (
        alt.Chart(chart_df)
        .mark_circle(size=130)
        .encode(
            x=alt.X("risk:Q", title="Risk"),
            y=alt.Y("fairness_risk:Q", title="Fairness risk"),
            color=alt.Color(color_field, title="Group"),
            shape=alt.Shape("is_pareto:N", title="Pareto"),
            size=alt.Size("performance:Q", title="Performance"),
            tooltip=[
                "display_name:N",
                "method:N",
                "risk:Q",
                "fairness_risk:Q",
                "performance:Q",
                "cost:Q",
                "is_pareto:N",
                "prompt_short:N",
            ],
        )
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


def render_recommendations_cards(recommendations_df: pd.DataFrame):
    st.subheader("Saved recommendations by preference mode")

    if recommendations_df.empty:
        st.info("No recommendation table found.")
        return

    rows = recommendation_summary(recommendations_df)

    cols = st.columns(2)

    for idx, row in enumerate(rows):
        with cols[idx % 2]:
            with st.container(border=True):
                st.markdown(f"**{row['preference_name']}**")
                st.write(f"Utility: `{row['utility']:.4f}`")
                st.write(row["reason"])
                st.code(row["instruction"], language="text")


def render_prompt_table(df: pd.DataFrame):
    st.subheader("Prompt table")

    if df.empty:
        st.info("No prompts to show.")
        return

    display_columns = [
        "is_pareto",
        "method",
        "category",
        "performance",
        "cost",
        "risk",
        "fairness_risk",
        "dashboard_utility",
        "detail_counterfactual_flip_rate",
        "detail_num_flips",
        "prompt",
    ]

    available_columns = [column for column in display_columns if column in df.columns]

    st.dataframe(
        df[available_columns],
        use_container_width=True,
        hide_index=True,
    )


def render_download_buttons(
    ranked_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
):
    st.subheader("Downloads")

    col1, col2 = st.columns(2)

    col1.download_button(
        label="Download filtered prompt table",
        data=ranked_df.to_csv(index=False).encode("utf-8"),
        file_name="heal_capo_filtered_prompts.csv",
        mime="text/csv",
    )

    if not recommendations_df.empty:
        col2.download_button(
            label="Download recommendations",
            data=recommendations_df.to_csv(index=False).encode("utf-8"),
            file_name="heal_capo_recommendations.csv",
            mime="text/csv",
        )


def render_prompt_comparison(df: pd.DataFrame):
    st.subheader("Prompt comparison")

    if df.empty:
        st.info("No prompts available for comparison.")
        return

    options = df["display_name"].tolist()

    selected = st.multiselect(
        "Select prompts to compare",
        options=options,
        default=options[: min(3, len(options))],
    )

    if not selected:
        st.info("Select at least one prompt.")
        return

    compare_df = df[df["display_name"].isin(selected)].copy()

    metrics = [
        "performance",
        "cost",
        "risk",
        "fairness_risk",
        "dashboard_utility",
    ]

    compare_columns = [
        "display_name",
        "method",
        "category",
        *[m for m in metrics if m in compare_df.columns],
        "prompt",
    ]

    compare_columns = [column for column in compare_columns if column in compare_df.columns]

    st.dataframe(
        compare_df[compare_columns],
        use_container_width=True,
        hide_index=True,
    )

    long_df = compare_df.melt(
        id_vars=["display_name"],
        value_vars=[m for m in metrics if m in compare_df.columns],
        var_name="metric",
        value_name="value",
    )

    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("metric:N", title="Metric"),
            y=alt.Y("value:Q", title="Value"),
            color=alt.Color("display_name:N", title="Prompt"),
            tooltip=["display_name:N", "metric:N", "value:Q"],
        )
    )

    st.altair_chart(chart, use_container_width=True)


def render_fairness_analysis(df: pd.DataFrame):
    st.subheader("Fairness analysis")

    if df.empty:
        st.info("No fairness data available.")
        return

    fairness_sorted = df.sort_values(
        by=["fairness_risk", "risk", "cost"],
        ascending=[True, True, True],
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Lowest fairness-risk prompts**")
        columns = [
            column
            for column in [
                "method",
                "category",
                "fairness_risk",
                "detail_counterfactual_flip_rate",
                "detail_num_flips",
                "risk",
                "performance",
                "cost",
                "is_pareto",
                "prompt",
            ]
            if column in fairness_sorted.columns
        ]

        st.dataframe(
            fairness_sorted[columns].head(10),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.markdown("**Fairness-risk distribution**")

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("fairness_risk:Q", bin=True, title="Fairness risk"),
                y=alt.Y("count():Q", title="Number of prompts"),
                color=alt.Color("is_pareto:N", title="Pareto"),
                tooltip=["count():Q"],
            )
        )

        st.altair_chart(chart, use_container_width=True)

def render_budget_summary_cards(budget_summary: dict):
    st.subheader("Budget summary")

    if not budget_summary:
        st.info("No budget summary found.")
        return

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Max budget", f"{float(budget_summary.get('max_budget', 0.0)):.2f}")
    col2.metric("Used budget", f"{float(budget_summary.get('used_budget', 0.0)):.2f}")
    col3.metric("Remaining", f"{float(budget_summary.get('remaining_budget', 0.0)):.2f}")
    col4.metric("Utilization", f"{float(budget_summary.get('utilization', 0.0)):.2%}")
    col5.metric("Records", int(budget_summary.get("num_records", 0)))
    col6.metric("Blocks", int(budget_summary.get("num_blocks", 0)))

    col7, col8, col9, col10, col11, col12 = st.columns(6)
    col7.metric("Input tokens", int(float(budget_summary.get("input_tokens", 0.0))))
    col8.metric("Output tokens", int(float(budget_summary.get("output_tokens", 0.0))))
    col9.metric("Total tokens", int(float(budget_summary.get("total_tokens", 0.0))))
    col10.metric("Population", int(budget_summary.get("population_size", 0)))
    col11.metric("Incumbents", int(budget_summary.get("num_incumbents", 0)))
    col12.metric(
        "Evaluated candidates",
        int(budget_summary.get("num_evaluated_candidates", 0)),
    )

    algorithm = budget_summary.get("algorithm", "unknown")
    evaluator = budget_summary.get("evaluator", "unknown")
    model_id = budget_summary.get("model_id", "unknown")

    st.caption(
        f"Algorithm: {algorithm} | Evaluator: {evaluator} | Model: {model_id}"
    )


def render_budget_events(events_df: pd.DataFrame):
    st.subheader("Evolutionary MO-CAPO events")

    st.caption(
        "Events include initial population evaluation, evolutionary intensification, "
        "environmental selection, and incumbent advancement."
    )

    if events_df.empty:
        st.info("No budget/evolutionary events found.")
        return

    summary = budget_events_summary(events_df)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Events", summary["num_events"])
    col2.metric("Accepted", summary["num_accepted"])
    col3.metric("Rejected", summary["num_rejected"])
    col4.metric("Initial pop.", summary["num_initial_population"])
    col5.metric("Evolutionary", summary["num_evolutionary_intensification"])
    col6.metric("Advance inc.", summary["num_advance_incumbent"])

    col7, col8, col9 = st.columns(3)
    col7.metric("Env. selections", summary["num_environmental_selection"])
    col8.metric("Budget stops", summary["num_budget_stops"])
    col9.metric("Budget errors", summary["num_budget_errors"])

    event_types = sorted(events_df["event_type"].dropna().astype(str).unique().tolist())

    selected_event_types = st.multiselect(
        "Event types",
        options=event_types,
        default=event_types,
    )

    filtered_events = events_df.copy()

    if selected_event_types:
        filtered_events = filtered_events[
            filtered_events["event_type"].isin(selected_event_types)
        ]

    event_chart = (
        alt.Chart(filtered_events)
        .mark_bar()
        .encode(
            x=alt.X("event_type:N", title="Event type"),
            y=alt.Y("count():Q", title="Count"),
            color=alt.Color("accepted:N", title="Accepted"),
            tooltip=["event_type:N", "accepted:N", "rejected:N", "count():Q"],
        )
    )

    st.altair_chart(event_chart, use_container_width=True)

    budget_line = (
        alt.Chart(filtered_events.reset_index().rename(columns={"index": "step"}))
        .mark_line(point=True)
        .encode(
            x=alt.X("step:Q", title="Event step"),
            y=alt.Y("budget_used:Q", title="Budget used"),
            color=alt.Color("event_type:N", title="Event type"),
            tooltip=[
                "step:Q",
                "iteration:Q",
                "method:N",
                "event_type:N",
                "operator:N",
                "budget_used:Q",
                "remaining_budget:Q",
                "reason:N",
            ],
        )
    )

    st.altair_chart(budget_line, use_container_width=True)

    display_columns = [
        "event_type",
        "iteration",
        "offspring_index",
        "method",
        "operator",
        "parent_ids",
        "accepted",
        "rejected",
        "evaluated_blocks",
        "block_id",
        "budget_used",
        "remaining_budget",
        "budget_utilization",
        "kept_ids",
        "removed_ids",
        "reason",
    ]

    available_columns = [
        column for column in display_columns if column in filtered_events.columns
    ]

    st.dataframe(
        filtered_events[available_columns],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Raw event metadata"):
        metadata_columns = [
            column
            for column in ["metadata", "parent_selection"]
            if column in filtered_events.columns
        ]

        if metadata_columns:
            st.dataframe(
                filtered_events[["event_type", "method", *metadata_columns]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No raw metadata columns available.")


def render_budget_analysis_tab(config: dict):
    data_cfg = config.get("data", {})

    events_path = data_cfg.get(
        "budgeted_mocapo_events_csv",
        "outputs/phase2_budgeted_mocapo_subj/budgeted_mocapo_events.csv",
    )
    summary_path = data_cfg.get(
        "budgeted_mocapo_budget_summary_json",
        "outputs/phase2_budgeted_mocapo_subj/budget_summary.json",
    )

    try:
        budget_summary = load_budget_summary(summary_path)
    except Exception as exc:
        st.warning(f"Could not load budget summary: {exc}")
        budget_summary = {}

    try:
        events_df = load_budget_events(events_path)
    except Exception as exc:
        st.warning(f"Could not load budget events: {exc}")
        events_df = pd.DataFrame()

    render_budget_summary_cards(budget_summary)

    if cfg_get(config, "display", "show_intensification_events", True):
        render_budget_events(events_df)

def render_overview_tab(
    filtered_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    config: dict,
):
    render_summary_cards(filtered_df)
    render_top_recommendation(ranked_df, config=config)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        render_accuracy_cost_chart(ranked_df)

    with chart_col2:
        render_risk_fairness_chart(ranked_df)

    if cfg_get(config, "display", "show_downloads", True):
        render_download_buttons(ranked_df, recommendations_df)


def render_all_candidates_tab(ranked_df: pd.DataFrame):
    render_prompt_table(ranked_df)


def render_pareto_tab(ranked_df: pd.DataFrame):
    pareto_df = ranked_df[ranked_df["is_pareto"] == True].copy()
    render_prompt_table(pareto_df)


def render_recommendations_tab(recommendations_df: pd.DataFrame):
    render_recommendations_cards(recommendations_df)


def main():
    config = load_dashboard_config(DEFAULT_CONFIG_PATH)

    render_header(config)
    settings = render_sidebar(config)

    sidebar_config_path = settings["config_path"]
    if sidebar_config_path != DEFAULT_CONFIG_PATH:
        config = load_dashboard_config(sidebar_config_path)

    try:
        portfolio_df = load_portfolio(settings["portfolio_path"])
    except Exception as exc:
        st.error(f"Could not load portfolio CSV: {exc}")
        return

    try:
        recommendations_df = load_recommendations(settings["recommendations_path"])
    except Exception:
        recommendations_df = pd.DataFrame()

    methods = sorted(
        [
            str(method)
            for method in portfolio_df["method"].dropna().unique().tolist()
        ]
    )

    selected_methods = st.sidebar.multiselect(
        "Methods",
        options=methods,
        default=methods,
    )

    selected_categories = None
    if "category" in portfolio_df.columns:
        categories = sorted(
            [
                str(category)
                for category in portfolio_df["category"].dropna().unique().tolist()
            ]
        )
        selected_categories = st.sidebar.multiselect(
            "Categories",
            options=categories,
            default=categories,
        )

    max_cost = None
    if not portfolio_df.empty:
        cost_max_value = float(portfolio_df["cost"].max())
        configured_max_cost = cfg_get(config, "defaults", "max_cost", None)

        if configured_max_cost is None:
            default_max_cost = max(1.0, cost_max_value)
        else:
            default_max_cost = min(float(configured_max_cost), max(1.0, cost_max_value))

        max_cost = st.sidebar.slider(
            "Maximum cost",
            min_value=0.0,
            max_value=max(1.0, cost_max_value),
            value=default_max_cost,
            step=0.1,
        )

    filtered_df = filter_portfolio(
        portfolio_df,
        show_only_pareto=settings["show_only_pareto"],
        method_filter=selected_methods,
        min_performance=settings["min_performance"],
        max_cost=max_cost,
        max_risk=settings["max_risk"],
        max_fairness_risk=settings["max_fairness_risk"],
    )

    if selected_categories and "category" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]

    ranked_df = compute_weighted_utility(
        filtered_df,
        accuracy_weight=settings["accuracy_weight"],
        cost_weight=settings["cost_weight"],
        risk_weight=settings["risk_weight"],
        fairness_weight=settings["fairness_weight"],
    )

    tabs = config.get(
        "tabs",
        [
            "Overview",
            "All Candidates",
            "Pareto Portfolio",
            "Recommendations",
            "Prompt Comparison",
            "Fairness Analysis",
        ],
    )

    tab_objects = st.tabs(tabs)

    for tab_name, tab in zip(tabs, tab_objects):
        with tab:
            if tab_name == "Overview":
                render_overview_tab(
                    filtered_df=filtered_df,
                    ranked_df=ranked_df,
                    recommendations_df=recommendations_df,
                    config=config,
                )

            elif tab_name == "All Candidates":
                render_all_candidates_tab(ranked_df)

            elif tab_name == "Pareto Portfolio":
                render_pareto_tab(ranked_df)

            elif tab_name == "Recommendations":
                if cfg_get(config, "display", "show_recommendation_cards", True):
                    render_recommendations_tab(recommendations_df)
                else:
                    st.dataframe(recommendations_df, use_container_width=True)

            elif tab_name == "Prompt Comparison":
                if cfg_get(config, "display", "show_prompt_comparison", True):
                    render_prompt_comparison(ranked_df)
                else:
                    st.info("Prompt comparison is disabled in dashboard config.")

            elif tab_name == "Fairness Analysis":
                if cfg_get(config, "display", "show_fairness_analysis", True):
                    render_fairness_analysis(ranked_df)
                else:
                    st.info("Fairness analysis is disabled in dashboard config.")

            elif tab_name == "Budget Analysis":
                if cfg_get(config, "display", "show_budget_analysis", True):
                    render_budget_analysis_tab(config)
                else:
                    st.info("Budget analysis is disabled in dashboard config.")

            else:
                st.info(f"No renderer registered for tab: {tab_name}")


if __name__ == "__main__":
    main()