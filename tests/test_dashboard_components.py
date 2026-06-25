import pandas as pd

from dashboard.components import (
    clean_portfolio_dataframe,
    compute_weighted_utility,
    filter_portfolio,
    get_top_prompt,
    portfolio_summary,
)


def _sample_df():
    return pd.DataFrame(
        [
            {
                "candidate_id": "a123",
                "method": "initial",
                "performance": 0.8,
                "cost": 10.0,
                "risk": 0.2,
                "fairness_risk": 0.1,
                "is_pareto": True,
                "prompt": "Prompt A",
            },
            {
                "candidate_id": "b123",
                "method": "cheap",
                "performance": 0.6,
                "cost": 1.0,
                "risk": 0.4,
                "fairness_risk": 0.3,
                "is_pareto": False,
                "prompt": "Prompt B",
            },
        ]
    )


def test_clean_portfolio_dataframe_adds_counterfactual_columns():
    df = clean_portfolio_dataframe(
        pd.DataFrame(
            [
                {
                    "candidate_id": "c123",
                    "method": "fairness",
                    "performance": 0.7,
                    "cost": 5.0,
                    "risk": 0.2,
                    "fairness_risk": 0.05,
                    "is_pareto": True,
                    "prompt": "Prompt C",
                    "detail_counterfactual_flip_rate": 0.05,
                    "detail_num_flips": 1,
                    "detail_num_pairs": 20,
                }
            ]
        )
    )

    assert "counterfactual_flip_rate" in df.columns
    assert "num_fairness_flips" in df.columns
    assert "num_fairness_pairs" in df.columns
    assert df.iloc[0]["counterfactual_flip_rate"] == 0.05
    assert df.iloc[0]["num_fairness_flips"] == 1
    assert df.iloc[0]["num_fairness_pairs"] == 20

def test_filter_portfolio_only_pareto():
    df = clean_portfolio_dataframe(_sample_df())

    filtered = filter_portfolio(df, show_only_pareto=True)

    assert len(filtered) == 1
    assert filtered.iloc[0]["candidate_id"] == "a123"


def test_compute_weighted_utility():
    df = clean_portfolio_dataframe(_sample_df())

    ranked = compute_weighted_utility(
        df,
        accuracy_weight=1.0,
        cost_weight=0.0,
        risk_weight=0.0,
        fairness_weight=0.0,
    )

    assert ranked.iloc[0]["candidate_id"] == "a123"
    assert "dashboard_utility" in ranked.columns


def test_portfolio_summary():
    df = clean_portfolio_dataframe(_sample_df())

    summary = portfolio_summary(df)

    assert summary["num_prompts"] == 2
    assert summary["num_pareto"] == 1
    assert summary["best_performance"] == 0.8
    assert summary["lowest_cost"] == 1.0
    assert summary["lowest_risk"] == 0.2
    assert summary["lowest_fairness_risk"] == 0.1


def test_get_top_prompt():
    df = clean_portfolio_dataframe(_sample_df())
    ranked = compute_weighted_utility(df)

    top = get_top_prompt(ranked)

    assert top is not None
    assert "prompt" in top
    assert "utility" in top