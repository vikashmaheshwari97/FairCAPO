from __future__ import annotations

from scripts.build_experiment_table import (
    TABLE_COLUMNS,
    best_per_objective,
    build_experiment_table,
    empty_row,
    filter_pareto_rows,
    make_latex_table,
    pareto_set_row,
    single_point_row,
)


def write_single_point_csv(path):
    path.write_text(
        "dataset,method,dev_score,test_score,dev_cost,test_cost,prompt\n"
        "subj,initial_prompt,0.7,0.8,55.0,29.0,Classify the sentence.\n"
        "subj,promptolution_capo,0.7,0.85,55.0,30.0,Classify carefully.\n",
        encoding="utf-8",
    )


def write_pareto_csv(path):
    # Two non-dominated points and one dominated point.
    path.write_text(
        "candidate_id,is_pareto,method,performance,cost,risk,fairness_risk\n"
        "a,True,m_a,0.90,4.0,0.10,0.30\n"
        "b,True,m_b,0.80,2.0,0.05,0.12\n"
        "c,False,m_c,0.50,9.0,0.50,0.50\n",
        encoding="utf-8",
    )


def base_table_config() -> dict:
    return {
        "dataset": "SUBJ",
        "model": "test-model",
        "objectives": [
            {"name": "performance", "direction": "maximize"},
            {"name": "cost", "direction": "minimize"},
            {"name": "risk", "direction": "minimize"},
            {"name": "fairness_risk", "direction": "minimize"},
        ],
        "bounds": {
            "performance": [0.0, 1.0],
            "cost": [0.0, 50.0],
            "risk": [0.0, 1.0],
            "fairness_risk": [0.0, 1.0],
        },
        "metrics": {"num_preference_vectors": 8, "seed": 0},
    }


def test_single_point_row_reads_test_score(tmp_path):
    csv_path = tmp_path / "baseline.csv"
    write_single_point_csv(csv_path)

    row = single_point_row(
        {"name": "CAPO", "csv": str(csv_path), "match": "promptolution_capo"}
    )

    assert row["method"] == "CAPO"
    assert row["performance"] == 0.85
    assert row["cost"] == 30.0
    assert row["risk"] == 0.0
    assert row["fairness_risk"] == 0.0
    assert row["portfolio_size"] == 1
    # Single point => no Pareto-set metrics.
    assert row["hypervolume"] is None
    assert row["nr2"] is None


def test_single_point_row_missing_file_returns_blank():
    row = single_point_row(
        {"name": "Ghost", "csv": "does/not/exist.csv", "match": "whatever"}
    )

    assert row["method"] == "Ghost"
    assert row["performance"] is None
    assert row["portfolio_size"] == 0


def test_single_point_row_missing_match_returns_blank(tmp_path):
    csv_path = tmp_path / "baseline.csv"
    write_single_point_csv(csv_path)

    row = single_point_row(
        {"name": "Nope", "csv": str(csv_path), "match": "nonexistent_method"}
    )

    assert row["performance"] is None
    assert row["portfolio_size"] == 0


def test_filter_pareto_rows_keeps_only_pareto():
    rows = [
        {"is_pareto": "True", "method": "a"},
        {"is_pareto": "False", "method": "b"},
    ]
    filtered = filter_pareto_rows(rows, only_pareto=True)
    assert len(filtered) == 1
    assert filtered[0]["method"] == "a"


def test_filter_pareto_rows_no_column_returns_all():
    rows = [{"method": "a"}, {"method": "b"}]
    filtered = filter_pareto_rows(rows, only_pareto=True)
    assert len(filtered) == 2


def test_best_per_objective():
    from heal_capo.core import EvaluationResult

    results = [
        EvaluationResult("a", performance=0.9, cost=4.0, risk=0.1, fairness_risk=0.3),
        EvaluationResult("b", performance=0.8, cost=2.0, risk=0.05, fairness_risk=0.12),
    ]

    best = best_per_objective(results)

    assert best["performance"] == 0.9
    assert best["cost"] == 2.0
    assert best["risk"] == 0.05
    assert best["fairness_risk"] == 0.12


def test_pareto_set_row_computes_best_and_metrics(tmp_path):
    csv_path = tmp_path / "candidates.csv"
    write_pareto_csv(csv_path)

    row = pareto_set_row(
        {
            "name": "HEAL-CAPO",
            "candidates_csv": str(csv_path),
            "only_pareto": True,
        },
        base_table_config(),
    )

    assert row["method"] == "HEAL-CAPO"
    # Best-per-objective over the 2 Pareto rows (dominated row 'c' filtered out).
    assert row["performance"] == 0.90
    assert row["cost"] == 2.0
    assert row["risk"] == 0.05
    assert row["fairness_risk"] == 0.12
    assert row["portfolio_size"] == 2
    assert row["hypervolume"] is not None
    assert row["nr2"] is not None
    assert 0.0 <= float(row["hypervolume"]) <= 1.0


def test_pareto_set_row_with_budget_json(tmp_path):
    csv_path = tmp_path / "candidates.csv"
    write_pareto_csv(csv_path)

    budget_path = tmp_path / "budget.json"
    budget_path.write_text('{"used_budget": 1234.5}', encoding="utf-8")

    row = pareto_set_row(
        {
            "name": "HEAL-CAPO",
            "candidates_csv": str(csv_path),
            "budget_json": str(budget_path),
        },
        base_table_config(),
    )

    assert row["budget_used"] == 1234.5


def test_pareto_set_row_missing_file_blank():
    row = pareto_set_row(
        {"name": "Ghost", "candidates_csv": "nope.csv"},
        base_table_config(),
    )
    assert row["method"] == "Ghost"
    assert row["portfolio_size"] == 0


def test_build_experiment_table_full(tmp_path):
    single_csv = tmp_path / "baseline.csv"
    write_single_point_csv(single_csv)

    pareto_csv = tmp_path / "candidates.csv"
    write_pareto_csv(pareto_csv)

    config = base_table_config()
    config["methods"] = [
        {
            "name": "Initial",
            "type": "single_point",
            "csv": str(single_csv),
            "match": "initial_prompt",
        },
        {
            "name": "HEAL-CAPO",
            "type": "pareto_set",
            "candidates_csv": str(pareto_csv),
        },
        {
            "name": "Ghost",
            "type": "single_point",
            "csv": "missing.csv",
            "match": "x",
        },
    ]

    rows = build_experiment_table(config)

    assert len(rows) == 3
    # Column ordering enforced.
    for row in rows:
        assert list(row.keys()) == TABLE_COLUMNS

    by_method = {row["method"]: row for row in rows}
    assert by_method["Initial"]["performance"] == 0.8
    assert by_method["HEAL-CAPO"]["portfolio_size"] == 2
    assert by_method["HEAL-CAPO"]["hypervolume"] is not None
    assert by_method["Ghost"]["performance"] is None


def test_make_latex_table_renders(tmp_path):
    pareto_csv = tmp_path / "candidates.csv"
    write_pareto_csv(pareto_csv)

    config = base_table_config()
    config["methods"] = [
        {"name": "HEAL-CAPO", "type": "pareto_set", "candidates_csv": str(pareto_csv)},
    ]

    rows = build_experiment_table(config)
    latex = make_latex_table(rows, config)

    assert r"\begin{table}" in latex
    assert "HEAL-CAPO" in latex
    assert r"\end{table}" in latex


def test_empty_row_shape():
    row = empty_row("X")
    assert set(row.keys()) == set(TABLE_COLUMNS)
    assert row["method"] == "X"


def write_holdout_csv(path):
    # Held-out Dtest format: total `cost` folds the fairness audit in;
    # detail_fairness_eval_cost is separable; detail_total is the call count.
    path.write_text(
        "candidate_id,is_pareto,performance,cost,detail_fairness_eval_cost,detail_total\n"
        # cheapest inference: (300-100)/50 = 4.0 per call
        "a,True,0.95,300.0,100.0,50\n"
        # pricier: (600-200)/50 = 8.0 per call
        "b,True,0.97,600.0,200.0,50\n"
        # dominated row excluded by only_pareto
        "c,False,0.50,999.0,500.0,50\n",
        encoding="utf-8",
    )


def test_holdout_inference_cost_separates_audit(tmp_path):
    from scripts.build_experiment_table import holdout_inference_cost

    csv_path = tmp_path / "holdout.csv"
    write_holdout_csv(csv_path)
    r = holdout_inference_cost(str(csv_path))
    # Cheapest-to-deploy front member: row a, 4.0 per call, 200 total inference.
    assert r["inference_cost_per_call"] == 4.0
    assert r["inference_cost_total"] == 200.0
    # Audit cost is the mean over the (pareto) front, reported separately: (100+200)/2.
    assert r["fairness_eval_cost"] == 150.0


def test_holdout_inference_cost_missing_file_is_empty():
    from scripts.build_experiment_table import holdout_inference_cost

    assert holdout_inference_cost("does/not/exist.csv") == {}
