from __future__ import annotations

import csv

from experiments.datasets import (
    get_dataset_classes,
    get_task_description,
    load_adult_income,
    load_paper_dataset,
)


ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education.num",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
    "native.country",
    "income",
]


def _write_adult_csv(path):
    rows = []
    cells = [
        ("Male", "<=50K", "Craft-repair"),
        ("Male", ">50K", "Exec-managerial"),
        ("Female", "<=50K", "Adm-clerical"),
        ("Female", ">50K", "Prof-specialty"),
    ]
    for idx in range(4):
        for sex, income, occupation in cells:
            rows.append(
                {
                    "age": str(25 + idx),
                    "workclass": "Private",
                    "fnlwgt": str(100000 + idx),
                    "education": "Bachelors",
                    "education.num": "13",
                    "marital.status": "Never-married",
                    "occupation": occupation,
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": sex,
                    "capital.gain": "0",
                    "capital.loss": "0",
                    "hours.per.week": "40",
                    "native.country": "United-States",
                    "income": income,
                }
            )

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ADULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_adult_loader_renders_text_without_target_or_protected_columns(tmp_path):
    csv_path = tmp_path / "adult.csv"
    _write_adult_csv(csv_path)

    split = load_adult_income(
        data_path=str(csv_path),
        dev_size=4,
        shots_size=4,
        test_size=4,
        seed=0,
        stratified=True,
        stratify_group_key="sex",
    )

    assert split.classes == ["<=50K", ">50K"]
    assert split.task_type == "classification"
    assert len(split.dev) == 4
    assert len(split.shots) == 4
    assert len(split.test) == 4

    example = split.dev[0]
    assert "Person record:" in example.text
    assert "Occupation:" in example.text
    assert "income" not in example.text.lower()
    assert "fnlwgt" not in example.text.lower()
    assert "education.num" not in example.text.lower()
    assert "Sex:" not in example.text
    assert "Race:" not in example.text
    assert "Male" not in example.text
    assert "Female" not in example.text

    assert example.label in {"<=50K", ">50K"}
    assert example.metadata["sex"] in {"Male", "Female"}
    assert example.metadata["race"] == "White"
    assert example.metadata["group"] == example.metadata["sex"]


def test_adult_unified_loader_and_helpers(tmp_path):
    csv_path = tmp_path / "adult.csv"
    _write_adult_csv(csv_path)

    split = load_paper_dataset(
        "adult",
        data_path=str(csv_path),
        dev_size=4,
        shots_size=4,
        test_size=4,
        seed=1,
        stratified=True,
        stratify_group_key="sex",
    )

    assert split.name == "adult"
    assert get_dataset_classes("adult") == ["<=50K", ">50K"]
    assert "annual income" in get_task_description("adult")
