from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = [
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

WORKCLASS_GLOSS = {
    "Private": "private-sector employee",
    "Self-emp-not-inc": "self-employed, not incorporated",
    "Self-emp-inc": "self-employed, incorporated",
    "Federal-gov": "federal government employee",
    "Local-gov": "local government employee",
    "State-gov": "state government employee",
    "Without-pay": "unpaid worker",
    "Never-worked": "never worked",
}

OCCUPATION_GLOSS = {
    "Exec-managerial": "executive or managerial occupation",
    "Prof-specialty": "professional specialty occupation",
    "Tech-support": "technical support occupation",
    "Sales": "sales occupation",
    "Adm-clerical": "administrative or clerical occupation",
    "Craft-repair": "craft or repair occupation",
    "Machine-op-inspct": "machine operator or inspector",
    "Transport-moving": "transportation or material-moving occupation",
    "Handlers-cleaners": "handler or cleaner occupation",
    "Farming-fishing": "farming or fishing occupation",
    "Protective-serv": "protective service occupation",
    "Other-service": "other service occupation",
    "Priv-house-serv": "private household service",
    "Armed-Forces": "armed forces occupation",
}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _age_band(age: int) -> str:
    if age < 25:
        return "early-career age"
    if age < 35:
        return "young-adult working age"
    if age < 50:
        return "mid-career age"
    if age < 65:
        return "late-career age"
    return "older working age"


def _hours_band(hours: int) -> str:
    if hours < 30:
        return "part-time hours"
    if hours <= 40:
        return "standard full-time hours"
    if hours <= 50:
        return "extended full-time hours"
    return "long working hours"


def _money_text(value: Any, kind: str) -> str:
    amount = _int(value)
    if amount <= 0:
        return f"0 (no recorded {kind})"
    if amount < 3000:
        band = "modest"
    elif amount < 8000:
        band = "substantial"
    else:
        band = "very high"
    return f"{amount} ({band} recorded {kind})"


def enrich_row(row: dict[str, str]) -> dict[str, str]:
    """Add target-independent semantic descriptions to non-protected fields."""
    out = dict(row)
    age = _int(row.get("age"))
    hours = _int(row.get("hours.per.week"))
    education_num = _int(row.get("education.num"))

    education = str(row.get("education", "")).strip() or "Unknown"
    out["age"] = f"{age} ({_age_band(age)})"
    out["education"] = (
        f"{education} (ordinal education level {education_num} of 16)"
    )
    out["hours.per.week"] = f"{hours} ({_hours_band(hours)})"
    out["capital.gain"] = _money_text(row.get("capital.gain"), "capital gain")
    out["capital.loss"] = _money_text(row.get("capital.loss"), "capital loss")

    workclass = str(row.get("workclass", "")).strip()
    if workclass in WORKCLASS_GLOSS:
        out["workclass"] = f"{workclass} ({WORKCLASS_GLOSS[workclass]})"

    occupation = str(row.get("occupation", "")).strip()
    if occupation in OCCUPATION_GLOSS:
        out["occupation"] = f"{occupation} ({OCCUPATION_GLOSS[occupation]})"

    # Protected columns stay untouched for metadata-only fairness evaluation.
    # The Adult loader deliberately excludes sex and race from rendered prompt text.
    return out


def validate(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Adult CSV contains no data rows.")

    labels = {
        str(row.get("income", "")).strip().replace(".", "")
        for row in rows
    }
    if not labels.issubset({"<=50K", ">50K"}):
        raise ValueError(f"Unexpected Adult labels: {sorted(labels)}")

    for row in rows[:100]:
        for protected in ("sex", "race"):
            if protected not in row:
                raise ValueError(f"Missing protected metadata column: {protected}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a semantically enriched Adult CSV while preserving labels "
            "and protected metadata."
        )
    )
    parser.add_argument("--input", default="data/adult.csv")
    parser.add_argument("--output", default="data/adult_semantic_v3.csv")
    args = parser.parse_args()

    source = Path(args.input)
    target = Path(args.output)
    if not source.exists():
        raise FileNotFoundError(source)

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(f"Adult CSV is missing required columns: {missing}")
        raw_rows = [dict(row) for row in reader]

    validate(raw_rows)
    enriched = [enrich_row(row) for row in raw_rows]

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(enriched)

    print(f"Wrote {len(enriched)} rows to {target}")
    print(
        "Protected attributes remain only in sex/race columns; "
        "the Adult loader omits them from prompt text."
    )
    print("Example enriched non-protected fields:")
    sample = enriched[0]
    for key in (
        "age",
        "workclass",
        "education",
        "occupation",
        "capital.gain",
        "capital.loss",
        "hours.per.week",
    ):
        print(f"  {key}: {sample.get(key)}")


if __name__ == "__main__":
    main()
