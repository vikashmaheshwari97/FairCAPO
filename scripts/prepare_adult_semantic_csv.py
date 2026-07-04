from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education.num",
    "marital.status", "occupation", "relationship", "race", "sex",
    "capital.gain", "capital.loss", "hours.per.week", "native.country",
    "income",
]
PROTECTED_COLUMNS = ("sex", "race")
TARGET_COLUMN = "income"

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    """Add deterministic, target-independent text to non-protected fields."""
    out = dict(row)
    age = _int(row.get("age"))
    hours = _int(row.get("hours.per.week"))
    education_num = _int(row.get("education.num"))

    education = str(row.get("education", "")).strip() or "Unknown"
    out["age"] = f"{age} ({_age_band(age)})"
    out["education"] = f"{education} (ordinal education level {education_num} of 16)"
    out["hours.per.week"] = f"{hours} ({_hours_band(hours)})"
    out["capital.gain"] = _money_text(row.get("capital.gain"), "capital gain")
    out["capital.loss"] = _money_text(row.get("capital.loss"), "capital loss")

    workclass = str(row.get("workclass", "")).strip()
    if workclass in WORKCLASS_GLOSS:
        out["workclass"] = f"{workclass} ({WORKCLASS_GLOSS[workclass]})"

    occupation = str(row.get("occupation", "")).strip()
    if occupation in OCCUPATION_GLOSS:
        out["occupation"] = f"{occupation} ({OCCUPATION_GLOSS[occupation]})"

    # Never transform protected attributes or the target. They remain byte-for-byte
    # values from the source row and are used only for metadata/evaluation.
    for column in (*PROTECTED_COLUMNS, TARGET_COLUMN):
        out[column] = row.get(column, "")
    return out


def validate_source(rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    if not rows:
        raise ValueError("Adult CSV contains no data rows.")
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(f"Adult CSV is missing required columns: {missing}")
    labels = {str(row.get(TARGET_COLUMN, "")).strip().replace(".", "") for row in rows}
    if not labels.issubset({"<=50K", ">50K"}):
        raise ValueError(f"Unexpected Adult labels: {sorted(labels)}")
    for index, row in enumerate(rows):
        for column in PROTECTED_COLUMNS:
            if column not in row:
                raise ValueError(f"Row {index} is missing protected metadata column {column!r}.")


def validate_transformation(raw_rows: list[dict[str, str]], enriched: list[dict[str, str]]) -> None:
    if len(raw_rows) != len(enriched):
        raise ValueError("Semantic enrichment changed the Adult row count.")
    for index, (source, target) in enumerate(zip(raw_rows, enriched)):
        for column in (*PROTECTED_COLUMNS, TARGET_COLUMN, "fnlwgt", "education.num"):
            if str(source.get(column, "")) != str(target.get(column, "")):
                raise ValueError(
                    f"Semantic enrichment changed protected/identity field {column!r} at row {index}."
                )


def atomic_write_csv(rows: list[dict[str, str]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic semantic Adult CSV with provenance metadata."
    )
    parser.add_argument("--input", default="data/adult.csv")
    parser.add_argument("--output", default="data/adult_semantic_v3.csv")
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Optional JSON provenance path; defaults to <output>.provenance.json.",
    )
    args = parser.parse_args()

    source = Path(args.input)
    target = Path(args.output)
    metadata_target = Path(args.metadata_output) if args.metadata_output else Path(f"{target}.provenance.json")
    if not source.is_file():
        raise FileNotFoundError(source)

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        raw_rows = [dict(row) for row in reader]

    validate_source(raw_rows, fieldnames)
    enriched = [enrich_row(row) for row in raw_rows]
    validate_transformation(raw_rows, enriched)
    atomic_write_csv(enriched, target)

    provenance = {
        "schema": "adult_semantic_csv_v3",
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "output_path": str(target),
        "output_sha256": sha256_file(target),
        "row_count": len(enriched),
        "columns": REQUIRED_COLUMNS,
        "protected_columns_unchanged": list(PROTECTED_COLUMNS),
        "target_column_unchanged": TARGET_COLUMN,
    }
    metadata_target.parent.mkdir(parents=True, exist_ok=True)
    metadata_target.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
