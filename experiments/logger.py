import csv
import json
from pathlib import Path
from typing import Any


def _json_default(obj: Any):
    """
    Fallback serializer for objects that json cannot handle directly.
    Useful for Promptolution Prompt objects, numpy values, etc.
    """
    try:
        return obj.item()
    except AttributeError:
        return str(obj)


def save_json(result: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=_json_default)


def _flatten_result(result: dict) -> dict:
    """
    Convert one experiment result into a stable CSV row.
    Keep this table simple and readable.
    Detailed predictions/raw Promptolution objects stay in JSON files.
    """
    return {
        "dataset": result.get("dataset"),
        "task_type": result.get("task_type"),
        "llm_backend": result.get("llm_backend"),
        "method": result.get("method"),
        "dev_size": result.get("dev_size"),
        "shots_size": result.get("shots_size"),
        "test_size": result.get("test_size"),
        "seed": result.get("seed"),

        "dev_score": result.get("dev_score", result.get("score")),
        "test_score": result.get("test_score"),
        "dev_cost": result.get("dev_cost", result.get("cost")),
        "test_cost": result.get("test_cost"),

        "dev_input_tokens": result.get("dev_input_tokens", result.get("input_tokens")),
        "dev_output_tokens": result.get("dev_output_tokens", result.get("output_tokens")),
        "test_input_tokens": result.get("test_input_tokens"),
        "test_output_tokens": result.get("test_output_tokens"),

        "candidate_id": result.get("candidate_id"),
        "pareto_rank": result.get("pareto_rank"),
        "num_candidates_evaluated": result.get("num_candidates_evaluated"),

        "prompt": result.get("prompt"),
    }


def _read_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_key(row: dict) -> tuple:
    """
    Use a practical key to prevent duplicate rows when rerunning
    the same experiment into the same output directory.
    """
    return (
        row.get("dataset"),
        row.get("llm_backend"),
        row.get("method"),
        row.get("dev_size"),
        row.get("test_size"),
        row.get("seed"),
        row.get("prompt"),
    )


def save_csv_row(result: dict, path: str):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    flat = _flatten_result(result)

    existing_rows = _read_existing_rows(path_obj)
    new_key = _row_key(flat)

    # Replace existing duplicate row instead of appending another copy.
    rows = [row for row in existing_rows if _row_key(row) != new_key]
    rows.append(flat)

    fieldnames = list(flat.keys())

    with open(path_obj, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)