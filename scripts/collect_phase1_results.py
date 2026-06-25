# from __future__ import annotations
#
# import argparse
# import csv
# from pathlib import Path
#
#
# DEFAULT_INPUT_TABLES = [
#     "outputs/phase1_subj_lmstudio_mistral32_30/baseline_table.csv",
#     "outputs/phase1_agnews_lmstudio_mistral32_30/baseline_table.csv",
#     "outputs/phase1_subj_lmstudio_mistral32_all_optimizers_fast/baseline_table.csv",
# ]
#
#
# def read_csv(path: Path) -> list[dict]:
#     with open(path, "r", encoding="utf-8", newline="") as f:
#         return list(csv.DictReader(f))
#
#
# def write_csv(rows: list[dict], path: Path):
#     path.parent.mkdir(parents=True, exist_ok=True)
#
#     # Keep a stable, readable column order.
#     preferred_columns = [
#         "dataset",
#         "task_type",
#         "llm_backend",
#         "method",
#         "dev_size",
#         "shots_size",
#         "test_size",
#         "seed",
#         "dev_score",
#         "test_score",
#         "dev_cost",
#         "test_cost",
#         "dev_input_tokens",
#         "dev_output_tokens",
#         "test_input_tokens",
#         "test_output_tokens",
#         "prompt",
#         "source_file",
#     ]
#
#     all_columns = set()
#     for row in rows:
#         all_columns.update(row.keys())
#
#     columns = [col for col in preferred_columns if col in all_columns]
#     columns += sorted(col for col in all_columns if col not in columns)
#
#     with open(path, "w", encoding="utf-8", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=columns)
#         writer.writeheader()
#         writer.writerows(rows)
#
#
# def to_float(value, default: float = 0.0) -> float:
#     try:
#         return float(value)
#     except (TypeError, ValueError):
#         return default
#
#
# def add_summary_fields(row: dict) -> dict:
#     row = dict(row)
#
#     dev_score = to_float(row.get("dev_score"))
#     test_score = to_float(row.get("test_score"))
#     dev_cost = to_float(row.get("dev_cost"))
#     test_cost = to_float(row.get("test_cost"))
#
#     row["score_gap_test_minus_dev"] = test_score - dev_score
#     row["dev_score_per_cost"] = dev_score / dev_cost if dev_cost > 0 else ""
#     row["test_score_per_cost"] = test_score / test_cost if test_cost > 0 else ""
#
#     return row
#
#
# def collect_tables(input_paths: list[str]) -> list[dict]:
#     rows = []
#
#     for input_path in input_paths:
#         path = Path(input_path)
#
#         if not path.exists():
#             print(f"Skipping missing file: {path}")
#             continue
#
#         table_rows = read_csv(path)
#
#         for row in table_rows:
#             row["source_file"] = str(path)
#             row = add_summary_fields(row)
#             rows.append(row)
#
#     return rows
#
#
# def print_summary(rows: list[dict]):
#     if not rows:
#         print("No rows collected.")
#         return
#
#     print("Collected Phase 1 rows:")
#     print("-" * 80)
#
#     for row in rows:
#         print(
#             f"{row.get('dataset')} | "
#             f"{row.get('method')} | "
#             f"dev_score={row.get('dev_score')} | "
#             f"test_score={row.get('test_score')} | "
#             f"dev_cost={row.get('dev_cost')} | "
#             f"test_cost={row.get('test_cost')}"
#         )
#
#     print("-" * 80)
#     print(f"Total rows: {len(rows)}")
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--inputs",
#         nargs="*",
#         default=DEFAULT_INPUT_TABLES,
#         help="Input baseline_table.csv files to combine.",
#     )
#     parser.add_argument(
#         "--output",
#         default="outputs/phase1_summary/final_phase1_baseline_table.csv",
#         help="Output CSV path.",
#     )
#     args = parser.parse_args()
#
#     rows = collect_tables(args.inputs)
#
#     if not rows:
#         raise RuntimeError(
#             "No rows collected. Check that the input CSV paths exist."
#         )
#
#     output_path = Path(args.output)
#     write_csv(rows, output_path)
#
#     print_summary(rows)
#     print(f"Saved combined table to: {output_path}")
#
#
# if __name__ == "__main__":
#     main()


import argparse
import csv
from pathlib import Path


DEFAULT_INPUT_TABLES = [
    "outputs/phase1_subj_lmstudio_mistral32_all_optimizers_fast/baseline_table.csv",
    "outputs/phase1_subj_lmstudio_mistral32_mocapo_style_30/baseline_table.csv",
]


def _to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except ValueError:
        return default


def _row_key(row: dict) -> tuple:
    return (
        row.get("dataset"),
        row.get("task_type"),
        row.get("llm_backend"),
        row.get("method"),
        row.get("dev_size"),
        row.get("shots_size"),
        row.get("test_size"),
        row.get("seed"),
        row.get("prompt"),
    )


def _add_derived_metrics(row: dict) -> dict:
    dev_score = _to_float(row.get("dev_score"))
    test_score = _to_float(row.get("test_score"))
    dev_cost = _to_float(row.get("dev_cost"))
    test_cost = _to_float(row.get("test_cost"))

    row["source_file"] = row.get("source_file", "")

    row["dev_score_per_cost"] = dev_score / dev_cost if dev_cost > 0 else ""
    row["test_score_per_cost"] = test_score / test_cost if test_cost > 0 else ""
    row["score_gap_test_minus_dev"] = test_score - dev_score

    return row


def collect_rows(input_tables: list[str]) -> list[dict]:
    rows = []
    seen = set()

    for table in input_tables:
        path = Path(table)

        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                row["source_file"] = str(path)

                key = _row_key(row)
                if key in seen:
                    continue

                seen.add(key)
                rows.append(_add_derived_metrics(row))

    return rows


def save_rows(rows: list[dict], output_path: str):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows found. Check DEFAULT_INPUT_TABLES or --inputs.")

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]):
    print("Collected Phase 1 rows:")
    print("-" * 80)

    for row in rows:
        print(
            f"{row.get('dataset')} | "
            f"{row.get('method')} | "
            f"dev_score={row.get('dev_score')} | "
            f"test_score={row.get('test_score')} | "
            f"dev_cost={row.get('dev_cost')} | "
            f"test_cost={row.get('test_cost')}"
        )

    print("-" * 80)
    print(f"Total rows: {len(rows)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=DEFAULT_INPUT_TABLES,
        help="Input baseline_table.csv files to combine.",
    )
    parser.add_argument(
        "--output",
        default="outputs/phase1_summary/final_phase1_subj_complete.csv",
        help="Output combined CSV path.",
    )
    args = parser.parse_args()

    rows = collect_rows(args.inputs)
    print_summary(rows)
    save_rows(rows, args.output)
    print(f"Saved combined table to: {args.output}")


if __name__ == "__main__":
    main()