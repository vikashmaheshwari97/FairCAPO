"""
Measure REAL BBQ bias-score fairness for single-objective baselines.

Single-objective optimizers (GEPA, CAPO, EvoPrompt, OPRO) ignore fairness, so
their candidate CSVs carry a placeholder ``fairness_risk`` (e.g. GEPA's hardcoded
0.30). This script takes each baseline's final prompt(s), runs FairCAPO's
canonical BBQ bias-score evaluator on a BBQ fairness item set, and writes the
MEASURED ``fairness_risk`` (plus ``bbq_sAMB`` / ``bbq_sDIS``) back out — so the
experiment table reports the bias these methods actually exhibit on BBQ rather
than a placeholder.

Usage:
    PYTHONPATH=. python scripts/measure_baseline_fairness.py \
        --fairness-config configs/evaluate_pareto_bbq.yaml \
        --inputs outputs/baselines/gepa_bbq/gepa_candidate.csv \
                 outputs/baselines/capo_bbq/capo_candidates.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from heal_capo.core import PromptCandidate
from scripts.run_phase2_budgeted_mocapo import LLMObjectiveEvaluator


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _save_csv(rows: list[dict], path: str) -> None:
    if not rows:
        raise ValueError(f"No rows to save for {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_few_shot_examples(raw) -> list[dict]:
    if not raw:
        return []
    parsed = raw if isinstance(raw, list) else None
    if parsed is None:
        try:
            parsed = json.loads(str(raw))
        except (ValueError, TypeError):
            return []
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        if isinstance(item, dict) and ("input" in item or "output" in item):
            out.append({"input": str(item.get("input", "")), "output": str(item.get("output", ""))})
    return out


def measure_file(
    evaluator: LLMObjectiveEvaluator,
    path: str,
    prompt_column: str,
    output_suffix: str = "_bbqfair",
) -> str:
    rows = _load_csv(path)
    for row in rows:
        instruction = str(row.get(prompt_column) or row.get("instruction") or "").strip()
        if not instruction:
            continue
        examples = _parse_few_shot_examples(row.get("few_shot_examples"))
        candidate = PromptCandidate(instruction=instruction, examples=examples)
        risk, details, _ = evaluator._evaluate_candidate_fairness_bbq(candidate)
        row["fairness_risk"] = risk
        row["bbq_sAMB"] = details.get("bbq_sAMB")
        row["bbq_sDIS"] = details.get("bbq_sDIS")
        row["fairness_source"] = "bbq_bias_score_posthoc"

    suffix = output_suffix if output_suffix.startswith("_") else f"_{output_suffix}"
    out_path = str(Path(path).with_name(Path(path).stem + suffix + ".csv"))
    _save_csv(rows, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure real BBQ bias for baselines.")
    parser.add_argument(
        "--fairness-config",
        required=True,
        help="YAML providing llm/cost/fairness blocks (e.g. configs/evaluate_pareto_bbq.yaml). "
        "Its fairness block must enable the BBQ bias-score path.",
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Baseline candidate CSV path(s).")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument(
        "--output-suffix",
        default="_bbqfair",
        help=(
            "Suffix for measured CSVs. Use _bbqfair for standard held-out and "
            "_bbqfair_large for Stage A large-held-out diagnostics."
        ),
    )
    args = parser.parse_args()

    config = _load_yaml(args.fairness_config)
    # Force the BBQ bias-score in-loop path on for measurement.
    config.setdefault("fairness", {})
    config["fairness"]["in_loop"] = True
    config["fairness"].setdefault("mode", "bbq_bias_score")
    config.setdefault("dataset", "bbq")
    config.setdefault("task_type", "multiple_choice")

    evaluator = LLMObjectiveEvaluator(config)
    if not evaluator.fairness_in_loop or evaluator.fairness_mode != "bbq_bias_score":
        raise SystemExit(
            "BBQ fairness path is not active. Ensure --fairness-config has a "
            "fairness block with mode bbq_bias_score and a valid fairness_data file."
        )

    for path in args.inputs:
        out = measure_file(evaluator, path, args.prompt_column, args.output_suffix)
        print(f"Measured BBQ fairness for {path} -> {out}")


if __name__ == "__main__":
    main()
