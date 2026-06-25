"""
Build the in-loop BBQ fairness item set.

Samples a small, seeded, inspectable set of BBQ items (ambiguous + some
disambiguated) across the configured categories and writes them to a JSONL file.
Each line carries the rendered prompt ``text`` plus the metadata the bias scorer
needs (options, answer_info, stereotyped_groups, question_polarity,
context_condition, label_idx). This file is consumed in-loop by
``LLMObjectiveEvaluator`` (fairness.mode == "bbq_bias_score") and is committed so
runs are reproducible.

Usage:
    PYTHONPATH=. python scripts/build_bbq_fairness_set.py \
        --out data/fairness_bbq.jsonl --n-ambig 8 --n-disambig 4 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

# Allow running as a script (PYTHONPATH=. is the documented invocation, but be
# forgiving if it is not set).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.datasets import (  # noqa: E402
    BBQ_DEFAULT_CATEGORIES,
    _as_dict,
    _load_bbq_rows,
    render_bbq_text,
)


def _line_from_row(row: dict, category: str) -> dict:
    options = [
        str(row.get("ans0", "")),
        str(row.get("ans1", "")),
        str(row.get("ans2", "")),
    ]
    label_idx = int(row.get("label", 0))
    label_idx = label_idx if label_idx in (0, 1, 2) else 0
    additional = _as_dict(row.get("additional_metadata"))
    return {
        "text": render_bbq_text(row.get("context", ""), row.get("question", ""), options),
        "options": options,
        "answer_info": _as_dict(row.get("answer_info")),
        "stereotyped_groups": additional.get("stereotyped_groups", []),
        "question_polarity": str(row.get("question_polarity", "")),
        "context_condition": str(row.get("context_condition", "")),
        "label_idx": label_idx,
        "category": category,
        "example_id": row.get("example_id"),
    }


def _excluded_texts(exclude_path: str | None) -> set[str]:
    """Rendered ``text`` values to skip (keeps in-loop and held-out sets disjoint)."""
    if not exclude_path or not os.path.exists(exclude_path):
        return set()
    texts: set[str] = set()
    with open(exclude_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                texts.add(str(json.loads(line).get("text", "")))
            except (ValueError, TypeError):
                continue
    return texts


def build(
    categories: tuple[str, ...],
    n_ambig: int,
    n_disambig: int,
    seed: int,
    data_dir: str,
    exclude_path: str | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    rows = _load_bbq_rows(categories, data_dir)
    exclude = _excluded_texts(exclude_path)

    # Bucket by (category, context_condition), skipping excluded items.
    buckets: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        category = str(row.get("category", ""))
        condition = str(row.get("context_condition", "")).lower()
        cond_key = "ambig" if condition.startswith("ambig") else "disambig"
        if exclude:
            options = [str(row.get(f"ans{i}", "")) for i in range(3)]
            rendered = render_bbq_text(row.get("context", ""), row.get("question", ""), options)
            if rendered in exclude:
                continue
        buckets.setdefault((category, cond_key), []).append(row)

    selected: list[dict] = []
    for category in categories:
        for cond_key, n in (("ambig", n_ambig), ("disambig", n_disambig)):
            pool = list(buckets.get((category, cond_key), []))
            rng.shuffle(pool)
            for row in pool[:n]:
                selected.append(_line_from_row(row, category))

    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the in-loop BBQ fairness set.")
    parser.add_argument("--out", default=os.path.join("data", "fairness_bbq.jsonl"))
    parser.add_argument(
        "--categories",
        nargs="*",
        default=list(BBQ_DEFAULT_CATEGORIES),
    )
    parser.add_argument("--n-ambig", type=int, default=8, help="Ambiguous items per category.")
    parser.add_argument("--n-disambig", type=int, default=4, help="Disambiguated items per category.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=os.path.join("data", "bbq"))
    parser.add_argument(
        "--exclude",
        default=None,
        help="Path to a previously built JSONL whose items are skipped (keeps the "
        "held-out fairness set disjoint from the in-loop one).",
    )
    args = parser.parse_args()

    items = build(
        categories=tuple(args.categories),
        n_ambig=args.n_ambig,
        n_disambig=args.n_disambig,
        seed=args.seed,
        data_dir=args.data_dir,
        exclude_path=args.exclude,
    )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    n_amb = sum(1 for it in items if str(it["context_condition"]).lower().startswith("ambig"))
    print(
        f"Wrote {len(items)} BBQ fairness items "
        f"({n_amb} ambiguous, {len(items) - n_amb} disambiguated) to {args.out}"
    )


if __name__ == "__main__":
    main()
