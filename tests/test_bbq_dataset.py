from __future__ import annotations

import json

from experiments.datasets import load_bbq, render_bbq_text


def _write_category(data_dir, category="Gender_identity", n=6):
    rows = []
    for i in range(n):
        rows.append(
            {
                "example_id": i,
                "question_polarity": "neg" if i % 2 == 0 else "nonneg",
                "context_condition": "ambig" if i % 2 == 0 else "disambig",
                "category": category,
                "context": f"Context {i} about a man and a woman.",
                "question": "Who was unprofessional?",
                "ans0": "The man",
                "ans1": "The woman",
                "ans2": "Cannot be determined",
                "label": i % 3,
                "answer_info": {
                    "ans0": ["The man", "M"],
                    "ans1": ["The woman", "F"],
                    "ans2": ["Cannot be determined", "unknown"],
                },
                "additional_metadata": {"stereotyped_groups": ["M"]},
            }
        )
    path = data_dir / f"{category}.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return rows


def test_render_bbq_text_has_lettered_options():
    text = render_bbq_text("ctx", "q?", ["a", "b", "c"])
    assert "(A) a" in text and "(B) b" in text and "(C) c" in text
    assert "Question: q?" in text


def test_load_bbq_from_local_jsonl(tmp_path):
    _write_category(tmp_path, "Gender_identity", n=6)

    split = load_bbq(
        dev_size=3,
        shots_size=1,
        test_size=2,
        seed=0,
        allow_smaller=True,
        categories=("Gender_identity",),
        data_dir=str(tmp_path),
    )

    assert split.task_type == "multiple_choice"
    assert split.classes is None
    all_examples = split.dev + split.shots + split.test
    assert all_examples, "expected BBQ examples to be loaded"

    ex = all_examples[0]
    assert ex.label in {"A", "B", "C"}
    assert "(A)" in ex.text
    meta = ex.metadata
    assert meta["dataset"] == "bbq"
    assert meta["context_condition"] in {"ambig", "disambig"}
    assert meta["question_polarity"] in {"neg", "nonneg"}
    assert meta["answer_info"]["ans2"][1] == "unknown"
    assert meta["stereotyped_groups"] == ["M"]
    assert meta["options"][0] == "The man"
    assert meta["label_idx"] in {0, 1, 2}
    # Label letter and label_idx agree.
    assert ex.label == "ABC"[meta["label_idx"]]
