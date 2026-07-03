from __future__ import annotations

from heal_capo.core import PromptCandidate
from scripts.prepare_adult_semantic_csv import enrich_row


def _adult_row() -> dict[str, str]:
    return {
        "age": "42",
        "workclass": "Self-emp-inc",
        "fnlwgt": "123456",
        "education": "Masters",
        "education.num": "14",
        "marital.status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital.gain": "15024",
        "capital.loss": "0",
        "hours.per.week": "55",
        "native.country": "United-States",
        "income": ">50K",
    }


def test_semantic_enrichment_preserves_target_and_protected_metadata():
    raw = _adult_row()
    enriched = enrich_row(raw)

    assert enriched["income"] == raw["income"]
    assert enriched["sex"] == raw["sex"]
    assert enriched["race"] == raw["race"]
    assert "ordinal education level 14" in enriched["education"]
    assert "late-career" not in enriched["age"]
    assert "mid-career" in enriched["age"]
    assert "long working hours" in enriched["hours.per.week"]
    assert "very high recorded capital gain" in enriched["capital.gain"]
    assert "executive or managerial" in enriched["occupation"]


def test_adult_reasoning_shots_are_optional_and_protected_free(monkeypatch):
    record = (
        "Person record:\n"
        "Age: 42 (mid career age)\n"
        "Work class: Self emp inc (self employed incorporated)\n"
        "Education: Masters (ordinal education level 14 of 16)\n"
        "Occupation: Exec managerial (executive or managerial occupation)\n"
        "Capital gain: 15024 (very high recorded capital gain)\n"
        "Capital loss: 0 (no recorded capital loss)\n"
        "Hours per week: 55 (long working hours)"
    )
    candidate = PromptCandidate(
        instruction="Predict income.",
        examples=[
            {
                "input": record,
                "output": "<final_answer>>50K</final_answer>",
            }
        ],
    )

    monkeypatch.setenv("FAIRCAPO_ADULT_REASONING_SHOTS", "1")
    rendered = candidate.render(record)

    assert "Reasoning: combine the economic evidence" in rendered
    assert "<final_answer>>50K</final_answer>" in rendered
    assert "sex:" not in rendered.lower()
    assert "race:" not in rendered.lower()
