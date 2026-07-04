from __future__ import annotations

from collections import Counter

from experiments.datasets import Example
from heal_capo.adult_retrieval import AdultSimilarityRetriever
from heal_capo.core import PromptCandidate
from heal_capo.retrieval_candidate import RetrievalPromptCandidate


def _record(
    age: int,
    work_class: str,
    education: str,
    occupation: str,
    capital_gain: int,
    hours: int,
) -> str:
    return (
        "Person record:\n"
        f"Age: {age}\n"
        f"Work class: {work_class}\n"
        f"Education: {education}\n"
        f"Occupation: {occupation}\n"
        f"Capital gain: {capital_gain}\n"
        f"Hours per week: {hours}"
    )


def test_retrieval_is_label_balanced_and_similarity_ranked() -> None:
    retriever = AdultSimilarityRetriever.__new__(AdultSimilarityRetriever)
    retriever.k = 4
    retriever.weights = {
        "education": 3.0,
        "occupation": 3.0,
        "work class": 2.0,
        "capital gain": 2.5,
        "hours per week": 1.5,
        "age": 1.0,
    }
    retriever.bank = [
        Example(
            _record(40, "Private", "Masters", "Exec managerial", 5000, 50),
            ">50K",
            {"source_index": 1},
        ),
        Example(
            _record(22, "Private", "HS grad", "Other service", 0, 25),
            "<=50K",
            {"source_index": 2},
        ),
        Example(
            _record(41, "Private", "Masters", "Prof specialty", 4000, 48),
            ">50K",
            {"source_index": 3},
        ),
        Example(
            _record(35, "Private", "HS grad", "Adm clerical", 0, 40),
            "<=50K",
            {"source_index": 4},
        ),
    ]

    selected = retriever.select(
        _record(40, "Private", "Masters", "Exec managerial", 4500, 50)
    )

    assert len(selected) == 4
    assert Counter(row["label"] for row in selected) == {
        "<=50K": 2,
        ">50K": 2,
    }
    assert selected[0]["source_index"] == 1
    assert all("sex:" not in row["input"].lower() for row in selected)
    assert all("race:" not in row["input"].lower() for row in selected)


def test_retrieval_candidate_renders_query_specific_examples() -> None:
    class DummyRetriever:
        def select(self, text: str) -> list[dict]:
            assert "Bachelors" in text
            return [
                {
                    "input": "Person record:\nEducation: Masters",
                    "output": "<final_answer>>50K</final_answer>",
                    "source_index": 9,
                    "label": ">50K",
                    "similarity": 1.0,
                }
            ]

    source = PromptCandidate(instruction="Predict annual income.")
    wrapped = RetrievalPromptCandidate(source, DummyRetriever())
    rendered = wrapped.render("Person record:\nEducation: Bachelors")

    assert "Examples:" in rendered
    assert "Education: Masters" in rendered
    assert "<final_answer>>50K</final_answer>" in rendered
    assert "Input: Person record:\nEducation: Bachelors" in rendered
    assert len(wrapped.trace) == 1
