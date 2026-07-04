from __future__ import annotations

from collections import Counter

from experiments.datasets import Example
from heal_capo.adult_retrieval import AdultSimilarityRetriever
from heal_capo.adult_similarity import DEFAULT_WEIGHTS
from heal_capo.core import PromptCandidate
from heal_capo.retrieval_candidate import RetrievalPromptCandidate
from heal_capo.retrieval_diagnostics import summarize_prediction_retrieval


def _record(
    age: int,
    work_class: str,
    education: str,
    occupation: str,
    capital_gain: int,
    capital_loss: int,
    hours: int,
    marital: str = "Never married",
    relationship: str = "Not in family",
) -> str:
    return (
        "Person record:\n"
        f"Age: {age}\n"
        f"Work class: {work_class}\n"
        f"Education: {education}\n"
        f"Marital status: {marital}\n"
        f"Occupation: {occupation}\n"
        f"Relationship: {relationship}\n"
        f"Capital gain: {capital_gain}\n"
        f"Capital loss: {capital_loss}\n"
        f"Hours per week: {hours}"
    )


def _retriever(policy: str, k: int = 4) -> AdultSimilarityRetriever:
    retriever = AdultSimilarityRetriever.__new__(AdultSimilarityRetriever)
    retriever.k = k
    retriever.policy = policy
    retriever.weights = dict(DEFAULT_WEIGHTS)
    retriever.bank = [
        Example(
            _record(40, "Private", "Masters (ordinal education level 14 of 16)", "Exec managerial", 5000, 0, 50),
            ">50K",
            {"source_index": 1},
        ),
        Example(
            _record(41, "Private", "Masters (ordinal education level 14 of 16)", "Prof specialty", 4000, 0, 48),
            ">50K",
            {"source_index": 2},
        ),
        Example(
            _record(39, "Private", "Bachelors (ordinal education level 13 of 16)", "Exec managerial", 3000, 0, 52),
            ">50K",
            {"source_index": 3},
        ),
        Example(
            _record(35, "Private", "HS grad (ordinal education level 9 of 16)", "Adm clerical", 0, 0, 40),
            "<=50K",
            {"source_index": 4},
        ),
        Example(
            _record(22, "Private", "HS grad (ordinal education level 9 of 16)", "Other service", 0, 0, 25),
            "<=50K",
            {"source_index": 5},
        ),
    ]
    return retriever


def test_natural_retrieval_preserves_similarity_neighbourhood() -> None:
    retriever = _retriever("natural_similarity", k=4)
    selected = retriever.select(
        _record(40, "Private", "Masters (ordinal education level 14 of 16)", "Exec managerial", 4500, 0, 50)
    )
    assert len(selected) == 4
    assert selected[0]["source_index"] == 1
    assert [row["similarity"] for row in selected] == sorted(
        [row["similarity"] for row in selected], reverse=True
    )
    assert Counter(row["label"] for row in selected)[">50K"] >= 3
    assert [row["rank"] for row in selected] == [1, 2, 3, 4]


def test_label_balanced_policy_is_explicit_control() -> None:
    retriever = _retriever("label_balanced", k=4)
    selected = retriever.select(
        _record(40, "Private", "Masters (ordinal education level 14 of 16)", "Exec managerial", 4500, 0, 50)
    )
    assert Counter(row["label"] for row in selected) == {"<=50K": 2, ">50K": 2}
    assert all("sex:" not in row["input"].lower() for row in selected)
    assert all("race:" not in row["input"].lower() for row in selected)


def test_similarity_uses_ordinal_education_and_capital_loss() -> None:
    retriever = _retriever("natural_similarity", k=1)
    query = _record(40, "Private", "Masters (ordinal education level 14 of 16)", "Exec managerial", 0, 2000, 50)
    close = _record(41, "Private", "Masters (ordinal education level 14 of 16)", "Exec managerial", 0, 1900, 49)
    far = _record(41, "Private", "HS grad (ordinal education level 9 of 16)", "Other service", 0, 0, 49)
    assert retriever.score(query, close) > retriever.score(query, far)


def test_retrieval_candidate_renders_query_last_and_concisely() -> None:
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
                    "rank": 1,
                }
            ]

    wrapped = RetrievalPromptCandidate(PromptCandidate(instruction="Predict annual income."), DummyRetriever())
    rendered = wrapped.render("Person record:\nEducation: Bachelors")
    assert "Reference 1:" in rendered
    assert "Education: Masters" in rendered
    assert "Current record:\nPerson record:\nEducation: Bachelors" in rendered
    assert rendered.index("Reference 1:") < rendered.index("Current record:")
    assert len(wrapped.trace) == 1


def test_retrieval_diagnostics_measure_nearest_label_and_similarity() -> None:
    summary = summarize_prediction_retrieval(
        [
            {
                "gold": ">50K",
                "correct": True,
                "retrieval_labels": [">50K", "<=50K"],
                "retrieval_similarities": [0.9, 0.5],
            },
            {
                "gold": "<=50K",
                "correct": False,
                "retrieval_labels": [">50K", "<=50K"],
                "retrieval_similarities": [0.6, 0.4],
            },
        ]
    )
    assert summary["retrieval_nearest_label_accuracy"] == 0.5
    assert summary["retrieval_mean_top_similarity_correct"] == 0.9
    assert summary["retrieval_mean_top_similarity_incorrect"] == 0.6
