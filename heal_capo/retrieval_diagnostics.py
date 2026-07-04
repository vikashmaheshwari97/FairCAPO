from __future__ import annotations

from collections import Counter


def summarize_prediction_retrieval(rows: list[dict]) -> dict:
    nearest, similarities, correct, incorrect = [], [], [], []
    labels = Counter()
    for row in rows:
        selected_labels = list(row.get("retrieval_labels") or [])
        selected_scores = [float(value) for value in row.get("retrieval_similarities") or []]
        labels.update(selected_labels)
        if selected_labels:
            nearest.append(
                selected_labels[0].strip().lower() == str(row.get("gold", "")).strip().lower()
            )
        if selected_scores:
            similarities.append(selected_scores[0])
            (correct if bool(row.get("correct")) else incorrect).append(selected_scores[0])
    mean = lambda values: sum(values) / len(values) if values else None
    return {
        "retrieval_nearest_label_accuracy": sum(nearest) / len(nearest) if nearest else None,
        "retrieval_mean_top_similarity": mean(similarities),
        "retrieval_mean_top_similarity_correct": mean(correct),
        "retrieval_mean_top_similarity_incorrect": mean(incorrect),
        "retrieval_label_counts": dict(sorted(labels.items())),
    }
