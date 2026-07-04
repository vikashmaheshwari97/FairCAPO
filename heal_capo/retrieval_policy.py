from __future__ import annotations

from collections import defaultdict


def select_ranked(ranked, k: int, policy: str):
    """Select natural nearest neighbours or an explicitly balanced control."""
    k = max(0, int(k))
    mode = str(policy or "natural_similarity").strip().lower()
    natural = {"natural", "nearest", "nearest_neighbors", "natural_similarity"}
    balanced = {"balanced", "label_balanced", "nonprotected_similarity_label_balanced"}
    if mode in natural:
        return list(ranked[:k]), "natural_similarity"
    if mode not in balanced:
        raise ValueError(f"Unknown retrieval policy: {mode}")
    buckets = defaultdict(list)
    for item in ranked:
        buckets[item[1].label].append(item)
    labels = sorted(buckets)
    if not labels:
        return [], "label_balanced"
    base, extra = divmod(k, len(labels))
    chosen = []
    for index, label in enumerate(labels):
        chosen.extend(buckets[label][: base + int(index < extra)])
    return sorted(chosen, key=lambda item: -item[0])[:k], "label_balanced"
