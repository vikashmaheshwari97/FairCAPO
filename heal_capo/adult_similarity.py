from __future__ import annotations

import math

from heal_capo.similarity_text import category, numeric, ordinal, parse_fields, proximity

DEFAULT_WEIGHTS = {
    "education": 2.0, "education ordinal": 2.5, "occupation": 3.0,
    "work class": 1.5, "marital status": 0.5, "relationship": 0.5,
    "native country": 0.25, "capital gain": 2.5, "capital loss": 0.75,
    "hours per week": 1.25, "age": 0.75,
}


def similarity(query: str, candidate: str, weights: dict) -> float:
    left, right = parse_fields(query), parse_fields(candidate)
    score = 0.0
    for key in ("education", "occupation", "work class", "marital status", "relationship", "native country"):
        a, b = category(left.get(key, "")), category(right.get(key, ""))
        if a and b and a == b:
            score += float(weights.get(key, 0.0))
    a, b = ordinal(left.get("education", "")), ordinal(right.get("education", ""))
    if a and b:
        score += float(weights.get("education ordinal", 0.0)) * proximity(a, b, 2.0)
    for key in ("age", "hours per week"):
        score += float(weights.get(key, 0.0)) * proximity(
            numeric(left.get(key, "")), numeric(right.get(key, "")), 12.0
        )
    for key in ("capital gain", "capital loss"):
        a = math.log1p(max(0.0, numeric(left.get(key, ""))))
        b = math.log1p(max(0.0, numeric(right.get(key, ""))))
        score += float(weights.get(key, 0.0)) * proximity(a, b, 1.5)
    return float(score)
