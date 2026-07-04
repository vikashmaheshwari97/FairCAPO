from __future__ import annotations

import math
import re
from collections import defaultdict

from heal_capo.adult_v4_data import examples_from_manifest


def parse_fields(text: str) -> dict[str, str]:
    result = {}
    for line in str(text).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip().lower()] = value.strip()
    return result


def numeric(value: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
    return float(match.group(0)) if match else 0.0


def category(value: str) -> str:
    return str(value or "").split("(", 1)[0].strip().lower()


class AdultSimilarityRetriever:
    def __init__(self, config: dict):
        cfg = config.get("retrieval") or {}
        dev = config.get("dev") or {}
        self.k = int(cfg.get("k", 12))
        data_path = dev.get("data_path", "data/adult_semantic_v3.csv")
        manifest_path = dev.get("split_manifest", "data/adult_v4_fixed_split_seed0.json")
        self.bank = examples_from_manifest("shots", data_path, manifest_path)
        self.weights = cfg.get("weights") or {
            "education": 3.0,
            "occupation": 3.0,
            "work class": 2.0,
            "capital gain": 2.5,
            "hours per week": 1.5,
            "age": 1.0,
        }

    def score(self, query: str, candidate: str) -> float:
        left, right = parse_fields(query), parse_fields(candidate)
        score = 0.0
        for key in ("education", "occupation", "work class"):
            score += float(self.weights[key]) * (
                category(left.get(key, "")) == category(right.get(key, ""))
            )
        for key, scale in (("age", 20.0), ("hours per week", 20.0)):
            distance = abs(numeric(left.get(key, "")) - numeric(right.get(key, "")))
            score += float(self.weights[key]) * math.exp(-distance / scale)
        a = math.log1p(max(0.0, numeric(left.get("capital gain", ""))))
        b = math.log1p(max(0.0, numeric(right.get("capital gain", ""))))
        score += float(self.weights["capital gain"]) * math.exp(-abs(a - b) / 2.0)
        return score

    def select(self, query: str) -> list[dict]:
        ranked = sorted(
            ((self.score(query, row.text), row) for row in self.bank),
            key=lambda item: (-item[0], int((item[1].metadata or {}).get("source_index", -1))),
        )
        buckets = defaultdict(list)
        for score, row in ranked:
            buckets[row.label].append((score, row))
        per_label = max(1, self.k // max(1, len(buckets)))
        chosen = []
        for label in sorted(buckets):
            chosen.extend(buckets[label][:per_label])
        chosen = sorted(chosen, key=lambda item: -item[0])[: self.k]
        return [
            {
                "input": row.text,
                "output": f"<final_answer>{row.label}</final_answer>",
                "source_index": int((row.metadata or {})["source_index"]),
                "label": row.label,
                "similarity": float(score),
            }
            for score, row in chosen
        ]
