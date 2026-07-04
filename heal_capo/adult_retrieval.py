from __future__ import annotations

from heal_capo.adult_similarity import DEFAULT_WEIGHTS, similarity
from heal_capo.adult_v4_data import examples_from_manifest
from heal_capo.retrieval_policy import select_ranked
from heal_capo.similarity_text import category, numeric, parse_fields


class AdultSimilarityRetriever:
    def __init__(self, config: dict):
        cfg, dev = config.get("retrieval") or {}, config.get("dev") or {}
        self.k = max(0, int(cfg.get("k", 4)))
        self.policy = str(cfg.get("policy", "natural_similarity"))
        self.weights = {**DEFAULT_WEIGHTS, **dict(cfg.get("weights") or {})}
        self.bank = examples_from_manifest(
            "shots",
            dev.get("data_path", "data/adult_semantic_v3.csv"),
            dev.get("split_manifest", "data/adult_v4_fixed_split_seed0.json"),
        )

    def score(self, query: str, candidate: str) -> float:
        return similarity(query, candidate, self.weights)

    def select(self, query: str) -> list[dict]:
        ranked = sorted(
            ((self.score(query, row.text), row) for row in self.bank),
            key=lambda item: (-item[0], int((item[1].metadata or {})["source_index"])),
        )
        chosen, self.policy = select_ranked(ranked, self.k, self.policy)
        return [
            {
                "input": row.text,
                "output": f"<final_answer>{row.label}</final_answer>",
                "source_index": int((row.metadata or {})["source_index"]),
                "label": row.label,
                "similarity": float(score),
                "rank": rank,
            }
            for rank, (score, row) in enumerate(chosen, 1)
        ]
