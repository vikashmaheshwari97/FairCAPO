from __future__ import annotations

import hashlib

from heal_capo.adult_retrieval import AdultSimilarityRetriever
from heal_capo.retrieval_candidate import RetrievalPromptCandidate
from scripts.run_phase2_budgeted_mocapo import LLMObjectiveEvaluator


class RetrievalLLMObjectiveEvaluator(LLMObjectiveEvaluator):
    def __init__(self, config: dict, llm=None):
        super().__init__(config, llm=llm)
        self.retriever = AdultSimilarityRetriever(config)

    def evaluate(self, candidate, data):
        wrapped = RetrievalPromptCandidate(candidate, self.retriever)
        result = super().evaluate(wrapped, data)
        result.details["retrieval_enabled"] = True
        result.details["retrieval_k"] = self.retriever.k
        result.details["retrieval_pool_size"] = len(self.retriever.bank)
        for row in result.details.get("predictions", []):
            key = hashlib.sha256(str(row.get("text", "")).encode("utf-8")).hexdigest()
            selected = wrapped.trace.get(key, [])
            row["retrieval_source_indices"] = [item["source_index"] for item in selected]
            row["retrieval_similarities"] = [item["similarity"] for item in selected]
        return result


def build_retrieval_evaluator(config: dict, force_no_llm: bool = False):
    if force_no_llm or not bool((config.get("evaluation") or {}).get("use_llm", False)):
        from heal_capo.objectives import ToyObjectiveEvaluator
        return ToyObjectiveEvaluator()
    return RetrievalLLMObjectiveEvaluator(config)
