from __future__ import annotations

import hashlib

from heal_capo.adult_retrieval import AdultSimilarityRetriever
from heal_capo.retrieval_candidate import RetrievalPromptCandidate
from heal_capo.retrieval_diagnostics import summarize_prediction_retrieval
from scripts.run_phase2_budgeted_mocapo import LLMObjectiveEvaluator


class RetrievalLLMObjectiveEvaluator(LLMObjectiveEvaluator):
    def __init__(self, config: dict, llm=None):
        super().__init__(config, llm=llm)
        self.retriever = AdultSimilarityRetriever(config)
        self.performance_floor = float(
            (config.get("selection") or {}).get("min_performance_feasibility", 0.0) or 0.0
        )

    def evaluate(self, candidate, data):
        wrapped = RetrievalPromptCandidate(candidate, self.retriever)
        result = super().evaluate(wrapped, data)
        predictions = list(result.details.get("predictions", []) or [])

        for row in predictions:
            key = hashlib.sha256(str(row.get("text", "")).encode("utf-8")).hexdigest()
            selected = wrapped.trace.get(key, [])
            labels = [item["label"] for item in selected]
            similarities = [float(item["similarity"]) for item in selected]
            row["retrieval_source_indices"] = [item["source_index"] for item in selected]
            row["retrieval_labels"] = labels
            row["retrieval_similarities"] = similarities
            row["retrieval_nearest_label"] = labels[0] if labels else ""
            row["retrieval_top_similarity"] = similarities[0] if similarities else None
            row["retrieval_nearest_matches_gold"] = bool(labels) and (
                str(labels[0]).strip().lower() == str(row.get("gold", "")).strip().lower()
            )

        shortfall = max(0.0, self.performance_floor - float(result.performance))
        result.details.update(
            {
                "retrieval_enabled": True,
                "retrieval_policy": self.retriever.policy,
                "retrieval_k": self.retriever.k,
                "retrieval_pool_size": len(self.retriever.bank),
                "performance_feasibility_threshold": self.performance_floor,
                "performance_feasibility_shortfall": shortfall,
                "performance_feasible": shortfall <= 0.0,
                **summarize_prediction_retrieval(predictions),
            }
        )
        return result


def build_retrieval_evaluator(config: dict, force_no_llm: bool = False):
    if force_no_llm or not bool((config.get("evaluation") or {}).get("use_llm", False)):
        from heal_capo.objectives import ToyObjectiveEvaluator

        return ToyObjectiveEvaluator()
    return RetrievalLLMObjectiveEvaluator(config)
