from __future__ import annotations

from heal_capo.adult_v4_fairness import merged_equalized_odds
from heal_capo.optimizers.block_evaluator import BlockEvaluator


class AdultV4BlockEvaluator(BlockEvaluator):
    """Capture errors and recompute fairness over merged predictions."""

    def evaluate_block(self, candidate, block_id: int, use_cache: bool = True):
        evaluation = super().evaluate_block(candidate, block_id, use_cache)
        existing = [dict(row) for row in candidate.metadata.get("error_feedback_rows", [])]
        seen = {(row.get("input"), row.get("gold"), row.get("prediction")) for row in existing}
        for row in (evaluation.result.details or {}).get("predictions", []) or []:
            if bool(row.get("correct")):
                continue
            item = {"input": str(row.get("text", ""))[:700],
                    "gold": str(row.get("gold", "")),
                    "prediction": str(row.get("prediction", ""))}
            key = (item["input"], item["gold"], item["prediction"])
            if key not in seen:
                existing.append(item)
                seen.add(key)
        candidate.metadata["error_feedback_rows"] = existing[-12:]
        candidate.metadata["last_block_performance"] = evaluation.result.performance
        return evaluation

    def aggregate_candidate(self, candidate_id: str, block_ids=None):
        result = super().aggregate_candidate(candidate_id, block_ids)
        evaluations = self.history.get_candidate_evaluations(candidate_id) if block_ids is None else [
            self.history.get(candidate_id, block_id)
            for block_id in block_ids if self.history.has(candidate_id, block_id)
        ]
        rows = [row for evaluation in evaluations
                for row in ((evaluation.result.details or {}).get("predictions", []) or [])]
        mode = str(getattr(self.evaluator, "fairness_mode", "")).strip().lower()
        if mode == "equalized_odds":
            positive = str(getattr(self.evaluator, "fairness_cfg", {}).get("positive_label", ">50K"))
            fairness = merged_equalized_odds(rows, positive)
            if fairness is not None:
                result.fairness_risk = float(fairness)
                result.details["fairness_aggregation"] = "recomputed_from_merged_predictions"
                result.details["fairness_risk_recomputed"] = float(fairness)
        floor = float(getattr(self.evaluator, "performance_floor", 0.0) or 0.0)
        shortfall = max(0.0, floor - result.performance)
        result.details["performance_feasibility_threshold"] = floor
        result.details["performance_feasibility_shortfall"] = shortfall
        result.details["performance_feasible"] = shortfall <= 0.0
        return result
