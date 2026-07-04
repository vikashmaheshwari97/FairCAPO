from __future__ import annotations

from heal_capo.optimizers.block_evaluator import BlockEvaluator


class AdultV4BlockEvaluator(BlockEvaluator):
    """Capture recent mistakes for error-driven prompt repair."""

    def evaluate_block(self, candidate, block_id: int, use_cache: bool = True):
        evaluation = super().evaluate_block(candidate, block_id, use_cache)
        existing = [dict(row) for row in candidate.metadata.get("error_feedback_rows", [])]
        seen = {(row.get("input"), row.get("gold"), row.get("prediction")) for row in existing}
        for row in (evaluation.result.details or {}).get("predictions", []) or []:
            if bool(row.get("correct")):
                continue
            item = {
                "input": str(row.get("text", ""))[:700],
                "gold": str(row.get("gold", "")),
                "prediction": str(row.get("prediction", "")),
            }
            key = (item["input"], item["gold"], item["prediction"])
            if key not in seen:
                existing.append(item)
                seen.add(key)
        candidate.metadata["error_feedback_rows"] = existing[-12:]
        candidate.metadata["last_block_performance"] = evaluation.result.performance
        return evaluation
