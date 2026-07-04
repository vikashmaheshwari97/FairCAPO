from __future__ import annotations

from scripts import run_phase2_budgeted_mocapo as runner


class FeedbackBlockEvaluator(runner.BlockEvaluator):
    def evaluate_block(self, candidate, block_id: int, use_cache: bool = True):
        evaluation = super().evaluate_block(candidate, block_id, use_cache)
        rows = []
        for item in (evaluation.result.details or {}).get("predictions", []) or []:
            if not item.get("correct"):
                rows.append(
                    {
                        "input": str(item.get("text", ""))[:600],
                        "gold": str(item.get("gold", "")),
                        "prediction": str(item.get("prediction", "")),
                    }
                )
        candidate.metadata["error_feedback_rows"] = rows[:8]
        return evaluation
