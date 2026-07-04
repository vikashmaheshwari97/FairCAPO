from __future__ import annotations

from typing import Optional, Sequence

from heal_capo.core import EvaluationResult, PromptCandidate
from heal_capo.fairness import (
    equalized_odds_gap,
    group_accuracy_gap,
    label_conditioned_group_accuracy_gap,
)
from heal_capo.optimizers import block_evaluator_legacy as _legacy

DataBlock = _legacy.DataBlock
BlockEvaluation = _legacy.BlockEvaluation
EvaluationHistory = _legacy.EvaluationHistory
prompt_signature = _legacy.prompt_signature
clone_evaluation_for_candidate = _legacy.clone_evaluation_for_candidate
make_blocks = _legacy.make_blocks


def _prediction_rows(evaluations: Sequence[BlockEvaluation]) -> list[dict]:
    return [
        dict(row)
        for evaluation in evaluations
        for row in ((evaluation.result.details or {}).get("predictions", []) or [])
        if isinstance(row, dict)
    ]


def merge_results(candidate_id: str, evaluations: Sequence[BlockEvaluation]) -> EvaluationResult:
    result = _legacy.merge_results(candidate_id, evaluations)
    rows = _prediction_rows(evaluations)
    if rows:
        result.details["predictions"] = rows
        result.details["num_merged_predictions"] = len(rows)
    return result


class BlockEvaluator(_legacy.BlockEvaluator):
    """Block evaluator with error capture and statistically correct fairness merging."""

    def evaluate_block(
        self,
        candidate: PromptCandidate,
        block_id: int,
        use_cache: bool = True,
    ) -> BlockEvaluation:
        evaluation = super().evaluate_block(candidate, block_id, use_cache)
        existing = [
            dict(row)
            for row in candidate.metadata.get("error_feedback_rows", [])
            if isinstance(row, dict)
        ]
        seen = {
            (str(row.get("input", "")), str(row.get("gold", "")), str(row.get("prediction", "")))
            for row in existing
        }
        for row in (evaluation.result.details or {}).get("predictions", []) or []:
            if bool(row.get("correct")):
                continue
            item = {
                "input": str(row.get("text", ""))[:700],
                "gold": str(row.get("gold", "")),
                "prediction": str(row.get("prediction", "")),
                "group": str(row.get("group", "")),
                "block_id": int(block_id),
            }
            key = (item["input"], item["gold"], item["prediction"])
            if key not in seen:
                existing.append(item)
                seen.add(key)
        candidate.metadata["error_feedback_rows"] = existing[-12:]
        candidate.metadata["last_block_performance"] = float(evaluation.result.performance)
        return evaluation

    def aggregate_candidate(
        self,
        candidate_id: str,
        block_ids: Optional[Sequence[int]] = None,
    ) -> EvaluationResult:
        if block_ids is None:
            evaluations = self.history.get_candidate_evaluations(candidate_id)
        else:
            evaluations = [
                self.history.get(candidate_id, block_id)
                for block_id in block_ids
                if self.history.has(candidate_id, block_id)
            ]

        result = merge_results(candidate_id, evaluations)
        rows = list(result.details.get("predictions", []) or [])
        predictions = [str(row.get("prediction", "")) for row in rows]
        labels = [str(row.get("gold", "")) for row in rows]
        groups = [str(row.get("group", "")) for row in rows]
        complete = bool(rows) and all(groups) and len(predictions) == len(labels) == len(groups)

        mode = str(getattr(self.evaluator, "fairness_mode", "")).strip().lower()
        fairness_cfg = dict(getattr(self.evaluator, "fairness_cfg", {}) or {})
        recomputed = None
        details = {}
        if complete and mode == "equalized_odds":
            positive_label = str(fairness_cfg.get("positive_label", ">50K"))
            recomputed = equalized_odds_gap(predictions, labels, groups, positive_label)
            details = {"fairness_method": "equalized_odds"}
        elif complete and mode in {"group", "group_fairness", "group_accuracy_gap"}:
            recomputed = group_accuracy_gap(predictions, labels, groups)
            details = {"fairness_method": "group_accuracy_gap"}
        elif complete and mode in {
            "label_conditioned_group_accuracy_gap",
            "label_group_accuracy_gap",
        }:
            recomputed, label_details = label_conditioned_group_accuracy_gap(
                predictions,
                labels,
                groups,
                min_count_per_group=int(fairness_cfg.get("min_count_per_group", 1)),
            )
            details = dict(label_details)

        if recomputed is not None:
            result.fairness_risk = float(recomputed)
            result.details.update(details)
            result.details["fairness_aggregation"] = "recomputed_from_merged_predictions"
            result.details["fairness_risk_recomputed"] = float(recomputed)

        floor = float(getattr(self.evaluator, "performance_floor", 0.0) or 0.0)
        shortfall = max(0.0, floor - float(result.performance))
        result.details["performance_feasibility_threshold"] = floor
        result.details["performance_feasibility_shortfall"] = shortfall
        result.details["performance_feasible"] = shortfall <= 0.0
        return result
