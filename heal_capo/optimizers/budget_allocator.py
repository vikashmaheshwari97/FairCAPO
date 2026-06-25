from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from heal_capo.optimizers.block_evaluator import BlockEvaluation


@dataclass
class BudgetRecord:
    """
    One recorded budget usage event.
    """

    candidate_id: str
    block_id: Optional[int]
    cost: float
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class BudgetState:
    """
    Current state of the evaluation budget.
    """

    max_budget: float
    used_budget: float = 0.0
    records: list[BudgetRecord] = field(default_factory=list)

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.max_budget - self.used_budget)

    @property
    def exhausted(self) -> bool:
        return self.used_budget >= self.max_budget

    @property
    def utilization(self) -> float:
        if self.max_budget <= 0:
            return 1.0

        return min(1.0, self.used_budget / self.max_budget)


class BudgetAllocator:
    """
    Tracks token/cost budget during block-wise prompt evaluation.

    In MO-CAPO-style optimization, this is used to:
      - stop when the total evaluation budget is exhausted
      - reject evaluations that exceed the remaining budget
      - record per-candidate and per-block cost
      - produce budget summaries for experiment logs
    """

    def __init__(
        self,
        max_budget: float,
        allow_overspend: bool = False,
        budget_unit: str = "cost",
    ):
        if max_budget <= 0:
            raise ValueError("max_budget must be positive.")

        if budget_unit not in ("cost", "tokens"):
            raise ValueError(
                f"budget_unit must be 'cost' or 'tokens', got {budget_unit!r}."
            )

        self.state = BudgetState(max_budget=float(max_budget))
        self.allow_overspend = allow_overspend
        # What `max_budget`/`used_budget` are denominated in:
        #   "cost"   -> weighted cost (input*w_in + output*w_out)  [default]
        #   "tokens" -> raw tokens (input_tokens + output_tokens)  [MO-CAPO 7.5M]
        # The cost OBJECTIVE (candidate_cost/block_cost) stays weighted cost
        # either way; this only changes what spending is metered against.
        self.budget_unit = budget_unit

    @property
    def max_budget(self) -> float:
        return self.state.max_budget

    @property
    def used_budget(self) -> float:
        return self.state.used_budget

    @property
    def remaining_budget(self) -> float:
        return self.state.remaining_budget

    @property
    def exhausted(self) -> bool:
        return self.state.exhausted

    @property
    def utilization(self) -> float:
        return self.state.utilization

    def can_spend(self, cost: float) -> bool:
        """
        Return True if this cost can be spent under the budget.
        """
        cost = float(cost)

        if cost < 0:
            raise ValueError("cost must be non-negative.")

        if self.allow_overspend:
            return True

        return cost <= self.remaining_budget

    def require_budget(self, cost: float) -> None:
        """
        Raise RuntimeError if the requested cost exceeds the remaining budget.
        """
        if not self.can_spend(cost):
            raise RuntimeError(
                f"Budget exhausted: requested={cost}, "
                f"remaining={self.remaining_budget}, max_budget={self.max_budget}"
            )

    def charge_for(
        self,
        cost: float,
        input_tokens: float = 0.0,
        output_tokens: float = 0.0,
    ) -> float:
        """
        Amount deducted from the budget for one evaluation, per ``budget_unit``.
        "cost" -> weighted cost; "tokens" -> raw input+output tokens.
        """
        if self.budget_unit == "tokens":
            return float(input_tokens) + float(output_tokens)

        return float(cost)

    def record(
        self,
        candidate_id: str,
        cost: float,
        block_id: Optional[int] = None,
        input_tokens: float = 0.0,
        output_tokens: float = 0.0,
        metadata: Optional[dict] = None,
    ) -> BudgetRecord:
        """
        Record budget usage. The amount charged against the budget depends on
        ``budget_unit`` (weighted cost vs. raw tokens); ``cost`` is always stored
        for the cost objective regardless of the metering unit.
        """
        cost = float(cost)
        input_tokens = float(input_tokens)
        output_tokens = float(output_tokens)

        if cost < 0:
            raise ValueError("cost must be non-negative.")

        charge = self.charge_for(cost, input_tokens, output_tokens)
        self.require_budget(charge)

        record = BudgetRecord(
            candidate_id=candidate_id,
            block_id=block_id,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata=dict(metadata or {}),
        )

        self.state.records.append(record)
        self.state.used_budget += charge

        return record

    def record_block_evaluation(
        self,
        evaluation: BlockEvaluation,
    ) -> BudgetRecord:
        """
        Record budget usage from a BlockEvaluation.

        The EvaluationResult.details may optionally contain:
          - input_tokens
          - output_tokens
          - dev_input_tokens
          - dev_output_tokens
        """
        details = evaluation.result.details or {}

        input_tokens = details.get(
            "input_tokens",
            details.get("dev_input_tokens", 0.0),
        )
        output_tokens = details.get(
            "output_tokens",
            details.get("dev_output_tokens", 0.0),
        )

        return self.record(
            candidate_id=evaluation.candidate_id,
            block_id=evaluation.block_id,
            cost=evaluation.result.cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={
                "performance": evaluation.result.performance,
                "risk": evaluation.result.risk,
                "fairness_risk": evaluation.result.fairness_risk,
                "n_examples": evaluation.result.n_examples,
            },
        )

    def candidate_cost(self, candidate_id: str) -> float:
        return sum(
            record.cost
            for record in self.state.records
            if record.candidate_id == candidate_id
        )

    def candidate_tokens(self, candidate_id: str) -> dict:
        input_tokens = sum(
            record.input_tokens
            for record in self.state.records
            if record.candidate_id == candidate_id
        )
        output_tokens = sum(
            record.output_tokens
            for record in self.state.records
            if record.candidate_id == candidate_id
        )

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def block_cost(self, block_id: int) -> float:
        return sum(
            record.cost
            for record in self.state.records
            if record.block_id == block_id
        )

    def candidate_block_costs(self, candidate_id: str) -> Dict[int, float]:
        costs: Dict[int, float] = {}

        for record in self.state.records:
            if record.candidate_id != candidate_id:
                continue

            if record.block_id is None:
                continue

            costs[record.block_id] = costs.get(record.block_id, 0.0) + record.cost

        return costs

    def num_records(self) -> int:
        return len(self.state.records)

    def reset(self) -> None:
        self.state.used_budget = 0.0
        self.state.records.clear()

    def summary(self) -> dict:
        candidate_ids = sorted(
            {
                record.candidate_id
                for record in self.state.records
            }
        )

        block_ids = sorted(
            {
                record.block_id
                for record in self.state.records
                if record.block_id is not None
            }
        )

        total_input_tokens = sum(record.input_tokens for record in self.state.records)
        total_output_tokens = sum(record.output_tokens for record in self.state.records)

        return {
            "max_budget": self.max_budget,
            "used_budget": self.used_budget,
            "remaining_budget": self.remaining_budget,
            "utilization": self.utilization,
            "exhausted": self.exhausted,
            "budget_unit": self.budget_unit,
            "num_records": self.num_records(),
            "num_candidates": len(candidate_ids),
            "num_blocks": len(block_ids),
            "candidate_costs": {
                candidate_id: self.candidate_cost(candidate_id)
                for candidate_id in candidate_ids
            },
            "block_costs": {
                block_id: self.block_cost(block_id)
                for block_id in block_ids
            },
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }

    def to_rows(self) -> list[dict]:
        rows = []

        for record in self.state.records:
            row = {
                "candidate_id": record.candidate_id,
                "block_id": record.block_id,
                "cost": record.cost,
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
            }

            for key, value in record.metadata.items():
                row[f"metadata_{key}"] = value

            rows.append(row)

        return rows