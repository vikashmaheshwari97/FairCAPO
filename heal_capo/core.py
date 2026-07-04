from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


def _adult_demo_reasoning(text: str, output: str) -> str:
    """Build a concise, protected-attribute-free rationale for Adult demos."""
    match = re.search(
        r"<final_answer>\s*(<=50K|>50K)\s*</final_answer>",
        str(output or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return str(output or "")

    label = "<=50K" if match.group(1).lower() == "<=50k" else ">50K"
    fields: dict[str, str] = {}
    for line in str(text or "").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()

    evidence = []
    for key in (
        "education",
        "occupation",
        "work class",
        "hours per week",
        "capital gain",
        "capital loss",
        "age",
    ):
        value = fields.get(key)
        if value:
            evidence.append(f"{key}={value}")

    joined = "; ".join(evidence)
    rationale = (
        "Reasoning: combine the economic evidence rather than relying on one "
        f"field. This labeled example has {joined}. Applying the same decision "
        f"standard across records gives {label}."
    )
    return f"{rationale}\n<final_answer>{label}</final_answer>"


@dataclass
class PromptCandidate:
    """A prompt candidate plus optional few-shot examples and metadata."""

    instruction: str
    examples: List[Dict[str, str]] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)
    candidate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render(self, x: str) -> str:
        """Render a stable instruction -> references -> current-input prompt."""
        use_adult_reasoning = os.environ.get(
            "FAIRCAPO_ADULT_REASONING_SHOTS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}

        sections = [f"Task instruction:\n{str(self.instruction).strip()}"]
        if self.examples:
            rendered_examples = []
            for index, example in enumerate(self.examples, start=1):
                example_input = str(example.get("input", ""))
                example_output = str(example.get("output", ""))
                if use_adult_reasoning and "person record:" in example_input.lower():
                    example_output = _adult_demo_reasoning(example_input, example_output)
                rendered_examples.append(
                    f"Reference example {index}:\n"
                    f"Input:\n{example_input}\n"
                    f"Output:\n{example_output}"
                )
            sections.append("Reference examples:\n" + "\n\n".join(rendered_examples))

        sections.append(f"Current input:\n{x}\n\nAnswer:")
        return "\n\n".join(section for section in sections if section)


@dataclass
class EvaluationResult:
    """Evaluation metrics for one PromptCandidate."""

    candidate_id: str
    performance: float
    cost: float
    risk: float
    fairness_risk: float = 0.0
    drift: float = 0.0
    n_examples: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def objective_vector(self) -> tuple[float, float, float]:
        return (-self.performance, self.cost, self.fairness_risk)

    @property
    def objective_vector_with_drift(self) -> tuple[float, float, float, float]:
        return (-self.performance, self.cost, self.fairness_risk, self.drift)

    def to_row(self) -> Dict[str, Any]:
        row = {
            "candidate_id": self.candidate_id,
            "performance": self.performance,
            "cost": self.cost,
            "risk": self.risk,
            "fairness_risk": self.fairness_risk,
            "drift": self.drift,
            "n_examples": self.n_examples,
        }
        for key, value in self.details.items():
            row[f"detail_{key}"] = value
        return row


@dataclass
class PromptPortfolio:
    """Unique prompt candidates keyed by candidate_id plus their latest result."""

    candidates: List[PromptCandidate] = field(default_factory=list)
    evaluations: Dict[str, EvaluationResult] = field(default_factory=dict)

    def add(
        self,
        candidate: PromptCandidate,
        result: Optional[EvaluationResult] = None,
    ) -> None:
        """Insert or replace a candidate without duplicating portfolio rows."""
        for index, existing in enumerate(self.candidates):
            if existing.candidate_id == candidate.candidate_id:
                self.candidates[index] = candidate
                break
        else:
            self.candidates.append(candidate)

        if result is not None:
            if result.candidate_id != candidate.candidate_id:
                raise ValueError(
                    "Portfolio candidate/result ID mismatch: "
                    f"{candidate.candidate_id!r} != {result.candidate_id!r}"
                )
            self.evaluations[candidate.candidate_id] = result

    def get(self, candidate_id: str) -> PromptCandidate:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise KeyError(candidate_id)

    def get_result(self, candidate_id: str) -> EvaluationResult:
        if candidate_id not in self.evaluations:
            raise KeyError(candidate_id)
        return self.evaluations[candidate_id]

    def evaluated_candidates(self) -> Sequence[PromptCandidate]:
        return [
            candidate
            for candidate in self.candidates
            if candidate.candidate_id in self.evaluations
        ]

    def evaluated_results(self) -> Sequence[EvaluationResult]:
        return list(self.evaluations.values())

    def best_by_performance(self) -> Optional[PromptCandidate]:
        evaluated = self.evaluated_candidates()
        if not evaluated:
            return None
        return max(
            evaluated,
            key=lambda candidate: self.evaluations[candidate.candidate_id].performance,
        )

    def lowest_risk(self) -> Optional[PromptCandidate]:
        evaluated = self.evaluated_candidates()
        if not evaluated:
            return None
        return min(
            evaluated,
            key=lambda candidate: self.evaluations[candidate.candidate_id].risk,
        )

    def lowest_fairness_risk(self) -> Optional[PromptCandidate]:
        evaluated = self.evaluated_candidates()
        if not evaluated:
            return None
        return min(
            evaluated,
            key=lambda candidate: self.evaluations[candidate.candidate_id].fairness_risk,
        )

    def lowest_cost(self) -> Optional[PromptCandidate]:
        evaluated = self.evaluated_candidates()
        if not evaluated:
            return None
        return min(
            evaluated,
            key=lambda candidate: self.evaluations[candidate.candidate_id].cost,
        )
