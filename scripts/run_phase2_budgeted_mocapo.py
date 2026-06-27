from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any

# Task types whose answer is NOT drawn from a fixed label set, so prompts ask for
# a free-form final answer in tags and few-shot demos don't enumerate labels
# (e.g. GSM8K math, code/text generation, and BBQ multiple-choice).
GENERATION_TASK_TYPES = {
    "math_reasoning",
    "generation",
    "code_generation",
    "qa",
    "multiple_choice",
}

# Multiple-choice tasks (e.g. BBQ): the answer is one option letter (A/B/C).
# Scored by letter match, NOT numeric normalization.
MULTIPLE_CHOICE_TASK_TYPES = {"multiple_choice"}

import yaml

from heal_capo.components.router import FairnessAwareRouter, RiskAwareRouter
from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
from heal_capo.fairness import CombinedFairnessConfig, evaluate_combined_fairness
from heal_capo.fairness_bbq import evaluate_bbq_fairness, item_from_meta
from heal_capo.objectives import ObjectiveEvaluator, ToyObjectiveEvaluator
from heal_capo.optimizers.advance_incumbents import (
    AdvanceIncumbentsConfig,
    advance_front_one_level,
    common_incumbent_blocks,
)
from heal_capo.optimizers.block_evaluator import BlockEvaluator
from heal_capo.optimizers.budget_allocator import BudgetAllocator
from heal_capo.optimizers.environmental_selection import (
    EnvironmentalSelectionConfig,
    EnvironmentalSelector,
)
from heal_capo.optimizers.evolutionary_ops import (
    EvolutionaryOpsConfig,
    EvolutionaryPromptOps,
)
from heal_capo.optimizers.intensification import (
    IntensificationConfig,
    Intensifier,
)
from heal_capo.optimizers.parent_selection import (
    ParentSelectionConfig,
    ParentSelector,
)
from heal_capo.evaluation.mo_metrics import (
    DEFAULT_OBJECTIVE_SPECS,
    fixed_bounds_from_config,
    infer_bounds,
    summarize_mo_metrics,
)
from heal_capo.pareto import non_dominated_ids, sort_pareto_results


def _json_default(obj: Any):
    try:
        return obj.item()
    except AttributeError:
        return str(obj)


def load_yaml(path: str) -> dict:
    yaml_path = Path(path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_csv(rows: list[dict], path: str):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"No rows to save for {path}")

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(data: Any, path: str):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def simple_token_count(text: str) -> int:
    return len(str(text or "").split())


def extract_label(response: str, labels: list[str]) -> str:
    text = str(response or "").strip()
    lowered = text.lower()

    if "<final_answer>" in lowered and "</final_answer>" in lowered:
        start = lowered.find("<final_answer>") + len("<final_answer>")
        end = lowered.find("</final_answer>", start)
        text = text[start:end].strip()
        lowered = text.lower()

    cleaned = lowered.strip(" .,:;!?\"'`")

    for label in labels:
        if cleaned == str(label).lower():
            return str(label)

    for label in labels:
        if str(label).lower() in lowered:
            return str(label)

    return cleaned


def extract_final_answer(response: str) -> str:
    """
    Pull the content of <final_answer>...</final_answer> from a response.

    Falls back to the whole stripped response when the tags are absent. Used for
    generation tasks (e.g. GSM8K) where the answer is free-form, not a label.
    """
    text = str(response or "").strip()
    lowered = text.lower()

    if "<final_answer>" in lowered and "</final_answer>" in lowered:
        start = lowered.find("<final_answer>") + len("<final_answer>")
        end = lowered.find("</final_answer>", start)
        return text[start:end].strip()

    return text


def normalize_numeric_answer(value: str) -> str:
    """
    Normalize a numeric answer for robust comparison (GSM8K-style).

    Strips $/%/commas/whitespace, takes the last number token, and canonicalizes
    integers (``18.0`` -> ``18``). Returns the lowercased text unchanged when no
    number is present so non-numeric generation answers still compare sensibly.
    """
    cleaned = (
        str(value or "")
        .replace(",", "")
        .replace("$", "")
        .replace("%", "")
        .strip()
    )

    matches = re.findall(r"-?\d+\.?\d*", cleaned)
    if not matches:
        return cleaned.lower()

    token = matches[-1]
    try:
        number = float(token)
        if number.is_integer():
            return str(int(number))
        return repr(number)
    except ValueError:
        return token


def normalize_multiple_choice_answer(value: str) -> str:
    """
    Extract a single option letter (A/B/C/...) from a multiple-choice response.

    Looks for a parenthesized option like ``(A)`` first, then a standalone letter
    token (``A``, ``B.``), then the first alphabetic character. Returns an
    uppercase letter (or "" when none is found).
    """
    text = str(value or "").strip().upper()
    if not text:
        return ""

    match = re.search(r"\(([A-Z])\)", text)
    if match:
        return match.group(1)

    match = re.search(r"\b([A-Z])\b", text)
    if match:
        return match.group(1)

    for char in text:
        if char.isalpha():
            return char

    return text


def build_llm_from_config(config: dict):
    llm_config = config.get("llm", {})

    try:
        from baselines.promptolution_runner import build_llm

        return build_llm(llm_config)
    except Exception as exc:
        raise RuntimeError(
            "Could not build LLM from baselines.promptolution_runner.build_llm. "
            "Check LM Studio is running and llm config is valid."
        ) from exc


def get_llm_response(llm, prompt: str) -> str:
    if hasattr(llm, "get_response"):
        response = llm.get_response(prompt)
    elif hasattr(llm, "generate"):
        response = llm.generate(prompt)
    elif callable(llm):
        response = llm(prompt)
    else:
        raise AttributeError(
            "LLM object must expose get_response(), generate(), or be callable."
        )

    if isinstance(response, list):
        return str(response[0])

    return str(response)


def get_example_text(example: dict) -> str:
    return str(
        example.get("text")
        or example.get("input")
        or example.get("sentence")
        or example.get("x")
        or ""
    )


def get_example_label(example: dict) -> str:
    return str(
        example.get("label")
        or example.get("label_text")
        or example.get("answer")
        or example.get("output")
        or example.get("target")
        or example.get("y")
        or ""
    )


def make_llm_prompt(
    candidate: PromptCandidate,
    text: str,
    labels: list[str],
    require_final_answer_tags: bool = False,
    task_type: str = "classification",
) -> str:
    rendered = candidate.render(text)

    # Multiple-choice (e.g. BBQ): the options are already rendered in the text; we
    # only need the model to return the chosen option letter inside the tags.
    if task_type in MULTIPLE_CHOICE_TASK_TYPES:
        return (
            f"{rendered}\n\n"
            f"Choose the single best option. "
            f"Return ONLY the letter of your chosen option (A, B, or C) inside "
            f"<final_answer> and </final_answer> tags."
        )

    # Generation tasks (e.g. GSM8K) have no fixed label set: ask for a free-form
    # final answer inside the tags. The task itself is described by the prompt /
    # task_description, so we only enforce the answer format here.
    if task_type in GENERATION_TASK_TYPES:
        return (
            f"{rendered}\n\n"
            f"Solve the problem and show your reasoning. "
            f"Return ONLY the final answer inside "
            f"<final_answer> and </final_answer> tags."
        )

    label_text = ", ".join(labels)

    if require_final_answer_tags:
        return (
            f"{rendered}\n\n"
            f"Allowed labels: {label_text}\n"
            f"Return only the answer inside <final_answer> and </final_answer> tags."
        )

    return (
        f"{rendered}\n\n"
        f"Allowed labels: {label_text}\n"
        f"Return only one label: {label_text}."
    )


def heuristic_fairness_risk(prompt: str) -> float:
    lowered = str(prompt).lower()

    fairness_terms = [
        "gender",
        "race",
        "ethnicity",
        "nationality",
        "religion",
        "age",
        "location",
        "demographic",
        "stereotype",
        "names",
    ]

    if any(term in lowered for term in fairness_terms):
        return 0.12

    if "do not infer" in lowered or "avoid assumptions" in lowered:
        return 0.18

    return 0.30


def heuristic_risk_adjustment(prompt: str) -> float:
    lowered = str(prompt).lower()

    risk_terms = [
        "do not hallucinate",
        "use only",
        "provided context",
        "given sentence",
        "insufficient",
        "cannot be determined",
        "unsupported",
        "missing context",
    ]

    if any(term in lowered for term in risk_terms):
        return -0.10

    return 0.0


def load_inloop_fairness_pairs(config: dict) -> list[dict]:
    """
    Load the fixed counterfactual pairs used for in-loop fairness evaluation.

    Returns at most ``fairness.eval_pairs`` pairs from ``fairness.fairness_data``
    (falling back to the top-level ``fairness_data``). Returns [] when in-loop
    fairness is disabled or no data is available, in which case the evaluator
    falls back to the cheap prompt-keyword heuristic.
    """
    fairness_cfg = config.get("fairness", {}) or {}

    if not bool(fairness_cfg.get("in_loop", False)):
        return []

    data_path = fairness_cfg.get("fairness_data") or config.get("fairness_data")
    if not data_path:
        return []

    fairness_data = load_yaml(data_path)
    pairs = fairness_data.get("pairs", []) or []

    cap = int(fairness_cfg.get("eval_pairs", 10))
    if cap > 0:
        pairs = pairs[:cap]

    return pairs


def _fairness_mode(config: dict) -> str:
    """
    Resolve the in-loop fairness mode.

    ``fairness.mode`` is honored when set; otherwise BBQ datasets default to the
    canonical bias-score path and everything else to the counterfactual path.
    """
    fairness_cfg = config.get("fairness", {}) or {}
    mode = str(fairness_cfg.get("mode", "")).strip().lower()
    if mode:
        return mode
    if str(config.get("dataset", "")).strip().lower() == "bbq":
        return "bbq_bias_score"
    return "counterfactual"


def load_inloop_bbq_items(config: dict) -> list[dict]:
    """
    Load the BBQ items used for in-loop bias-score fairness evaluation.

    Reads a JSONL file (``fairness.fairness_data``) produced by
    ``scripts/build_bbq_fairness_set.py``; each line carries a rendered ``text``
    plus the metadata needed to score bias (options, answer_info,
    stereotyped_groups, question_polarity, context_condition, label_idx). Returns
    at most ``fairness.eval_pairs`` items, or [] when in-loop fairness is off or
    no data is available.
    """
    fairness_cfg = config.get("fairness", {}) or {}

    if not bool(fairness_cfg.get("in_loop", False)):
        return []

    data_path = fairness_cfg.get("fairness_data") or config.get("fairness_data")
    if not data_path or not Path(data_path).exists():
        return []

    items: list[dict] = []
    with open(data_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    cap = int(fairness_cfg.get("eval_pairs", 10))
    if cap > 0:
        items = items[:cap]

    return items


class LLMObjectiveEvaluator(ObjectiveEvaluator):
    """
    LM Studio-backed objective evaluator for budgeted MO-CAPO.

    It evaluates one candidate on one block and returns:
      - performance: block accuracy
      - cost: weighted input/output token estimate
      - risk: 1 - accuracy with prompt-risk adjustment
      - fairness_risk: real combined counterfactual fairness when in-loop
        fairness is enabled (``fairness.in_loop: true``), otherwise a
        prompt-level keyword heuristic

    When in-loop fairness is on, each *unique* candidate prompt is additionally
    evaluated once on a fixed set of counterfactual pairs (cached by instruction
    text), so the fairness_risk in the objective vector reflects the model's
    actual demographic sensitivity. The one-time fairness-eval token cost is
    folded into that candidate's first block cost so it counts against the
    optimization budget.
    """

    def __init__(self, config: dict, llm: Any = None):
        self.config = config
        self.labels = config.get("labels", ["subjective", "objective"])
        self.task_type = str(config.get("task_type", "classification"))
        self.is_generation = self.task_type in GENERATION_TASK_TYPES
        self.is_multiple_choice = self.task_type in MULTIPLE_CHOICE_TASK_TYPES
        self.evaluation_cfg = config.get("evaluation", {})
        self.cost_cfg = config.get("cost", {})
        self.fairness_cfg = config.get("fairness", {}) or {}
        # Allow injecting an LLM (used by tests); otherwise build from config.
        self.llm = llm if llm is not None else build_llm_from_config(config)

        self.input_weight = float(self.cost_cfg.get("input_weight", 0.08))
        self.output_weight = float(self.cost_cfg.get("output_weight", 0.32))
        self.require_final_answer_tags = bool(
            self.evaluation_cfg.get("require_final_answer_tags", False)
        )

        # In-loop fairness setup. BBQ uses the canonical bias-score path (a set of
        # items); everything else uses counterfactual pairs.
        self.fairness_mode = _fairness_mode(config)
        if self.fairness_mode == "bbq_bias_score":
            self.fairness_pairs = []
            self.bbq_fairness_items = load_inloop_bbq_items(config)
            self.fairness_in_loop = bool(self.bbq_fairness_items)
        else:
            self.fairness_pairs = load_inloop_fairness_pairs(config)
            self.bbq_fairness_items = []
            self.fairness_in_loop = bool(self.fairness_pairs)
        self.use_expected_same = bool(
            self.fairness_cfg.get("use_expected_same_prediction", True)
        )
        # How the BBQ scalar fairness_risk is distilled from sAMB/sDIS. Default
        # "samb" (original); "max_amb_dis" folds sDIS in so the objective stays
        # discriminative when |sAMB| saturates. See heal_capo.fairness_bbq.
        self.bbq_score = str(self.fairness_cfg.get("bbq_score", "samb"))
        self.combined_fairness_config = CombinedFairnessConfig(
            flip_weight=float(self.fairness_cfg.get("flip_weight", 0.50)),
            group_gap_weight=float(self.fairness_cfg.get("group_gap_weight", 0.25)),
            bias_weight=float(self.fairness_cfg.get("bias_weight", 0.15)),
            debt_weight=float(self.fairness_cfg.get("debt_weight", 0.10)),
            clamp=bool(self.fairness_cfg.get("clamp", True)),
        )
        # Cache real fairness by instruction text so it is computed once per
        # unique prompt: {instruction: (fairness_risk, details)}.
        self._fairness_cache: dict[str, tuple[float, dict]] = {}

    def _evaluate_candidate_fairness(
        self,
        candidate: PromptCandidate,
    ) -> tuple[float, dict, float]:
        """
        Compute real combined counterfactual fairness for one candidate.

        Returns ``(fairness_risk, details, extra_cost)``. On a cache hit,
        ``extra_cost`` is 0.0 (the candidate's fairness was already paid for).
        """
        key = candidate.instruction

        if key in self._fairness_cache:
            risk, details = self._fairness_cache[key]
            return risk, details, 0.0

        base_predictions: list[str] = []
        counterfactual_predictions: list[str] = []
        expected_same: list[bool] = []
        f_input_tokens = 0
        f_output_tokens = 0

        for pair in self.fairness_pairs:
            base_text = str(pair.get("base_text", ""))
            cf_text = str(pair.get("counterfactual_text", ""))

            base_prompt = make_llm_prompt(
                candidate=candidate,
                text=base_text,
                labels=self.labels,
                require_final_answer_tags=self.require_final_answer_tags,
                task_type=self.task_type,
            )
            cf_prompt = make_llm_prompt(
                candidate=candidate,
                text=cf_text,
                labels=self.labels,
                require_final_answer_tags=self.require_final_answer_tags,
                task_type=self.task_type,
            )

            base_response = get_llm_response(self.llm, base_prompt)
            cf_response = get_llm_response(self.llm, cf_prompt)

            base_predictions.append(
                extract_label(base_response, self.labels).strip().lower()
            )
            counterfactual_predictions.append(
                extract_label(cf_response, self.labels).strip().lower()
            )
            expected_same.append(bool(pair.get("expected_same_prediction", True)))

            f_input_tokens += simple_token_count(base_prompt) + simple_token_count(cf_prompt)
            f_output_tokens += simple_token_count(base_response) + simple_token_count(cf_response)

        fairness_result = evaluate_combined_fairness(
            base_predictions=base_predictions,
            counterfactual_predictions=counterfactual_predictions,
            expected_same_prediction=expected_same if self.use_expected_same else None,
            config=self.combined_fairness_config,
        )

        extra_cost = (
            self.input_weight * f_input_tokens
            + self.output_weight * f_output_tokens
        )

        details = {
            "fairness_method": fairness_result.details.get("method"),
            "counterfactual_flip_rate": fairness_result.counterfactual_flip_rate,
            "fairness_num_pairs": fairness_result.num_pairs,
            "fairness_num_flips": fairness_result.num_flips,
            "fairness_eval_input_tokens": f_input_tokens,
            "fairness_eval_output_tokens": f_output_tokens,
            "fairness_eval_cost": extra_cost,
        }

        self._fairness_cache[key] = (fairness_result.fairness_risk, details)

        return fairness_result.fairness_risk, details, extra_cost

    def _evaluate_candidate_fairness_bbq(
        self,
        candidate: PromptCandidate,
    ) -> tuple[float, dict, float]:
        """
        Compute the canonical BBQ bias score (|sAMB| drives fairness_risk) for one
        candidate over the in-loop BBQ fairness items.

        Returns ``(fairness_risk, details, extra_cost)``; ``extra_cost`` is 0.0 on
        a cache hit (the candidate's fairness was already paid for). Cached by
        instruction text, exactly like the counterfactual path.
        """
        key = candidate.instruction

        if key in self._fairness_cache:
            risk, details = self._fairness_cache[key]
            return risk, details, 0.0

        letter_to_index = {"A": 0, "B": 1, "C": 2}
        items = []
        predicted_indices: list[int] = []
        f_input_tokens = 0
        f_output_tokens = 0

        for raw_item in self.bbq_fairness_items:
            text = str(raw_item.get("text", ""))

            prompt = make_llm_prompt(
                candidate=candidate,
                text=text,
                labels=self.labels,
                require_final_answer_tags=self.require_final_answer_tags,
                task_type="multiple_choice",
            )
            response = get_llm_response(self.llm, prompt)

            letter = normalize_multiple_choice_answer(extract_final_answer(response))
            predicted_indices.append(letter_to_index.get(letter, -1))
            items.append(item_from_meta(raw_item))

            f_input_tokens += simple_token_count(prompt)
            f_output_tokens += simple_token_count(response)

        fairness_result = evaluate_bbq_fairness(
            items, predicted_indices, score=self.bbq_score
        )

        extra_cost = (
            self.input_weight * f_input_tokens
            + self.output_weight * f_output_tokens
        )

        scores = fairness_result.details
        details = {
            "fairness_method": scores.get("method"),
            "bbq_sAMB": scores.get("sAMB"),
            "bbq_sDIS": scores.get("sDIS"),
            "bbq_acc_ambig": scores.get("acc_ambig"),
            "bbq_acc_disambig": scores.get("acc_disambig"),
            "fairness_num_pairs": fairness_result.num_pairs,
            "fairness_eval_input_tokens": f_input_tokens,
            "fairness_eval_output_tokens": f_output_tokens,
            "fairness_eval_cost": extra_cost,
        }

        self._fairness_cache[key] = (fairness_result.fairness_risk, details)

        return fairness_result.fairness_risk, details, extra_cost

    def _score_prediction(self, raw_response: str, gold: str) -> tuple[str, bool]:
        """
        Extract the prediction and decide correctness for this task type.

        classification -> fixed-label match; multiple-choice (e.g. BBQ) -> option
        letter match; generation (e.g. GSM8K) -> free-form final-answer extraction
        with numeric normalization.
        """
        if self.is_multiple_choice:
            pred = normalize_multiple_choice_answer(extract_final_answer(raw_response))
            return pred, pred == str(gold).strip().upper()

        if self.is_generation:
            pred = normalize_numeric_answer(extract_final_answer(raw_response))
            return pred, pred == normalize_numeric_answer(gold)

        pred = extract_label(raw_response, self.labels).strip().lower()
        return pred, pred == gold

    def evaluate(self, candidate: PromptCandidate, data) -> EvaluationResult:
        correct = 0
        total = 0
        input_tokens = 0
        output_tokens = 0
        rows = []

        for example in data:
            text = get_example_text(example)
            gold = get_example_label(example).strip().lower()

            prompt = make_llm_prompt(
                candidate=candidate,
                text=text,
                labels=self.labels,
                require_final_answer_tags=self.require_final_answer_tags,
                task_type=self.task_type,
            )

            raw_response = get_llm_response(self.llm, prompt)
            pred, is_correct = self._score_prediction(raw_response, gold)

            correct += int(is_correct)
            total += 1

            prompt_tokens = simple_token_count(prompt)
            response_tokens = simple_token_count(raw_response)

            input_tokens += prompt_tokens
            output_tokens += response_tokens

            rows.append(
                {
                    "text": text,
                    "gold": gold,
                    "prediction": pred,
                    "raw_response": raw_response,
                    "correct": is_correct,
                    "input_tokens": prompt_tokens,
                    "output_tokens": response_tokens,
                }
            )

        performance = correct / total if total else 0.0

        cost = (
            self.input_weight * input_tokens
            + self.output_weight * output_tokens
        )

        base_risk = 1.0 - performance
        risk = min(
            1.0,
            max(
                0.0,
                base_risk + heuristic_risk_adjustment(candidate.instruction),
            ),
        )

        fairness_details: dict = {}

        if self.fairness_in_loop:
            if self.fairness_mode == "bbq_bias_score":
                fairness_risk, fairness_details, fairness_cost = (
                    self._evaluate_candidate_fairness_bbq(candidate)
                )
                fairness_source = "bbq_bias_score"
            else:
                fairness_risk, fairness_details, fairness_cost = (
                    self._evaluate_candidate_fairness(candidate)
                )
                fairness_source = "counterfactual_in_loop"
            # Fold the one-time fairness-eval cost into this block's cost so it
            # counts against the optimization budget (0.0 on cache reuse).
            cost += fairness_cost
            input_tokens += int(fairness_details.get("fairness_eval_input_tokens", 0))
            output_tokens += int(fairness_details.get("fairness_eval_output_tokens", 0))
        else:
            fairness_risk = heuristic_fairness_risk(candidate.instruction)
            fairness_source = "prompt_heuristic"

        return EvaluationResult(
            candidate_id=candidate.candidate_id,
            performance=performance,
            cost=cost,
            risk=risk,
            fairness_risk=fairness_risk,
            drift=0.0,
            n_examples=total,
            details={
                "evaluator": "lmstudio",
                "model_id": self.config.get("llm", {}).get("model_id"),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "correct": correct,
                "total": total,
                "fairness_source": fairness_source,
                **fairness_details,
                "predictions": rows,
            },
        )


def build_objective_evaluator(
    config: dict,
    force_no_llm: bool = False,
) -> ObjectiveEvaluator:
    evaluation_cfg = config.get("evaluation", {})
    use_llm = bool(evaluation_cfg.get("use_llm", False))

    if force_no_llm:
        use_llm = False

    if use_llm:
        return LLMObjectiveEvaluator(config)

    return ToyObjectiveEvaluator()


def get_prompt_pool(config: dict) -> list[dict]:
    prompt_pool_path = config.get("prompt_pool")

    if prompt_pool_path:
        prompt_pool_config = load_yaml(prompt_pool_path)
        prompt_pool = prompt_pool_config.get("prompt_pool", [])
    else:
        prompt_pool = config.get("prompt_pool_inline", [])

    if not prompt_pool:
        raise ValueError(
            "Config must contain prompt_pool path or prompt_pool_inline list."
        )

    normalized = []

    for idx, item in enumerate(prompt_pool):
        if isinstance(item, str):
            normalized.append(
                {
                    "id": f"prompt_{idx}",
                    "category": "unknown",
                    "prompt": item,
                }
            )
            continue

        prompt = str(item.get("prompt", "")).strip()

        if not prompt:
            continue

        normalized.append(
            {
                "id": str(item.get("id", f"prompt_{idx}")),
                "category": str(item.get("category", "unknown")),
                "prompt": prompt,
            }
        )

    if not normalized:
        raise ValueError("No valid prompts found.")

    return normalized


def _example_to_row(ex) -> dict:
    """
    Flatten a dataset Example into a runner row, preserving a chain-of-thought
    rationale when the dataset provides one (e.g. GSM8K's full worked answer).
    The rationale is used to build reasoning-bearing few-shot demonstrations.
    """
    metadata = getattr(ex, "metadata", None) or {}
    row = {
        "text": str(getattr(ex, "text", "")),
        "label": str(getattr(ex, "label", "")),
    }
    rationale = metadata.get("full_answer") or metadata.get("rationale")
    if rationale:
        row["rationale"] = str(rationale)
    # Preserve dataset metadata (e.g. BBQ's context_condition / question_polarity /
    # answer_info) so the fairness scorer and held-out eval can read it.
    if metadata:
        row["meta"] = dict(metadata)
    return row


def get_dev_data(config: dict) -> list[dict]:
    # 1. Explicit inline dev_data wins (used by tests / tiny demos).
    inline = config.get("dev_data")
    if inline:
        return inline

    # 2. Dataset-backed dev split (paper-style Ddev). Opt in via dev.dataset
    #    or a top-level `dev_dataset` name; loads a real stratified split.
    dev_cfg = config.get("dev") if isinstance(config.get("dev"), dict) else {}
    dataset_name = dev_cfg.get("dataset") or config.get("dev_dataset")
    if dataset_name:
        from experiments.datasets import load_paper_dataset

        split = load_paper_dataset(
            name=str(dataset_name),
            dev_size=int(dev_cfg.get("dev_size", config.get("dev_size", 40))),
            shots_size=int(dev_cfg.get("shots_size", 2)),
            test_size=int(dev_cfg.get("test_size", 5)),
            seed=int(config.get("seed", 0)),
            allow_smaller=bool(dev_cfg.get("allow_smaller", True)),
            stratified=bool(dev_cfg.get("stratified", True)),
        )
        return [_example_to_row(ex) for ex in split.dev]

    # 3. Default tiny hardcoded SUBJ demo set.
    return [
            {"text": "The movie was released in 1999.", "label": "objective"},
            {"text": "The acting is absolutely wonderful.", "label": "subjective"},
            {"text": "Paris is the capital of France.", "label": "objective"},
            {
                "text": "This boring film wastes its talented cast.",
                "label": "subjective",
            },
            {"text": "The book contains twelve chapters.", "label": "objective"},
            {
                "text": "The speech was moving and unforgettable.",
                "label": "subjective",
            },
            {"text": "The meeting started at 9 a.m.", "label": "objective"},
            {
                "text": "The design looks beautiful and modern.",
                "label": "subjective",
            },
    ]


def _shot_output(row: dict, label: str, task_type: str, use_rationale: bool) -> str:
    """
    Build the demonstration ``output`` string for one few-shot example.

    For generation/reasoning tasks (e.g. GSM8K) with ``use_rationale`` on, the
    demonstration shows the worked chain-of-thought ending in the final-answer
    tag -- this is both what teaches the model to reason AND the cost lever
    (longer demos cost more input tokens). Classification (and rationale-less)
    demos collapse to the bare tagged label.
    """
    if (
        use_rationale
        and task_type in GENERATION_TASK_TYPES
        and row.get("rationale")
    ):
        rationale = str(row["rationale"]).strip()
        # GSM8K rationales end in "#### <answer>"; replace that marker with the
        # tagged final answer so the demo matches the required output format.
        if "####" in rationale:
            reasoning = rationale.split("####")[0].strip()
        else:
            reasoning = rationale
        return f"{reasoning}\n<final_answer>{label}</final_answer>"

    return f"<final_answer>{label}</final_answer>"


def build_shot_pool(config: dict, dev_data: list[dict]) -> list[dict]:
    """
    Build the few-shot example pool the evolutionary few-shot operator draws
    from. Returns [] when few-shot search is disabled (``few_shot.enabled``).

    Each entry is ``{"input": text, "output": <demonstration answer>}`` so the
    rendered demonstration matches the required answer format. For reasoning
    tasks the output carries the chain-of-thought (see ``_shot_output``); for
    classification it is the bare tagged label. The pool is drawn from a
    dedicated ``few_shot.shots_data`` dataset split when given, otherwise from
    the dev split (capped by ``few_shot.pool_size``).
    """
    fs_cfg = config.get("few_shot", {}) or {}
    if not bool(fs_cfg.get("enabled", False)):
        return []

    task_type = str(config.get("task_type", "classification"))
    # Default to chain-of-thought demos for generation tasks; opt-out via
    # few_shot.use_rationale: false.
    use_rationale = bool(fs_cfg.get("use_rationale", True))

    rows = dev_data
    shots_dataset = fs_cfg.get("shots_data")
    if shots_dataset:
        from experiments.datasets import load_paper_dataset

        split = load_paper_dataset(
            name=str(shots_dataset),
            dev_size=int(fs_cfg.get("pool_size", 20)),
            shots_size=2,
            test_size=int(fs_cfg.get("test_size", 5)),
            seed=int(config.get("seed", 0)),
            allow_smaller=True,
            stratified=bool(fs_cfg.get("stratified", task_type == "classification")),
        )
        rows = [_example_to_row(ex) for ex in split.dev]

    pool_size = int(fs_cfg.get("pool_size", 20))
    pool: list[dict] = []
    for row in rows[:pool_size]:
        text = str(row.get("text", "")).strip()
        label = str(row.get("label", "")).strip()
        if not text or not label:
            continue
        pool.append(
            {
                "input": text,
                "output": _shot_output(row, label, task_type, use_rationale),
            }
        )
    return pool


def make_candidates(config: dict) -> list[PromptCandidate]:
    dataset = config.get("dataset", "subj")
    task_type = config.get("task_type", "classification")

    candidates = []

    for idx, item in enumerate(get_prompt_pool(config)):
        candidates.append(
            PromptCandidate(
                instruction=item["prompt"],
                metadata={
                    "prompt_pool_id": item["id"],
                    "method": item["id"],
                    "category": item["category"],
                    "row_index": idx,
                    "dataset": dataset,
                    "task_type": task_type,
                    "source": "budgeted_mocapo",
                },
            )
        )

    return candidates


def get_task_description(config: dict) -> str:
    return str(
        config.get(
            "task_description",
            (
                "The dataset contains sentences labeled as either subjective or objective. "
                "The task is to classify each sentence as either subjective or objective. "
                "The class will be extracted between the markers "
                "<final_answer>answer</final_answer>."
            ),
        )
    )


def make_meta_llm(config: dict, force_no_llm: bool = False):
    evolutionary_cfg = config.get("evolutionary", {})
    use_meta_llm = bool(evolutionary_cfg.get("use_meta_llm", False))

    if force_no_llm or not use_meta_llm:
        return None

    return build_llm_from_config(config)


def get_incumbents_from_portfolio(
    population: list[PromptCandidate],
    portfolio: PromptPortfolio,
) -> list[PromptCandidate]:
    population_ids = {candidate.candidate_id for candidate in population}

    evaluated_results = [
        result
        for candidate_id, result in portfolio.evaluations.items()
        if candidate_id in population_ids
    ]

    if not evaluated_results:
        return []

    incumbent_ids = set(non_dominated_ids(evaluated_results))

    return [
        candidate
        for candidate in population
        if candidate.candidate_id in incumbent_ids
    ]


def front_candidates_from_portfolio(
    portfolio: PromptPortfolio,
) -> list[PromptCandidate]:
    """
    Non-dominated candidates over the WHOLE portfolio.

    Unlike :func:`get_incumbents_from_portfolio` (restricted to the current
    population), this returns exactly the set the final reported Pareto front is
    drawn from (see the ``non_dominated_ids(all_portfolio.evaluations.values())``
    call at the end of :func:`run_budgeted_mocapo`). The final deepening pass
    races precisely these candidates so no reported front member rests on a
    1-block (optimistic, ~10-example) estimate.
    """
    if not portfolio.evaluations:
        return []

    front_ids = set(non_dominated_ids(portfolio.evaluations.values()))

    candidates = []
    for front_id in front_ids:
        candidate = portfolio.get(front_id)
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def compact_bbq_instruction(instruction: str) -> str:
    """
    Build a short zero-shot BBQ instruction from a longer evolved prompt.

    The repair keeps the important fairness behavior while aggressively cutting
    token cost. It is deterministic and LLM-free so it can run inside the
    budgeted optimizer without spending extra meta-model calls.
    """
    lowered = str(instruction or "").lower()

    if "demographic" in lowered or "stereotype" in lowered:
        return (
            "Use only stated context. Ignore demographic stereotypes. If the "
            "answer is unsupported, choose the unknown option. Return only "
            "<final_answer>A</final_answer>, <final_answer>B</final_answer>, "
            "or <final_answer>C</final_answer>."
        )

    if "unknown" in lowered or "cannot be determined" in lowered:
        return (
            "Pick the option directly supported by the context. If unclear, "
            "choose the unknown/cannot-be-determined option. Return only the "
            "tagged option letter."
        )

    return (
        "Choose the option supported by the context. Do not guess. Return only "
        "<final_answer>A</final_answer>, <final_answer>B</final_answer>, or "
        "<final_answer>C</final_answer>."
    )


def make_cost_repair_candidates(
    candidates: list[PromptCandidate],
    portfolio: PromptPortfolio,
    max_source_candidates: int = 4,
    variants_per_candidate: int = 3,
) -> list[PromptCandidate]:
    """
    Generate cheap repair variants from the current front.

    Variants deliberately remove or reduce few-shot examples and compact the
    instruction. The goal is to recover cheaper Pareto points near the same
    accuracy/fairness region, which the large-held-out v2/v3 diagnostics showed
    FairCAPO was missing relative to NSGA.
    """
    scored: list[tuple[float, PromptCandidate]] = []

    for candidate in candidates:
        result = portfolio.evaluations.get(candidate.candidate_id)
        if result is None:
            continue

        score = (
            float(result.performance)
            - 0.8 * float(result.fairness_risk)
            - 0.25 * (float(result.cost) / 6000.0)
            - 0.5 * float(result.risk)
        )
        scored.append((score, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)

    repairs: list[PromptCandidate] = []
    seen: set[tuple[str, int]] = set()

    for _, source in scored[: max(1, max_source_candidates)]:
        variants: list[tuple[str, str, list[dict]]] = [
            ("zero_shot_same_instruction", source.instruction, []),
            ("one_shot_same_instruction", source.instruction, list(source.examples[:1])),
            ("zero_shot_compact_instruction", compact_bbq_instruction(source.instruction), []),
        ]

        for operator, instruction, examples in variants[: max(1, variants_per_candidate)]:
            key = (instruction.strip(), len(examples))
            if key in seen:
                continue
            seen.add(key)

            repair = PromptCandidate(
                instruction=instruction.strip(),
                examples=[dict(example) for example in examples],
                parent_ids=[source.candidate_id],
                metadata={
                    "method": f"cost_repair_{operator}",
                    "category": "cost_repair",
                    "source": "cost_repair",
                    "operator": operator,
                    "parent_ids": [source.candidate_id],
                    "repair_source_candidate_id": source.candidate_id,
                    "repair_source_num_few_shot": len(source.examples),
                },
            )
            repairs.append(repair)

    return repairs


def portfolio_contains_candidate(
    portfolio: PromptPortfolio,
    candidate: PromptCandidate,
) -> bool:
    return candidate.candidate_id in portfolio.evaluations


def evaluate_initial_population(
    candidates: list[PromptCandidate],
    block_evaluator: BlockEvaluator,
    budget_allocator: BudgetAllocator,
    num_seed_blocks: int = 1,
) -> tuple[list[PromptCandidate], PromptPortfolio, list[dict]]:
    """
    MO-CAPO initialization.

    Evaluate each initial population member on the same initial block set,
    then create the incumbent set from non-dominated candidates.
    """
    portfolio = PromptPortfolio()
    events: list[dict] = []

    seed_block_ids = block_evaluator.block_ids()[: max(1, num_seed_blocks)]

    for candidate in candidates:
        evaluated_blocks = []

        for block_id in seed_block_ids:
            if budget_allocator.exhausted:
                events.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "method": candidate.metadata.get("method"),
                        "event_type": "initialization_budget_stop",
                        "accepted": False,
                        "rejected": True,
                        "reason": (
                            "Budget exhausted during initial population evaluation."
                        ),
                        "evaluated_blocks": str(evaluated_blocks),
                        "budget_used": budget_allocator.used_budget,
                        "remaining_budget": budget_allocator.remaining_budget,
                    }
                )
                break

            try:
                evaluation = block_evaluator.evaluate_block(
                    candidate=candidate,
                    block_id=block_id,
                    use_cache=True,
                )
                budget_allocator.record_block_evaluation(evaluation)
            except RuntimeError as exc:
                events.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "method": candidate.metadata.get("method"),
                        "event_type": "initialization_budget_error",
                        "accepted": False,
                        "rejected": True,
                        "reason": str(exc),
                        "evaluated_blocks": str(evaluated_blocks),
                        "budget_used": budget_allocator.used_budget,
                        "remaining_budget": budget_allocator.remaining_budget,
                    }
                )
                break

            evaluated_blocks.append(block_id)

        if not evaluated_blocks:
            continue

        candidate.metadata["evaluated_blocks"] = sorted(evaluated_blocks)

        aggregate = block_evaluator.aggregate_candidate(
            candidate_id=candidate.candidate_id,
            block_ids=evaluated_blocks,
        )

        portfolio.add(candidate, aggregate)

        events.append(
            {
                "candidate_id": candidate.candidate_id,
                "method": candidate.metadata.get("method"),
                "event_type": "initial_population",
                "accepted": True,
                "rejected": False,
                "reason": "Initial population candidate evaluated on seed blocks.",
                "evaluated_blocks": str(evaluated_blocks),
                "budget_used": budget_allocator.used_budget,
                "remaining_budget": budget_allocator.remaining_budget,
            }
        )

    evaluated_population = [
        candidate
        for candidate in candidates
        if candidate.candidate_id in portfolio.evaluations
    ]

    return evaluated_population, portfolio, events


def make_evolutionary_event(
    iteration: int,
    offspring_index: int,
    op_result,
    decision,
) -> dict:
    return {
        "candidate_id": op_result.candidate.candidate_id,
        "method": op_result.candidate.metadata.get("method"),
        "event_type": "evolutionary_intensification",
        "iteration": iteration,
        "offspring_index": offspring_index,
        "operator": op_result.operator,
        "parent_ids": str(op_result.parent_ids),
        "used_meta_llm": op_result.used_meta_llm,
        "accepted": decision.accepted,
        "rejected": decision.rejected,
        "reason": decision.reason,
        "evaluated_blocks": str(decision.evaluated_blocks),
        "compared_against": decision.compared_against,
        "budget_used": decision.budget_used,
        "remaining_budget": decision.metadata.get("remaining_budget"),
        "budget_utilization": decision.metadata.get("budget_utilization"),
    }


def make_environmental_event(
    iteration: int,
    decision,
) -> dict:
    return {
        "candidate_id": "",
        "method": "environmental_selection",
        "event_type": "environmental_selection",
        "iteration": iteration,
        "accepted": True,
        "rejected": False,
        "reason": decision.reason,
        "kept_ids": str(decision.kept_ids),
        "removed_ids": str(decision.removed_ids),
        "evaluated_blocks": "[]",
        "budget_used": None,
        "remaining_budget": None,
        "metadata": json.dumps(decision.metadata, default=_json_default),
    }


def make_advance_event(
    iteration: int,
    decision,
) -> dict:
    return {
        "candidate_id": decision.candidate_id,
        "method": "advance_incumbents",
        "event_type": "advance_incumbent",
        "iteration": iteration,
        "accepted": decision.advanced,
        "rejected": not decision.advanced,
        "reason": decision.reason,
        "block_id": decision.block_id,
        "evaluated_blocks": str(decision.evaluated_blocks_after),
        "budget_used": decision.budget_used,
        "remaining_budget": decision.remaining_budget,
        "metadata": json.dumps(decision.metadata, default=_json_default),
    }


def _front_snapshot(
    label: str,
    iteration: int,
    front: list[PromptCandidate],
    portfolio: PromptPortfolio,
    budget_allocator: BudgetAllocator,
) -> dict:
    """Lightweight (LLM-free) snapshot of the current incumbent front.

    Records only floats already computed during search — the budget spent so
    far and each front member's objectives — so a post-hoc pass can rebuild the
    HV/nR2-vs-budget trajectory (paper Fig 2) without re-running anything.
    """
    points = []
    for candidate in front:
        result = portfolio.evaluations.get(candidate.candidate_id)
        if result is None:
            continue
        points.append(
            {
                "performance": float(result.performance),
                "cost": float(result.cost),
                "risk": float(result.risk),
                "fairness_risk": float(result.fairness_risk),
            }
        )

    return {
        "label": label,
        "iteration": int(iteration),
        "budget_used": float(budget_allocator.used_budget),
        "budget_utilization": float(budget_allocator.utilization),
        "front_size": len(points),
        "front": points,
    }


def compute_trajectory_metrics(
    snapshots: list[dict],
    bounds_config: dict | None = None,
    num_preference_vectors: int = 500,
    seed: int = 0,
) -> list[dict]:
    """Enrich raw front snapshots with HV / HV_opt / HV_pes / Gap / nR2.

    A SINGLE global normalization basis is used for every snapshot so the
    hypervolumes are comparable across the trajectory (the paper normalizes over
    the union of all incumbents). Bounds come from ``bounds_config`` when given,
    else are inferred from the union of every point seen across all snapshots.
    Pure function: no I/O, no LLM — safe to unit-test with synthetic snapshots.
    """
    objective_specs = DEFAULT_OBJECTIVE_SPECS

    all_results = [
        EvaluationResult(
            candidate_id=f"snap{i}_pt{j}",
            performance=pt["performance"],
            cost=pt["cost"],
            risk=pt["risk"],
            fairness_risk=pt["fairness_risk"],
        )
        for i, snap in enumerate(snapshots)
        for j, pt in enumerate(snap.get("front", []))
    ]

    bounds = fixed_bounds_from_config(bounds_config, objective_specs)
    if bounds is None and all_results:
        bounds = infer_bounds(all_results, objective_specs)

    enriched = []

    # Use the LAST snapshot's front as the fixed reference for nR2, so nR2
    # measures convergence: "how much of the final front's utility does each
    # intermediate front already capture?"  Without a separate reference set,
    # nR2(cand, ref=cand) = 0.0 by definition (distance to self is zero).
    final_results: list[EvaluationResult] = []
    if snapshots:
        final_snap = snapshots[-1]
        final_results = [
            EvaluationResult(
                candidate_id=f"ref_{j}",
                performance=pt["performance"],
                cost=pt["cost"],
                risk=pt["risk"],
                fairness_risk=pt["fairness_risk"],
            )
            for j, pt in enumerate(final_snap.get("front", []))
        ]

    for snap in snapshots:
        results = [
            EvaluationResult(
                candidate_id=f"pt{j}",
                performance=pt["performance"],
                cost=pt["cost"],
                risk=pt["risk"],
                fairness_risk=pt["fairness_risk"],
            )
            for j, pt in enumerate(snap.get("front", []))
        ]

        row = dict(snap)
        if results and bounds is not None:
            summary = summarize_mo_metrics(
                candidate_results=results,
                reference_results=final_results if final_results else None,
                objective_specs=objective_specs,
                num_preference_vectors=num_preference_vectors,
                seed=seed,
                bounds=bounds,
            )
            row["hypervolume"] = summary.hypervolume
            row["optimistic_hypervolume"] = summary.optimistic_hypervolume
            row["pessimistic_hypervolume"] = summary.pessimistic_hypervolume
            row["approximation_gap"] = summary.approximation_gap
            row["nr2"] = summary.nr2
        else:
            row["hypervolume"] = None
            row["optimistic_hypervolume"] = None
            row["pessimistic_hypervolume"] = None
            row["approximation_gap"] = None
            row["nr2"] = None
        enriched.append(row)

    return enriched


def run_budgeted_mocapo(
    config: dict,
    force_no_llm: bool = False,
) -> tuple[PromptPortfolio, PromptPortfolio, list[dict], dict, list[dict]]:
    candidates = make_candidates(config)
    dev_data = get_dev_data(config)

    budget_cfg = config.get("budget", {})
    intensification_cfg = config.get("intensification", {})
    evolutionary_cfg = config.get("evolutionary", {})

    block_size = int(config.get("block_size", 2))
    max_blocks_per_challenger = intensification_cfg.get(
        "max_blocks_per_challenger",
        None,
    )
    # After the evolutionary loop, race the final non-dominated front up to the
    # full block depth so no reported front member rests on a 1-block
    # (optimistic, ~10-example) estimate. Default on; opt-out for ablations.
    final_intensification = bool(
        intensification_cfg.get("final_intensification", True)
    )
    max_budget = float(budget_cfg.get("max_budget", 500.0))
    cost_repair_cfg = config.get("cost_repair", {}) or {}
    cost_repair_enabled = bool(cost_repair_cfg.get("enabled", False))
    cost_repair_reserved_budget = 0.0
    if cost_repair_enabled:
        cost_repair_reserved_budget = max(
            float(cost_repair_cfg.get("reserve_budget", 0.0)),
            max_budget
            * max(0.0, float(cost_repair_cfg.get("reserve_budget_fraction", 0.0))),
        )
    allow_overspend = bool(budget_cfg.get("allow_overspend", False))
    # "cost" (weighted, default) or "tokens" (raw tokens, e.g. MO-CAPO's 7.5M).
    budget_unit = str(budget_cfg.get("unit", budget_cfg.get("budget_unit", "cost")))

    mu = int(
        evolutionary_cfg.get(
            "population_size",
            config.get("population_size", 10),
        )
    )
    max_iterations = int(evolutionary_cfg.get("max_iterations", 5))
    offspring_per_iteration = int(
        evolutionary_cfg.get("offspring_per_iteration", 4)
    )
    num_seed_blocks = int(config.get("num_seed_blocks", 1))
    random_seed = int(config.get("seed", evolutionary_cfg.get("random_seed", 0)))

    mutate_after_crossover = bool(
        evolutionary_cfg.get("mutate_after_crossover", True)
    )
    advance_after_iteration = bool(
        evolutionary_cfg.get("advance_after_iteration", True)
    )

    rng = random.Random(random_seed)

    evaluator = build_objective_evaluator(
        config=config,
        force_no_llm=force_no_llm,
    )

    block_evaluator = BlockEvaluator.from_data(
        evaluator=evaluator,
        data=dev_data,
        block_size=block_size,
        drop_last=bool(config.get("drop_last_block", False)),
    )

    budget_allocator = BudgetAllocator(
        max_budget=max_budget,
        allow_overspend=allow_overspend,
        budget_unit=budget_unit,
    )

    intensifier = Intensifier(
        block_evaluator=block_evaluator,
        budget_allocator=budget_allocator,
        config=IntensificationConfig(
            max_blocks_per_challenger=max_blocks_per_challenger,
            reject_when_dominated=bool(
                intensification_cfg.get("reject_when_dominated", True)
            ),
            accept_when_not_dominated_on_common_blocks=bool(
                intensification_cfg.get(
                    "accept_when_not_dominated_on_common_blocks",
                    True,
                )
            ),
            add_rejected_to_population=bool(
                intensification_cfg.get("add_rejected_to_population", True)
            ),
            use_cache=bool(intensification_cfg.get("use_cache", True)),
        ),
    )

    parent_selection_cfg = config.get("parent_selection", {}) or {}
    weighted_tiebreak_cfg = (
        parent_selection_cfg.get("weighted_tiebreak", {}) or {}
    )

    parent_selector = ParentSelector(
        config=ParentSelectionConfig(
            prefer_incumbents=bool(
                parent_selection_cfg.get("prefer_incumbents", True)
            ),
            use_crowding_distance=bool(
                parent_selection_cfg.get("use_crowding_distance", True)
            ),
            use_subset_dominance=bool(
                parent_selection_cfg.get("use_subset_dominance", True)
            ),
            use_weighted_tiebreak=bool(
                parent_selection_cfg.get("use_weighted_tiebreak", False)
            ),
            weighted_tiebreak={
                "performance": float(
                    weighted_tiebreak_cfg.get("performance", 1.0)
                ),
                "cost": float(weighted_tiebreak_cfg.get("cost", 0.0)),
                "risk": float(weighted_tiebreak_cfg.get("risk", 0.0)),
                "fairness_risk": float(
                    weighted_tiebreak_cfg.get("fairness_risk", 0.0)
                ),
                "drift": float(weighted_tiebreak_cfg.get("drift", 0.0)),
                "cost_scale": float(
                    weighted_tiebreak_cfg.get("cost_scale", 1.0)
                ),
            },
            random_seed=random_seed,
        ),
        rng=rng,
    )

    meta_llm = make_meta_llm(
        config=config,
        force_no_llm=force_no_llm,
    )

    few_shot_cfg = config.get("few_shot", {}) or {}
    shot_pool = build_shot_pool(config, dev_data)

    evolutionary_ops = EvolutionaryPromptOps(
        config=EvolutionaryOpsConfig(
            random_seed=random_seed,
            mutation_probability=float(
                evolutionary_cfg.get("mutation_probability", 1.0)
            ),
            crossover_probability=float(
                evolutionary_cfg.get("crossover_probability", 1.0)
            ),
            require_prompt_tags=bool(
                evolutionary_cfg.get("require_prompt_tags", True)
            ),
            preserve_output_format=bool(
                evolutionary_cfg.get("preserve_output_format", True)
            ),
            few_shot_probability=float(
                few_shot_cfg.get("few_shot_probability", 0.5)
            ),
            max_few_shot_examples=int(
                few_shot_cfg.get("max_few_shot_examples", 4)
            ),
        ),
        meta_llm=meta_llm,
        rng=rng,
    )

    environmental_cfg = config.get("environmental_selection", {}) or {}
    environmental_selector = EnvironmentalSelector(
        config=EnvironmentalSelectionConfig(
            population_size=mu,
            prefer_keep_incumbents=bool(
                environmental_cfg.get("prefer_keep_incumbents", True)
            ),
            remove_unevaluated_first=bool(
                environmental_cfg.get("remove_unevaluated_first", True)
            ),
            use_crowding_distance=bool(
                environmental_cfg.get("use_crowding_distance", True)
            ),
            use_dominance_fronts=bool(
                environmental_cfg.get("use_dominance_fronts", True)
            ),
            protect_low_cost_quantile=float(
                environmental_cfg.get("protect_low_cost_quantile", 0.0)
            ),
            protected_low_cost_min_count=int(
                environmental_cfg.get("protected_low_cost_min_count", 0)
            ),
            random_seed=random_seed,
        ),
        rng=rng,
    )

    initial_population = candidates[: max(2, min(mu, len(candidates)))]

    population, all_portfolio, event_rows = evaluate_initial_population(
        candidates=initial_population,
        block_evaluator=block_evaluator,
        budget_allocator=budget_allocator,
        num_seed_blocks=num_seed_blocks,
    )

    incumbents = get_incumbents_from_portfolio(
        population=population,
        portfolio=all_portfolio,
    )

    # §F trajectory: snapshot the incumbent front + budget spent at each stage so
    # we can later plot HV/nR2 vs budget (paper Fig 2). Floats only — no LLM cost.
    trajectory: list[dict] = [
        _front_snapshot(
            label="initial_population",
            iteration=0,
            front=incumbents,
            portfolio=all_portfolio,
            budget_allocator=budget_allocator,
        )
    ]

    task_description = get_task_description(config)

    def repair_budget_reached() -> bool:
        return (
            cost_repair_enabled
            and cost_repair_reserved_budget > 0.0
            and budget_allocator.remaining_budget <= cost_repair_reserved_budget
        )

    if len(population) < 2:
        event_rows.append(
            {
                "candidate_id": "",
                "method": "evolutionary_loop",
                "event_type": "evolutionary_stop",
                "iteration": 0,
                "accepted": False,
                "rejected": True,
                "reason": (
                    "Not enough evaluated population candidates for parent selection."
                ),
                "evaluated_blocks": "[]",
                "budget_used": budget_allocator.used_budget,
                "remaining_budget": budget_allocator.remaining_budget,
            }
        )

    for iteration in range(1, max_iterations + 1):
        if budget_allocator.exhausted:
            event_rows.append(
                {
                    "candidate_id": "",
                    "method": "evolutionary_loop",
                    "event_type": "budget_stop",
                    "iteration": iteration,
                    "accepted": False,
                    "rejected": True,
                    "reason": "Budget exhausted before evolutionary iteration.",
                    "evaluated_blocks": "[]",
                    "budget_used": budget_allocator.used_budget,
                    "remaining_budget": budget_allocator.remaining_budget,
                }
            )
            break

        if repair_budget_reached():
            event_rows.append(
                {
                    "candidate_id": "",
                    "method": "cost_repair_reserve",
                    "event_type": "budget_stop",
                    "iteration": iteration,
                    "accepted": False,
                    "rejected": True,
                    "reason": "Reserved remaining budget for final cost repair.",
                    "evaluated_blocks": "[]",
                    "budget_used": budget_allocator.used_budget,
                    "remaining_budget": budget_allocator.remaining_budget,
                }
            )
            break

        if len(population) < 2:
            break

        incumbent_ids = {
            incumbent.candidate_id
            for incumbent in incumbents
            if incumbent.candidate_id in all_portfolio.evaluations
        }

        for offspring_index in range(offspring_per_iteration):
            if budget_allocator.exhausted:
                event_rows.append(
                    {
                        "candidate_id": "",
                        "method": "evolutionary_loop",
                        "event_type": "budget_stop",
                        "iteration": iteration,
                        "offspring_index": offspring_index,
                        "accepted": False,
                        "rejected": True,
                        "reason": "Budget exhausted before offspring evaluation.",
                        "evaluated_blocks": "[]",
                        "budget_used": budget_allocator.used_budget,
                        "remaining_budget": budget_allocator.remaining_budget,
                    }
                )
                break

            if repair_budget_reached():
                event_rows.append(
                    {
                        "candidate_id": "",
                        "method": "cost_repair_reserve",
                        "event_type": "budget_stop",
                        "iteration": iteration,
                        "offspring_index": offspring_index,
                        "accepted": False,
                        "rejected": True,
                        "reason": "Reserved remaining budget for final cost repair.",
                        "evaluated_blocks": "[]",
                        "budget_used": budget_allocator.used_budget,
                        "remaining_budget": budget_allocator.remaining_budget,
                    }
                )
                break

            try:
                mother, father, parent_decisions = parent_selector.select_two_parents(
                    population=population,
                    incumbent_ids=incumbent_ids,
                    evaluations=all_portfolio.evaluations,
                )
            except ValueError as exc:
                event_rows.append(
                    {
                        "candidate_id": "",
                        "method": "parent_selection",
                        "event_type": "parent_selection_error",
                        "iteration": iteration,
                        "offspring_index": offspring_index,
                        "accepted": False,
                        "rejected": True,
                        "reason": str(exc),
                        "evaluated_blocks": "[]",
                        "budget_used": budget_allocator.used_budget,
                        "remaining_budget": budget_allocator.remaining_budget,
                    }
                )
                break

            op_result = evolutionary_ops.crossover(
                mother=mother,
                father=father,
                task_description=task_description,
            )

            if mutate_after_crossover:
                mutation_result = evolutionary_ops.mutate(
                    parent=op_result.candidate,
                    task_description=task_description,
                )
                op_result = mutation_result

            # Few-shot count mutation (MO-CAPO accuracy/cost lever). Applied
            # with probability few_shot_probability when a shot pool exists.
            if shot_pool and rng.random() <= evolutionary_ops.config.few_shot_probability:
                op_result = evolutionary_ops.mutate_few_shot(
                    parent=op_result.candidate,
                    shot_pool=shot_pool,
                )

            challenger = op_result.candidate

            try:
                decision = intensifier.intensify(
                    challenger=challenger,
                    incumbents=incumbents,
                    portfolio=all_portfolio,
                )
            except RuntimeError as exc:
                event_rows.append(
                    {
                        "candidate_id": challenger.candidate_id,
                        "method": challenger.metadata.get("method"),
                        "event_type": "budget_error",
                        "iteration": iteration,
                        "offspring_index": offspring_index,
                        "operator": op_result.operator,
                        "accepted": False,
                        "rejected": True,
                        "reason": str(exc),
                        "evaluated_blocks": "[]",
                        "budget_used": budget_allocator.used_budget,
                        "remaining_budget": budget_allocator.remaining_budget,
                    }
                )
                break

            event_row = make_evolutionary_event(
                iteration=iteration,
                offspring_index=offspring_index,
                op_result=op_result,
                decision=decision,
            )
            event_row["parent_selection"] = json.dumps(
                [
                    parent_decision.__dict__
                    for parent_decision in parent_decisions
                ],
                default=_json_default,
            )
            event_rows.append(event_row)

            if decision.accepted or portfolio_contains_candidate(
                all_portfolio,
                challenger,
            ):
                current_population_ids = {
                    candidate.candidate_id for candidate in population
                }

                if challenger.candidate_id not in current_population_ids:
                    population.append(challenger)

            incumbents = get_incumbents_from_portfolio(
                population=population,
                portfolio=all_portfolio,
            )
            incumbent_ids = {
                incumbent.candidate_id
                for incumbent in incumbents
                if incumbent.candidate_id in all_portfolio.evaluations
            }

            if len(population) > mu:
                population, environmental_decision = environmental_selector.select(
                    population=population,
                    incumbent_ids=incumbent_ids,
                    evaluations=all_portfolio.evaluations,
                )
                event_rows.append(
                    make_environmental_event(
                        iteration=iteration,
                        decision=environmental_decision,
                    )
                )

                incumbents = get_incumbents_from_portfolio(
                    population=population,
                    portfolio=all_portfolio,
                )

        if (
            advance_after_iteration
            and incumbents
            and not budget_allocator.exhausted
            and not repair_budget_reached()
        ):
            # Intensify the WHOLE incumbent front by (up to) one aligned block
            # this iteration. A single advance/iteration cannot keep the front
            # aligned, so the intensifier ends up racing every challenger on the
            # intersection of incumbent blocks -> a single block -> winners
            # decided on ~10 examples. Advancing the front one level per
            # iteration grows that common block set (capped by
            # max_blocks_per_challenger) so later generations are judged on more
            # data.
            try:
                advance_decisions = advance_front_one_level(
                    incumbents=incumbents,
                    portfolio=all_portfolio,
                    block_evaluator=block_evaluator,
                    budget_allocator=budget_allocator,
                    config=AdvanceIncumbentsConfig(
                        random_seed=random_seed,
                        update_pareto_archive=False,
                    ),
                    rng=rng,
                    max_blocks=max_blocks_per_challenger,
                    refresh_incumbents=lambda: get_incumbents_from_portfolio(
                        population=population,
                        portfolio=all_portfolio,
                    ),
                )
            except RuntimeError as exc:
                event_rows.append(
                    {
                        "candidate_id": "",
                        "method": "advance_incumbents",
                        "event_type": "advance_budget_error",
                        "iteration": iteration,
                        "accepted": False,
                        "rejected": True,
                        "reason": str(exc),
                        "evaluated_blocks": "[]",
                        "budget_used": budget_allocator.used_budget,
                        "remaining_budget": budget_allocator.remaining_budget,
                    }
                )
                break

            for advance_decision in advance_decisions:
                event_rows.append(
                    make_advance_event(
                        iteration=iteration,
                        decision=advance_decision,
                    )
                )

            incumbents = get_incumbents_from_portfolio(
                population=population,
                portfolio=all_portfolio,
            )

        trajectory.append(
            _front_snapshot(
                label=f"iteration_{iteration}",
                iteration=iteration,
                front=incumbents,
                portfolio=all_portfolio,
                budget_allocator=budget_allocator,
            )
        )

    if cost_repair_enabled and not budget_allocator.exhausted:
        repair_sources = front_candidates_from_portfolio(all_portfolio) or incumbents
        repair_candidates = make_cost_repair_candidates(
            candidates=repair_sources,
            portfolio=all_portfolio,
            max_source_candidates=int(
                cost_repair_cfg.get("max_source_candidates", 4)
            ),
            variants_per_candidate=int(
                cost_repair_cfg.get("variants_per_candidate", 3)
            ),
        )
        max_repair_candidates = int(
            cost_repair_cfg.get("max_repair_candidates", len(repair_candidates))
        )

        for repair_index, repair_candidate in enumerate(
            repair_candidates[: max(0, max_repair_candidates)]
        ):
            if budget_allocator.exhausted:
                break

            repair_incumbents = front_candidates_from_portfolio(all_portfolio)
            if not repair_incumbents:
                repair_incumbents = incumbents

            try:
                decision = intensifier.intensify(
                    challenger=repair_candidate,
                    incumbents=repair_incumbents,
                    portfolio=all_portfolio,
                )
            except RuntimeError as exc:
                event_rows.append(
                    {
                        "candidate_id": repair_candidate.candidate_id,
                        "method": repair_candidate.metadata.get("method"),
                        "event_type": "cost_repair_budget_error",
                        "iteration": max_iterations + 1,
                        "offspring_index": repair_index,
                        "operator": repair_candidate.metadata.get("operator"),
                        "accepted": False,
                        "rejected": True,
                        "reason": str(exc),
                        "evaluated_blocks": "[]",
                        "budget_used": budget_allocator.used_budget,
                        "remaining_budget": budget_allocator.remaining_budget,
                    }
                )
                break

            event_rows.append(
                {
                    "candidate_id": repair_candidate.candidate_id,
                    "method": repair_candidate.metadata.get("method"),
                    "event_type": "cost_repair",
                    "iteration": max_iterations + 1,
                    "offspring_index": repair_index,
                    "operator": repair_candidate.metadata.get("operator"),
                    "parent_ids": repair_candidate.metadata.get("parent_ids"),
                    "accepted": decision.accepted,
                    "rejected": decision.rejected,
                    "reason": decision.reason,
                    "evaluated_blocks": str(decision.evaluated_blocks),
                    "compared_against": decision.compared_against,
                    "budget_used": decision.budget_used,
                    "remaining_budget": decision.metadata.get("remaining_budget"),
                    "metadata": json.dumps(
                        {
                            **repair_candidate.metadata,
                            "repair_reserved_budget": cost_repair_reserved_budget,
                        },
                        default=_json_default,
                    ),
                }
            )

            if decision.accepted or portfolio_contains_candidate(
                all_portfolio,
                repair_candidate,
            ):
                if repair_candidate.candidate_id not in {
                    candidate.candidate_id for candidate in population
                }:
                    population.append(repair_candidate)

        incumbents = get_incumbents_from_portfolio(
            population=population,
            portfolio=all_portfolio,
        )
        trajectory.append(
            _front_snapshot(
                label="cost_repair",
                iteration=max_iterations + 1,
                front=front_candidates_from_portfolio(all_portfolio),
                portfolio=all_portfolio,
                budget_allocator=budget_allocator,
            )
        )

    # Final deepening pass. The evolutionary loop can accept a challenger on a
    # single block in the LAST iteration (no later advance pass deepens it), and
    # the final front is plain Pareto dominance over all candidates regardless of
    # block depth -- so a lucky 1-block estimate (e.g. 10/10 -> perf 1.0) can
    # dominate honestly-deepened candidates and land on the reported front. Race
    # the final non-dominated front up to the full block depth here, then let the
    # recompute below select over honest scores.
    if (
        final_intensification
        and max_blocks_per_challenger
        and not budget_allocator.exhausted
    ):
        previous_depth = -1

        for _ in range(int(max_blocks_per_challenger) + 2):
            if budget_allocator.exhausted:
                break

            front = front_candidates_from_portfolio(all_portfolio)

            if not front:
                break

            depth = len(common_incumbent_blocks(front, all_portfolio))

            if depth >= max_blocks_per_challenger or depth == previous_depth:
                break

            previous_depth = depth

            try:
                final_advance_decisions = advance_front_one_level(
                    incumbents=front,
                    portfolio=all_portfolio,
                    block_evaluator=block_evaluator,
                    budget_allocator=budget_allocator,
                    config=AdvanceIncumbentsConfig(
                        random_seed=random_seed,
                        update_pareto_archive=False,
                    ),
                    rng=rng,
                    max_blocks=max_blocks_per_challenger,
                    refresh_incumbents=lambda: front_candidates_from_portfolio(
                        all_portfolio
                    ),
                )
            except RuntimeError as exc:
                if "Budget exhausted" not in str(exc):
                    raise

                event_rows.append(
                    {
                        "candidate_id": "",
                        "method": "final_front_advance",
                        "event_type": "budget_stop",
                        "iteration": max_iterations + 1,
                        "accepted": False,
                        "rejected": True,
                        "reason": (
                            "Budget exhausted during final front advancement."
                        ),
                        "error": str(exc),
                        "evaluated_blocks": "[]",
                        "budget_used": budget_allocator.used_budget,
                        "remaining_budget": budget_allocator.remaining_budget,
                    }
                )
                break

            for advance_decision in final_advance_decisions:
                event_rows.append(
                    make_advance_event(
                        iteration=max_iterations + 1,
                        decision=advance_decision,
                    )
                )

            if not any(
                getattr(decision, "advanced", False)
                for decision in final_advance_decisions
            ):
                break

        incumbents = get_incumbents_from_portfolio(
            population=population,
            portfolio=all_portfolio,
        )

    pareto_ids = set(non_dominated_ids(all_portfolio.evaluations.values()))

    pareto_portfolio = PromptPortfolio()

    sorted_results = sort_pareto_results(
        [
            result
            for result in all_portfolio.evaluations.values()
            if result.candidate_id in pareto_ids
        ]
    )

    for result in sorted_results:
        candidate = all_portfolio.get(result.candidate_id)
        pareto_portfolio.add(candidate, result)

    # §F final trajectory point: the reported Pareto front at end-of-budget,
    # drawn from the SAME basis as the saved front (non-dominated over the whole
    # portfolio) so the trajectory's last HV matches the table's HV.
    trajectory.append(
        _front_snapshot(
            label="final",
            iteration=max_iterations + 1,
            front=front_candidates_from_portfolio(all_portfolio),
            portfolio=all_portfolio,
            budget_allocator=budget_allocator,
        )
    )

    budget_summary = budget_allocator.summary()
    budget_summary["evaluator"] = (
        "lmstudio"
        if config.get("evaluation", {}).get("use_llm", False) and not force_no_llm
        else "toy"
    )
    budget_summary["model_id"] = config.get("llm", {}).get("model_id")
    budget_summary["algorithm"] = "evolutionary_budgeted_mocapo"
    budget_summary["population_size"] = mu
    budget_summary["max_iterations"] = max_iterations
    budget_summary["offspring_per_iteration"] = offspring_per_iteration
    budget_summary["num_population_candidates"] = len(population)
    budget_summary["num_incumbents"] = len(incumbents)
    budget_summary["num_evaluated_candidates"] = len(all_portfolio.evaluations)

    trajectory = compute_trajectory_metrics(
        snapshots=trajectory,
        bounds_config=config.get("bounds"),
        num_preference_vectors=int(
            config.get("metrics", {}).get("num_preference_vectors", 500)
        ),
        seed=random_seed,
    )

    return all_portfolio, pareto_portfolio, event_rows, budget_summary, trajectory


def portfolio_to_rows(
    portfolio: PromptPortfolio,
    pareto_ids: set[str] | None = None,
) -> list[dict]:
    pareto_ids = pareto_ids or set()
    rows = []

    for candidate in portfolio.evaluated_candidates():
        result = portfolio.get_result(candidate.candidate_id)

        row = {
            "candidate_id": candidate.candidate_id,
            "is_pareto": candidate.candidate_id in pareto_ids,
            "method": candidate.metadata.get("method")
            or candidate.metadata.get("prompt_pool_id"),
            "category": candidate.metadata.get("category"),
            "dataset": candidate.metadata.get("dataset"),
            "task_type": candidate.metadata.get("task_type"),
            "performance": result.performance,
            "cost": result.cost,
            "risk": result.risk,
            "fairness_risk": result.fairness_risk,
            "drift": result.drift,
            "n_examples": result.n_examples,
            "objective_vector": str(result.objective_vector),
            "prompt": candidate.instruction,
            # Persist the few-shot demonstrations so the held-out test evaluator
            # can reconstruct the EXACT prompt that won (instruction + shots).
            # Without this, evaluate_pareto_on_test rebuilds a zero-shot prompt
            # and a few-shot winner is silently tested without its demos -- which
            # collapses the accuracy/cost staircase on held-out data.
            "num_few_shot": len(candidate.examples),
            "few_shot_examples": json.dumps(
                candidate.examples, default=_json_default
            ),
            "source": candidate.metadata.get("source"),
            "operator": candidate.metadata.get("operator"),
            "parent_ids": candidate.metadata.get("parent_ids"),
            "used_meta_llm": candidate.metadata.get("used_meta_llm"),
            "evaluated_blocks": candidate.metadata.get("evaluated_blocks"),
        }

        for key, value in result.details.items():
            if key == "predictions":
                row[f"detail_{key}"] = json.dumps(value, default=_json_default)
            elif isinstance(value, (dict, list, tuple)):
                row[f"detail_{key}"] = json.dumps(value, default=_json_default)
            else:
                row[f"detail_{key}"] = value

        rows.append(row)

    return rows


def routing_preferences_from_config(config: dict) -> list[dict]:
    return config.get(
        "routing_preferences",
        [
            {"name": "accuracy_first", "mode": "accuracy_first"},
            {"name": "cost_first", "mode": "cost_first"},
            {"name": "risk_first", "mode": "risk_first"},
            {"name": "fairness_first", "mode": "fairness_first"},
            {
                "name": "balanced",
                "mode": "balanced",
                "performance": 1.0,
                "cost": 0.3,
                "risk": 1.0,
                "fairness": 1.0,
            },
        ],
    )


def routing_to_rows(
    portfolio: PromptPortfolio,
    preferences: list[dict],
) -> list[dict]:
    rows = []

    default_router = RiskAwareRouter()
    fairness_router = FairnessAwareRouter()

    for preference in preferences:
        name = preference.get("name", preference.get("mode", "balanced"))

        if preference.get("mode") == "fairness_first":
            router = fairness_router
        else:
            router = default_router

        decision = router.select(
            x="phase2_budgeted_mocapo",
            portfolio=portfolio,
            preference=preference,
        )

        row = decision.to_row()
        row["preference_name"] = name
        rows.append(row)

    return rows


def print_portfolio(title: str, rows: list[dict]):
    print(title)
    print("-" * 80)

    for row in rows:
        pareto_mark = "Pareto" if row.get("is_pareto") else "Non-Pareto"

        print(
            f"{pareto_mark} | "
            f"{row.get('method')} | "
            f"category={row.get('category')} | "
            f"performance={row.get('performance')} | "
            f"cost={row.get('cost')} | "
            f"risk={row.get('risk')} | "
            f"fairness_risk={row.get('fairness_risk')} | "
            f"prompt={row.get('prompt')}"
        )

    print("-" * 80)
    print(f"Total rows: {len(rows)}")


def print_events(events: list[dict]):
    print("Budgeted evolutionary MO-CAPO events")
    print("-" * 80)

    for row in events:
        print(
            f"{row.get('event_type')} | "
            f"iter={row.get('iteration', '')} | "
            f"{row.get('method')} | "
            f"accepted={row.get('accepted')} | "
            f"rejected={row.get('rejected')} | "
            f"blocks={row.get('evaluated_blocks')} | "
            f"reason={row.get('reason')}"
        )

    print("-" * 80)


def print_recommendations(rows: list[dict]):
    print("Budgeted evolutionary MO-CAPO recommendations")
    print("-" * 80)

    for row in rows:
        print(
            f"{row['preference_name']} -> "
            f"{row['instruction']} | "
            f"utility={row['utility']} | "
            f"reason={row['reason']}"
        )

    print("-" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/phase2_budgeted_mocapo_subj.yaml",
        help="Budgeted MO-CAPO YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Force ToyObjectiveEvaluator even if evaluation.use_llm is true.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config['seed']. Seeds the dev/shot split sampling and the "
        "evolutionary RNG (per MO-CAPO, the run's seed seeds the data sampling). "
        "Use for multi-seed sweeps from a single config.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)

    if args.seed is not None:
        config["seed"] = args.seed

    output_dir = args.output_dir or config.get(
        "output_dir",
        "outputs/phase2_budgeted_mocapo_subj",
    )

    all_portfolio, pareto_portfolio, event_rows, budget_summary, trajectory = run_budgeted_mocapo(
        config=config,
        force_no_llm=args.no_llm,
    )

    pareto_ids = set(non_dominated_ids(all_portfolio.evaluations.values()))

    all_rows = portfolio_to_rows(
        portfolio=all_portfolio,
        pareto_ids=pareto_ids,
    )
    pareto_rows = portfolio_to_rows(
        portfolio=pareto_portfolio,
        pareto_ids=pareto_ids,
    )

    recommendations = routing_to_rows(
        portfolio=pareto_portfolio,
        preferences=routing_preferences_from_config(config),
    )

    save_csv(all_rows, f"{output_dir}/phase2_all_candidates.csv")
    save_json(all_rows, f"{output_dir}/phase2_all_candidates.json")

    save_csv(pareto_rows, f"{output_dir}/phase2_prompt_portfolio.csv")
    save_json(pareto_rows, f"{output_dir}/phase2_prompt_portfolio.json")

    save_csv(recommendations, f"{output_dir}/phase2_prompt_recommendations.csv")
    save_json(recommendations, f"{output_dir}/phase2_prompt_recommendations.json")

    save_csv(event_rows, f"{output_dir}/budgeted_mocapo_events.csv")
    save_json(event_rows, f"{output_dir}/budgeted_mocapo_events.json")

    save_json(budget_summary, f"{output_dir}/budget_summary.json")
    save_json(trajectory, f"{output_dir}/budgeted_mocapo_trajectory.json")

    print_events(event_rows)
    print_portfolio("Budgeted evolutionary MO-CAPO Pareto portfolio", pareto_rows)
    print_recommendations(recommendations)

    print("Budget summary")
    print("-" * 80)
    print(json.dumps(budget_summary, indent=2, default=_json_default))
    print("-" * 80)

    print(f"Saved all candidates to: {output_dir}/phase2_all_candidates.csv")
    print(f"Saved Pareto portfolio to: {output_dir}/phase2_prompt_portfolio.csv")
    print(f"Saved recommendations to: {output_dir}/phase2_prompt_recommendations.csv")
    print(f"Saved evolutionary events to: {output_dir}/budgeted_mocapo_events.csv")
    print(f"Saved budget summary to: {output_dir}/budget_summary.json")
    print(
        f"Saved trajectory ({len(trajectory)} snapshots) to:"
        f" {output_dir}/budgeted_mocapo_trajectory.json"
    )


if __name__ == "__main__":
    main()
