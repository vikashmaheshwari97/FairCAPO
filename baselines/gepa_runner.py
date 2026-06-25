from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from heal_capo.core import EvaluationResult, PromptCandidate


@dataclass
class GEPARunnerConfig:
    task_description: str
    labels: list[str]
    initial_instruction: str
    max_iterations: int = 10
    budget_tokens: int = 50_000
    require_final_answer_tags: bool = True
    input_weight: float = 0.08
    output_weight: float = 0.32
    use_native_gepa: bool = True
    allow_fallback: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GEPARunResult:
    candidate: PromptCandidate
    optimized_instruction: str
    used_native_gepa: bool
    num_iterations: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NativeGEPAScore:
    """
    Score object expected by GEPA's default adapter.

    GEPA expects evaluator outputs to expose:
      - score
      - objective_scores

    We keep one objective called "accuracy" for the GEPA single-objective
    baseline, while our HEAL-CAPO evaluation later computes cost/risk/fairness.
    """

    score: float
    feedback: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    objective_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.objective_scores:
            self.objective_scores = {"accuracy": float(self.score)}


def simple_token_count(text: str) -> int:
    return len(str(text or "").split())


def extract_between_tags(
    text: str,
    start_tag: str,
    end_tag: str,
) -> Optional[str]:
    raw = str(text or "")
    lowered = raw.lower()

    start_lower = start_tag.lower()
    end_lower = end_tag.lower()

    if start_lower not in lowered or end_lower not in lowered:
        return None

    start = lowered.find(start_lower) + len(start_lower)
    end = lowered.find(end_lower, start)

    if end < start:
        return None

    return raw[start:end].strip()


def extract_prompt(text: str) -> str:
    extracted = extract_between_tags(text, "<prompt>", "</prompt>")
    if extracted:
        return extracted

    return str(text or "").strip()


def extract_label(response: str, labels: list[str]) -> str:
    extracted = extract_between_tags(response, "<final_answer>", "</final_answer>")

    if extracted:
        text = extracted
    else:
        text = str(response or "")

    lowered = text.strip().lower()
    cleaned = lowered.strip(" .,:;!?\"'`")

    for label in labels:
        if cleaned == str(label).lower():
            return str(label)

    for label in labels:
        if str(label).lower() in lowered:
            return str(label)

    return cleaned


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


def call_llm(llm: Any, prompt: str) -> str:
    if hasattr(llm, "get_response"):
        response = llm.get_response(prompt)
    elif hasattr(llm, "generate"):
        response = llm.generate(prompt)
    elif callable(llm):
        response = llm(prompt)
    else:
        raise AttributeError(
            "LLM must expose get_response(), generate(), or be callable."
        )

    if isinstance(response, list):
        return str(response[0])

    return str(response)


def make_prediction_prompt(
    instruction: str,
    text: str,
    labels: list[str],
    require_final_answer_tags: bool = True,
) -> str:
    labels_text = ", ".join(labels)

    if require_final_answer_tags:
        return (
            f"{instruction}\n\n"
            f"Input:\n{text}\n\n"
            f"Allowed labels: {labels_text}\n"
            f"Return exactly one label between <final_answer> and </final_answer> tags."
        )

    return (
        f"{instruction}\n\n"
        f"Input:\n{text}\n\n"
        f"Allowed labels: {labels_text}\n"
        f"Return exactly one label: {labels_text}."
    )


def make_reflection_prompt(
    task_description: str,
    instruction: str,
    failures: list[dict],
) -> str:
    failure_text = "\n".join(
        [
            (
                f"- Input: {failure['text']}\n"
                f"  Gold: {failure['gold']}\n"
                f"  Prediction: {failure['prediction']}\n"
                f"  Response: {failure['raw_response']}"
            )
            for failure in failures[:8]
        ]
    )

    return (
        "You are improving a prompt for the following task.\n\n"
        f"Task description:\n{task_description}\n\n"
        f"Current prompt:\n{instruction}\n\n"
        "The current prompt made these mistakes:\n"
        f"{failure_text}\n\n"
        "Revise the prompt to reduce these mistakes while preserving the task and output format. "
        "Return only the revised prompt inside <prompt> and </prompt> tags."
    )


def evaluate_instruction(
    instruction: str,
    data: list[dict],
    llm: Any,
    labels: list[str],
    input_weight: float = 0.08,
    output_weight: float = 0.32,
    require_final_answer_tags: bool = True,
) -> tuple[EvaluationResult, list[dict]]:
    correct = 0
    total = 0
    input_tokens = 0
    output_tokens = 0
    rows = []

    candidate_id = str(uuid.uuid4())

    for example in data:
        text = get_example_text(example)
        gold = get_example_label(example).strip().lower()

        prompt = make_prediction_prompt(
            instruction=instruction,
            text=text,
            labels=labels,
            require_final_answer_tags=require_final_answer_tags,
        )

        raw_response = call_llm(llm, prompt)
        prediction = extract_label(raw_response, labels).strip().lower()
        is_correct = prediction == gold

        prompt_tokens = simple_token_count(prompt)
        response_tokens = simple_token_count(raw_response)

        input_tokens += prompt_tokens
        output_tokens += response_tokens

        correct += int(is_correct)
        total += 1

        rows.append(
            {
                "text": text,
                "gold": gold,
                "prediction": prediction,
                "raw_response": raw_response,
                "correct": is_correct,
                "input_tokens": prompt_tokens,
                "output_tokens": response_tokens,
            }
        )

    performance = correct / total if total else 0.0
    cost = input_weight * input_tokens + output_weight * output_tokens
    risk = 1.0 - performance

    result = EvaluationResult(
        candidate_id=candidate_id,
        performance=performance,
        cost=cost,
        risk=risk,
        fairness_risk=0.30,
        drift=0.0,
        n_examples=total,
        details={
            "evaluator": "gepa_runner",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "correct": correct,
            "total": total,
            "predictions": rows,
        },
    )

    return result, rows


def _extract_instruction_from_gepa_result(result: Any) -> str:
    if result is None:
        return ""

    if isinstance(result, str):
        return extract_prompt(result)

    if isinstance(result, dict):
        for key in [
            "instruction",
            "prompt",
            "system_prompt",
            "optimized_prompt",
            "best_prompt",
            "best_instruction",
        ]:
            if key in result:
                return extract_prompt(str(result[key]))

    if hasattr(result, "best_candidate"):
        best_candidate = getattr(result, "best_candidate")

        if isinstance(best_candidate, dict):
            for key in ["instruction", "prompt", "system_prompt"]:
                if key in best_candidate:
                    return extract_prompt(str(best_candidate[key]))

        if hasattr(best_candidate, "__dict__"):
            for key in ["instruction", "prompt", "system_prompt"]:
                if key in best_candidate.__dict__:
                    return extract_prompt(str(best_candidate.__dict__[key]))

    for attr in [
        "optimized_prompt",
        "prompt",
        "instruction",
        "best_prompt",
        "best_instruction",
    ]:
        if hasattr(result, attr):
            return extract_prompt(str(getattr(result, attr)))

    return ""


def normalize_candidate_instruction(candidate: Any, default_instruction: str) -> str:
    if isinstance(candidate, dict):
        return str(
            candidate.get("instruction")
            or candidate.get("prompt")
            or candidate.get("system_prompt")
            or default_instruction
        )

    if isinstance(candidate, str):
        return candidate

    if hasattr(candidate, "__dict__"):
        candidate_dict = candidate.__dict__
        return str(
            candidate_dict.get("instruction")
            or candidate_dict.get("prompt")
            or candidate_dict.get("system_prompt")
            or default_instruction
        )

    return default_instruction


def normalize_example(example: Any) -> dict:
    if isinstance(example, dict):
        item = dict(example)
    else:
        item = {"input": str(example), "text": str(example)}

    if "input" not in item:
        item["input"] = item.get("text") or item.get("sentence") or item.get("x") or ""

    if "text" not in item:
        item["text"] = item["input"]

    if "label" not in item and "label_text" in item:
        item["label"] = item["label_text"]

    if "label_text" not in item and "label" in item:
        item["label_text"] = item["label"]

    return item


def try_native_gepa(
    config: GEPARunnerConfig,
    train_data: list[dict],
    dev_data: list[dict],
    llm: Any,
) -> tuple[Optional[str], dict[str, Any]]:
    """
    Call official GEPA through a custom GEPAAdapter.

    This avoids GEPA's DefaultAdapter scoring mismatch and ensures GEPA's
    internal validation score uses the same label extraction logic as our
    final HEAL-CAPO evaluator.
    """
    metadata: dict[str, Any] = {
        "native_gepa_attempted": True,
        "native_gepa_adapter": "SubjGEPAAdapter",
    }

    try:
        import gepa  # type: ignore
    except Exception as exc:
        metadata["native_gepa_error"] = f"import_error: {exc}"
        return None, metadata

    train_data = [normalize_example(example) for example in train_data]
    dev_data = [normalize_example(example) for example in dev_data]

    def task_lm(prompt: Any = None, **kwargs) -> str:
        prompt_value = (
            prompt
            or kwargs.get("prompt")
            or kwargs.get("input")
            or kwargs.get("messages")
            or kwargs.get("text")
            or ""
        )

        if isinstance(prompt_value, list):
            parts = []
            for item in prompt_value:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    parts.append(f"{role}: {content}")
                else:
                    parts.append(str(item))
            prompt_text = "\n".join(parts)
        elif isinstance(prompt_value, dict):
            prompt_text = json.dumps(prompt_value, ensure_ascii=False)
        else:
            prompt_text = str(prompt_value)

        return call_llm(llm, prompt_text)

    class SubjGEPAAdapter:
        """
        Custom GEPA adapter for SUBJ-style classification.

        GEPA internal scoring and our final evaluation now share:
          - same prompt construction
          - same <final_answer> extraction
          - same label comparison
        """

        def __init__(
            self,
            labels: list[str],
            require_final_answer_tags: bool = True,
        ):
            self.labels = labels
            self.require_final_answer_tags = require_final_answer_tags

        def evaluate(
            self,
            batch: list[dict],
            candidate: dict[str, str],
            capture_traces: bool = False,
        ):
            instruction = normalize_candidate_instruction(
                candidate=candidate,
                default_instruction=config.initial_instruction,
            )

            outputs = []
            scores = []
            objective_scores = []
            trajectories = [] if capture_traces else None

            for raw_example in batch:
                example = normalize_example(raw_example)
                text = get_example_text(example)
                gold = get_example_label(example).strip().lower()

                prompt = make_prediction_prompt(
                    instruction=instruction,
                    text=text,
                    labels=self.labels,
                    require_final_answer_tags=self.require_final_answer_tags,
                )

                raw_response = call_llm(llm, prompt)
                prediction = extract_label(raw_response, self.labels).strip().lower()

                score = 1.0 if gold and prediction == gold else 0.0

                feedback = (
                    f"Input: {text}\n"
                    f"Gold label: {gold}\n"
                    f"Predicted label: {prediction}\n"
                    f"Raw response: {raw_response}\n"
                    f"Correct: {bool(score)}"
                )

                output = {
                    "full_assistant_response": raw_response,
                    "prediction": prediction,
                    "gold": gold,
                }

                outputs.append(output)
                scores.append(score)
                objective_scores.append({"accuracy": score})

                if trajectories is not None:
                    trajectories.append(
                        {
                            "data": example,
                            "instruction": instruction,
                            "prompt": prompt,
                            "full_assistant_response": raw_response,
                            "prediction": prediction,
                            "gold": gold,
                            "score": score,
                            "feedback": feedback,
                        }
                    )

            return gepa.EvaluationBatch(
                outputs=outputs,
                scores=scores,
                trajectories=trajectories,
                objective_scores=objective_scores,
            )

        def make_reflective_dataset(
            self,
            candidate: dict[str, str],
            eval_batch,
            components_to_update: list[str],
        ):
            trajectories = eval_batch.trajectories

            if trajectories is None:
                raise ValueError(
                    "Trajectories are required to build GEPA reflective dataset."
                )

            reflective = {}

            for component in components_to_update:
                records = []

                for traj in trajectories:
                    records.append(
                        {
                            "Inputs": {
                                "sentence": traj.get("data", {}).get("input", ""),
                                "task": config.task_description,
                            },
                            "Generated Outputs": traj.get(
                                "full_assistant_response",
                                "",
                            ),
                            "Feedback": traj.get("feedback", ""),
                            "score": traj.get("score", 0.0),
                            "gold": traj.get("gold", ""),
                            "prediction": traj.get("prediction", ""),
                        }
                    )

                if not records:
                    raise ValueError(
                        f"No reflective records created for component: {component}"
                    )

                reflective[component] = records

            return reflective

    seed_candidate = {
        "instruction": config.initial_instruction,
    }

    adapter = SubjGEPAAdapter(
        labels=config.labels,
        require_final_answer_tags=config.require_final_answer_tags,
    )

    try:
        result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=train_data,
            valset=dev_data,
            adapter=adapter,
            reflection_lm=task_lm,
            reflection_lm_kwargs={},
            max_metric_calls=config.max_iterations,
            reflection_minibatch_size=min(3, max(1, len(train_data))),
            display_progress_bar=False,
            raise_on_exception=False,
            cache_evaluation=False,
            seed=int(config.metadata.get("seed", 0)),
        )
    except Exception as exc:
        metadata["native_gepa_error"] = f"optimize_error: {type(exc).__name__}: {exc}"
        return None, metadata

    metadata["native_gepa_result_type"] = type(result).__name__

    for attr in [
        "total_metric_calls",
        "num_candidates",
        "num_full_val_evals",
        "num_val_instances",
        "best_idx",
        "seed",
        "run_dir",
    ]:
        if hasattr(result, attr):
            try:
                metadata[f"gepa_{attr}"] = getattr(result, attr)
            except Exception:
                pass

    optimized_instruction = _extract_instruction_from_gepa_result(result)

    if not optimized_instruction:
        metadata["native_gepa_error"] = "could_not_extract_best_candidate"
        return None, metadata

    metadata["native_gepa_success"] = True
    return optimized_instruction, metadata


def local_reflective_prompt_optimization(
    config: GEPARunnerConfig,
    train_data: list[dict],
    dev_data: list[dict],
    llm: Any,
) -> tuple[str, int, dict]:
    best_instruction = config.initial_instruction
    best_result, best_rows = evaluate_instruction(
        instruction=best_instruction,
        data=dev_data,
        llm=llm,
        labels=config.labels,
        input_weight=config.input_weight,
        output_weight=config.output_weight,
        require_final_answer_tags=config.require_final_answer_tags,
    )

    history = [
        {
            "iteration": 0,
            "instruction": best_instruction,
            "performance": best_result.performance,
            "cost": best_result.cost,
            "risk": best_result.risk,
        }
    ]

    used_tokens = (
        best_result.details.get("input_tokens", 0)
        + best_result.details.get("output_tokens", 0)
    )

    for iteration in range(1, config.max_iterations + 1):
        if used_tokens >= config.budget_tokens:
            break

        failures = [row for row in best_rows if not row["correct"]]

        if not failures:
            break

        reflection_prompt = make_reflection_prompt(
            task_description=config.task_description,
            instruction=best_instruction,
            failures=failures,
        )

        raw_revised = call_llm(llm, reflection_prompt)
        revised_instruction = extract_prompt(raw_revised)

        if not revised_instruction:
            continue

        result, rows = evaluate_instruction(
            instruction=revised_instruction,
            data=dev_data,
            llm=llm,
            labels=config.labels,
            input_weight=config.input_weight,
            output_weight=config.output_weight,
            require_final_answer_tags=config.require_final_answer_tags,
        )

        used_tokens += (
            simple_token_count(reflection_prompt)
            + simple_token_count(raw_revised)
            + result.details.get("input_tokens", 0)
            + result.details.get("output_tokens", 0)
        )

        history.append(
            {
                "iteration": iteration,
                "instruction": revised_instruction,
                "performance": result.performance,
                "cost": result.cost,
                "risk": result.risk,
            }
        )

        if result.performance > best_result.performance:
            best_instruction = revised_instruction
            best_result = result
            best_rows = rows

    return (
        best_instruction,
        len(history) - 1,
        {
            "history": history,
            "used_tokens_estimate": used_tokens,
            "fallback": "local_reflective_prompt_optimization",
        },
    )


class GEPARunner:
    def __init__(
        self,
        config: GEPARunnerConfig,
        llm: Any,
    ):
        self.config = config
        self.llm = llm

    def run(
        self,
        train_data: list[dict],
        dev_data: list[dict],
    ) -> GEPARunResult:
        used_native_gepa = False
        metadata: dict[str, Any] = {}

        optimized_instruction: Optional[str] = None
        num_iterations = 0

        if self.config.use_native_gepa:
            optimized_instruction, native_metadata = try_native_gepa(
                config=self.config,
                train_data=train_data,
                dev_data=dev_data,
                llm=self.llm,
            )
            metadata.update(native_metadata)

            if optimized_instruction:
                used_native_gepa = True
                metadata["native_gepa"] = True
                num_iterations = int(
                    metadata.get("gepa_total_metric_calls", self.config.max_iterations)
                    or self.config.max_iterations
                )

        if not optimized_instruction:
            if not self.config.allow_fallback:
                raise RuntimeError(
                    "Could not call native GEPA API and fallback is disabled. "
                    f"Metadata: {metadata}"
                )

            optimized_instruction, num_iterations, fallback_metadata = (
                local_reflective_prompt_optimization(
                    config=self.config,
                    train_data=train_data,
                    dev_data=dev_data,
                    llm=self.llm,
                )
            )
            metadata.update(fallback_metadata)

        candidate = PromptCandidate(
            instruction=optimized_instruction,
            metadata={
                "method": "gepa",
                "category": "single_objective_baseline",
                "source": "third_party/gepa",
                "used_native_gepa": used_native_gepa,
                **self.config.metadata,
            },
        )

        return GEPARunResult(
            candidate=candidate,
            optimized_instruction=optimized_instruction,
            used_native_gepa=used_native_gepa,
            num_iterations=num_iterations,
            metadata=metadata,
        )


def evaluate_gepa_candidate(
    candidate: PromptCandidate,
    test_data: list[dict],
    llm: Any,
    labels: list[str],
    input_weight: float = 0.08,
    output_weight: float = 0.32,
    require_final_answer_tags: bool = True,
) -> EvaluationResult:
    result, rows = evaluate_instruction(
        instruction=candidate.instruction,
        data=test_data,
        llm=llm,
        labels=labels,
        input_weight=input_weight,
        output_weight=output_weight,
        require_final_answer_tags=require_final_answer_tags,
    )

    result.candidate_id = candidate.candidate_id
    result.details["method"] = "gepa"
    result.details["predictions"] = rows

    return result


def gepa_result_to_row(
    run_result: GEPARunResult,
    evaluation: EvaluationResult,
) -> dict:
    return {
        "candidate_id": run_result.candidate.candidate_id,
        "method": "gepa",
        "category": "single_objective_baseline",
        "is_pareto": False,
        "performance": evaluation.performance,
        "cost": evaluation.cost,
        "risk": evaluation.risk,
        "fairness_risk": evaluation.fairness_risk,
        "drift": evaluation.drift,
        "n_examples": evaluation.n_examples,
        "prompt": run_result.optimized_instruction,
        "used_native_gepa": run_result.used_native_gepa,
        "num_iterations": run_result.num_iterations,
        "detail_input_tokens": evaluation.details.get("input_tokens", 0),
        "detail_output_tokens": evaluation.details.get("output_tokens", 0),
        "detail_correct": evaluation.details.get("correct", 0),
        "detail_total": evaluation.details.get("total", 0),
        "detail_predictions": json.dumps(
            evaluation.details.get("predictions", []),
            ensure_ascii=False,
        ),
        "detail_gepa_metadata": json.dumps(
            run_result.metadata,
            ensure_ascii=False,
            default=str,
        ),
    }