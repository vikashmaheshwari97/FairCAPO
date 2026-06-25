from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import requests

from promptolution.llms.base_llm import BaseLLM
from promptolution.llms import APILLM
from promptolution.optimizers import CAPO, EvoPromptDE, EvoPromptGA, OPRO
from promptolution.utils.templates import (
    EVOPROMPT_DE_TEMPLATE_TD,
    EVOPROMPT_GA_TEMPLATE,
)
from promptolution.predictors import MarkerBasedPredictor
from promptolution.tasks import ClassificationTask

from experiments.metrics import accuracy
from experiments.token_cost import estimate_dataset_cost


class ToyPromptolutionLLM(BaseLLM):
    """
    Tiny fake LLM for Phase 1 integration testing.

    It does two jobs:
    1. For meta-optimization prompts, it returns a new prompt inside <prompt> tags.
    2. For downstream task prompts, it returns a label inside <final_answer> tags.

    This lets us test Promptolution optimizers without API/local LLM cost.
    """

    SUBJECTIVE_WORDS = {
        "boring",
        "wonderful",
        "beautiful",
        "badly",
        "painfully",
        "best",
        "worst",
        "amazing",
        "amusing",
        "important",
        "dull",
        "funny",
        "touching",
        "poor",
        "great",
        "terrible",
        "excellent",
        "awful",
        "faintly",
        "takes hold",
    }

    def _get_response(self, prompts: list[str], system_prompts: list[str]) -> list[str]:
        responses = []

        for prompt in prompts:
            lower = prompt.lower()

            is_meta_prompt = (
                "return the new prompt" in lower
                or "generate a better prompt" in lower
                or "merge the two prompts" in lower
                or "rephrase the prompt" in lower
                or ("<prompt>" in lower and "new prompt" in lower)
                or "crossover" in lower
                or "mutate" in lower
                or "differential evolution" in lower
                or "previous prompts" in lower
                or "score:" in lower
            )

            if is_meta_prompt:
                responses.append(
                    "<prompt>"
                    "Classify the given input into the correct label. "
                    "Return only the label inside <final_answer> and </final_answer> tags."
                    "</prompt>"
                )
                continue

            label = "objective"
            if any(word in lower for word in self.SUBJECTIVE_WORDS):
                label = "subjective"

            responses.append(f"<final_answer>{label}</final_answer>")

        return responses


class OllamaPromptolutionLLM(BaseLLM):
    """
    Promptolution-compatible local Ollama backend.

    It calls Ollama's local HTTP API:
        POST http://localhost:11434/api/generate
    """

    def __init__(
        self,
        model_id: str,
        api_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        num_predict: int = 512,
        timeout: int = 180,
    ):
        super().__init__()
        self.model_id = model_id
        self.api_url = api_url.rstrip("/")
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout

    def _get_response(self, prompts: list[str], system_prompts: list[str]) -> list[str]:
        responses = []

        for i, prompt in enumerate(prompts):
            system_prompt = ""
            if system_prompts and i < len(system_prompts):
                system_prompt = system_prompts[i] or ""

            final_prompt = prompt
            if system_prompt:
                final_prompt = f"System: {system_prompt}\n\nUser: {prompt}"

            payload = {
                "model": self.model_id,
                "prompt": final_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.num_predict,
                },
            }

            try:
                response = requests.post(
                    f"{self.api_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as exc:
                raise RuntimeError(
                    f"Ollama request failed. Make sure Ollama is running and model "
                    f"'{self.model_id}' is available. Original error: {exc}"
                ) from exc

            data = response.json()
            responses.append(data.get("response", "") or "")

        return responses


class LMStudioPromptolutionLLM(BaseLLM):
    """
    Promptolution-compatible LM Studio backend.

    It calls LM Studio's OpenAI-compatible local API directly:
        POST http://localhost:1234/v1/chat/completions

    This avoids Promptolution APILLM async timeout issues with local models.
    """

    def __init__(
        self,
        model_id: str,
        api_url: str = "http://localhost:1234/v1",
        temperature: float = 0.0,
        num_predict: int = 128,
        timeout: int = 900,
    ):
        super().__init__()
        self.model_id = model_id
        self.api_url = api_url.rstrip("/")
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout

    def _get_response(self, prompts: list[str], system_prompts: list[str]) -> list[str]:
        responses = []

        for i, prompt in enumerate(prompts):
            messages = []

            if system_prompts and i < len(system_prompts) and system_prompts[i]:
                messages.append(
                    {
                        "role": "system",
                        "content": system_prompts[i],
                    }
                )

            messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            payload = {
                "model": self.model_id,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.num_predict,
                "stream": False,
            }

            try:
                response = requests.post(
                    f"{self.api_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                    headers={"Authorization": "Bearer lm-studio"},
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as exc:
                raise RuntimeError(
                    f"LM Studio request failed. Make sure LM Studio server is running, "
                    f"the model '{self.model_id}' is loaded, and api_url is correct. "
                    f"Original error: {exc}"
                ) from exc

            data = response.json()

            try:
                text = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                text = str(data)

            responses.append(text or "")

        return responses


def build_llm(llm_config: Optional[dict] = None) -> BaseLLM:
    """
    Create the LLM backend.

    Supported:
      - toy: deterministic fake LLM, no API/local model cost
      - ollama: local Ollama model
      - api: Promptolution APILLM / OpenAI-compatible API
      - lmstudio: direct LM Studio local API backend
    """
    llm_config = llm_config or {}
    backend = str(llm_config.get("backend", "toy")).lower().strip()

    if backend == "toy":
        return ToyPromptolutionLLM()

    if backend == "ollama":
        model_id = os.environ.get(
            "FAIRCAPO_MODEL_ID",
            llm_config.get("model_id", "mistral-small:24b"),
        )
        api_url = os.environ.get(
            "FAIRCAPO_LLM_API_URL",
            llm_config.get("api_url", "http://localhost:11434"),
        )
        temperature = float(llm_config.get("temperature", 0.0))
        num_predict = int(llm_config.get("num_predict", 512))
        timeout = int(llm_config.get("timeout", 180))

        return OllamaPromptolutionLLM(
            model_id=model_id,
            api_url=api_url,
            temperature=temperature,
            num_predict=num_predict,
            timeout=timeout,
        )

    if backend == "lmstudio":
        model_id = os.environ.get(
            "FAIRCAPO_MODEL_ID",
            llm_config.get("model_id", "mistralai/mistral-small-3.2"),
        )
        api_url = os.environ.get(
            "FAIRCAPO_LLM_API_URL",
            llm_config.get("api_url", "http://localhost:1234/v1"),
        )
        temperature = float(llm_config.get("temperature", 0.0))
        num_predict = int(llm_config.get("num_predict", 128))
        timeout = int(llm_config.get("timeout", 900))

        return LMStudioPromptolutionLLM(
            model_id=model_id,
            api_url=api_url,
            temperature=temperature,
            num_predict=num_predict,
            timeout=timeout,
        )

    if backend == "api":
        model_id = os.environ.get("FAIRCAPO_MODEL_ID", llm_config.get("model_id"))
        api_url = os.environ.get("FAIRCAPO_LLM_API_URL", llm_config.get("api_url"))
        api_key_env = llm_config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env)

        if not model_id:
            raise ValueError("llm.backend='api' requires llm.model_id in config.")

        if not api_url:
            raise ValueError("llm.backend='api' requires llm.api_url in config.")

        if not api_key:
            raise ValueError(
                f"API key not found. Set environment variable {api_key_env}, "
                f"or change llm.api_key_env in config."
            )

        return APILLM(
            model_id=model_id,
            api_url=api_url,
            api_key=api_key,
        )

    raise ValueError(
        f"Unknown llm backend: {backend}. Supported: toy, ollama, api, lmstudio."
    )


def _dataset_to_dataframe(dataset) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [ex.text for ex in dataset],
            "label": [ex.label for ex in dataset],
        }
    )


def _normalize_prompt(prompt_obj) -> str:
    """
    Promptolution may return strings or Prompt objects.
    This function safely converts either to readable text.
    """
    if isinstance(prompt_obj, str):
        return prompt_obj

    for attr in ["instruction", "text", "prompt"]:
        if hasattr(prompt_obj, attr):
            value = getattr(prompt_obj, attr)
            if isinstance(value, str):
                return value

    return str(prompt_obj)


def _extract_between_markers(text: str, start: str, end: str) -> Optional[str]:
    lower_text = text.lower()
    lower_start = start.lower()
    lower_end = end.lower()

    start_idx = lower_text.find(lower_start)
    if start_idx == -1:
        return None

    start_idx += len(start)
    end_idx = lower_text.find(lower_end, start_idx)

    if end_idx == -1:
        return None

    return text[start_idx:end_idx].strip()


def _extract_class_from_response(response: str, classes: Optional[list[str]]) -> str:
    marker_answer = _extract_between_markers(
        response,
        "<final_answer>",
        "</final_answer>",
    )

    candidate = marker_answer if marker_answer is not None else response
    candidate_lower = candidate.lower()

    if classes:
        for cls in classes:
            cls_text = str(cls)
            if cls_text.lower() == candidate_lower.strip():
                return cls_text

        for cls in classes:
            cls_text = str(cls)
            if cls_text.lower() in candidate_lower:
                return cls_text

    if "subjective" in candidate_lower:
        return "subjective"

    if "objective" in candidate_lower:
        return "objective"

    return candidate.strip()


def evaluate_prompt_with_llm(
    dataset,
    prompt: str,
    llm: BaseLLM,
    classes: Optional[list[str]] = None,
):
    predictions = []
    labels = []

    class_text = ""
    if classes:
        class_text = "Allowed labels: " + ", ".join(str(c) for c in classes) + "\n"

    for ex in dataset:
        full_prompt = (
            f"{prompt}\n\n"
            f"{class_text}"
            f"Input: {ex.text}\n\n"
            f"Return only the answer inside <final_answer> and </final_answer> tags."
        )

        response = llm.get_response(full_prompt)[0]
        pred = _extract_class_from_response(response, classes)

        predictions.append(pred)
        labels.append(ex.label)

    score = accuracy(predictions, labels)

    cost_info = estimate_dataset_cost(
        prompt=prompt,
        inputs=[ex.text for ex in dataset],
        outputs=predictions,
        classes=classes,
    )

    token_info = llm.get_token_count()

    return score, predictions, labels, cost_info, token_info


def _build_promptolution_task(
    dataset,
    optimizer_name: str,
    task_description: Optional[str] = None,
    task_n_subsamples: Optional[int] = None,
    task_eval_strategy: Optional[str] = None,
):
    df = _dataset_to_dataframe(dataset)

    if task_n_subsamples is None:
        task_n_subsamples = max(1, min(5, len(df)))

    if task_eval_strategy is None:
        if optimizer_name == "capo":
            task_eval_strategy = "sequential_block"
        else:
            task_eval_strategy = "full"

    task = ClassificationTask(
        df,
        task_description=task_description
        or (
            "The dataset contains text examples and class labels. "
            "The task is to classify each input into the correct label. "
            "The answer should be placed between <final_answer> and </final_answer> tags."
        ),
        x_column="text",
        y_column="label",
        n_subsamples=task_n_subsamples,
        eval_strategy=task_eval_strategy,
    )

    return task, df, task_n_subsamples, task_eval_strategy


def _validate_optimizer_inputs(
    optimizer_name: str,
    initial_prompts: list[str],
):
    if not initial_prompts:
        raise ValueError("initial_prompts must contain at least one prompt.")

    if optimizer_name in {"evoprompt_de", "evopromptde", "evoprompt-de"}:
        if len(initial_prompts) < 4:
            raise ValueError(
                "EvoPromptDE requires at least 4 initial prompts because it samples "
                "three donor prompts for each current prompt. Add a fourth prompt "
                "to initial_prompts in the config."
            )


def run_promptolution_optimizer(
    dataset,
    initial_prompts: list[str],
    optimizer_name: str,
    n_steps: int = 1,
    llm_config: Optional[dict] = None,
    classes: Optional[list[str]] = None,
    task_description: Optional[str] = None,
    task_n_subsamples: Optional[int] = None,
    task_eval_strategy: Optional[str] = None,
    capo_max_n_blocks_eval: Optional[int] = None,
    capo_upper_shots: int = 0,
    capo_crossovers_per_iter: int = 1,
    evoprompt_de_donor_random: bool = False,
    opro_max_num_instructions: int = 10,
    opro_num_instructions_per_step: int = 2,
    opro_num_few_shots: int = 2,
):
    """
    Run Promptolution optimizers:
      - CAPO
      - EvoPromptGA
      - EvoPromptDE
      - OPRO
    """

    optimizer_name = optimizer_name.lower().strip()
    _validate_optimizer_inputs(optimizer_name, initial_prompts)

    task, df, task_n_subsamples, task_eval_strategy = _build_promptolution_task(
        dataset=dataset,
        optimizer_name=optimizer_name,
        task_description=task_description,
        task_n_subsamples=task_n_subsamples,
        task_eval_strategy=task_eval_strategy,
    )

    llm = build_llm(llm_config)
    predictor = MarkerBasedPredictor(llm, classes=classes)

    if optimizer_name == "capo":
        if capo_max_n_blocks_eval is None:
            capo_max_n_blocks_eval = max(1, len(df) // task_n_subsamples)

        optimizer = CAPO(
            task=task,
            predictor=predictor,
            meta_llm=llm,
            initial_prompts=initial_prompts,
            crossovers_per_iter=capo_crossovers_per_iter,
            upper_shots=capo_upper_shots,
            max_n_blocks_eval=capo_max_n_blocks_eval,
            check_fs_accuracy=False,
            create_fs_reasoning=False,
            callbacks=[],
        )

    elif optimizer_name in {"evoprompt", "evoprompt_ga", "evopromptga", "evoprompt-ga"}:
        optimizer = EvoPromptGA(
            task=task,
            prompt_template=EVOPROMPT_GA_TEMPLATE,
            predictor=predictor,
            meta_llm=llm,
            initial_prompts=initial_prompts,
            callbacks=[],
        )

    elif optimizer_name in {"evoprompt_de", "evopromptde", "evoprompt-de"}:
        optimizer = EvoPromptDE(
            task=task,
            prompt_template=EVOPROMPT_DE_TEMPLATE_TD,
            predictor=predictor,
            meta_llm=llm,
            initial_prompts=initial_prompts,
            donor_random=evoprompt_de_donor_random,
            callbacks=[],
        )

    elif optimizer_name == "opro":
        optimizer = OPRO(
            task=task,
            predictor=predictor,
            meta_llm=llm,
            initial_prompts=initial_prompts,
            max_num_instructions=opro_max_num_instructions,
            num_instructions_per_step=opro_num_instructions_per_step,
            num_few_shots=opro_num_few_shots,
            callbacks=[],
        )

    else:
        raise ValueError(
            f"Unknown Promptolution optimizer: {optimizer_name}. "
            f"Supported: capo, evoprompt_ga, evoprompt_de, opro."
        )

    best_prompts = optimizer.optimize(n_steps=n_steps)

    if isinstance(best_prompts, list) and best_prompts:
        selected_prompt = _normalize_prompt(best_prompts[0])
    else:
        selected_prompt = _normalize_prompt(best_prompts)

    eval_llm = build_llm(llm_config)

    score, predictions, labels, cost_info, token_info = evaluate_prompt_with_llm(
        dataset=dataset,
        prompt=selected_prompt,
        llm=eval_llm,
        classes=classes,
    )

    return {
        "method": f"promptolution_{optimizer_name}",
        "prompt": selected_prompt,
        "score": score,
        "predictions": predictions,
        "labels": labels,
        "promptolution_best_prompts_raw": str(best_prompts),
        "promptolution_token_info": token_info,
        **cost_info,
    }


def run_promptolution_optimizer_with_test(
    dev_dataset,
    test_dataset,
    initial_prompts: list[str],
    optimizer_name: str,
    n_steps: int = 1,
    llm_config: Optional[dict] = None,
    classes: Optional[list[str]] = None,
    task_description: Optional[str] = None,
    task_n_subsamples: Optional[int] = None,
    task_eval_strategy: Optional[str] = None,
    capo_max_n_blocks_eval: Optional[int] = None,
    capo_upper_shots: int = 0,
    capo_crossovers_per_iter: int = 1,
    evoprompt_de_donor_random: bool = False,
    opro_max_num_instructions: int = 10,
    opro_num_instructions_per_step: int = 2,
    opro_num_few_shots: int = 2,
):
    dev_result = run_promptolution_optimizer(
        dataset=dev_dataset,
        initial_prompts=initial_prompts,
        optimizer_name=optimizer_name,
        n_steps=n_steps,
        llm_config=llm_config,
        classes=classes,
        task_description=task_description,
        task_n_subsamples=task_n_subsamples,
        task_eval_strategy=task_eval_strategy,
        capo_max_n_blocks_eval=capo_max_n_blocks_eval,
        capo_upper_shots=capo_upper_shots,
        capo_crossovers_per_iter=capo_crossovers_per_iter,
        evoprompt_de_donor_random=evoprompt_de_donor_random,
        opro_max_num_instructions=opro_max_num_instructions,
        opro_num_instructions_per_step=opro_num_instructions_per_step,
        opro_num_few_shots=opro_num_few_shots,
    )

    selected_prompt = dev_result["prompt"]
    test_llm = build_llm(llm_config)

    (
        test_score,
        test_predictions,
        test_labels,
        test_cost_info,
        test_token_info,
    ) = evaluate_prompt_with_llm(
        dataset=test_dataset,
        prompt=selected_prompt,
        llm=test_llm,
        classes=classes,
    )

    return {
        "method": dev_result["method"],
        "prompt": selected_prompt,
        "dev_score": dev_result["score"],
        "test_score": test_score,
        "dev_predictions": dev_result["predictions"],
        "test_predictions": test_predictions,
        "dev_labels": dev_result["labels"],
        "test_labels": test_labels,
        "dev_input_tokens": dev_result["input_tokens"],
        "dev_output_tokens": dev_result["output_tokens"],
        "dev_cost": dev_result["cost"],
        "test_input_tokens": test_cost_info["input_tokens"],
        "test_output_tokens": test_cost_info["output_tokens"],
        "test_cost": test_cost_info["cost"],
        "promptolution_best_prompts_raw": dev_result.get("promptolution_best_prompts_raw"),
        "promptolution_token_info": dev_result.get("promptolution_token_info"),
        "test_token_info": test_token_info,
    }
