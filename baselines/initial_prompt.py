# from typing import Optional
#
# from experiments.metrics import accuracy
# from experiments.token_cost import estimate_cost
#
#
# class RuleBasedToyLLM:
#     """
#     Temporary fake model for Phase 1 debugging.
#     Currently works best for Subj-style subjective/objective classification.
#     """
#
#     SUBJECTIVE_WORDS = {
#         "boring",
#         "wonderful",
#         "beautiful",
#         "badly",
#         "painfully",
#         "best",
#         "worst",
#         "amazing",
#         "amusing",
#         "important",
#         "dull",
#         "funny",
#         "touching",
#         "poor",
#         "great",
#         "terrible",
#         "excellent",
#         "awful",
#     }
#
#     def predict(self, prompt: str, text: str) -> str:
#         lowered = text.lower()
#
#         if any(word in lowered for word in self.SUBJECTIVE_WORDS):
#             return "subjective"
#
#         return "objective"
#
#
# def _extract_class_from_response(response: str, classes: Optional[list[str]] = None) -> str:
#     lowered = response.lower()
#
#     if "<final_answer>" in lowered and "</final_answer>" in lowered:
#         start = lowered.find("<final_answer>") + len("<final_answer>")
#         end = lowered.find("</final_answer>", start)
#         answer = response[start:end].strip()
#         lowered = answer.lower()
#
#     if classes:
#         for cls in classes:
#             if str(cls).lower() == lowered.strip():
#                 return str(cls)
#
#         for cls in classes:
#             if str(cls).lower() in lowered:
#                 return str(cls)
#
#     if "subjective" in lowered:
#         return "subjective"
#
#     if "objective" in lowered:
#         return "objective"
#
#     return response.strip()
#
#
# def evaluate_initial_prompt_rule_based(dataset, prompt: str):
#     model = RuleBasedToyLLM()
#
#     predictions = []
#     labels = []
#
#     for ex in dataset:
#         pred = model.predict(prompt, ex.text)
#         predictions.append(pred)
#         labels.append(ex.label)
#
#     score = accuracy(predictions, labels)
#     cost_info = estimate_cost(prompt, predictions)
#
#     return {
#         "score": score,
#         "predictions": predictions,
#         "labels": labels,
#         **cost_info,
#     }
#
#
# def evaluate_initial_prompt_with_llm(dataset, prompt: str, llm, classes: Optional[list[str]] = None):
#     predictions = []
#     labels = []
#
#     class_text = ""
#     if classes:
#         class_text = "Allowed labels: " + ", ".join(str(c) for c in classes) + "\n"
#
#     for ex in dataset:
#         full_prompt = (
#             f"{prompt}\n\n"
#             f"{class_text}"
#             f"Input: {ex.text}\n\n"
#             f"Return only the answer inside <final_answer> and </final_answer> tags."
#         )
#
#         response = llm.get_response(full_prompt)[0]
#         pred = _extract_class_from_response(response, classes)
#
#         predictions.append(pred)
#         labels.append(ex.label)
#
#     score = accuracy(predictions, labels)
#     cost_info = estimate_cost(prompt, predictions)
#
#     return {
#         "score": score,
#         "predictions": predictions,
#         "labels": labels,
#         **cost_info,
#     }
#
#
# def run_initial_prompt_baseline(dataset, prompt: str):
#     eval_result = evaluate_initial_prompt_rule_based(dataset, prompt)
#
#     return {
#         "method": "initial_prompt",
#         "prompt": prompt,
#         **eval_result,
#     }
#
#
# def run_initial_prompt_baseline_with_test(
#     dev_dataset,
#     test_dataset,
#     prompt: str,
#     llm=None,
#     classes=None,
# ):
#     if llm is None:
#         dev_result = evaluate_initial_prompt_rule_based(dev_dataset, prompt)
#         test_result = evaluate_initial_prompt_rule_based(test_dataset, prompt)
#     else:
#         dev_result = evaluate_initial_prompt_with_llm(dev_dataset, prompt, llm, classes)
#         test_result = evaluate_initial_prompt_with_llm(test_dataset, prompt, llm, classes)
#
#     return {
#         "method": "initial_prompt",
#         "prompt": prompt,
#         "dev_score": dev_result["score"],
#         "test_score": test_result["score"],
#         "dev_predictions": dev_result["predictions"],
#         "test_predictions": test_result["predictions"],
#         "dev_labels": dev_result["labels"],
#         "test_labels": test_result["labels"],
#         "dev_input_tokens": dev_result["input_tokens"],
#         "dev_output_tokens": dev_result["output_tokens"],
#         "dev_cost": dev_result["cost"],
#         "test_input_tokens": test_result["input_tokens"],
#         "test_output_tokens": test_result["output_tokens"],
#         "test_cost": test_result["cost"],
#     }

from typing import Optional

from experiments.metrics import accuracy
from experiments.token_cost import estimate_dataset_cost


class RuleBasedToyLLM:
    """
    Temporary fake model for Phase 1 debugging.
    Currently works best for Subj-style subjective/objective classification.
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
    }

    def predict(self, prompt: str, text: str) -> str:
        lowered = text.lower()

        if any(word in lowered for word in self.SUBJECTIVE_WORDS):
            return "subjective"

        return "objective"


def _extract_class_from_response(response: str, classes: Optional[list[str]] = None) -> str:
    lowered = response.lower()

    if "<final_answer>" in lowered and "</final_answer>" in lowered:
        start = lowered.find("<final_answer>") + len("<final_answer>")
        end = lowered.find("</final_answer>", start)
        answer = response[start:end].strip()
        lowered = answer.lower()

    if classes:
        for cls in classes:
            if str(cls).lower() == lowered.strip():
                return str(cls)

        for cls in classes:
            if str(cls).lower() in lowered:
                return str(cls)

    if "subjective" in lowered:
        return "subjective"

    if "objective" in lowered:
        return "objective"

    return response.strip()


def evaluate_initial_prompt_rule_based(
    dataset,
    prompt: str,
    classes: Optional[list[str]] = None,
):
    model = RuleBasedToyLLM()

    predictions = []
    labels = []

    for ex in dataset:
        pred = model.predict(prompt, ex.text)
        predictions.append(pred)
        labels.append(ex.label)

    score = accuracy(predictions, labels)

    cost_info = estimate_dataset_cost(
        prompt=prompt,
        inputs=[ex.text for ex in dataset],
        outputs=predictions,
        classes=classes,
    )

    return {
        "score": score,
        "predictions": predictions,
        "labels": labels,
        **cost_info,
    }


def evaluate_initial_prompt_with_llm(
    dataset,
    prompt: str,
    llm,
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

    return {
        "score": score,
        "predictions": predictions,
        "labels": labels,
        **cost_info,
    }


def run_initial_prompt_baseline(
    dataset,
    prompt: str,
    classes: Optional[list[str]] = None,
):
    eval_result = evaluate_initial_prompt_rule_based(
        dataset=dataset,
        prompt=prompt,
        classes=classes,
    )

    return {
        "method": "initial_prompt",
        "prompt": prompt,
        **eval_result,
    }


def run_initial_prompt_baseline_with_test(
    dev_dataset,
    test_dataset,
    prompt: str,
    llm=None,
    classes=None,
):
    if llm is None:
        dev_result = evaluate_initial_prompt_rule_based(
            dataset=dev_dataset,
            prompt=prompt,
            classes=classes,
        )
        test_result = evaluate_initial_prompt_rule_based(
            dataset=test_dataset,
            prompt=prompt,
            classes=classes,
        )
    else:
        dev_result = evaluate_initial_prompt_with_llm(
            dataset=dev_dataset,
            prompt=prompt,
            llm=llm,
            classes=classes,
        )
        test_result = evaluate_initial_prompt_with_llm(
            dataset=test_dataset,
            prompt=prompt,
            llm=llm,
            classes=classes,
        )

    return {
        "method": "initial_prompt",
        "prompt": prompt,
        "dev_score": dev_result["score"],
        "test_score": test_result["score"],
        "dev_predictions": dev_result["predictions"],
        "test_predictions": test_result["predictions"],
        "dev_labels": dev_result["labels"],
        "test_labels": test_result["labels"],
        "dev_input_tokens": dev_result["input_tokens"],
        "dev_output_tokens": dev_result["output_tokens"],
        "dev_cost": dev_result["cost"],
        "test_input_tokens": test_result["input_tokens"],
        "test_output_tokens": test_result["output_tokens"],
        "test_cost": test_result["cost"],
    }