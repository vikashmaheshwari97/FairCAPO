from __future__ import annotations

import re
from typing import Optional


def simple_token_count(text: str) -> int:
    """
    Lightweight tokenizer fallback.

    This is not exact model tokenization, but it is stable and consistent
    across local backends. Good enough for Phase 1 cost comparison.
    """
    if text is None:
        return 0

    text = str(text).strip()

    if not text:
        return 0

    # Slightly better than text.split(): counts punctuation as separate tokens.
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return len(tokens)


def build_evaluation_prompt(
    prompt: str,
    input_text: str,
    classes: Optional[list[str]] = None,
) -> str:
    """
    Reconstruct the same evaluation prompt shape used by evaluate_prompt_with_llm.
    """
    class_text = ""
    if classes:
        class_text = "Allowed labels: " + ", ".join(str(c) for c in classes) + "\n"

    return (
        f"{prompt}\n\n"
        f"{class_text}"
        f"Input: {input_text}\n\n"
        f"Return only the answer inside <final_answer> and </final_answer> tags."
    )


def estimate_cost(
    prompt: str,
    outputs: list[str],
    input_weight: float = 0.08,
    output_weight: float = 0.32,
):
    """
    Backward-compatible simple cost estimate.

    Counts only the instruction prompt once plus all outputs.
    Kept for older code paths.
    """
    input_tokens = simple_token_count(prompt)
    output_tokens = sum(simple_token_count(output) for output in outputs)

    cost = input_weight * input_tokens + output_weight * output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
    }


def estimate_dataset_cost(
    prompt: str,
    inputs: list[str],
    outputs: list[str],
    classes: Optional[list[str]] = None,
    input_weight: float = 0.08,
    output_weight: float = 0.32,
):
    """
    Better Phase 1 cost estimate.

    Counts the full evaluation prompt for every example, including:
      - instruction prompt
      - allowed labels
      - input text
      - final-answer instruction
    """
    input_tokens = 0

    for input_text in inputs:
        full_prompt = build_evaluation_prompt(
            prompt=prompt,
            input_text=input_text,
            classes=classes,
        )
        input_tokens += simple_token_count(full_prompt)

    output_tokens = sum(simple_token_count(output) for output in outputs)

    cost = input_weight * input_tokens + output_weight * output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
    }