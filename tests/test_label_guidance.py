import pytest

from heal_capo.core import PromptCandidate
from scripts.run_phase2_budgeted_mocapo import (
    BIOS_PROFESSION_BOUNDARY_GUIDE,
    label_guidance_from_config,
    make_llm_prompt,
)


def test_bios_label_guidance_from_config_returns_boundary_guide():
    guide = label_guidance_from_config(
        {"evaluation": {"label_guidance": "bios_profession_boundaries"}}
    )

    assert "surgeon requires surgery" in guide
    assert "paralegal assists lawyers" in guide


def test_label_guidance_from_config_accepts_custom_text():
    guide = label_guidance_from_config(
        {
            "evaluation": {
                "label_guidance": "custom",
                "label_guidance_text": "Use the narrowest supported label.",
            }
        }
    )

    assert guide == "Use the narrowest supported label."


def test_label_guidance_from_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown evaluation.label_guidance"):
        label_guidance_from_config({"evaluation": {"label_guidance": "bogus"}})


def test_make_llm_prompt_includes_label_guidance_for_classification():
    candidate = PromptCandidate(instruction="Classify the biography.")

    prompt = make_llm_prompt(
        candidate=candidate,
        text="Biography: performs surgical operations.",
        labels=["physician", "surgeon"],
        require_final_answer_tags=True,
        label_guidance=BIOS_PROFESSION_BOUNDARY_GUIDE,
    )

    assert "Profession boundary guide:" in prompt
    assert "surgeon requires surgery" in prompt
    assert "Allowed labels: physician, surgeon" in prompt


def test_make_llm_prompt_omits_label_guidance_for_generation_tasks():
    candidate = PromptCandidate(instruction="Solve.")

    prompt = make_llm_prompt(
        candidate=candidate,
        text="What is 2 + 2?",
        labels=[],
        task_type="math_reasoning",
        label_guidance=BIOS_PROFESSION_BOUNDARY_GUIDE,
    )

    assert "Profession boundary guide:" not in prompt
