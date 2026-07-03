from __future__ import annotations

import pytest

from scripts.run_phase2_budgeted_mocapo import get_prompt_pool


def test_get_prompt_pool_prioritizes_initial_prompt_ids():
    config = {
        "prompt_pool_inline": [
            {"id": "a", "category": "x", "prompt": "Prompt A"},
            {"id": "b", "category": "x", "prompt": "Prompt B"},
            {"id": "c", "category": "x", "prompt": "Prompt C"},
        ],
        "initial_prompt_ids": ["c", "a"],
    }

    prompts = get_prompt_pool(config)

    assert [item["id"] for item in prompts] == ["c", "a", "b"]


def test_get_prompt_pool_rejects_unknown_initial_prompt_id():
    config = {
        "prompt_pool_inline": [
            {"id": "a", "category": "x", "prompt": "Prompt A"},
        ],
        "initial_prompt_ids": ["missing"],
    }

    with pytest.raises(ValueError, match="Unknown initial_prompt_ids"):
        get_prompt_pool(config)
