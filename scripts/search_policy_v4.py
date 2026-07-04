from __future__ import annotations

import random


def choose_examples(pool: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = [dict(item) for item in pool]
    rng.shuffle(rows)
    return rows[:max(0, count)]
