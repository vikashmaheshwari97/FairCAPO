from __future__ import annotations

import math
import re


def parse_fields(text: str) -> dict[str, str]:
    fields = {}
    for line in str(text).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
    return fields


def numeric(value: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
    return float(match.group(0)) if match else 0.0


def category(value: str) -> str:
    return str(value or "").split("(", 1)[0].strip().lower()


def ordinal(value: str) -> float:
    match = re.search(r"ordinal education level\s+(\d+)", str(value or ""), re.I)
    return float(match.group(1)) if match else 0.0


def proximity(left: float, right: float, scale: float) -> float:
    return math.exp(-abs(left - right) / max(scale, 1e-9))
