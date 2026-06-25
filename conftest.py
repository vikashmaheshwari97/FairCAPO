"""
Pytest bootstrap.

Ensures the repository root is importable so tests can import the
non-packaged `scripts` modules (only `heal_capo*` is an installed package).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
