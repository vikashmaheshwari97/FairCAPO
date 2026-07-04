from __future__ import annotations

import os
from scripts import run_phase2_budgeted_mocapo as runner

os.environ["FAIRCAPO_ADULT_REASONING_SHOTS"] = "0"

if __name__ == "__main__":
    runner.main()
