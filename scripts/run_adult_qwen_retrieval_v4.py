from __future__ import annotations

from heal_capo.adult_v4_data import rows_from_manifest
from heal_capo.retrieval_evaluator import build_retrieval_evaluator
from scripts import run_phase2_budgeted_mocapo as runner


runner.get_dev_data = lambda config: rows_from_manifest(config, "dev")
runner.build_objective_evaluator = build_retrieval_evaluator


if __name__ == "__main__":
    runner.main()
