from __future__ import annotations

from heal_capo.adult_v4_data import rows_from_manifest
from heal_capo.retrieval_evaluator import build_retrieval_evaluator
from scripts import evaluate_pareto_on_test as evaluator


evaluator.load_test_data = lambda config, seed: rows_from_manifest(config, "test")
evaluator.build_objective_evaluator = build_retrieval_evaluator


if __name__ == "__main__":
    evaluator.main()
