from __future__ import annotations

import json

from heal_capo.optimizers.block_evaluator import prompt_signature
from scripts import evaluate_pareto_on_test as evaluator


_original_build_candidate_rows = evaluator.build_candidate_rows


def build_candidate_rows_with_identity(results, candidates, pareto_ids):
    rows = _original_build_candidate_rows(results, candidates, pareto_ids)
    candidates_by_id = {
        candidate.candidate_id: candidate
        for candidate in candidates
    }

    for row in rows:
        candidate = candidates_by_id[row["candidate_id"]]
        examples = [dict(example) for example in candidate.examples]
        row["num_few_shot"] = len(examples)
        row["few_shot_examples"] = json.dumps(
            examples,
            ensure_ascii=False,
        )
        row["prompt_signature"] = prompt_signature(candidate)

    return rows


evaluator.build_candidate_rows = build_candidate_rows_with_identity


if __name__ == "__main__":
    evaluator.main()
