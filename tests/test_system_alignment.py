from heal_capo.components.failure_memory import FailureCase, FailureMemory
from heal_capo.components.repair import TemplateRepairer
from heal_capo.components.verifier import VerificationResult
from heal_capo.continual import ContinualHealer, HealingReport
from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
from dashboard.components import clean_portfolio_dataframe, compute_weighted_utility

import pandas as pd


def test_core_evaluation_result_has_trust_fields():
    result = EvaluationResult(
        candidate_id="c1",
        performance=0.8,
        cost=1.0,
        risk=0.2,
        fairness_risk=0.1,
        drift=0.0,
    )

    assert result.objective_vector == (-0.8, 1.0, 0.2, 0.1)


def test_failure_memory_and_repair_align():
    memory = FailureMemory()
    memory.add(
        FailureCase(
            x="The old employee arrived at 9 a.m.",
            output="subjective",
            candidate_id="p1",
            failure_type="fairness",
            explanation="Counterfactual flip.",
        )
    )

    repairer = TemplateRepairer()
    candidate = PromptCandidate(
        instruction="Classify as subjective or objective.",
        candidate_id="p1",
    )
    feedback = VerificationResult(
        risk_score=0.9,
        failure_type="fairness",
        explanation="Counterfactual flip.",
    )

    repaired = repairer.repair(candidate, feedback)

    assert memory.fairness_debt("p1") == 1
    assert "demographic attributes" in repaired.instruction
    assert repaired.metadata["repair_failure_type"] == "fairness"


def test_dashboard_accepts_counterfactual_portfolio_shape():
    df = pd.DataFrame(
        [
            {
                "candidate_id": "p1",
                "method": "fairness_prompt",
                "category": "fairness_first",
                "performance": 0.8,
                "cost": 10.0,
                "risk": 0.2,
                "fairness_risk": 0.05,
                "is_pareto": True,
                "prompt": "Classify fairly.",
                "detail_counterfactual_flip_rate": 0.05,
                "detail_num_flips": 1,
                "detail_num_pairs": 20,
            }
        ]
    )

    cleaned = clean_portfolio_dataframe(df)
    ranked = compute_weighted_utility(cleaned)

    assert cleaned.iloc[0]["counterfactual_flip_rate"] == 0.05
    assert cleaned.iloc[0]["num_fairness_flips"] == 1
    assert "dashboard_utility" in ranked.columns


def test_continual_healer_report_type_exists():
    report = HealingReport()

    assert report.num_repairs_attempted == 0
    assert report.repair_acceptance_rate == 0.0