from __future__ import annotations

import argparse
import yaml

from heal_capo.objectives import ToyObjectiveEvaluator
from heal_capo.optimizers.risk_aware_mo_capo import RiskAwareMOCAPO, RiskAwareMOCAPOConfig
from heal_capo.components.drift_guard import KeywordDriftGuard
from heal_capo.components.router import RiskAwareRouter
from heal_capo.components.verifier import RuleBasedVerifier
from heal_capo.components.repair import TemplateRepairer
from heal_capo.continual import ContinualHealer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    initial_prompts = [
        "Answer the question and provide the final answer. Do not hallucinate.",
        "Use context to answer. Provide the final answer. Do not hallucinate.",
        "Give a concise final answer.",
    ]
    dev_data = [{"x": "toy question", "y": "toy answer"}]

    guard = KeywordDriftGuard(cfg["semantic_drift"]["required_terms"])
    optimizer = RiskAwareMOCAPO(ToyObjectiveEvaluator(), guard, RiskAwareMOCAPOConfig(**cfg["optimizer"]))
    portfolio = optimizer.optimize(initial_prompts, dev_data)

    router = RiskAwareRouter(lambda_perf=1.0, lambda_cost=0.2, lambda_risk=1.0)
    decision = router.select("toy question", portfolio)
    print("Selected prompt:", decision)

    healer = ContinualHealer(RuleBasedVerifier(), TemplateRepairer(), guard, ToyObjectiveEvaluator())
    healer.observe("toy question", "Obviously this is always true", decision.candidate_id, context="limited context")
    portfolio = healer.repair_portfolio(portfolio, initial_prompts[0], dev_data)
    print("Portfolio size after healing:", len(portfolio.candidates))
    for cid, result in portfolio.evaluations.items():
        print(cid, result)


if __name__ == "__main__":
    main()
