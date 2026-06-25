from heal_capo.components.drift_guard import KeywordDriftGuard
from heal_capo.components.repair import TemplateRepairer
from heal_capo.components.verifier import RuleBasedVerifier
from heal_capo.continual import ContinualHealer
from heal_capo.core import EvaluationResult, PromptCandidate, PromptPortfolio
from heal_capo.objectives import ToyObjectiveEvaluator


ORIGINAL_INSTRUCTION = (
    "Classify the input using the provided context. "
    "Do not hallucinate. If there is not enough information, say so."
)


def _portfolio_with_one_candidate():
    candidate = PromptCandidate(
        instruction=ORIGINAL_INSTRUCTION,
        metadata={"source": "initial_prompt"},
    )
    portfolio = PromptPortfolio()
    portfolio.add(
        candidate,
        EvaluationResult(
            candidate_id=candidate.candidate_id,
            performance=0.5,
            cost=0.2,
            risk=0.35,
            fairness_risk=0.1,
        ),
    )
    return portfolio, candidate


def _healer(keep_rejected_repairs: bool = False) -> ContinualHealer:
    return ContinualHealer(
        verifier=RuleBasedVerifier(),
        repairer=TemplateRepairer(),
        # Permissive guard so a repaired prompt is accepted (exercises the
        # post-drift evaluate + portfolio-add path).
        drift_guard=KeywordDriftGuard(
            required_terms=["classify", "input"],
            max_missing_ratio=1.0,
        ),
        evaluator=ToyObjectiveEvaluator(),
        keep_rejected_repairs=keep_rejected_repairs,
    )


def test_observe_records_failure():
    healer = _healer()

    result = healer.observe(
        x="some text",
        output="The answer is definitely correct.",
        candidate_id="cand-1",
        context="unrelated grounding text about weather",
    )

    assert result is not None
    assert result.failure_type == "hallucination"
    assert len(healer.memory) == 1


def test_observe_returns_none_when_no_failure():
    healer = _healer()

    result = healer.observe(
        x="some text",
        output="not enough information",
        candidate_id="cand-1",
    )

    assert result is None
    assert healer.memory.is_empty()


def test_repair_portfolio_accepts_repair_and_updates_archive():
    """End-to-end HEAL loop: observe -> cluster -> repair -> drift -> evaluate.

    This is the path that previously crashed because repair_portfolio called
    drift_guard.check() with the wrong keyword argument.
    """
    healer = _healer()
    portfolio, candidate = _portfolio_with_one_candidate()

    healer.observe(
        x="some text",
        output="The answer is definitely correct.",
        candidate_id=candidate.candidate_id,
        context="unrelated grounding text about weather",
    )
    assert len(healer.memory) == 1

    updated = healer.repair_portfolio(
        portfolio=portfolio,
        original_instruction=ORIGINAL_INSTRUCTION,
        dev_data=[{"input": "x", "label": "objective"}],
    )

    report = healer.get_last_report()
    assert report.num_failures_seen == 1
    assert report.num_repairs_attempted == 1
    assert report.num_repairs_accepted == 1
    assert report.repair_acceptance_rate == 1.0

    # The repaired candidate should be evaluated and present in the archive.
    accepted_event = report.events[0]
    assert accepted_event.accepted is True
    assert accepted_event.repaired_candidate_id is not None
    assert accepted_event.repaired_candidate_id in updated.evaluations


def test_repair_portfolio_rejected_by_drift_guard():
    """A strict drift guard should reject the repair without crashing."""
    healer = ContinualHealer(
        verifier=RuleBasedVerifier(),
        repairer=TemplateRepairer(),
        drift_guard=KeywordDriftGuard(
            required_terms=["this-term-will-never-be-present"],
            max_missing_ratio=0.0,
        ),
        evaluator=ToyObjectiveEvaluator(),
    )
    portfolio, candidate = _portfolio_with_one_candidate()

    healer.observe(
        x="some text",
        output="The answer is definitely correct.",
        candidate_id=candidate.candidate_id,
        context="unrelated grounding text about weather",
    )

    healer.repair_portfolio(
        portfolio=portfolio,
        original_instruction=ORIGINAL_INSTRUCTION,
        dev_data=[{"input": "x", "label": "objective"}],
    )

    report = healer.get_last_report()
    assert report.num_repairs_attempted == 1
    assert report.num_repairs_rejected == 1
    assert report.events[0].accepted is False
    assert "drift" in report.events[0].reason.lower()


def test_repair_portfolio_source_candidate_missing():
    healer = _healer()
    portfolio = PromptPortfolio()  # empty: source candidate won't be found

    healer.observe(
        x="some text",
        output="The answer is definitely correct.",
        candidate_id="missing-candidate-id",
        context="unrelated grounding text about weather",
    )

    healer.repair_portfolio(
        portfolio=portfolio,
        original_instruction=ORIGINAL_INSTRUCTION,
        dev_data=[{"input": "x", "label": "objective"}],
    )

    report = healer.get_last_report()
    assert report.num_repairs_accepted == 0
    assert report.events[0].accepted is False
    assert "not found" in report.events[0].reason.lower()


def test_clear_memory():
    healer = _healer()
    healer.observe(
        x="some text",
        output="The answer is definitely correct.",
        candidate_id="cand-1",
        context="unrelated grounding text about weather",
    )
    assert len(healer.memory) == 1

    healer.clear_memory()
    assert healer.memory.is_empty()
