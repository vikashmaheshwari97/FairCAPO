from heal_capo.components.repair import TemplateRepairer
from heal_capo.components.verifier import VerificationResult
from heal_capo.core import PromptCandidate


def _candidate():
    return PromptCandidate(
        instruction="Classify the sentence as subjective or objective.",
        examples=[{"input": "A", "output": "objective"}],
        metadata={"source": "test"},
    )


def _feedback(failure_type: str):
    return VerificationResult(
        risk_score=0.8,
        failure_type=failure_type,
        explanation=f"{failure_type} failure",
    )


def test_repair_hallucination():
    repairer = TemplateRepairer()
    repaired = repairer.repair(_candidate(), _feedback("hallucination"))

    assert "Do not hallucinate" in repaired.instruction
    assert repaired.metadata["repair_failure_type"] == "hallucination"


def test_repair_fairness():
    repairer = TemplateRepairer()
    repaired = repairer.repair(_candidate(), _feedback("fairness"))

    assert "demographic attributes" in repaired.instruction
    assert "gender" in repaired.instruction
    assert repaired.metadata["repair_failure_type"] == "fairness"


def test_repair_over_refusal():
    repairer = TemplateRepairer()
    repaired = repairer.repair(_candidate(), _feedback("over_refusal"))

    assert "Do not refuse" in repaired.instruction
    assert repaired.metadata["repair_failure_type"] == "over_refusal"


def test_repair_drift():
    repairer = TemplateRepairer()
    repaired = repairer.repair(_candidate(), _feedback("drift"))

    assert "Preserve the original task" in repaired.instruction
    assert repaired.metadata["repair_failure_type"] == "drift"


def test_repair_format():
    repairer = TemplateRepairer()
    repaired = repairer.repair(_candidate(), _feedback("format"))

    assert "required output format" in repaired.instruction


def test_repair_unknown_failure_type_uses_default():
    repairer = TemplateRepairer()
    repaired = repairer.repair(_candidate(), _feedback("unknown"))

    assert "avoid unsupported assumptions" in repaired.instruction


def test_repair_preserves_examples_and_parent():
    candidate = _candidate()
    repairer = TemplateRepairer()

    repaired = repairer.repair(candidate, _feedback("incorrect"))

    assert repaired.examples == candidate.examples
    assert repaired.parent_ids == [candidate.candidate_id]


def test_repair_preserves_existing_metadata():
    repairer = TemplateRepairer()
    repaired = repairer.repair(_candidate(), _feedback("incorrect"))

    assert repaired.metadata["source"] == "test"
    assert "repair_from" in repaired.metadata


def test_repair_does_not_duplicate_same_rule():
    candidate = _candidate()
    repairer = TemplateRepairer()

    first = repairer.repair(candidate, _feedback("fairness"))
    second = repairer.repair(first, _feedback("fairness"))

    assert second.instruction.count("Do not infer ability") == 1


def test_repair_can_truncate_instruction():
    repairer = TemplateRepairer(max_instruction_chars=60)
    repaired = repairer.repair(_candidate(), _feedback("hallucination"))

    assert len(repaired.instruction) <= 60