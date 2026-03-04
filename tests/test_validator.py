import pytest

from core.kernel import ReasoningResult, ReasoningStep
from core.validator import AdversarialValidator


def test_validator_accepts_strong_local_result():
    validator = AdversarialValidator(api_key="dummy_key_for_testing")

    result = ReasoningResult(
        conclusion=(
            "The dominant failure modes cluster around boundary-condition failures and hidden assumptions. "
            "Preserve formal invariants and resource constraints at every step."
        ),
        reasoning_chain=[
            ReasoningStep("1", "step 1"),
            ReasoningStep("2", "step 2"),
        ],
        violated_constraints=[],
        epistemic_confidence=0.9,
        lens_contributions={"MockLens1": 1.0},
    )

    report = validator.validate(result, "Test Problem")

    assert report.is_valid is True
    assert report.recommendation == "accept"
    assert report.confidence_adjusted > 0.6


def test_validator_fails_on_violated_constraints():
    validator = AdversarialValidator(api_key="dummy_key_for_testing")

    result = ReasoningResult(
        conclusion="Test conclusion",
        reasoning_chain=[ReasoningStep("1", "step 1")],
        violated_constraints=["Violated rule 1"],
        epistemic_confidence=0.9,
        lens_contributions={"MockLens1": 1.0},
    )

    report = validator.validate(result, "Test Problem")

    assert report.is_valid is False
    assert report.recommendation == "re-reason"


def test_validator_rejects_weak_result():
    validator = AdversarialValidator(api_key="dummy_key_for_testing")

    result = ReasoningResult(
        conclusion="Too short.",
        reasoning_chain=[],
        violated_constraints=[],
        epistemic_confidence=0.4,
        lens_contributions={},
    )

    report = validator.validate(result, "Analyze this distributed API")

    assert report.is_valid is False
    assert report.recommendation in {"re-reason", "escalate_to_human"}
    assert report.attacks


def test_validator_remote_only_requires_live_credentials():
    validator = AdversarialValidator(api_key="dummy_key_for_testing", execution_mode="remote-only")

    result = ReasoningResult(
        conclusion=(
            "A sufficiently long conclusion to bypass local short-output validation while keeping the test focused "
            "on execution mode behavior."
        ),
        reasoning_chain=[ReasoningStep("1", "step 1"), ReasoningStep("2", "step 2")],
        violated_constraints=[],
        epistemic_confidence=0.8,
        lens_contributions={"MockLens1": 1.0},
    )

    with pytest.raises(RuntimeError):
        validator.validate(result, "Validate this API reasoning")
