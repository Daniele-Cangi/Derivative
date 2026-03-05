import pytest

from core.execution_loop import ExecutionCycle, ExecutionResult
from core.kernel import ReasoningResult, ReasoningStep
from core.problem_classifier import ProblemClassifier
from core.validator import AdversarialValidator, NumericAnswerCheck


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


def test_validator_rejects_numeric_problem_without_numeric_answer():
    validator = AdversarialValidator(api_key="dummy_key_for_testing")
    problem = (
        "Find the minimum number of rounds required so that the probability of interception "
        "falls below 0.001 under both independent and correlated models."
    )
    result = ReasoningResult(
        conclusion="The system appears stable after execution, but the reasoning stayed qualitative.",
        reasoning_chain=[ReasoningStep("1", "step 1"), ReasoningStep("2", "step 2")],
        violated_constraints=[],
        epistemic_confidence=0.99,
        lens_contributions={"MockLens1": 1.0},
        execution_result=ExecutionResult(
            conclusion="No numeric answer",
            cycles_used=1,
            converged=True,
            history=[
                ExecutionCycle(
                    cycle=1,
                    hypothesis="h",
                    code="",
                    output='{"result": {"mode": "generic"}, "confirms_hypothesis": true}',
                    delta=0.0,
                    converged=True,
                )
            ],
            final_code="",
            final_output='{"result": {"mode": "generic"}, "confirms_hypothesis": true}',
        ),
    )

    report = validator.validate(result, problem)

    assert report.is_valid is False
    assert report.recommendation == "re-reason"
    assert "numeric answer" in report.attacks[0].lower()


def test_validator_reduces_confidence_when_models_diverge():
    validator = AdversarialValidator(api_key="dummy_key_for_testing")
    problem = (
        "Find the minimum number of rounds required so that the probability of interception "
        "falls below 0.001 under both independent and correlated models."
    )
    result = ReasoningResult(
        conclusion=(
            "The minimum is 5 rounds. The independent model needs 4 rounds while the correlated "
            "model needs 5."
        ),
        reasoning_chain=[ReasoningStep("1", "step 1"), ReasoningStep("2", "step 2")],
        violated_constraints=[],
        epistemic_confidence=0.95,
        lens_contributions={"MockLens1": 1.0},
        execution_result=ExecutionResult(
            conclusion="Has numeric answer",
            cycles_used=2,
            converged=True,
            history=[
                ExecutionCycle(
                    cycle=2,
                    hypothesis="h",
                    code="",
                    output=(
                        '{"result": {"mode": "probabilistic_numeric", "result": 5, '
                        '"minimum_rounds": 5, "independent_model_result": 4, '
                        '"correlated_model_result": 5}, "confirms_hypothesis": true}'
                    ),
                    delta=0.0,
                    converged=True,
                )
            ],
            final_code="",
            final_output=(
                '{"result": {"mode": "probabilistic_numeric", "result": 5, '
                '"minimum_rounds": 5, "independent_model_result": 4, '
                '"correlated_model_result": 5}, "confirms_hypothesis": true}'
            ),
        ),
    )

    report = validator.validate(result, problem)

    assert report.is_valid is True
    assert report.confidence_adjusted < 0.95
    assert any("diverged" in edge_case.lower() for edge_case in report.edge_cases)


def test_validator_rejects_numeric_token_without_objective_coverage():
    validator = AdversarialValidator(api_key="dummy_key_for_testing")
    problem = (
        "Given the linear recurrence f(n) = 3f(n-1) - f(n-2), with f(0)=1 and f(1)=2, "
        "find the closed-form formula, verify it satisfies the recurrence, determine the first n "
        "such that f(n) > 10^6, and compute the limit of f(n+1)/f(n)."
    )
    result = ReasoningResult(
        conclusion="The answer is 1.",
        reasoning_chain=[ReasoningStep("1", "step 1"), ReasoningStep("2", "step 2")],
        violated_constraints=[],
        epistemic_confidence=0.99,
        lens_contributions={"MockLens1": 1.0},
        execution_result=ExecutionResult(
            conclusion="Only initial token surfaced",
            cycles_used=1,
            converged=True,
            history=[
                ExecutionCycle(
                    cycle=1,
                    hypothesis="h",
                    code="",
                    output='{"result": {"mode": "numeric", "result": 1}, "confirms_hypothesis": true}',
                    delta=0.0,
                    converged=True,
                )
            ],
            final_code="",
            final_output='{"result": {"mode": "numeric", "result": 1}, "confirms_hypothesis": true}',
        ),
    )

    report = validator.validate(result, problem)

    assert report.is_valid is False
    assert report.recommendation == "re-reason"
    assert "unresolved objectives" in report.attacks[0].lower()


def test_numeric_check_accepts_symbolic_numeric_payload_with_objective_coverage():
    checker = NumericAnswerCheck()
    classifier = ProblemClassifier()
    problem = (
        "Given the linear recurrence f(n) = 3f(n-1) - f(n-2), with f(0)=1 and f(1)=2, "
        "find the closed-form formula, verify it satisfies the recurrence, determine the first n "
        "such that f(n) > 10^6, and compute the limit of f(n+1)/f(n)."
    )
    classification = classifier.classify(problem)
    conclusion = (
        "SymPy derived f(n)=((3-sqrt(5))**n*(5-sqrt(5)) + (3+sqrt(5))**n*(5+sqrt(5)))/(10*2**n), "
        "verified the recurrence on the verification window, found threshold 10^6 at n=15, and computed "
        "the ratio limit as sqrt(5)/2 + 3/2."
    )
    execution_output = (
        '{"result": {"mode": "symbolic_numeric", "closed_form": "expr", "recurrence_verified": true, '
        '"threshold_target": 1000000, "threshold_index": 15, "ratio_limit": "sqrt(5)/2 + 3/2", '
        '"result_count": 4, "verification_window": 10}, "confirms_hypothesis": true}'
    )

    ok, failure = checker.check(classification, conclusion, execution_output, problem)

    assert ok is True
    assert failure == ""


def test_numeric_check_accepts_infeasible_payload():
    checker = NumericAnswerCheck()
    classifier = ProblemClassifier()
    problem = "Find x such that x > 2 and x <= 2."
    classification = classifier.classify(problem)
    conclusion = "The constraints are contradictory, so no satisfiable solution exists."
    execution_output = (
        '{"result": {"mode": "infeasible", "is_satisfiable": false, "contradiction_count": 1}, '
        '"confirms_hypothesis": true}'
    )

    ok, failure = checker.check(classification, conclusion, execution_output, problem)

    assert ok is True
    assert failure == ""
