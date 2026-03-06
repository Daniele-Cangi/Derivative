import json

from core.obligation_compiler import ObligationCompiler
from core.problem_classifier import ProblemClassifier, ProblemType


RECURRENCE_QUERY = (
    "Given the linear recurrence f(n) = 3f(n-1) - f(n-2), with f(0)=1 and f(1)=2, "
    "find the closed-form formula, verify it satisfies the recurrence, determine the first n "
    "such that f(n) > 10^6, and compute the limit of f(n+1)/f(n)."
)

RECURRENCE_QUERY_FOR_CLAUSE = (
    "A recursive function f(n) is defined as f(0)=1, f(1)=2, f(n)=3*f(n-1) - f(n-2) for n>=2. "
    "Find the closed-form expression for f(n), verify it satisfies the recurrence for n=0 through n=10, "
    "and determine for which values of n the function exceeds 10^6. Then prove that the ratio f(n+1)/f(n) "
    "converges and find the limit."
)

CRYPTO_QUERY = (
    "A cryptographic key exchange protocol uses 5 participants in a ring topology. "
    "Each participant i can only communicate directly with participant i+1 mod 5. "
    "Each communication channel has a probability of interception p=0.03. "
    "Find the minimum number of independent key-agreement rounds required so that the "
    "probability of at least one undetected interception across the full ring falls below 0.001. "
    "Then verify that your answer satisfies the constraint under both independent and correlated "
    "channel failure models."
)

SURVIVAL_QUERY = (
    "A reliability network has 5 channels with per-hour failure probability p=0.03. "
    "Compute the probability the system survives 8 hours, and compute the expected number "
    "of hours before failure under independent and correlated failure models."
)

SURVIVAL_COMBINATORIAL_QUERY = (
    "A system has 4 independent components, each failing with probability p=0.1 per hour. "
    "The system fails if 2 or more components fail in the same hour. "
    "Find the probability that the system survives exactly 8 consecutive hours without failure. "
    "Then find the expected number of hours until first system failure."
)


def test_obligation_compiler_builds_symbolic_schema():
    compiler = ObligationCompiler()
    classification = ProblemClassifier().classify(RECURRENCE_QUERY)

    assert classification.primary_type == ProblemType.SYMBOLIC
    compiled = compiler.compile(RECURRENCE_QUERY, classification)

    assert compiled.mode == "symbolic_numeric"
    assert compiled.schema["closed_form_ok"] == "bool"
    assert compiled.schema["verify_0_10_ok"] == "bool"
    assert compiled.schema["threshold_n"] == "int"
    assert compiled.schema["ratio_limit"] == "expr"


def test_obligation_compiler_forces_schema_failure_on_missing_required_fields():
    compiler = ObligationCompiler()
    classification = ProblemClassifier().classify(RECURRENCE_QUERY)
    compiled = compiler.compile(RECURRENCE_QUERY, classification)

    payload = {
        "result": {
            "mode": "symbolic_numeric",
            "closed_form_ok": True,
            "verify_0_10_ok": True,
            "ratio_limit": "sqrt(5)/2 + 3/2",
        },
        "confirms_hypothesis": True,
    }
    assessment = compiler.evaluate(compiled, json.dumps(payload, sort_keys=True))

    assert assessment.schema_valid is False
    assert assessment.all_required_passed is False
    assert "threshold_n" in assessment.missing_or_null_fields


def test_obligation_compiler_rejects_wrong_symbolic_closed_form_even_if_reported_ok():
    compiler = ObligationCompiler()
    classification = ProblemClassifier().classify(RECURRENCE_QUERY)
    compiled = compiler.compile(RECURRENCE_QUERY, classification)

    payload = {
        "result": {
            "mode": "symbolic_numeric",
            "closed_form": "n + 1",
            "closed_form_ok": True,
            "verify_0_10_ok": True,
            "threshold_n": 15,
            "ratio_limit": "1",
        },
        "confirms_hypothesis": True,
    }
    assessment = compiler.evaluate(compiled, json.dumps(payload, sort_keys=True))

    assert assessment.schema_valid is True
    assert assessment.all_required_passed is False
    assert assessment.required_results["closed_form_ok"] is False
    assert assessment.required_results["verify_0_10_ok"] is False


def test_obligation_compiler_validates_probabilistic_obligations():
    compiler = ObligationCompiler()
    classification = ProblemClassifier().classify(CRYPTO_QUERY)
    compiled = compiler.compile(CRYPTO_QUERY, classification)

    payload = {
        "result": {
            "mode": "probabilistic_numeric",
            "minimum_rounds": 5,
            "independent_model_result": 4.0,
            "correlated_model_result": 5.0,
        },
        "confirms_hypothesis": True,
    }
    assessment = compiler.evaluate(compiled, json.dumps(payload, sort_keys=True))

    assert compiled.mode == "probabilistic_numeric"
    assert assessment.schema_valid is True
    assert assessment.all_required_passed is True


def test_obligation_compiler_validates_probabilistic_survival_obligations():
    compiler = ObligationCompiler()
    classification = ProblemClassifier().classify(SURVIVAL_QUERY)
    compiled = compiler.compile(SURVIVAL_QUERY, classification)

    payload = {
        "result": {
            "mode": "probabilistic_numeric",
            "template": "survival",
            "survival_probability": compiled.context["survival_probability"],
            "expected_hours": compiled.context["expected_hours"],
            "independent_model_survival": compiled.context["independent_model_survival"],
            "correlated_model_survival": compiled.context["correlated_model_survival"],
            "independent_model_expected_hours": compiled.context["independent_model_expected_hours"],
            "correlated_model_expected_hours": compiled.context["correlated_model_expected_hours"],
        },
        "confirms_hypothesis": True,
    }
    assessment = compiler.evaluate(compiled, json.dumps(payload, sort_keys=True))

    assert compiled.mode == "probabilistic_numeric"
    assert compiled.context["template"] == "survival"
    assert assessment.schema_valid is True
    assert assessment.all_required_passed is True


def test_obligation_compiler_validates_probabilistic_survival_combinatorial_obligations():
    compiler = ObligationCompiler()
    classification = ProblemClassifier().classify(SURVIVAL_COMBINATORIAL_QUERY)
    compiled = compiler.compile(SURVIVAL_COMBINATORIAL_QUERY, classification)

    payload = {
        "result": {
            "mode": "probabilistic_numeric",
            "template": "survival_combinatorial",
            "survival_probability": compiled.context["survival_probability"],
            "expected_hours": compiled.context["expected_hours"],
            "independent_model_survival": compiled.context["independent_model_survival"],
            "correlated_model_survival": compiled.context["correlated_model_survival"],
            "independent_model_expected_hours": compiled.context["independent_model_expected_hours"],
            "correlated_model_expected_hours": compiled.context["correlated_model_expected_hours"],
        },
        "confirms_hypothesis": True,
    }
    assessment = compiler.evaluate(compiled, json.dumps(payload, sort_keys=True))

    assert compiled.context["template"] == "survival_combinatorial"
    assert assessment.schema_valid is True
    assert assessment.all_required_passed is True


def test_obligation_compiler_handles_for_clause_recurrence_format():
    compiler = ObligationCompiler()
    classification = ProblemClassifier().classify(RECURRENCE_QUERY_FOR_CLAUSE)
    compiled = compiler.compile(RECURRENCE_QUERY_FOR_CLAUSE, classification)

    payload = {
        "result": {
            "mode": "symbolic_numeric",
            "closed_form": "n + 1",
            "closed_form_ok": True,
            "verify_0_10_ok": True,
            "threshold_n": 15,
            "ratio_limit": "1",
        },
        "confirms_hypothesis": True,
    }
    assessment = compiler.evaluate(compiled, json.dumps(payload, sort_keys=True))

    assert compiled.mode == "symbolic_numeric"
    assert assessment.all_required_passed is False
    assert assessment.required_results["closed_form_ok"] is False
