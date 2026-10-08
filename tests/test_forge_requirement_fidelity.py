"""Deterministic regressions for meaning lost before or during validation."""
import pytest

from core.forge.conditional_evidence import ConditionalEvidenceValidator
from core.forge.contracts import FeasiblePlan
from core.forge.execution import LocalProcessExecutor
from core.forge.planner_stage import PlannerStage
from core.forge.repair_support import behavioral_contract_seal
from core.forge.requirement_compiler import RequirementCompiler


def _mapping_plan(spec):
    planner = PlannerStage.__new__(PlannerStage)
    tests = planner._derive_required_tests(spec)
    return FeasiblePlan(
        plan_id=f"plan-{spec.build_id}",
        build_spec=spec,
        architecture_summary="Diagnostic requirement mapping.",
        required_tests=tests,
        requirement_coverage=planner._build_requirement_coverage(spec, [], tests),
        conditional_obligation_coverage=planner._build_conditional_obligation_coverage(
            spec, tests,
        ),
    )


@pytest.mark.parametrize(
    "expected",
    ["a  b", "a\tb", "a\nb", "a, returns b; writes c", "it's  exact", "exit code 9; raise ValueError"],
)
def test_quoted_output_preserves_value_and_does_not_create_fake_obligations(expected):
    requirement = (
        f"Build a Python CLI. If the file is empty, output exactly {expected!r} "
        "to stdout with exit code 0."
    )
    spec = RequirementCompiler().compile(requirement)

    assert {item.observable_channel for item in spec.conditional_obligations} == {
        "stdout", "exit_code",
    }
    output = next(item for item in spec.conditional_obligations if item.observable_channel == "stdout")
    status = next(item for item in spec.conditional_obligations if item.observable_channel == "exit_code")
    assert output.expected_value == expected
    assert status.expected_value == 0
    assert expected.__repr__() in output.source_fragment


def test_significant_output_whitespace_changes_build_and_behavioral_identity():
    compiler = RequirementCompiler()
    two_spaces = compiler.compile(
        "Build a Python CLI. If the file is empty, output exactly 'a  b' to stdout."
    )
    one_space = compiler.compile(
        "Build a Python CLI. If the file is empty, output exactly 'a b' to stdout."
    )

    assert two_spaces.build_id != one_space.build_id
    assert behavioral_contract_seal(_mapping_plan(two_spaces)) != behavioral_contract_seal(
        _mapping_plan(one_space)
    )


def test_case_distinct_quoted_outputs_are_not_deduplicated():
    spec = RequirementCompiler().compile(
        "Build a Python CLI. Output exactly 'A' to stdout. Output exactly 'a' to stdout."
    )

    texts = [atom.text for atom in spec.requirement_atoms]
    assert "Output exactly 'A' to stdout" in texts
    assert "Output exactly 'a' to stdout" in texts


def test_prose_whitespace_and_apostrophes_remain_normalized():
    spec = RequirementCompiler().compile(
        "  Build  a Python library.\n The function must not close the caller's stream.  "
    )

    assert spec.normalized_requirement == (
        "Build a Python library. The function must not close the caller's stream."
    )


def test_conditional_negative_retains_antecedent_instead_of_becoming_unconditional():
    spec = RequirementCompiler().compile(
        "Build a Python library. If the stream is borrowed, the function must not close it."
    )
    obligation = next(item for item in spec.conditional_obligations if item.observable_channel == "resource_state")

    assert obligation.trigger == "the stream is borrowed"
    assert obligation.precondition == {"kind": "textual_precondition", "text": "the stream is borrowed"}
    assert obligation.polarity == "negative"
    assert obligation.comparison_relation == "not_equals"
    assert obligation.expected_value == "closed"
    assert obligation.obligation_id in _mapping_plan(spec).conditional_obligation_coverage


def test_unless_is_rejected_explicitly_when_negated_trigger_cannot_be_probed(tmp_path):
    spec = RequirementCompiler().compile(
        "Build a Python CLI. Unless the file is empty, output is empty with exit code 0."
    )

    assert spec.conditional_obligations == []
    assert spec.conditional_normalization_issues
    assert all(item.hard for item in spec.conditional_normalization_issues)
    validator = ConditionalEvidenceValidator(LocalProcessExecutor(), timeout_seconds=10)
    failures, signatures, _ = validator.validate(spec, _mapping_plan(spec), {}, tmp_path)
    assert failures
    assert "uncompiled_hard_conditional" in signatures


def test_unconditional_stream_negative_keeps_resource_contract():
    spec = RequirementCompiler().compile(
        "Build a Python library. The function must not close the stream."
    )
    obligation = next(item for item in spec.conditional_obligations if item.observable_channel == "resource_state")

    assert obligation.trigger == "always"
    assert obligation.precondition == {"kind": "always"}
    assert obligation.expected_value == "closed"
