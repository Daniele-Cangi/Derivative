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


def test_repeated_consequences_remain_attached_to_each_condition():
    spec = RequirementCompiler().compile(
        "Build a Python CLI. If the file is empty, exit code 0 and output nothing. "
        "If the input is invalid, exit code 2 and output nothing."
    )
    for trigger, status in [("the file is empty", 0), ("the input is invalid", 2)]:
        obligations = [item for item in spec.conditional_obligations if item.trigger == trigger]
        assert {(item.observable_channel, item.expected_value) for item in obligations} == {
            ("exit_code", status), ("stdout", ""),
        }
        parents = {item.parent_requirement_id for item in obligations}
        assert len(parents) == 1
        assert any("and output nothing" in atom.text for atom in spec.requirement_atoms if atom.requirement_id in parents)


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


@pytest.mark.parametrize(
    "instruction",
    [
        "If N is negative, use its absolute value.",
        "Input is never modified.",
        "The range is limited to integers 0 through 9999999 inclusive.",
        "Otherwise, print the English representation.",
    ],
)
def test_precise_instructions_reach_acceptance_and_planned_evidence(instruction):
    spec = RequirementCompiler().compile("Build a Python library. " + instruction)
    atom = next(item for item in spec.requirement_atoms if item.text == instruction.rstrip("."))
    plan = _mapping_plan(spec)

    assert atom.category != "ambiguity"
    assert atom.strength == "hard"
    assert any(atom.requirement_id in item.requirement_ids for item in spec.acceptance_contract.criteria)
    assert plan.requirement_coverage[atom.requirement_id]["acceptance_criteria"]
    assert plan.requirement_coverage[atom.requirement_id]["tests"]


@pytest.mark.parametrize(
    ("requirement", "persistent", "scope"),
    [
        ("Build a Python service. Use persistent per-user rate limiting.", True, "per_user"),
        ("Build a Python service. Rate limiting must survive restart.", True, "per_user"),
        ("Build a Python service. Use distributed rate limiting across instances.", True, "distributed"),
        ("Build a Python service. Do not use persistent rate limiting. State must not survive restart.", False, "per_user"),
        ("Build a Python library that writes persistent records to SQLite.", False, "per_user"),
        ("Build a Python service with rate limiting and persistent storage for records.", False, "per_user"),
        ("Build a Python service. Use rate limiting and persistent storage for limiter counters.", True, "per_user"),
        ("Build a Python service. Use rate limiting and do not use persistent storage for limiter counters.", False, "per_user"),
        ("Build a Python service with rate limiting. The limiter must not survive restart.", False, "per_user"),
    ],
)
def test_rate_limit_quality_respects_persistence_scope_and_negation(requirement, persistent, scope):
    quality = RequirementCompiler().compile(requirement).quality_contract

    assert quality.rate_limit_persistent is persistent
    assert quality.rate_limit_scope == scope


@pytest.mark.parametrize("percent", [60, 80, 95])
def test_explicit_coverage_threshold_is_retained_and_requires_independent_measurement(percent, tmp_path):
    from dataclasses import asdict
    from core.forge.contracts import CodeArtifact
    from core.forge.validation.quality import QualityContractChecker

    spec = RequirementCompiler().compile(
        f"Build a Python service with test coverage at least {percent} percent."
    )
    assert spec.quality_contract.test_coverage_target == percent / 100
    artifact = CodeArtifact(
        artifact_id="diagnostic", plan_id="diagnostic",
        artifact_manifest={"quality_contract": asdict(spec.quality_contract)},
    )
    failures, evidence = QualityContractChecker().check({}, artifact, spec)
    assert failures
    assert evidence["checks"]["explicit_coverage_target_evidenced"] is False
