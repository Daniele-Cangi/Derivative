import copy
import os

import pytest

from core.forge.coder_stage import CoderStage
from core.forge.contracts import FeasiblePlan
from core.forge.planner_stage import PlannerStage
from core.forge.requirement_compiler import RequirementCompiler
from core.forge.validator_stage import ValidatorStage
from core.forge.repair_support import behavioral_contract_seal, behavioral_generation_contracts
from core.forge.expiration_contract import expiration_horizon_days
from core.forge.execution import DockerSandboxExecutor
from core.forge.packaging_stage import PackagingRefusedError, PackagingStage


def _numeric_plan(horizon, extra="", expiration_clause=None):
    clause = expiration_clause or f"flags contracts expiring in less than {horizon} days"
    spec = RequirementCompiler().compile(
        "Build a Python CLI that reads a CSV of contracts, extracts expiration dates, "
        f"{clause}, writes a summary CSV, and includes tests. {extra}"
    )
    planner = PlannerStage.__new__(PlannerStage)
    blueprint = planner._derive_implementation_blueprint(spec)
    files = planner._derive_file_tree_plan(spec, blueprint)
    tests = planner._derive_required_tests(spec)
    plan = FeasiblePlan(
        plan_id=f"plan-{spec.build_id}", build_spec=spec,
        architecture_summary="CSV contract expiration.", quality_contract=spec.quality_contract,
        implementation_blueprint=blueprint, file_tree_plan=files,
        interfaces=planner._derive_interfaces(spec), required_tests=tests,
        required_obligations=list(spec.obligation_contract.required_fields),
        acceptance_criterion_ids=[criterion.criterion_id for criterion in spec.acceptance_contract.criteria],
        requirement_coverage=planner._build_requirement_coverage(spec, files, tests),
        conditional_obligation_coverage=planner._build_conditional_obligation_coverage(spec, tests),
        obligation_mode=spec.obligation_contract.mode,
        packaging_target="python_cli_package",
    )
    return spec, plan


@pytest.mark.parametrize("horizon", [0, 30, 45, 120])
def test_expiration_bound_is_compiled_implemented_and_independently_probed(horizon):
    spec, plan = _numeric_plan(horizon)
    contract = spec.obligation_contract.context["expiration_horizon"]
    assert contract["threshold_days"] == horizon
    assert contract["comparison_relation"] == "less_than"
    assert contract["requirement_ids"]
    artifact = CoderStage().generate(plan)
    validation = ValidatorStage().validate(artifact, plan, spec)

    assert validation.passed, validation.failures
    checks = validation.layer2_result.evidence["numeric_contract_checks"]["contracts"]
    assert checks[0]["passed"]
    assert checks[0]["observed"]["function_flags"] == checks[0]["expected_flags"]
    assert checks[0]["observed"]["cli_flags"] == checks[0]["expected_flags"]


@pytest.mark.parametrize("mutation", ["function_default", "cli_default", "inclusive_comparison", "spurious_lower_bound"])
def test_validator_owned_bound_probe_rejects_wrong_default_or_relation(mutation):
    spec, plan = _numeric_plan(30)
    artifact = CoderStage().generate(plan)
    mutant = copy.deepcopy(artifact)
    for generated in mutant.files:
        if mutation == "function_default" and generated.path == "src/expiration_rules.py":
            generated.content = generated.content.replace("horizon_days: int = 30", "horizon_days: int = 90")
        elif mutation == "cli_default" and generated.path == "src/cli.py":
            generated.content = generated.content.replace("default=30", "default=90")
        elif mutation == "inclusive_comparison" and generated.path == "src/expiration_rules.py":
            generated.content = generated.content.replace("days < horizon_days", "days <= horizon_days")
        elif mutation == "spurious_lower_bound" and generated.path == "src/expiration_rules.py":
            generated.content = generated.content.replace("days < horizon_days", "10 < days < horizon_days")
    assert any(a.content != b.content for a, b in zip(artifact.files, mutant.files))

    validation = ValidatorStage().validate(mutant, plan, spec)
    assert validation.passed is False
    assert "numeric_contract_mismatch" in validation.failure_signatures
    if mutation == "spurious_lower_bound":
        assert "test_execution_failure" not in validation.failure_signatures
    checks = validation.layer2_result.evidence["numeric_contract_checks"]["contracts"]
    assert checks[0]["passed"] is False


def test_numeric_bound_is_exposed_and_sealed_for_untrusted_generation():
    spec, plan = _numeric_plan(30)
    original = behavioral_contract_seal(plan)
    assert behavioral_generation_contracts(plan)["numeric_constraints"][0]["threshold_days"] == 30
    spec.obligation_contract.context["expiration_horizon"]["threshold_days"] = 90
    assert behavioral_contract_seal(plan)["sha256"] != original["sha256"]


def test_conflicting_numeric_bounds_fail_closed_instead_of_selecting_a_default():
    spec = RequirementCompiler().compile(
        "Build a Python CLI that reads CSV and flags contracts expiring in less than 30 days. "
        "Flags contracts expiring in less than 90 days."
    )
    assert spec.obligation_contract.context["expiration_horizon"]["threshold_days"] is None
    assert any("Materially unspecified expiration horizon" in flag for flag in spec.ambiguity_flags)


@pytest.mark.parametrize("requirement", [
    "Build a Python library that flags contracts expiring in less than 30 days.",
    "Build a Python library that reads CSV and flags contracts expiring in less than 30 days.",
    "Build a Python CLI that reads JSON and flags contracts expiring in less than 30 days.",
])
def test_csv_numeric_probe_does_not_apply_to_other_domains(requirement):
    spec = RequirementCompiler().compile(requirement)
    assert "expiration_horizon" not in spec.obligation_contract.context


def test_conflicting_horizons_reach_fail_closed_validation_without_planning_exception():
    spec, plan = _numeric_plan(30, "Flags contracts expiring in less than 90 days.")
    artifact = CoderStage().generate(plan)
    validation = ValidatorStage().validate(artifact, plan, spec)
    assert not validation.passed
    assert "underspecified_requirement" in validation.failure_signatures
    assert "numeric_contract_unproven" in validation.failure_signatures


@pytest.mark.parametrize("clause", [
    "flags contracts that expire in less than 30 days",
    "flags a contract expiring in less than 30 days",
    "flags contracts which expire in less than 30 days",
])
def test_equivalent_expiration_clauses_bind_the_numeric_default(clause):
    spec, plan = _numeric_plan(30, expiration_clause=clause)
    contract = spec.obligation_contract.context["expiration_horizon"]
    assert contract["threshold_days"] == 30
    assert contract["requirement_ids"]
    artifact = CoderStage().generate(plan)
    validation = ValidatorStage().validate(artifact, plan, spec)
    assert validation.passed, validation.failures
    assert validation.layer2_result.evidence["numeric_contract_checks"]["contracts"][0]["passed"]


@pytest.mark.parametrize("reordered_channel", ["function", "cli"])
def test_numeric_probe_accepts_correct_flags_regardless_of_row_order(reordered_channel):
    spec, plan = _numeric_plan(120)
    artifact = CoderStage().generate(plan)
    for generated in artifact.files:
        if reordered_channel == "function" and generated.path == "src/expiration_rules.py":
            assert "return flagged" in generated.content
            generated.content = generated.content.replace(
                "return flagged", "return sorted(flagged, key=lambda row: row['contract_id'])",
            )
        elif reordered_channel == "cli" and generated.path == "src/cli.py":
            assert "write_summary_csv(flagged, args.output_csv)" in generated.content
            generated.content = generated.content.replace(
                "write_summary_csv(flagged, args.output_csv)",
                "write_summary_csv(sorted(flagged, key=lambda row: row['contract_id']), args.output_csv)",
            )
    validation = ValidatorStage().validate(artifact, plan, spec)
    assert validation.passed, validation.failures


@pytest.mark.parametrize("returned_rows", ["flagged[:-1]", "flagged + [flagged[0]]"])
@pytest.mark.parametrize("channel", ["function", "cli"])
def test_order_independent_numeric_probe_still_rejects_missing_or_duplicate_rows(returned_rows, channel):
    spec, plan = _numeric_plan(120)
    artifact = CoderStage().generate(plan)
    for generated in artifact.files:
        if channel == "function" and generated.path == "src/expiration_rules.py":
            generated.content = generated.content.replace("return flagged", f"return {returned_rows}")
        elif channel == "cli" and generated.path == "src/cli.py":
            generated.content = generated.content.replace(
                "write_summary_csv(flagged, args.output_csv)",
                f"write_summary_csv({returned_rows}, args.output_csv)",
            )
    validation = ValidatorStage().validate(artifact, plan, spec)
    assert not validation.passed
    assert "numeric_contract_mismatch" in validation.failure_signatures


UNSUPPORTED_EXPIRATION_CLAUSES = [
    "flags contracts expiring within 30 days",
    "flags contracts that expire in under 30 days",
    "flags contracts whose expiration is fewer than 30 days away",
    "identifies contracts expiring in less than 30 days",
    "reports contracts expiring in less than 30 days",
    "flags any contract expiring in less than 30 days",
    "flags contracts expiring at most 30 days",
    "flags contracts expiring no more than 30 days",
    "flags contracts expiring inside 30 days",
    "flags contracts expiring 30 days from now",
    "flags contracts expiring in less than 30.5 days",
    "sets the expiration horizon to 30 days",
    "flags contracts within 30 days of expiration",
    "flags contracts no more than 30 days before expiration",
    "flags contracts whose expiration does not exceed 30 days",
    "flags contracts expiring on or before 30 days from now",
    "flags contracts expiring in less than or equal to 30 days",
    "flags contracts expiring in less than 30 days and reports contracts expiring within 90 days",
    "flags contracts whose expiration must not exceed 30 days",
    "flags contracts whose expiration shall not exceed 30 days",
    "flags contracts whose expiration may not exceed 30 days",
    "flags contracts whose expiration dates do not exceed 30 days",
]


@pytest.mark.parametrize("clause", UNSUPPORTED_EXPIRATION_CLAUSES)
def test_uninterpreted_numeric_expiration_is_unresolved_not_defaulted(clause):
    spec, plan = _numeric_plan(30, expiration_clause=clause)
    contract = spec.obligation_contract.context.get("expiration_horizon")
    assert contract is not None
    assert contract["threshold_days"] is None
    assert contract["comparison_relation"] is None
    assert contract["unresolved_reason"] == "unsupported_numeric_expiration_wording"
    assert contract["requirement_ids"] == [atom.requirement_id for atom in spec.requirement_atoms if "30" in atom.text]
    assert expiration_horizon_days(spec) is None
    assert any("Materially unspecified expiration horizon: unsupported numeric wording" in flag
               for flag in spec.ambiguity_flags)
    assert behavioral_generation_contracts(plan)["numeric_constraints"] == [contract]


RESIDUAL_EXPIRATION_CLAUSES = [
    "flags contracts expiring, after the reader has normalised the date column, in less than 30 days",
    "flags contracts expiring after the reader has normalised the date column using the configured locale "
    "with the documented validation settings for the current input format in less than 30 days",
    "flags contracts expiring in less than thirty days",
    "flags contracts expiring in less than 1 month",
    "flags contracts expiring in less than 2 weeks",
]


@pytest.mark.parametrize("clause", RESIDUAL_EXPIRATION_CLAUSES + [
    "flags contracts expiring within twenty-one days",
    "flags contracts expiring within one hundred days",
    "flags contracts within two weeks of expiration",
])
def test_residual_expiration_policy_is_unresolved_with_source_traceability(clause):
    spec, plan = _numeric_plan(30, expiration_clause=clause)
    contract = spec.obligation_contract.context.get("expiration_horizon")
    assert contract is not None
    assert contract["threshold_days"] is None
    assert contract["comparison_relation"] is None
    assert contract["unresolved_reason"] == "unsupported_numeric_expiration_wording"
    assert contract["requirement_ids"] == [
        atom.requirement_id for atom in spec.requirement_atoms if clause in atom.text
    ]
    assert contract["requirement_ids"]
    assert expiration_horizon_days(spec) is None
    assert behavioral_generation_contracts(plan)["numeric_constraints"] == [contract]


@pytest.mark.parametrize("clause", UNSUPPORTED_EXPIRATION_CLAUSES[:4] + UNSUPPORTED_EXPIRATION_CLAUSES[17:20]
                         + RESIDUAL_EXPIRATION_CLAUSES)
@pytest.mark.parametrize("backend", [
    "local",
    pytest.param("docker", marks=pytest.mark.skipif(
        os.environ.get("FORGE_RUN_DOCKER_TESTS") != "1",
        reason="real Docker sandbox tests are CI-gated",
    )),
])
def test_issue32_uninterpreted_horizon_fails_closed_through_generation_validation(clause, backend, tmp_path):
    spec, plan = _numeric_plan(30, expiration_clause=clause)
    artifact = CoderStage().generate(plan)
    contents = {generated.path: generated.content for generated in artifact.files}
    assert "horizon_days: int = None" in contents["src/expiration_rules.py"]
    assert "default=None" in contents["src/cli.py"]
    validator = (
        ValidatorStage(executor=DockerSandboxExecutor(), require_isolation=True)
        if backend == "docker" else ValidatorStage()
    )
    validation = validator.validate(artifact, plan, spec)
    assert not validation.passed
    assert "underspecified_requirement" in validation.failure_signatures
    assert "numeric_contract_unproven" in validation.failure_signatures
    checks = validation.layer2_result.evidence["numeric_contract_checks"]["contracts"]
    assert len(checks) == 1
    assert checks[0]["passed"] is False
    assert checks[0]["reason"] == "probe_unavailable"
    assert validation.evidence["executed_tests"]["backend"] == backend
    if backend == "docker":
        assert validation.evidence["execution_policy"]["isolated"] is True
    with pytest.raises(PackagingRefusedError, match="passed ValidationArtifact"):
        PackagingStage(str(tmp_path / "packages")).package(spec, plan, artifact, validation)
    assert not (tmp_path / "packages").exists()


@pytest.mark.parametrize("unknown", [
    "Flags contracts expiring within 30 days.",
    "Flags contracts expiring within 90 days.",
    "Flags contracts expiring within thirty days.",
    "Flags contracts expiring within two weeks.",
])
def test_supported_bound_cannot_hide_an_uninterpreted_numeric_expiration(unknown):
    spec, _ = _numeric_plan(30, extra=unknown)
    contract = spec.obligation_contract.context["expiration_horizon"]
    assert contract["threshold_days"] is None
    assert contract["comparison_relation"] is None
    assert len(contract["requirement_ids"]) == 2
    assert expiration_horizon_days(spec) is None


@pytest.mark.parametrize("extra", [
    "Do not flag contracts expiring within 30 days.",
    "Never flag contracts expiring within 30 days.",
    "Do not flag contracts within 30 days of expiration.",
    "Should flag contracts expiring within 30 days.",
    'Print the literal "flags contracts expiring within 30 days".',
    'Print the literal "flags contracts expiring in less than 30 days".',
    'Label expiration records "within 30 days".',
    "Retain logs for 30 days.",
    "Extract expiration dates and retain logs for 30 days.",
    "Extract expiration dates, retain logs for 30 days.",
    "Expiration dates must not be interpreted as within 30 days.",
    "Extract expiration dates, retain logs for two weeks.",
    "Extract expiration dates, after reading the CSV, retain logs for 30 days.",
    "Extract expiration dates and retain logs for one month.",
    "Extract expiration dates, after retaining logs for 30 days, write a summary CSV.",
    'Print the literal "flags contracts expiring within thirty days".',
    "Should flag contracts expiring within two weeks.",
    "Do not flag contracts expiring within one month.",
    "Never flag contracts expiring, after reading the CSV, within thirty days.",
])
def test_unrelated_soft_negated_or_quoted_duration_is_not_an_expiration_bound(extra):
    spec = RequirementCompiler().compile(
        "Build a Python CLI that reads CSV and writes a summary CSV. " + extra
    )
    assert "expiration_horizon" not in spec.obligation_contract.context
    assert expiration_horizon_days(spec) == 90
    assert not any("Materially unspecified expiration horizon" in flag for flag in spec.ambiguity_flags)


def test_no_numeric_expiration_obligation_retains_legacy_default():
    spec, _ = _numeric_plan(90, expiration_clause="flags expiring contracts")
    assert "expiration_horizon" not in spec.obligation_contract.context
    assert expiration_horizon_days(spec) == 90


@pytest.mark.parametrize("requirement", [
    "Build a Python library that reads CSV and flags contracts expiring within 30 days.",
    "Build a Python CLI that reads JSON and flags contracts expiring within 30 days.",
])
def test_uninterpreted_expiration_detection_remains_scoped_to_csv_cli(requirement):
    spec = RequirementCompiler().compile(requirement)
    assert "expiration_horizon" not in spec.obligation_contract.context


@pytest.mark.parametrize("tail", [
    "and reports contracts expiring within 90 days",
    "and reports contracts expiring within 30 days",
    "but reports contracts expiring within 90 days",
    "and reports contracts expiring within thirty days",
    "and reports contracts expiring within two weeks",
])
def test_recognized_prefix_does_not_hide_a_second_policy_in_the_same_atom(tail):
    spec, _ = _numeric_plan(30, expiration_clause="flags contracts expiring in less than 30 days " + tail)
    numeric_atoms = [atom for atom in spec.requirement_atoms if "30 days" in atom.text]
    assert len(numeric_atoms) == 1
    assert tail in numeric_atoms[0].text
    contract = spec.obligation_contract.context["expiration_horizon"]
    assert contract["threshold_days"] is None
    assert contract["comparison_relation"] is None
    assert contract["requirement_ids"] == [numeric_atoms[0].requirement_id]


@pytest.mark.parametrize("tail", [
    'and prints "contracts expiring within 90 days"',
    "and retains logs for 90 days",
    "and does not report contracts expiring within 90 days",
])
def test_recognized_prefix_does_not_borrow_unrelated_literal_or_rejected_policies(tail):
    spec, _ = _numeric_plan(30, expiration_clause="flags contracts expiring in less than 30 days " + tail)
    contract = spec.obligation_contract.context["expiration_horizon"]
    assert contract["threshold_days"] == 30
    assert contract["comparison_relation"] == "less_than"
    assert "unresolved_reason" not in contract


@pytest.mark.parametrize("modal", ["must", "shall", "may", "does"])
def test_outer_rejection_is_not_masked_with_a_prohibitive_comparison(modal):
    spec = RequirementCompiler().compile(
        "Build a Python CLI that reads CSV and writes a summary CSV. "
        f"Do not flag contracts whose expiration {modal} not exceed 30 days."
    )
    assert "expiration_horizon" not in spec.obligation_contract.context
