import copy

import pytest

from core.forge.coder_stage import CoderStage
from core.forge.contracts import FeasiblePlan
from core.forge.planner_stage import PlannerStage
from core.forge.requirement_compiler import RequirementCompiler
from core.forge.validator_stage import ValidatorStage
from core.forge.repair_support import behavioral_contract_seal, behavioral_generation_contracts


def _numeric_plan(horizon, extra=""):
    spec = RequirementCompiler().compile(
        "Build a Python CLI that reads a CSV of contracts, extracts expiration dates, "
        f"flags contracts expiring in less than {horizon} days, writes a summary CSV, and includes tests. {extra}"
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
    assert checks[0]["observed"]["function_flags"] == ["True", "False", "False"]
    assert checks[0]["observed"]["cli_flags"] == ["True", "False", "False"]


@pytest.mark.parametrize("mutation", ["function_default", "cli_default", "inclusive_comparison"])
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
    assert any(a.content != b.content for a, b in zip(artifact.files, mutant.files))

    validation = ValidatorStage().validate(mutant, plan, spec)
    assert validation.passed is False
    assert "numeric_contract_mismatch" in validation.failure_signatures
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
