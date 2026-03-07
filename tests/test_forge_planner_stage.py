from dataclasses import is_dataclass
from pathlib import Path

from core.forge.contracts import (
    AcceptanceContract,
    ArtifactTargetType,
    BuildSpec,
    FeasiblePlan,
    InfeasibilityCertificate,
    ObligationContract,
    PlanFile,
    PlannerStageOutput,
    ValidationStrategy,
)
from core.forge.planner_stage import PlannerStage
from core.forge.requirement_compiler import RequirementCompiler


FEASIBLE_REQUIREMENT = (
    "Build a Python CLI that reads a CSV of contracts, extracts expiration dates, "
    "flags contracts expiring in less than 90 days, writes a summary CSV, and includes tests."
)

CONTRADICTORY_REQUIREMENT = (
    "Design a network on exactly 4 nodes such that every pair of nodes is directly connected, "
    "the network diameter is strictly greater than 2, vertex connectivity is at least 3, "
    "and the total number of edges does not exceed 3."
)


def _build_planner(tmp_path: Path) -> PlannerStage:
    return PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "forge_audit.json"),
        memory_file=str(tmp_path / "forge_memory.json"),
        gene_pool_file=str(tmp_path / "forge_gene_pool.json"),
    )


def test_requirement_compiler_and_planner_return_feasible_plan(tmp_path):
    compiler = RequirementCompiler()
    build_spec = compiler.compile(FEASIBLE_REQUIREMENT)
    planner = _build_planner(tmp_path)

    output = planner.plan(build_spec)

    assert isinstance(output, FeasiblePlan)
    assert output.build_spec.build_id == build_spec.build_id
    assert output.build_spec.target_artifact_type == ArtifactTargetType.CLI
    assert output.build_spec.obligation_contract is not None
    assert output.build_spec.obligation_contract.mode == "software_build"
    assert output.build_spec.obligation_contract.schema
    assert output.build_spec.obligation_contract.required_fields
    assert all("flags contracts expiring in less than 90 days" != c.lower() for c in output.build_spec.non_functional_constraints)
    assert output.file_tree_plan
    assert any(item.path == "src/cli.py" for item in output.file_tree_plan)
    assert "Constraint Lattice Compiler" not in output.architecture_summary
    assert "csv input loader" in output.architecture_summary.lower()
    assert output.interfaces
    assert any(interface.name == "flag_expiring_contracts" for interface in output.interfaces)
    assert output.required_tests
    assert any(test.test_name == "test_reads_contracts_csv" for test in output.required_tests)
    assert all(not test.test_name.startswith("test_acceptance_") for test in output.required_tests)
    assert output.acceptance_criterion_ids
    assert output.obligation_mode == "software_build"
    assert output.required_obligations
    assert output.validation_strategy.layer1_checks
    assert output.validation_strategy.layer2_checks
    assert output.validation_strategy.layer3_checks


def test_requirement_compiler_and_planner_return_infeasibility_certificate(tmp_path):
    compiler = RequirementCompiler()
    build_spec = compiler.compile(CONTRADICTORY_REQUIREMENT)
    planner = _build_planner(tmp_path)

    output = planner.plan(build_spec)

    assert isinstance(output, InfeasibilityCertificate)
    assert output.contradictions
    merged = " ".join(output.contradictions).lower()
    assert "diameter" in merged or "edge" in merged
    assert output.execution_evidence.get("result_mode") == "infeasible"
    assert output.execution_evidence.get("is_satisfiable") is False
    assert output.terminal_status == "infeasible_proven"
    assert output.execution_evidence.get("terminal_status") == "infeasible_proven"


def test_infeasibility_is_not_reported_as_generic_failure_or_not_converged(tmp_path):
    compiler = RequirementCompiler()
    build_spec = compiler.compile(CONTRADICTORY_REQUIREMENT)
    planner = _build_planner(tmp_path)

    output = planner.plan(build_spec)

    assert isinstance(output, InfeasibilityCertificate)
    proof_lower = output.proof_summary.lower()
    assert "not converged" not in proof_lower
    assert "generic" not in proof_lower
    assert "unsatisfiable" in proof_lower or "infeasible" in proof_lower
    assert output.terminal_status == "infeasible_proven"


def test_planner_outputs_are_typed_and_contract_compatible(tmp_path):
    compiler = RequirementCompiler()
    feasible_spec = compiler.compile(FEASIBLE_REQUIREMENT)
    contradictory_spec = compiler.compile(CONTRADICTORY_REQUIREMENT)
    planner = _build_planner(tmp_path)

    feasible_output: PlannerStageOutput = planner.plan(feasible_spec)
    infeasible_output: PlannerStageOutput = planner.plan(contradictory_spec)

    assert isinstance(feasible_output, FeasiblePlan)
    assert isinstance(infeasible_output, InfeasibilityCertificate)
    assert is_dataclass(feasible_output)
    assert is_dataclass(infeasible_output)
    assert isinstance(feasible_output.build_spec, BuildSpec)
    assert isinstance(feasible_output.build_spec.acceptance_contract, AcceptanceContract)
    assert isinstance(feasible_output.build_spec.obligation_contract, ObligationContract)
    assert isinstance(feasible_output.file_tree_plan[0], PlanFile)
    assert isinstance(feasible_output.validation_strategy, ValidationStrategy)
    assert feasible_output.acceptance_criterion_ids
    assert isinstance(feasible_output.required_tests[0].acceptance_criterion_ids, list)
    assert isinstance(feasible_output.required_tests[0].obligation_fields, list)
    assert infeasible_output.terminal_status == "infeasible_proven"


def test_forge_package_layout_uses_dunder_init():
    assert Path("core/forge/__init__.py").exists()
    assert not Path("core/forge/init.py").exists()
