from dataclasses import is_dataclass
from pathlib import Path

from core.execution_loop import ExecutionCycle, ExecutionResult
from core.forge.contracts import (
    AcceptanceContract,
    ArtifactTargetType,
    BuildSpec,
    CapabilitySpec,
    FeasiblePlan,
    ImplementationBlueprint,
    InfeasibilityCertificate,
    ObligationContract,
    PlanFile,
    PlannerStageOutput,
    ValidationStrategy,
)
from core.kernel import ReasoningResult
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

PIPELINE_REQUIREMENT = (
    "Build a production-grade Python data pipeline that reads CSV files from a watched directory, "
    "validates each row against a configurable schema, persists valid records to SQLite with full audit trail, "
    "rejects and quarantines invalid rows with structured error logging, and exposes a REST health endpoint "
    "showing pipeline statistics."
)

PRODUCTION_SERVICE_REQUIREMENT = (
    "Build a production-grade Python REST microservice with hashed API keys using bcrypt, "
    "persistent per-user rate limiting that survives restarts, a full audit trail of all requests, "
    "structured JSON logging, and integration tests."
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


def test_service_plan_contains_typed_capability_blueprint(tmp_path):
    build_spec = RequirementCompiler().compile(PRODUCTION_SERVICE_REQUIREMENT)
    output = _build_planner(tmp_path).plan(build_spec)

    assert isinstance(output, FeasiblePlan)
    assert isinstance(output.implementation_blueprint, ImplementationBlueprint)
    assert output.implementation_blueprint.target_artifact_type == ArtifactTargetType.SERVICE
    assert output.implementation_blueprint.entrypoint_path == "src/service.py"
    assert all(isinstance(capability, CapabilitySpec) for capability in output.implementation_blueprint.capabilities)

    capability_ids = {
        capability.capability_id for capability in output.implementation_blueprint.capabilities
    }
    assert capability_ids == {
        "cap_service_api",
        "cap_domain",
        "cap_storage",
        "cap_auth",
        "cap_rate_limit",
        "cap_audit",
        "cap_observability",
    }
    assert all(
        dependency in capability_ids
        for capability in output.implementation_blueprint.capabilities
        for dependency in capability.dependencies
    )
    capabilities_by_id = {
        capability.capability_id: capability
        for capability in output.implementation_blueprint.capabilities
    }
    assert "R002" in capabilities_by_id["cap_storage"].requirement_ids
    assert "R003" in capabilities_by_id["cap_audit"].requirement_ids
    assert "R004" in capabilities_by_id["cap_observability"].requirement_ids
    assert "R002" not in capabilities_by_id["cap_service_api"].requirement_ids
    assert "R003" not in capabilities_by_id["cap_rate_limit"].requirement_ids
    planned_paths = {plan_file.path for plan_file in output.file_tree_plan}
    assert {
        capability.module_path for capability in output.implementation_blueprint.capabilities
    }.issubset(planned_paths)


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


def test_planner_surfaces_persistence_warnings_in_evidence_and_plan_notes():
    class _StubSubstrate:
        def decompose(self, problem: str):
            return ["stub_framing"]

    class _StubKernel:
        def synthesize(self, problem: str, lenses, design_context=None, audit=None):
            execution_result = ExecutionResult(
                conclusion="synthetic execution result",
                cycles_used=1,
                converged=True,
                history=[
                    ExecutionCycle(
                        cycle=1,
                        hypothesis="h",
                        code="print(1)",
                        output='{"result":{"mode":"generic","is_satisfiable":true}}',
                        delta=0.0,
                        converged=True,
                    )
                ],
                final_code="print(1)",
                final_output='{"result":{"mode":"generic","is_satisfiable":true}}',
                final_prediction='{"expectations":{"unique_tag_count":1}}',
                final_residual=0.0,
            )
            return ReasoningResult(
                conclusion="kernel",
                reasoning_chain=[],
                violated_constraints=[],
                epistemic_confidence=0.9,
                lens_contributions={"stub": 1.0},
                execution_result=execution_result,
            )

    class _StubMemory:
        def retrieve_design_context(self, problem: str, top_k: int = 3):
            return []

        def record(self, result, problem: str):
            raise OSError("memory write failed")

    class _StubGenePool:
        def record_execution(self, result, problem: str):
            raise OSError("gene pool write failed")

    compiler = RequirementCompiler()
    build_spec = compiler.compile(FEASIBLE_REQUIREMENT)
    planner = PlannerStage(
        execution_mode="local-only",
        substrate=_StubSubstrate(),
        kernel=_StubKernel(),
        memory=_StubMemory(),
        gene_pool=_StubGenePool(),
    )

    output = planner.plan(build_spec)

    assert isinstance(output, FeasiblePlan)
    assert any("Persistence warning: memory_persist_failed:" in note for note in output.implementation_notes)
    assert any("Persistence warning: gene_pool_persist_failed:" in note for note in output.implementation_notes)


def test_planner_surfaces_persistence_warnings_in_infeasibility_evidence():
    class _StubSubstrate:
        def decompose(self, problem: str):
            return ["stub_framing"]

    class _StubKernel:
        def synthesize(self, problem: str, lenses, design_context=None, audit=None):
            execution_result = ExecutionResult(
                conclusion="infeasible execution result",
                cycles_used=1,
                converged=False,
                history=[
                    ExecutionCycle(
                        cycle=1,
                        hypothesis="h",
                        code="print(1)",
                        output='{"result":{"mode":"infeasible","is_satisfiable":false,"contradictions":["c"]}}',
                        delta=1.0,
                        converged=False,
                    )
                ],
                final_code="print(1)",
                final_output='{"result":{"mode":"infeasible","is_satisfiable":false,"contradictions":["c"]}}',
                final_prediction='{"expectations":{"constraint_count":1}}',
                final_residual=1.0,
            )
            return ReasoningResult(
                conclusion="kernel",
                reasoning_chain=[],
                violated_constraints=[],
                epistemic_confidence=0.2,
                lens_contributions={"stub": 1.0},
                execution_result=execution_result,
            )

    class _StubMemory:
        def retrieve_design_context(self, problem: str, top_k: int = 3):
            return []

        def record(self, result, problem: str):
            raise OSError("memory write failed")

    class _StubGenePool:
        def record_execution(self, result, problem: str):
            raise OSError("gene pool write failed")

    compiler = RequirementCompiler()
    build_spec = compiler.compile(CONTRADICTORY_REQUIREMENT)
    planner = PlannerStage(
        execution_mode="local-only",
        substrate=_StubSubstrate(),
        kernel=_StubKernel(),
        memory=_StubMemory(),
        gene_pool=_StubGenePool(),
    )

    output = planner.plan(build_spec)

    assert isinstance(output, InfeasibilityCertificate)
    warnings = output.execution_evidence.get("persistence_warnings", [])
    assert any("memory_persist_failed:" in warning for warning in warnings)
    assert any("gene_pool_persist_failed:" in warning for warning in warnings)


def test_planner_surfaces_execution_warnings_in_plan_notes():
    class _StubSubstrate:
        def decompose(self, problem: str):
            return ["stub_framing"]

    class _StubKernel:
        def synthesize(self, problem: str, lenses, design_context=None, audit=None):
            execution_result = ExecutionResult(
                conclusion="synthetic execution result",
                cycles_used=1,
                converged=True,
                history=[
                    ExecutionCycle(
                        cycle=1,
                        hypothesis="h",
                        code="print(1)",
                        output='{"result":{"mode":"generic","is_satisfiable":true}}',
                        delta=0.0,
                        converged=True,
                    )
                ],
                final_code="print(1)",
                final_output='{"result":{"mode":"generic","is_satisfiable":true}}',
                final_prediction='{"expectations":{"unique_tag_count":1}}',
                final_residual=0.0,
                warnings=["audit_log_failed:cycle=1:disk is read-only"],
            )
            return ReasoningResult(
                conclusion="kernel",
                reasoning_chain=[],
                violated_constraints=[],
                epistemic_confidence=0.9,
                lens_contributions={"stub": 1.0},
                execution_result=execution_result,
            )

    class _StubMemory:
        def retrieve_design_context(self, problem: str, top_k: int = 3):
            return []

        def record(self, result, problem: str):
            return None

    class _StubGenePool:
        def record_execution(self, result, problem: str):
            return []

    compiler = RequirementCompiler()
    build_spec = compiler.compile(FEASIBLE_REQUIREMENT)
    planner = PlannerStage(
        execution_mode="local-only",
        substrate=_StubSubstrate(),
        kernel=_StubKernel(),
        memory=_StubMemory(),
        gene_pool=_StubGenePool(),
    )

    output = planner.plan(build_spec)

    assert isinstance(output, FeasiblePlan)
    assert any("Execution warning: audit_log_failed:cycle=1" in note for note in output.implementation_notes)


def test_pipeline_requirement_generates_pipeline_plan_shape(tmp_path):
    compiler = RequirementCompiler()
    build_spec = compiler.compile(PIPELINE_REQUIREMENT)
    planner = _build_planner(tmp_path)

    output = planner.plan(build_spec)

    assert isinstance(output, FeasiblePlan)
    assert output.build_spec.target_artifact_type == ArtifactTargetType.PIPELINE
    assert output.obligation_mode == "software_build"
    assert output.required_obligations
    assert output.packaging_target == "python_pipeline_package"
    assert "data pipeline" in output.architecture_summary.lower()
    planned_paths = {item.path for item in output.file_tree_plan}
    assert "src/pipeline.py" in planned_paths
    assert "src/watcher.py" in planned_paths
    assert "src/validator.py" in planned_paths
    assert "src/quarantine.py" in planned_paths
    assert any(interface.name == "run" for interface in output.interfaces)
