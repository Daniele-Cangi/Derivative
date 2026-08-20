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

TELEMETRY_CLI_REQUIREMENT = (
    "Build a Python CLI that reads JSON Lines telemetry events with fields device_id, timestamp, "
    "and temperature_c, rejects malformed records, missing fields, and invalid timestamps into a "
    "quarantine JSONL file, computes per-device minimum, maximum, and average temperature, writes "
    "a summary CSV, and includes behavioral tests for parsing, quarantine handling, aggregation, "
    "and the complete CLI flow."
)

PRODUCTION_SERVICE_REQUIREMENT = (
    "Build a production-grade Python REST microservice with hashed API keys using bcrypt, "
    "persistent per-user rate limiting that survives restarts, a full audit trail of all requests, "
    "structured JSON logging, and integration tests."
)

ZERO_CAPACITY_REQUIREMENT = (
    "Build a bounded in-memory queue with capacity exactly zero that must successfully "
    "accept and retain at least one item without rejecting, blocking, overwriting, or discarding it."
)

ZERO_BYTE_JSON_REQUIREMENT = (
    "Build a generator whose output file must contain exactly zero bytes and must "
    "simultaneously contain a non-empty valid JSON object."
)

LOSSLESS_ENCODER_REQUIREMENT = (
    "Build a lossless encoder that maps every possible two-byte input to exactly one output byte "
    "and a decoder that reconstructs the original two-byte input for every encoded value, without "
    "external state, metadata, probabilistic behavior, or rejection."
)

ERASABLE_AUDIT_LOG_REQUIREMENT = (
    "Build an append-only audit log that must permanently retain every appended record and "
    "simultaneously provide an erase_all operation after which the log contains exactly zero records "
    "and every erased record remains retrievable from that same log, with no backup, external storage, "
    "hidden metadata, or reconstruction source."
)

LIBRARY_REQUIREMENT = (
    "Build a Python library exposing allocate_cents(total_cents: int, weights: list[int]) -> list[int]. "
    "It must use largest-remainder allocation, return integers whose sum equals total_cents, "
    "preserve input order, reject negative totals or weights, and include tests."
)

IDEMPOTENT_EVENT_SERVICE_REQUIREMENT = (
    "Build a Python REST service module with API-key authentication and SQLite persistence exposing "
    "create_event(api_key: str, event_id: str, payload: dict, db_path: str) -> tuple[int, dict]. "
    "Repeating the same event_id must be idempotent and must not insert a duplicate row. "
    "Invalid keys return 401. Include integration tests."
)

SALES_JSONL_PIPELINE_REQUIREMENT = (
    "Build a Python JSON Lines data pipeline exposing "
    "run(input_path: str, quarantine_path: str, summary_json_path: str) -> int. "
    "Each valid sales event has customer_id and amount. Write malformed events to quarantine and "
    "write per-customer transaction_count and total_amount to summary JSON. Include end-to-end tests."
)

GENERIC_JSONL_PIPELINE_REQUIREMENT = (
    "Build a Python JSON Lines data pipeline exposing "
    "run(input_path: str, output_path: str) -> int. "
    "Each valid event contains a non-empty sensor_id and a numeric value. Skip malformed events, "
    "write JSON containing valid_count, malformed_count, and a sensors object with count, min, max, "
    "and mean for each sensor, return 0 on success, produce deterministic keys, and include "
    "end-to-end tests."
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
    requirement_ids_by_text = {
        atom.text.lower(): atom.requirement_id for atom in build_spec.requirement_atoms
    }
    rate_limit_ids = {
        requirement_id
        for text, requirement_id in requirement_ids_by_text.items()
        if "rate limit" in text or "restart" in text
    }
    audit_ids = {
        requirement_id
        for text, requirement_id in requirement_ids_by_text.items()
        if "audit" in text or "logging" in text
    }
    logging_ids = {
        requirement_id
        for text, requirement_id in requirement_ids_by_text.items()
        if "logging" in text
    }
    assert rate_limit_ids <= set(capabilities_by_id["cap_storage"].requirement_ids)
    assert audit_ids <= set(capabilities_by_id["cap_audit"].requirement_ids)
    assert logging_ids <= set(capabilities_by_id["cap_observability"].requirement_ids)
    assert rate_limit_ids.isdisjoint(capabilities_by_id["cap_service_api"].requirement_ids)
    assert audit_ids.isdisjoint(capabilities_by_id["cap_rate_limit"].requirement_ids)
    planned_paths = {plan_file.path for plan_file in output.file_tree_plan}
    assert {
        capability.module_path for capability in output.implementation_blueprint.capabilities
    }.issubset(planned_paths)


def test_idempotent_event_service_plan_preserves_declared_api_and_omits_unrequested_rate_limit(tmp_path):
    output = _build_planner(tmp_path).plan(
        RequirementCompiler().compile(IDEMPOTENT_EVENT_SERVICE_REQUIREMENT)
    )

    assert isinstance(output, FeasiblePlan)
    assert {item.path for item in output.file_tree_plan} == {
        "src/service.py",
        "src/storage.py",
        "src/auth.py",
        "tests/test_service.py",
    }
    assert [(item.name, item.signature) for item in output.interfaces] == [
        ("register_user", "register_user(user_id: str, api_key: str, db_path: str) -> None"),
        (
            "create_event",
            "create_event(api_key: str, event_id: str, payload: dict, db_path: str) -> tuple[int, dict]",
        ),
    ]
    capability_types = {
        item.capability_type for item in output.implementation_blueprint.capabilities
    }
    assert capability_types == {"service_api", "sqlite_storage", "authentication"}
    assert "rate-limit" not in output.architecture_summary.lower()


def test_generic_universal_proof_uses_semantic_non_date_name():
    planner = PlannerStage.__new__(PlannerStage)
    test_name, test_type = planner._semantic_test_spec(
        "Prove universal constraint: For any row, if a field in field_order is missing, "
        "its rotated value becomes None",
        6,
    )
    assert test_name == "test_universal_for_any_row_if_a_field_06"
    assert "date_format" not in test_name
    assert test_type == "proof"


def test_universal_date_format_requirement_keeps_specialized_test_name():
    planner = PlannerStage.__new__(PlannerStage)
    test_name, test_type = planner._semantic_test_spec(
        "Prove universal constraint: supports every possible date format",
        4,
    )
    assert test_name == "test_universal_date_format_support"
    assert test_type == "proof"


def test_sales_jsonl_pipeline_plan_preserves_output_and_function_signature(tmp_path):
    output = _build_planner(tmp_path).plan(
        RequirementCompiler().compile(SALES_JSONL_PIPELINE_REQUIREMENT)
    )

    assert isinstance(output, FeasiblePlan)
    assert [(item.name, item.interface_type, item.signature) for item in output.interfaces] == [
        (
            "run",
            "entrypoint",
            "run(input_path: str, quarantine_path: str, summary_json_path: str) -> int",
        )
    ]
    assert "sales pipeline" in output.architecture_summary.lower()
    assert "summary json" in output.architecture_summary.lower()
    purposes = " ".join(item.purpose for item in output.file_tree_plan).lower()
    assert "customer_id" in purposes
    assert "telemetry" not in purposes


def test_generic_jsonl_pipeline_plan_preserves_declared_contract_without_telemetry_shape(tmp_path):
    output = _build_planner(tmp_path).plan(
        RequirementCompiler().compile(GENERIC_JSONL_PIPELINE_REQUIREMENT)
    )

    assert isinstance(output, FeasiblePlan)
    assert [(item.name, item.interface_type, item.signature) for item in output.interfaces] == [
        (
            "run",
            "entrypoint",
            "run(input_path: str, output_path: str) -> int",
        )
    ]
    assert "declared public interface" in output.architecture_summary.lower()
    purposes = " ".join(item.purpose for item in output.file_tree_plan).lower()
    assert "json lines" in purposes
    assert "telemetry" not in purposes
    assert "summary csv" not in purposes
    assert "device_id" not in purposes
    assert "src/quarantine.py" not in {item.path for item in output.file_tree_plan}


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


def test_cardinality_and_content_contradictions_are_terminal_infeasibility(tmp_path):
    planner = _build_planner(tmp_path)

    for requirement in (ZERO_CAPACITY_REQUIREMENT, ZERO_BYTE_JSON_REQUIREMENT):
        output = planner.plan(RequirementCompiler().compile(requirement))
        assert isinstance(output, InfeasibilityCertificate)
        assert output.terminal_status == "infeasible_proven"
        assert output.execution_evidence["is_satisfiable"] is False
        assert output.contradictions


def test_unicode_case_cardinality_witness_is_terminal_infeasibility(tmp_path):
    requirement = (
        "Implement a Python module returning a new string of the same length as input, "
        "where each Unicode letter has its case inverted. Use str.isupper(), "
        "str.islower(), and str.isalpha() to determine each character's status."
    )

    output = _build_planner(tmp_path).plan(
        RequirementCompiler().compile(requirement)
    )

    assert isinstance(output, InfeasibilityCertificate)
    assert output.terminal_status == "infeasible_proven"
    assert output.execution_evidence["is_satisfiable"] is False
    proof = " ".join(output.contradictions)
    assert "U+0130" in proof
    assert "2 code points" in proof


def test_information_and_retention_contradictions_are_terminal_infeasibility(tmp_path):
    planner = _build_planner(tmp_path)

    for requirement in (LOSSLESS_ENCODER_REQUIREMENT, ERASABLE_AUDIT_LOG_REQUIREMENT):
        output = planner.plan(RequirementCompiler().compile(requirement))
        assert isinstance(output, InfeasibilityCertificate)
        assert output.terminal_status == "infeasible_proven"
        assert output.execution_evidence["is_satisfiable"] is False
        assert output.contradictions
        assert output.minimal_relaxations


def test_library_plan_exposes_declared_api_without_fake_run_entrypoint(tmp_path):
    output = _build_planner(tmp_path).plan(RequirementCompiler().compile(LIBRARY_REQUIREMENT))

    assert isinstance(output, FeasiblePlan)
    assert [(item.name, item.interface_type) for item in output.interfaces] == [
        ("allocate_cents", "function")
    ]
    assert output.implementation_blueprint.entrypoint_path == ""
    assert all(item.name != "run" for item in output.interfaces)
    assert "src/library/core.py" in output.requirement_coverage["R001"]["files"]


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


def test_jsonl_cli_pipeline_separates_workflow_from_cli_entrypoint(tmp_path):
    compiler = RequirementCompiler()
    build_spec = compiler.compile(TELEMETRY_CLI_REQUIREMENT)
    planner = _build_planner(tmp_path)

    output = planner.plan(build_spec)

    assert isinstance(output, FeasiblePlan)
    interfaces = {interface.name: interface for interface in output.interfaces}
    assert interfaces["run"].interface_type == "function"
    assert interfaces["run"].signature == (
        "run(input_path: str, quarantine_path: str, summary_csv_path: str) -> int"
    )
    assert interfaces["main"].interface_type == "cli_entrypoint"
    assert output.implementation_blueprint.entrypoint_path == "src/pipeline.py"
    assert "json lines telemetry" in output.architecture_summary.lower()
    purposes = {item.path: item.purpose for item in output.file_tree_plan}
    assert "json lines telemetry" in purposes["src/pipeline.py"].lower()
    assert "line-number" in purposes["src/watcher.py"].lower()
    assert "timestamp validation" in purposes["src/validator.py"].lower()
    assert "quarantine jsonl" in purposes["src/quarantine.py"].lower()
    assert "sqlite" not in " ".join(purposes.values()).lower()


def test_declared_public_module_is_preserved_in_plan_layout_and_interface(tmp_path):
    build_spec = RequirementCompiler().compile(
        "Create a codec module exposing def encode_stream(stream: bytes) -> str "
        "that returns a deterministic digest and includes tests."
    )

    output = _build_planner(tmp_path).plan(build_spec)

    assert isinstance(output, FeasiblePlan)
    assert {item.path for item in output.file_tree_plan} == {
        "src/codec.py",
        "tests/test_codec.py",
    }
    interface = next(item for item in output.interfaces if item.name == "encode_stream")
    assert interface.module_path == "codec"
    assert "codec" in output.architecture_summary


def test_named_callable_component_becomes_public_module_contract(tmp_path):
    build_spec = RequirementCompiler().compile(
        "Design a data pipeline component 'filter_by_predicate' accepting an iterator and a predicate. "
        "It yields matching records in input order and includes tests."
    )

    output = _build_planner(tmp_path).plan(build_spec)

    assert isinstance(output, FeasiblePlan)
    assert {item.path for item in output.file_tree_plan} == {
        "src/filter_by_predicate.py",
        "tests/test_filter_by_predicate.py",
    }
    interface = next(item for item in output.interfaces if item.name == "filter_by_predicate")
    assert interface.module_path == "filter_by_predicate"


def test_module_with_forbidden_cli_and_service_gets_library_only_plan(tmp_path):
    build_spec = RequirementCompiler().compile(
        "Implement a Python data-pipeline module called 'rowrotate' exposing a single function "
        "rotate_fields(rows: list[dict], field_order: list[str], shift: int = 1) -> list[dict]. "
        "Only rotate_fields is public, and the module has no CLI or service interface. Include tests."
    )

    output = _build_planner(tmp_path).plan(build_spec)

    assert isinstance(output, FeasiblePlan)
    assert {item.path for item in output.file_tree_plan} == {
        "src/rowrotate.py",
        "tests/test_rowrotate.py",
    }
    assert output.packaging_target == "python_library_package"
    assert [(item.name, item.interface_type) for item in output.interfaces] == [
        ("rotate_fields", "function")
    ]
    assert "pipeline.py" not in output.architecture_summary


def test_named_cli_command_preserves_module_entrypoint_contract(tmp_path):
    build_spec = RequirementCompiler().compile(
        "Develop a CLI tool 'jsoncompact' that reads a JSON array from standard input, writes JSON "
        "to standard output, and includes tests."
    )

    output = _build_planner(tmp_path).plan(build_spec)

    assert isinstance(output, FeasiblePlan)
    paths = {item.path for item in output.file_tree_plan}
    assert "src/jsoncompact.py" in paths
    assert "src/cli.py" not in paths
    assert output.implementation_blueprint.entrypoint_path == "src/jsoncompact.py"
    interface = next(item for item in output.interfaces if item.interface_type == "cli_entrypoint")
    assert interface.name == "main"
    assert interface.module_path == "jsoncompact"
