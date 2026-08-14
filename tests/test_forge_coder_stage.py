import ast
from dataclasses import replace
from pathlib import Path

import pytest

from core.forge.coder_stage import CoderStage, MalformedPlanError
from core.forge.contracts import (
    ArtifactTargetType,
    CodeArtifact,
    FeasiblePlan,
    ImplementationBlueprint,
    PlanFile,
    PlanInterface,
    PlanTest,
)
from core.forge.domains.registry import DomainAdapterRegistry
from core.forge.planner_stage import PlannerStage
from core.forge.requirement_compiler import RequirementCompiler


FEASIBLE_REQUIREMENT = (
    "Build a Python CLI that reads a CSV of contracts, extracts expiration dates, "
    "flags contracts expiring in less than 90 days, writes a summary CSV, and includes tests."
)

INVOICE_REQUIREMENT = (
    "Build a Python CLI that reads a CSV of invoices with columns invoice_id, due_date, amount, "
    "customer_name, flags overdue invoices, writes a summary CSV with totals and counts, and "
    "includes tests for malformed rows and invalid dates."
)

SERVICE_REQUIREMENT = (
    "Build a Python REST microservice with authentication, rate limiting (100 requests per minute per user), "
    "persistent storage for users and API keys, and tests for unauthorized access, rate-limit enforcement, "
    "and persistence across restart."
)

PRODUCTION_SERVICE_REQUIREMENT = (
    "Build a production-grade Python REST microservice with hashed API keys using bcrypt, "
    "persistent per-user rate limiting that survives restarts, a full audit trail of all requests, "
    "structured JSON logging, and integration tests."
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

JSONL_LOG_CLI_REQUIREMENT = (
    "Build a Python CLI whose main(argv) reads a JSON Lines application log with level and message "
    "fields, skips malformed lines, and writes a JSON report containing total_valid, "
    "malformed_count, and counts_by_level. Include behavioral tests."
)

JSON_MERGE_CLI_REQUIREMENT = (
    "Build a Python CLI whose main(argv) merges a base JSON object with an override JSON object "
    "recursively, writes the merged JSON to an output path, replaces lists instead of concatenating "
    "them, and rejects a non-object root. Include tests."
)

EMAIL_LIBRARY_REQUIREMENT = (
    "Build a Python library exposing canonicalize_email(value: str) -> str and "
    "deduplicate_emails(values: list[str]) -> list[str]. Canonicalization must trim surrounding "
    "whitespace and lowercase the address. Deduplication must preserve first-seen order after "
    "canonicalization. Include tests."
)

ALLOCATION_LIBRARY_REQUIREMENT = (
    "Build a Python library exposing allocate_cents(total_cents: int, weights: list[int]) -> "
    "list[int]. It must use largest-remainder allocation, return integers whose sum equals "
    "total_cents, preserve input order, reject negative totals or weights, and include tests."
)

SEMVER_LIBRARY_REQUIREMENT = (
    "Build a Python library exposing compare_versions(left: str, right: str) -> int for "
    "Semantic Versioning 2.0.0. Return -1, 0, or 1 by precedence; compare major, minor, "
    "and patch numerically; order prerelease identifiers according to the SemVer rules; "
    "ignore build metadata; reject invalid versions with ValueError; and include tests."
)

INTERVAL_LIBRARY_REQUIREMENT = (
    "Build a Python library exposing merge_intervals(intervals: list[tuple[int, int]]) -> "
    "list[tuple[int, int]]. Sort intervals by start, merge overlapping and touching intervals, "
    "return a deterministic list without mutating the input, reject an interval whose start "
    "exceeds its end with ValueError, and include tests."
)

JSON_RECORD_SORT_REQUIREMENT = (
    "Build a Python CLI whose main(argv) reads a JSON array of objects containing unique string "
    "id and numeric score fields, stably orders records by descending score and then ascending id, "
    "writes the ordered array as JSON to an output path, rejects duplicate ids or malformed "
    "records with ValueError, and includes behavioral tests."
)


@pytest.fixture(scope="module")
def feasible_plan(tmp_path_factory) -> FeasiblePlan:
    root = tmp_path_factory.mktemp("forge_coder_stage")
    compiler = RequirementCompiler()
    spec = compiler.compile(FEASIBLE_REQUIREMENT)
    planner = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(root / "forge_audit.json"),
        memory_file=str(root / "forge_memory.json"),
        gene_pool_file=str(root / "forge_gene_pool.json"),
    )
    output = planner.plan(spec)
    assert isinstance(output, FeasiblePlan)
    return output


@pytest.fixture(scope="module")
def invoice_feasible_plan(tmp_path_factory) -> FeasiblePlan:
    root = tmp_path_factory.mktemp("forge_coder_stage_invoice")
    compiler = RequirementCompiler()
    spec = compiler.compile(INVOICE_REQUIREMENT)
    planner = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(root / "forge_audit.json"),
        memory_file=str(root / "forge_memory.json"),
        gene_pool_file=str(root / "forge_gene_pool.json"),
    )
    output = planner.plan(spec)
    assert isinstance(output, FeasiblePlan)
    return output


@pytest.fixture(scope="module")
def service_feasible_plan(tmp_path_factory) -> FeasiblePlan:
    root = tmp_path_factory.mktemp("forge_coder_stage_service")
    compiler = RequirementCompiler()
    spec = compiler.compile(SERVICE_REQUIREMENT)
    planner = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(root / "forge_audit.json"),
        memory_file=str(root / "forge_memory.json"),
        gene_pool_file=str(root / "forge_gene_pool.json"),
    )
    output = planner.plan(spec)
    assert isinstance(output, FeasiblePlan)
    return output


@pytest.fixture(scope="module")
def production_service_feasible_plan(tmp_path_factory) -> FeasiblePlan:
    root = tmp_path_factory.mktemp("forge_coder_stage_service_production")
    compiler = RequirementCompiler()
    spec = compiler.compile(PRODUCTION_SERVICE_REQUIREMENT)
    planner = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(root / "forge_audit.json"),
        memory_file=str(root / "forge_memory.json"),
        gene_pool_file=str(root / "forge_gene_pool.json"),
    )
    output = planner.plan(spec)
    assert isinstance(output, FeasiblePlan)
    return output


@pytest.fixture(scope="module")
def pipeline_feasible_plan(tmp_path_factory) -> FeasiblePlan:
    root = tmp_path_factory.mktemp("forge_coder_stage_pipeline")
    compiler = RequirementCompiler()
    spec = compiler.compile(PIPELINE_REQUIREMENT)
    planner = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(root / "forge_audit.json"),
        memory_file=str(root / "forge_memory.json"),
        gene_pool_file=str(root / "forge_gene_pool.json"),
    )
    output = planner.plan(spec)
    assert isinstance(output, FeasiblePlan)
    return output


def _find_generated_file(artifact: CodeArtifact, path: str):
    for generated_file in artifact.files:
        if generated_file.path == path:
            return generated_file
    return None


def test_domain_registry_routes_typed_plans(
    feasible_plan,
    service_feasible_plan,
    pipeline_feasible_plan,
):
    registry = DomainAdapterRegistry()

    assert registry.select(feasible_plan).name == "cli"
    assert registry.select(service_feasible_plan).name == "service"
    assert registry.select(pipeline_feasible_plan).name == "pipeline"

    assert CoderStage().generate(feasible_plan).artifact_manifest["metadata"]["domain_adapter"] == "cli"
    assert CoderStage().generate(service_feasible_plan).artifact_manifest["metadata"]["domain_adapter"] == "service"
    assert CoderStage().generate(pipeline_feasible_plan).artifact_manifest["metadata"]["domain_adapter"] == "pipeline"
    cli_capabilities = CoderStage().generate(feasible_plan).artifact_manifest["metadata"][
        "adapter_capabilities"
    ]
    assert "csv_input" in cli_capabilities
    assert "recursive_json_merge" not in cli_capabilities


def test_domain_registry_uses_typed_plan_instead_of_raw_requirement(pipeline_feasible_plan):
    misleading_spec = replace(
        pipeline_feasible_plan.build_spec,
        raw_requirement="Build a Python CLI for contracts CSV files.",
        normalized_requirement="Build a Python CLI for contracts CSV files.",
        target_artifact_type=ArtifactTargetType.PIPELINE,
    )
    typed_pipeline_plan = replace(pipeline_feasible_plan, build_spec=misleading_spec)

    assert DomainAdapterRegistry().select(typed_pipeline_plan).name == "pipeline"

    path_driven_spec = replace(misleading_spec, target_artifact_type=ArtifactTargetType.UNKNOWN)
    windows_path_plan = replace(
        typed_pipeline_plan,
        build_spec=path_driven_spec,
        file_tree_plan=[
            replace(plan_file, path=plan_file.path.replace("/", "\\"))
            for plan_file in typed_pipeline_plan.file_tree_plan
        ],
    )
    assert DomainAdapterRegistry().select(windows_path_plan).name == "pipeline"


def test_script_main_path_does_not_route_to_cli_adapter(feasible_plan):
    script_spec = replace(
        feasible_plan.build_spec,
        raw_requirement="Build a Python script with tests.",
        normalized_requirement="Build a Python script with tests.",
        target_artifact_type=ArtifactTargetType.SCRIPT,
    )
    script_plan = replace(
        feasible_plan,
        build_spec=script_spec,
        architecture_summary="Python executable with explicit entrypoint and tests.",
        implementation_blueprint=ImplementationBlueprint(
            target_artifact_type=ArtifactTargetType.SCRIPT,
            entrypoint_path="src/main.py",
        ),
        file_tree_plan=[
            PlanFile(path="src/main.py", purpose="Executable workflow."),
            PlanFile(path="tests/test_main.py", purpose="Workflow tests."),
        ],
        interfaces=[
            PlanInterface(
                name="run",
                interface_type="entrypoint",
                signature="run() -> int",
            )
        ],
        required_tests=[
            PlanTest(
                test_name="test_script_workflow",
                objective="Execute the script workflow.",
            )
        ],
    )

    assert DomainAdapterRegistry().select(script_plan).name == "generic"
    artifact = CoderStage().generate(script_plan)
    main_source = _find_generated_file(artifact, "src/main.py")
    assert main_source is not None
    assert "def run() -> int:" in main_source.content
    assert "contracts_csv" not in main_source.content
    planned_test = _find_generated_file(artifact, "tests/test_main.py")
    required_test = _find_generated_file(artifact, "tests/test_script_workflow.py")
    assert planned_test is not None
    assert required_test is not None
    assert "result = main.run()" in planned_test.content
    assert "assert result == 0" in planned_test.content
    assert "getattr(main, 'run', None)" in required_test.content
    assert "main.main(" not in required_test.content


def test_coder_stage_returns_typed_code_artifact(feasible_plan):
    coder = CoderStage()
    artifact = coder.generate(feasible_plan)

    assert isinstance(artifact, CodeArtifact)
    assert artifact.plan_id == feasible_plan.plan_id
    assert artifact.files
    assert artifact.artifact_manifest
    for file in artifact.files:
        if file.path.endswith(".py"):
            ast.parse(file.content)


def test_code_artifact_contains_plan_file_tree(feasible_plan):
    coder = CoderStage()
    artifact = coder.generate(feasible_plan)
    artifact_paths = {file.path for file in artifact.files}
    planned_paths = {plan_file.path for plan_file in feasible_plan.file_tree_plan}

    assert planned_paths.issubset(artifact_paths)


def test_cli_entrypoint_is_present_when_declared(feasible_plan):
    coder = CoderStage()
    artifact = coder.generate(feasible_plan)

    declared_cli = [interface for interface in feasible_plan.interfaces if interface.interface_type == "cli_entrypoint"]
    assert declared_cli
    assert artifact.runnable_entrypoints
    assert any(path in artifact.runnable_entrypoints for path in ("src/cli.py", "src/main.py"))


def test_test_files_align_with_required_tests(feasible_plan):
    coder = CoderStage()
    artifact = coder.generate(feasible_plan)
    artifact_paths = {file.path for file in artifact.files}
    expected_test_paths = {f"tests/{plan_test.test_name}.py" for plan_test in feasible_plan.required_tests}

    assert expected_test_paths.issubset(artifact_paths)
    assert set(artifact.test_paths).issuperset(expected_test_paths)


def test_provenance_exists_for_every_generated_file(feasible_plan):
    coder = CoderStage()
    artifact = coder.generate(feasible_plan)

    for generated_file in artifact.files:
        assert generated_file.generated_from_plan_sections
        assert generated_file.path in artifact.traceability
        assert artifact.traceability[generated_file.path] == generated_file.generated_from_plan_sections


def test_malformed_plan_fails_explicitly(feasible_plan):
    coder = CoderStage()
    malformed = replace(feasible_plan, file_tree_plan=[])

    with pytest.raises(MalformedPlanError):
        coder.generate(malformed)


def test_unmappable_required_test_fails_closed(feasible_plan):
    coder = CoderStage()
    constrained = replace(
        feasible_plan,
        file_tree_plan=[
            PlanFile(
                path="README.md",
                purpose="Non-runnable metadata placeholder.",
                source_requirement_refs=[],
            )
        ],
        interfaces=[
            PlanInterface(
                name="run",
                interface_type="entrypoint",
                signature="run() -> int",
                description="Generic entrypoint.",
            )
        ],
        required_tests=[
            PlanTest(
                test_name="test_custom_requirement_without_mapping",
                objective="Satisfy a custom requirement without callable mapping.",
                test_type="acceptance",
                required=True,
                acceptance_criterion_ids=[],
                obligation_fields=[],
                requirement_ids=[],
            )
        ],
    )

    with pytest.raises(MalformedPlanError, match="Unable to generate semantic test template"):
        coder.generate(constrained)


def test_invoice_business_tests_are_semantic(invoice_feasible_plan):
    coder = CoderStage()
    artifact = coder.generate(invoice_feasible_plan)

    reads_csv = _find_generated_file(artifact, "tests/test_reads_contracts_csv.py")
    overdue = _find_generated_file(artifact, "tests/test_implement_functional_goal_flags_overdue_invoices.py")
    totals = _find_generated_file(artifact, "tests/test_writes_summary_csv_with_totals_and_counts.py")
    malformed_invalid = _find_generated_file(artifact, "tests/test_handles_malformed_rows_and_invalid_dates.py")
    cli_flow = _find_generated_file(artifact, "tests/test_cli_flow.py")
    build_goal = _find_generated_file(artifact, "tests/test_implement_functional_goal_build_a_python.py")
    cli_module = _find_generated_file(artifact, "src/cli.py")

    assert reads_csv is not None
    assert overdue is not None
    assert totals is not None
    assert malformed_invalid is not None
    assert cli_flow is not None
    assert build_goal is not None
    assert cli_module is not None

    assert "load_contracts_csv(" in reads_csv.content
    assert "invoice_id,due_date,amount,customer_name" in reads_csv.content
    assert "assert len(rows) == 1" in reads_csv.content
    assert "assert rows[0]['due_date']" in reads_csv.content

    assert "flag_expiring_contracts(" in overdue.content
    assert "{'invoice_id': 'INV-1', 'due_date': '2026-01-10'}" in overdue.content
    assert "{'invoice_id': 'INV-2', 'due_date': '2026-01-20'}" in overdue.content
    assert "assert flagged_by_id['INV-1']['is_expiring_within_horizon'] == 'True'" in overdue.content
    assert "assert flagged_by_id['INV-2']['is_expiring_within_horizon'] == 'False'" in overdue.content
    assert "assert flagged_by_id['INV-1']['is_overdue'] == 'True'" in overdue.content

    assert "write_summary_csv(" in totals.content
    assert "csv.DictReader" in totals.content
    assert "assert parsed[0]['total_amount'] == '25'" in totals.content
    assert "assert parsed[1]['invoice_count'] == '2'" in totals.content

    assert "load_contracts_csv(" in malformed_invalid.content
    assert "input_path.write_text(" in malformed_invalid.content
    assert "flag_expiring_contracts(" in malformed_invalid.content
    assert "assert len(rows) == 1" in malformed_invalid.content
    assert "assert len(flagged) == 1" in malformed_invalid.content
    assert "assert flagged[0]['days_to_expiration'] == ''" in malformed_invalid.content

    assert "def test_cli_flow_end_to_end(tmp_path):" in cli_flow.content
    assert "invoice_id,due_date,amount,customer_name" in cli_flow.content
    assert "rows = list(csv.DictReader(handle))" in cli_flow.content
    assert "assert rows[0]['total_amount'] == '25'" in cli_flow.content

    assert "invoice_id,due_date,amount,customer_name" in build_goal.content
    assert "rows = list(csv.DictReader(handle))" in build_goal.content

    assert "Process invoice due dates from CSV input." in cli_module.content
    assert "_ = 'entrypoint_defined" not in cli_module.content


def test_invoice_required_tests_have_no_assert_true_placeholders(invoice_feasible_plan):
    coder = CoderStage()
    artifact = coder.generate(invoice_feasible_plan)
    required_paths = {f"tests/{plan_test.test_name}.py" for plan_test in invoice_feasible_plan.required_tests}

    for path in required_paths:
        generated = _find_generated_file(artifact, path)
        assert generated is not None
        assert "assert True" not in generated.content


def test_invoice_required_tests_keep_requirement_provenance(invoice_feasible_plan):
    coder = CoderStage()
    artifact = coder.generate(invoice_feasible_plan)
    required_paths = {f"tests/{plan_test.test_name}.py" for plan_test in invoice_feasible_plan.required_tests}

    for path in required_paths:
        generated = _find_generated_file(artifact, path)
        assert generated is not None
        assert any(section.startswith("requirement:") for section in generated.generated_from_plan_sections)


def test_invoice_test_generation_is_deterministic(invoice_feasible_plan):
    coder = CoderStage()
    first = coder.generate(invoice_feasible_plan)
    second = coder.generate(invoice_feasible_plan)
    required_paths = {f"tests/{plan_test.test_name}.py" for plan_test in invoice_feasible_plan.required_tests}

    for path in required_paths:
        first_file = _find_generated_file(first, path)
        second_file = _find_generated_file(second, path)
        assert first_file is not None and second_file is not None
        assert first_file.content == second_file.content


def test_service_plan_generates_service_artifacts(service_feasible_plan):
    coder = CoderStage()
    artifact = coder.generate(service_feasible_plan)

    service_module = _find_generated_file(artifact, "src/service.py")
    domain_module = _find_generated_file(artifact, "src/domain.py")
    storage_module = _find_generated_file(artifact, "src/storage.py")
    auth_module = _find_generated_file(artifact, "src/auth.py")
    rate_limit_module = _find_generated_file(artifact, "src/rate_limit.py")
    audit_module = _find_generated_file(artifact, "src/audit.py")
    observability_module = _find_generated_file(artifact, "src/observability.py")
    service_test = _find_generated_file(artifact, "tests/test_service.py")
    suite_test = _find_generated_file(artifact, "tests/test_suite_executes.py")

    assert service_module is not None
    assert domain_module is not None
    assert storage_module is not None
    assert auth_module is not None
    assert rate_limit_module is not None
    assert audit_module is not None
    assert observability_module is not None
    assert service_test is not None
    assert suite_test is not None

    assert "import sqlite3" not in service_module.content
    assert "from domain import handle_request" in service_module.content
    assert "from auth import authenticate, register_user" in service_module.content
    assert "from rate_limit import RATE_LIMIT_PER_MINUTE, enforce_rate_limit" in service_module.content
    assert "def run() -> int:" in service_module.content
    assert "create_app(" in service_module.content

    assert "def handle_request(" in domain_module.content
    assert "from auth import authenticate" in domain_module.content
    assert "def init_db(" in storage_module.content
    assert "def authenticate(" in auth_module.content
    assert "def enforce_rate_limit(" in rate_limit_module.content
    assert "def record_event(" in audit_module.content
    assert "def health_status(" in observability_module.content
    assert "import service" in service_test.content
    assert "import service" in suite_test.content
    assert "import cli" not in suite_test.content
    assert "import contracts_csv" not in suite_test.content


def test_service_file_tree_never_generates_cli_imports(service_feasible_plan):
    assert any(file.path.lower().endswith("src/service.py") for file in service_feasible_plan.file_tree_plan)

    coder = CoderStage()
    artifact = coder.generate(service_feasible_plan)
    python_files = [generated for generated in artifact.files if generated.path.endswith(".py")]

    for generated in python_files:
        assert "import cli" not in generated.content
        assert "import contracts_csv" not in generated.content


def test_production_service_quality_contract_changes_generated_code(production_service_feasible_plan):
    coder = CoderStage()
    artifact = coder.generate(production_service_feasible_plan)

    service_module = _find_generated_file(artifact, "src/service.py")
    storage_module = _find_generated_file(artifact, "src/storage.py")
    auth_module = _find_generated_file(artifact, "src/auth.py")
    rate_limit_module = _find_generated_file(artifact, "src/rate_limit.py")
    audit_module = _find_generated_file(artifact, "src/audit.py")
    suite_test = _find_generated_file(artifact, "tests/test_suite_executes.py")

    assert service_module is not None
    assert storage_module is not None
    assert auth_module is not None
    assert rate_limit_module is not None
    assert audit_module is not None
    assert suite_test is not None

    assert "import bcrypt" in auth_module.content
    assert "bcrypt.checkpw" in auth_module.content
    assert "FORGE_USE_BCRYPT" not in auth_module.content
    assert "sha256$" not in auth_module.content
    assert "api_key_hash TEXT UNIQUE NOT NULL" in storage_module.content
    assert "CREATE TABLE IF NOT EXISTS rate_limit_hits" in storage_module.content
    assert "_RATE_LIMIT_BUCKETS" not in rate_limit_module.content
    assert "CREATE TABLE IF NOT EXISTS events" in storage_module.content
    assert "CREATE TABLE IF NOT EXISTS schema_meta" in storage_module.content
    assert "INSERT INTO events" in audit_module.content
    assert "json.dumps" in audit_module.content
    assert "@app.get('/health')" in service_module.content

    assert "def test_integration_flow_handles_valid_request" in suite_test.content
    assert "def test_audit_trail_records_requests" in suite_test.content

    blueprint = production_service_feasible_plan.implementation_blueprint
    capability_ids = {capability.capability_id for capability in blueprint.capabilities}
    assert capability_ids == {
        "cap_service_api",
        "cap_domain",
        "cap_storage",
        "cap_auth",
        "cap_rate_limit",
        "cap_audit",
        "cap_observability",
    }
    assert artifact.artifact_manifest["implementation_blueprint"]["entrypoint_path"] == "src/service.py"
    for capability in blueprint.capabilities:
        generated = _find_generated_file(artifact, capability.module_path)
        assert generated is not None
        assert f"capability:{capability.capability_id}" in generated.generated_from_plan_sections
        for quality_field in capability.quality_fields:
            assert f"quality_field:{quality_field}" in generated.generated_from_plan_sections


def test_pipeline_mode_generates_pipeline_artifacts_and_respects_entrypoint_contract(pipeline_feasible_plan):
    coder = CoderStage()
    artifact = coder.generate(pipeline_feasible_plan)

    pipeline_module = _find_generated_file(artifact, "src/pipeline.py")
    watcher_module = _find_generated_file(artifact, "src/watcher.py")
    validator_module = _find_generated_file(artifact, "src/validator.py")
    quarantine_module = _find_generated_file(artifact, "src/quarantine.py")

    assert pipeline_module is not None
    assert watcher_module is not None
    assert validator_module is not None
    assert quarantine_module is not None
    assert artifact.runnable_entrypoints == ["src/pipeline.py"]

    assert "from watcher import discover_csv_files" in pipeline_module.content
    assert "from validator import DEFAULT_SCHEMA, validate_row" in pipeline_module.content
    assert "from quarantine import quarantine_row" in pipeline_module.content
    assert "INSERT INTO audit_events" in pipeline_module.content
    assert "@app.get('/health')" in pipeline_module.content

    assert "def run(" in pipeline_module.content
    assert "def main(" not in pipeline_module.content

    assert "import contracts_csv" not in pipeline_module.content
    assert "import expiration_rules" not in pipeline_module.content
    assert "import summary_writer" not in pipeline_module.content


def test_pipeline_cli_entrypoint_uses_blueprint_path_and_keeps_run_callable(tmp_path):
    spec = RequirementCompiler().compile(TELEMETRY_CLI_REQUIREMENT)
    planner = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    )
    plan = planner.plan(spec)
    assert isinstance(plan, FeasiblePlan)

    artifact = CoderStage().generate(plan)
    pipeline_module = _find_generated_file(artifact, "src/pipeline.py")

    assert pipeline_module is not None
    assert artifact.runnable_entrypoints == ["src/pipeline.py"]
    assert "def run(" in pipeline_module.content
    assert "def main(argv: list[str] | None = None) -> int:" in pipeline_module.content
    assert "@click.command" not in pipeline_module.content
    assert "interface:main" in pipeline_module.generated_from_plan_sections


def test_jsonl_telemetry_plan_generates_domain_specific_behavior(tmp_path):
    spec = RequirementCompiler().compile(TELEMETRY_CLI_REQUIREMENT)
    planner = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    )
    plan = planner.plan(spec)
    assert isinstance(plan, FeasiblePlan)

    artifact = CoderStage().generate(plan)
    pipeline_module = _find_generated_file(artifact, "src/pipeline.py")
    watcher_module = _find_generated_file(artifact, "src/watcher.py")
    validator_module = _find_generated_file(artifact, "src/validator.py")

    assert pipeline_module is not None
    assert watcher_module is not None
    assert validator_module is not None
    assert "def aggregate_per_device(" in pipeline_module.content
    assert "minimum = min(temperatures)" in pipeline_module.content
    assert "maximum = max(temperatures)" in pipeline_module.content
    assert "average = sum(temperatures) / len(temperatures)" in pipeline_module.content
    assert "def write_summary_csv(" in pipeline_module.content
    assert "def main(argv: list[str] | None = None) -> int:" in pipeline_module.content
    assert "def iter_jsonl(" in watcher_module.content
    assert '"malformed_records"' in watcher_module.content
    assert "invalid_timestamp" in validator_module.content
    assert "discover_csv_files" not in pipeline_module.content
    assert "sqlite3" not in pipeline_module.content

    generated_tests = [
        generated.content
        for generated in artifact.files
        if generated.path.startswith("tests/")
    ]
    corpus = "\n".join(generated_tests)
    assert "input_jsonl" in corpus
    assert "malformed_records" in corpus
    assert "missing_fields:temperature_c" in corpus
    assert "invalid_timestamp" in corpus
    assert "minimum" in corpus
    assert "maximum" in corpus
    assert "average" in corpus
    assert "summary_csv" in corpus
    assert "cli_flow" in corpus


@pytest.mark.parametrize(
    ("requirement", "expected_source", "expected_terms", "forbidden_terms"),
    [
        (
            IDEMPOTENT_EVENT_SERVICE_REQUIREMENT,
            "src/service.py",
            ("def create_event(", "insert_event_once", "return 401"),
            ("enforce_rate_limit", "handle_request"),
        ),
        (
            SALES_JSONL_PIPELINE_REQUIREMENT,
            "src/pipeline.py",
            ("def run(input_path: str, quarantine_path: str, summary_json_path: str)", "transaction_count", "total_amount"),
            ("temperature_c", "summary_csv"),
        ),
    ],
)
def test_contract_specific_domains_generate_executable_semantics(
    tmp_path,
    requirement,
    expected_source,
    expected_terms,
    forbidden_terms,
):
    spec = RequirementCompiler().compile(requirement)
    plan = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    ).plan(spec)
    assert isinstance(plan, FeasiblePlan)

    artifact = CoderStage().generate(plan)
    source = _find_generated_file(artifact, expected_source)
    assert source is not None
    all_source = "\n".join(
        item.content for item in artifact.files if item.path.startswith("src/")
    )
    assert all(term in all_source for term in expected_terms)
    assert all(term not in all_source for term in forbidden_terms)
    assert all("assert True" not in item.content for item in artifact.files if item.path.startswith("tests/"))


@pytest.mark.parametrize(
    ("requirement", "capabilities", "source_terms"),
    [
        (
            JSONL_LOG_CLI_REQUIREMENT,
            {"cli_entrypoint", "jsonl_log_input", "summary_json_output", "log_level_counts"},
            {"json.loads", "malformed_count", "counts_by_level"},
        ),
        (
            JSON_MERGE_CLI_REQUIREMENT,
            {
                "cli_entrypoint",
                "recursive_json_merge",
                "json_list_replacement",
                "json_object_root_validation",
            },
            {"recursive_json_merge", "replace_lists", "JSON root must be an object"},
        ),
    ],
)
def test_json_cli_profiles_are_deterministic_and_capability_exact(
    tmp_path,
    requirement,
    capabilities,
    source_terms,
):
    spec = RequirementCompiler().compile(requirement)
    plan = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    ).plan(spec)
    assert isinstance(plan, FeasiblePlan)

    artifact = CoderStage().generate(plan)
    source = _find_generated_file(artifact, "src/cli.py")
    assert source is not None
    assert set(artifact.artifact_manifest["metadata"]["adapter_capabilities"]) == capabilities
    assert all(term in source.content for term in source_terms)


def test_email_library_profile_is_narrow_deterministic_and_canonical(tmp_path):
    spec = RequirementCompiler().compile(EMAIL_LIBRARY_REQUIREMENT)
    plan = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    ).plan(spec)
    assert isinstance(plan, FeasiblePlan)
    assert DomainAdapterRegistry().select(plan).name == "library"

    artifact = CoderStage().generate(plan)
    core_module = _find_generated_file(artifact, "src/library/core.py")
    assert core_module is not None
    assert "result.append(canonical)" in core_module.content
    assert artifact.artifact_manifest["metadata"]["adapter_capabilities"] == [
        "library_public_api"
    ]


def test_largest_remainder_library_profile_uses_exact_tie_breaking(tmp_path):
    spec = RequirementCompiler().compile(ALLOCATION_LIBRARY_REQUIREMENT)
    plan = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    ).plan(spec)
    assert isinstance(plan, FeasiblePlan)
    assert DomainAdapterRegistry().select(plan).name == "library"

    artifact = CoderStage().generate(plan)
    core_module = _find_generated_file(artifact, "src/library/core.py")
    assert core_module is not None
    assert "divmod(total_cents * weight, total_weight)" in core_module.content
    assert "key=lambda item: (-item[0], item[1])" in core_module.content


@pytest.mark.parametrize(
    ("requirement", "source_terms", "test_terms"),
    [
        (
            SEMVER_LIBRARY_REQUIREMENT,
            {"_SEMVER_PATTERN", "_compare_prerelease", "compare_versions"},
            {"1.0.0-alpha.1", "build.99", "pytest.raises(ValueError)"},
        ),
        (
            INTERVAL_LIBRARY_REQUIREMENT,
            {"merge_intervals", "start > end", "ordered.sort"},
            {"intervals == original", "(5, 8)", "pytest.raises(ValueError)"},
        ),
    ],
)
def test_standard_library_profiles_generate_behavioral_implementations(
    tmp_path,
    requirement,
    source_terms,
    test_terms,
):
    spec = RequirementCompiler().compile(requirement)
    plan = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    ).plan(spec)
    assert isinstance(plan, FeasiblePlan)
    assert DomainAdapterRegistry().select(plan).name == "library"

    artifact = CoderStage().generate(plan)
    source = _find_generated_file(artifact, "src/library/core.py")
    assert source is not None
    tests = "\n".join(
        generated.content
        for generated in artifact.files
        if generated.path.startswith("tests/")
    )
    assert all(term in source.content for term in source_terms)
    assert all(term in tests for term in test_terms)
    assert artifact.artifact_manifest["metadata"]["adapter_capabilities"] == [
        "library_public_api"
    ]


def test_json_record_sort_profile_preserves_value_error_contract(tmp_path):
    spec = RequirementCompiler().compile(JSON_RECORD_SORT_REQUIREMENT)
    plan = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    ).plan(spec)
    assert isinstance(plan, FeasiblePlan)

    artifact = CoderStage().generate(plan)
    source = _find_generated_file(artifact, "src/cli.py")
    assert source is not None
    assert "except ValueError" not in source.content
    assert "raise ValueError" in source.content
    assert "key=lambda record: (-record[\"score\"], record[\"id\"])" in source.content
    tests = "\n".join(
        generated.content
        for generated in artifact.files
        if generated.path.startswith("tests/")
    )
    assert "test_duplicate_ids_and_malformed_records_raise_value_error" in tests
    assert "pytest.raises(ValueError)" in tests
