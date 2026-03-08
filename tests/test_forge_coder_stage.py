import ast
from dataclasses import replace
from pathlib import Path

import pytest

from core.forge.coder_stage import CoderStage, MalformedPlanError
from core.forge.contracts import CodeArtifact, FeasiblePlan
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


def _find_generated_file(artifact: CodeArtifact, path: str):
    for generated_file in artifact.files:
        if generated_file.path == path:
            return generated_file
    return None


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
