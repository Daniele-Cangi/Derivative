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

