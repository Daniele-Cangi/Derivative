import copy
from dataclasses import replace
from pathlib import Path

import pytest

import core.forge.validator_stage as validator_stage_module
from core.forge.coder_stage import CoderStage
from core.forge.contracts import CodeArtifact, FeasiblePlan, ValidationArtifact
from core.forge.planner_stage import PlannerStage
from core.forge.requirement_compiler import RequirementCompiler
from core.forge.validator_stage import ValidatorStage


FEASIBLE_REQUIREMENT = (
    "Build a Python CLI that reads a CSV of contracts, extracts expiration dates, "
    "flags contracts expiring in less than 90 days, writes a summary CSV, and includes tests."
)

INVOICE_REQUIREMENT = (
    "Build a Python CLI that reads a CSV of invoices with columns invoice_id, due_date, amount, "
    "customer_name, flags overdue invoices, writes a summary CSV with totals and counts, and "
    "includes tests for malformed rows and invalid dates."
)

PRODUCTION_SERVICE_REQUIREMENT = (
    "Build a production-grade Python REST microservice with hashed API keys using bcrypt, "
    "persistent per-user rate limiting that survives restarts, a full audit trail of all requests, "
    "structured JSON logging, and integration tests."
)


@pytest.fixture(scope="module")
def forge_pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("forge_validator_stage")
    compiler = RequirementCompiler()
    spec = compiler.compile(FEASIBLE_REQUIREMENT)
    planner = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(root / "forge_audit.json"),
        memory_file=str(root / "forge_memory.json"),
        gene_pool_file=str(root / "forge_gene_pool.json"),
    )
    planned = planner.plan(spec)
    assert isinstance(planned, FeasiblePlan)
    coder = CoderStage()
    artifact = coder.generate(planned)
    validator = ValidatorStage()
    return {
        "build_spec": spec,
        "plan": planned,
        "artifact": artifact,
        "validator": validator,
    }


def _find_file(artifact: CodeArtifact, path: str):
    for generated in artifact.files:
        if generated.path == path:
            return generated
    return None


def test_validator_passes_when_all_layers_pass(forge_pipeline):
    validator: ValidatorStage = forge_pipeline["validator"]
    artifact: CodeArtifact = forge_pipeline["artifact"]
    plan: FeasiblePlan = forge_pipeline["plan"]
    build_spec = forge_pipeline["build_spec"]

    result = validator.validate(artifact, plan, build_spec)

    assert isinstance(result, ValidationArtifact)
    assert result.passed is True
    assert result.layer1_result is not None and result.layer1_result.passed is True
    assert result.layer2_result is not None and result.layer2_result.passed is True
    assert result.layer3_result is not None and result.layer3_result.passed is True
    assert result.failure_signatures == []
    assert result.failure_category is None
    assert result.evidence
    assert "validated_entrypoints" in result.evidence
    assert "executed_tests" in result.evidence
    assert "manifest_provenance_checks" in result.evidence
    assert "obligation_acceptance_checks" in result.evidence


def test_missing_required_file_is_detected(forge_pipeline):
    validator: ValidatorStage = forge_pipeline["validator"]
    artifact: CodeArtifact = copy.deepcopy(forge_pipeline["artifact"])
    plan: FeasiblePlan = forge_pipeline["plan"]
    build_spec = forge_pipeline["build_spec"]

    artifact.files = [file for file in artifact.files if file.path != "src/summary_writer.py"]
    artifact.traceability.pop("src/summary_writer.py", None)

    result = validator.validate(artifact, plan, build_spec)

    assert result.passed is False
    assert "missing_required_file" in result.failure_signatures
    assert result.evidence
    assert "validated_entrypoints" in result.evidence
    assert "executed_tests" in result.evidence
    assert "manifest_provenance_checks" in result.evidence
    assert "obligation_acceptance_checks" in result.evidence


def test_missing_declared_entrypoint_is_detected(forge_pipeline):
    validator: ValidatorStage = forge_pipeline["validator"]
    artifact: CodeArtifact = copy.deepcopy(forge_pipeline["artifact"])
    plan: FeasiblePlan = forge_pipeline["plan"]
    build_spec = forge_pipeline["build_spec"]

    artifact.runnable_entrypoints = ["src/missing_cli.py"]

    result = validator.validate(artifact, plan, build_spec)

    assert result.passed is False
    assert "missing_entrypoint" in result.failure_signatures


def test_superficial_stub_fails_adversarial_layer(forge_pipeline):
    validator: ValidatorStage = forge_pipeline["validator"]
    artifact: CodeArtifact = copy.deepcopy(forge_pipeline["artifact"])
    plan: FeasiblePlan = forge_pipeline["plan"]
    build_spec = forge_pipeline["build_spec"]

    cli_file = _find_file(artifact, "src/cli.py")
    csv_file = _find_file(artifact, "src/contracts_csv.py")
    expiration_file = _find_file(artifact, "src/expiration_rules.py")
    summary_file = _find_file(artifact, "src/summary_writer.py")
    assert cli_file and csv_file and expiration_file and summary_file

    cli_file.content = (
        "def main(argv=None):\n"
        "    return 0\n"
    )
    csv_file.content = (
        "def load_contracts_csv(path: str):\n"
        "    return []\n"
    )
    expiration_file.content = (
        "def flag_expiring_contracts(records, horizon_days=90, today=None):\n"
        "    return records\n"
    )
    summary_file.content = (
        "def write_summary_csv(rows, output_path):\n"
        "    return None\n"
    )
    for generated in artifact.files:
        if generated.path.startswith("tests/"):
            generated.content = "def test_stub():\n    assert True\n"

    result = validator.validate(artifact, plan, build_spec)

    assert result.passed is False
    assert result.layer3_result is not None and result.layer3_result.passed is False
    assert "superficial_stub" in result.failure_signatures


def test_missing_obligation_acceptance_coverage_fails_even_when_layer1_passes(forge_pipeline):
    validator: ValidatorStage = forge_pipeline["validator"]
    artifact: CodeArtifact = copy.deepcopy(forge_pipeline["artifact"])
    plan: FeasiblePlan = forge_pipeline["plan"]
    build_spec = forge_pipeline["build_spec"]

    artifact.artifact_manifest["required_obligations"] = []
    for generated in artifact.files:
        generated.generated_from_plan_sections = [
            section
            for section in generated.generated_from_plan_sections
            if not (section.startswith("acceptance:") or section.startswith("obligation:"))
        ]
    artifact.traceability = {
        generated.path: list(generated.generated_from_plan_sections)
        for generated in artifact.files
    }

    result = validator.validate(artifact, plan, build_spec)

    assert result.layer1_result is not None and result.layer1_result.passed is True
    assert result.passed is False
    assert "missing_obligation" in result.failure_signatures or "missing_acceptance_coverage" in result.failure_signatures


def test_provenance_manifest_mismatch_is_detected(forge_pipeline):
    validator: ValidatorStage = forge_pipeline["validator"]
    artifact: CodeArtifact = copy.deepcopy(forge_pipeline["artifact"])
    plan: FeasiblePlan = forge_pipeline["plan"]
    build_spec = forge_pipeline["build_spec"]

    artifact.traceability["src/cli.py"] = ["plan_file:src/cli.py", "tampered:yes"]
    artifact.artifact_manifest["generated_files"][0]["path"] = "src/does_not_exist.py"

    result = validator.validate(artifact, plan, build_spec)

    assert result.passed is False
    assert "provenance_mismatch" in result.failure_signatures or "manifest_mismatch" in result.failure_signatures


def test_missing_semantic_requirement_coverage_is_detected(forge_pipeline):
    validator: ValidatorStage = forge_pipeline["validator"]
    artifact: CodeArtifact = copy.deepcopy(forge_pipeline["artifact"])
    plan: FeasiblePlan = forge_pipeline["plan"]
    build_spec = forge_pipeline["build_spec"]

    hard_atom = next(
        atom for atom in build_spec.requirement_atoms if atom.category != "ambiguity" and atom.strength in {"hard", "universal"}
    )
    mapped_tests = plan.requirement_coverage[hard_atom.requirement_id]["tests"]
    assert mapped_tests

    for test_name in mapped_tests:
        path = f"tests/{test_name}.py"
        generated = _find_file(artifact, path)
        assert generated is not None
        generated.content = "def test_placeholder():\n    assert True\n"

    result = validator.validate(artifact, plan, build_spec)

    assert result.passed is False
    assert "non_semantic_test" in result.failure_signatures
    assert "missing_semantic_requirement_coverage" in result.failure_signatures
    assert result.failure_category is not None and result.failure_category.value == "validation"


def test_validator_uses_invoice_smoke_input_for_invoice_specs():
    validator = ValidatorStage()
    invoice_spec = RequirementCompiler().compile(INVOICE_REQUIREMENT)

    sample = validator._sample_input_csv_content(invoice_spec)

    assert "invoice_id,due_date,amount,customer_name" in sample
    assert "INV-1,2026-01-15,100.00,Acme" in sample


def test_quality_contract_violation_detected_for_hashed_service(tmp_path):
    compiler = RequirementCompiler()
    spec = compiler.compile(PRODUCTION_SERVICE_REQUIREMENT)
    planner = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "forge_audit.json"),
        memory_file=str(tmp_path / "forge_memory.json"),
        gene_pool_file=str(tmp_path / "forge_gene_pool.json"),
    )
    plan_output = planner.plan(spec)
    assert isinstance(plan_output, FeasiblePlan)

    artifact = CoderStage().generate(plan_output)
    service_file = _find_file(artifact, "src/service.py")
    assert service_file is not None
    service_file.content = (
        "import sqlite3\n"
        "def init_db(db_path='service.db'):\n"
        "    with sqlite3.connect(db_path) as conn:\n"
        "        conn.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, api_key TEXT UNIQUE NOT NULL)')\n"
        "        conn.commit()\n"
        "def run():\n"
        "    return 0\n"
    )

    result = ValidatorStage().validate(artifact, plan_output, spec)

    assert result.passed is False
    assert "quality_contract_violation" in result.failure_signatures
    assert any("quality_contract_violation:" in failure for failure in result.failures)


def test_quality_contract_violation_when_bcrypt_not_available(tmp_path, monkeypatch):
    compiler = RequirementCompiler()
    spec = compiler.compile(PRODUCTION_SERVICE_REQUIREMENT)
    planner = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "forge_audit.json"),
        memory_file=str(tmp_path / "forge_memory.json"),
        gene_pool_file=str(tmp_path / "forge_gene_pool.json"),
    )
    plan_output = planner.plan(spec)
    assert isinstance(plan_output, FeasiblePlan)
    artifact = CoderStage().generate(plan_output)

    original_find_spec = validator_stage_module.importlib.util.find_spec

    def _fake_find_spec(module_name: str):
        if module_name == "bcrypt":
            return None
        return original_find_spec(module_name)

    monkeypatch.setattr(validator_stage_module.importlib.util, "find_spec", _fake_find_spec)

    result = ValidatorStage().validate(artifact, plan_output, spec)

    assert result.passed is False
    assert "quality_contract_violation" in result.failure_signatures
    assert any("bcrypt required but not available" in failure for failure in result.failures)
