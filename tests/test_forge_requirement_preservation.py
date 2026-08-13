from pathlib import Path

import pytest

from core.forge.coder_stage import CoderStage
from core.forge.contracts import FeasiblePlan
from core.forge.planner_stage import PlannerStage
from core.forge.requirement_compiler import RequirementCompiler
from core.forge.validator_stage import ValidatorStage
from core.forge.validation.obligations import ObligationValidationLayer
from forge import run_forge


EXAMPLE_A = (
    "Build a Python CLI that reads a CSV of contracts, extracts expiration dates, "
    "flags contracts expiring in less than 90 days, writes a summary CSV, includes tests, "
    "and guarantees support for every possible date format."
)

EXAMPLE_B = (
    "Build a Python CLI that reads a CSV of invoices with columns invoice_id, due_date, amount, "
    "customer_name, flags overdue invoices, writes a summary CSV with totals and counts, and "
    "includes tests for malformed rows and invalid dates."
)

BASE_REQUIREMENT = (
    "Build a Python CLI that reads a CSV of contracts, extracts expiration dates, "
    "flags contracts expiring in less than 90 days, writes a summary CSV, and includes tests."
)

TELEMETRY_REQUIREMENT = (
    "Build a Python CLI that reads JSON Lines telemetry events with fields device_id, timestamp, and temperature_c, "
    "rejects malformed records, missing fields, and invalid timestamps into a quarantine JSONL file, "
    "computes per-device minimum, maximum, and average temperature, writes a summary CSV, and includes "
    "behavioral tests for parsing, quarantine handling, aggregation, and the complete CLI flow."
)

CSV_FEASIBLE_REGRESSIONS = [
    (
        "Build a Python CLI that reads a CSV of invoices with columns invoice_id, due_date, amount, "
        "customer_name, flags overdue invoices, writes a summary CSV with totals and counts, and "
        "includes tests for malformed rows and invalid dates."
    ),
    (
        "Build a Python CLI that reads a CSV of contracts, extracts expiration dates, flags contracts "
        "expiring in less than 30 days, writes a summary CSV, and includes tests for invalid dates."
    ),
    (
        "Build a Python CLI that reads a CSV of contracts with columns contract_id and expiration_date, "
        "extracts expiration dates, flags contracts expiring in less than 120 days, writes a summary CSV, "
        "and includes tests for malformed rows."
    ),
    (
        "Build a Python CLI that reads a CSV of invoices with columns invoice_id, due_date, amount, "
        "customer_name, flags overdue invoices, writes a summary CSV with totals and counts, and includes "
        "tests for malformed rows and invalid dates."
    ),
    (
        "Build a Python CLI that reads a CSV of contracts, extracts expiration dates, flags contracts "
        "expiring in less than 45 days, writes a summary CSV with totals and counts, and includes tests "
        "for malformed rows."
    ),
]


def _planner(tmp_path: Path) -> PlannerStage:
    return PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "forge_audit.json"),
        memory_file=str(tmp_path / "forge_memory.json"),
        gene_pool_file=str(tmp_path / "forge_gene_pool.json"),
    )


def test_universal_requirement_is_preserved_as_atomic_unit():
    spec = RequirementCompiler().compile(EXAMPLE_A)

    universal_atoms = [atom for atom in spec.requirement_atoms if atom.strength == "universal"]
    assert universal_atoms
    assert any("every possible date format" in atom.text.lower() for atom in universal_atoms)
    assert any(atom.category == "universal_constraint" for atom in universal_atoms)


def test_universal_requirement_not_covered_cannot_end_as_verified(tmp_path):
    result = run_forge(
        requirement=EXAMPLE_A,
        output_root=str(tmp_path / "runs"),
        packaging_output_root=str(tmp_path / "packages"),
    )

    assert result.terminal_status == "validation_failed"
    assert result.validation is not None
    assert "universal_constraint_unproven" in result.validation.failure_signatures
    assert result.packaged_artifact is None


def test_business_requirement_atoms_are_preserved_and_propagated_to_plan(tmp_path):
    compiler = RequirementCompiler()
    spec = compiler.compile(EXAMPLE_B)
    atom_text = " ".join(atom.text.lower() for atom in spec.requirement_atoms)

    assert "build a python cli" in atom_text
    assert "invoice_id" in atom_text
    assert "due_date" in atom_text
    assert "amount" in atom_text
    assert "customer_name" in atom_text
    assert "malformed rows" in atom_text
    assert "invalid dates" in atom_text
    assert "totals and counts" in atom_text

    assert any("malformed" in atom.text.lower() and atom.category == "validation" for atom in spec.requirement_atoms)
    assert any("invalid" in atom.text.lower() and atom.category == "validation" for atom in spec.requirement_atoms)

    plan_output = _planner(tmp_path).plan(spec)
    assert isinstance(plan_output, FeasiblePlan)
    assert plan_output.requirement_coverage
    required_ids = {atom.requirement_id for atom in spec.requirement_atoms if atom.category != "ambiguity"}
    assert required_ids.issubset(set(plan_output.requirement_coverage.keys()))
    for requirement_id in required_ids:
        entry = plan_output.requirement_coverage[requirement_id]
        assert entry["acceptance_criteria"]


def test_trivial_generated_test_is_rejected_as_non_semantic(tmp_path):
    compiler = RequirementCompiler()
    spec = compiler.compile(BASE_REQUIREMENT)
    plan_output = _planner(tmp_path).plan(spec)
    assert isinstance(plan_output, FeasiblePlan)

    artifact = CoderStage().generate(plan_output)
    required_test_path = f"tests/{plan_output.required_tests[0].test_name}.py"
    for generated_file in artifact.files:
        if generated_file.path == required_test_path:
            generated_file.content = "def test_acceptance_requirement():\n    assert True\n"
            break

    validation = ValidatorStage().validate(artifact, plan_output, spec)

    assert validation.passed is False
    assert "non_semantic_test" in validation.failure_signatures
    assert "fake_acceptance_coverage" in validation.failure_signatures


def test_jsonl_telemetry_requirement_uses_matching_domain_and_verifies(tmp_path):
    compiler = RequirementCompiler()
    spec = compiler.compile(TELEMETRY_REQUIREMENT)
    plan_output = _planner(tmp_path).plan(spec)
    assert isinstance(plan_output, FeasiblePlan)
    assert "src/pipeline.py" in plan_output.requirement_coverage["R002"]["files"]
    assert "src/pipeline.py" in plan_output.requirement_coverage["R004"]["files"]
    assert "src/pipeline.py" in plan_output.requirement_coverage["R005"]["files"]
    assert "src/watcher.py" not in plan_output.requirement_coverage["R005"]["files"]

    artifact = CoderStage().generate(plan_output)
    validation = ValidatorStage().validate(artifact, plan_output, spec)

    assert validation.passed is True
    assert validation.failure_signatures == []
    pipeline_source = next(
        generated.content
        for generated in artifact.files
        if generated.path == "src/pipeline.py"
    )
    assert "aggregate_per_device" in pipeline_source
    assert "write_summary_csv" in pipeline_source
    assert "discover_csv_files" not in pipeline_source
    assert "invoice_id,due_date,amount,customer_name" not in pipeline_source
    checks = validation.layer2_result.evidence["requirement_semantic_checks"]
    assert checks["semantic_content_mismatches"] == []
    assert all(
        not details["missing_source_terms"] and not details["missing_test_terms"]
        for details in checks["requirements"].values()
    )


@pytest.mark.parametrize("requirement", CSV_FEASIBLE_REGRESSIONS)
def test_test_only_validation_atoms_use_executed_test_evidence(requirement, tmp_path):
    spec = RequirementCompiler().compile(requirement)
    plan_output = _planner(tmp_path).plan(spec)
    assert isinstance(plan_output, FeasiblePlan)
    artifact = CoderStage().generate(plan_output)

    validation = ValidatorStage().validate(artifact, plan_output, spec)

    assert validation.passed is True
    assert validation.failure_signatures == []
    semantic_checks = validation.layer2_result.evidence["requirement_semantic_checks"]
    test_only_atoms = [
        atom
        for atom in spec.requirement_atoms
        if atom.category == "validation" and "test" in atom.text.lower()
    ]
    assert test_only_atoms
    for atom in test_only_atoms:
        check = semantic_checks["requirements"][atom.requirement_id]
        assert check["source_evidence_required"] is False
        assert check["missing_test_terms"] == []


def test_operational_validation_atom_still_requires_source_evidence(tmp_path):
    spec = RequirementCompiler().compile(TELEMETRY_REQUIREMENT)
    plan_output = _planner(tmp_path).plan(spec)
    assert isinstance(plan_output, FeasiblePlan)
    artifact = CoderStage().generate(plan_output)
    operational_atom = next(
        atom
        for atom in spec.requirement_atoms
        if atom.text.lower().startswith("rejects malformed records")
    )
    for generated in artifact.files:
        if generated.path == "src/watcher.py":
            generated.content = generated.content.replace("json.JSONDecodeError", "ValueError")
            generated.content = generated.content.replace('"malformed_records"', '"parse_error"')

    validation = ValidatorStage().validate(artifact, plan_output, spec)

    assert validation.passed is False
    assert "semantic_content_mismatch" in validation.failure_signatures
    check = validation.layer2_result.evidence["requirement_semantic_checks"][
        "requirements"
    ][operational_atom.requirement_id]
    assert check["source_evidence_required"] is True
    assert "malformed_records" in check["missing_source_terms"]


def test_semantic_gate_recognizes_executed_cli_and_aggregation_apis():
    test_corpus = (
        "result = pipeline.main([str(input_path), str(quarantine_path), str(summary_path)])\n"
        "summary = pipeline.compute_summary(device_temps)\n"
        "assert summary[0]['average_temperature_c'] == 15.0\n"
    )
    source_corpus = (
        "def compute_summary(device_temps):\n"
        "    return {'minimum': min(device_temps), 'maximum': max(device_temps)}\n"
    )

    assert ObligationValidationLayer._semantic_term_present(
        "cli_entrypoint",
        test_corpus,
        is_test=True,
    )
    assert ObligationValidationLayer._semantic_term_present(
        "cli_flow",
        test_corpus,
        is_test=True,
    )
    assert ObligationValidationLayer._semantic_term_present("aggregation", source_corpus)
    assert ObligationValidationLayer._semantic_term_present(
        "aggregation",
        test_corpus,
        is_test=True,
    )
