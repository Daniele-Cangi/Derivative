from pathlib import Path

from core.forge.coder_stage import CoderStage
from core.forge.contracts import FeasiblePlan
from core.forge.planner_stage import PlannerStage
from core.forge.requirement_compiler import RequirementCompiler
from core.forge.validator_stage import ValidatorStage
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
