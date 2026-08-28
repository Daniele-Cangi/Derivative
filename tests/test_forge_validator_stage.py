import copy
from dataclasses import replace
from pathlib import Path

import pytest

import core.forge.validator_stage as validator_stage_module
from core.forge.coder_stage import CoderStage
from core.forge.contracts import (
    CodeArtifact,
    FeasiblePlan,
    GeneratedFile,
    PlanInterface,
    ValidationArtifact,
)
from core.forge.planner_stage import PlannerStage
from core.forge.requirement_compiler import RequirementCompiler
from core.forge.semantic_contracts import (
    behaviorally_evidences,
    has_end_to_end_file_workflow_test,
    structurally_evidences,
)
from core.forge.validator_stage import ValidatorStage
from core.forge.validation.adversarial import AdversarialValidationLayer
from core.forge.validation.adapter_capabilities import AdapterCapabilityContractChecker
from core.forge.validation.capabilities import CapabilityContractChecker
from core.forge.validation.obligations import ObligationValidationLayer
from core.forge.validation.runtime import RuntimeValidationLayer


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

PIPELINE_REQUIREMENT = (
    "Build a production-grade Python data pipeline that reads CSV files from a watched directory, "
    "validates each row against a configurable schema, persists valid records to SQLite with full audit trail, "
    "rejects and quarantines invalid rows with structured error logging, and exposes a REST health endpoint "
    "showing pipeline statistics."
)

JSON_MERGE_REQUIREMENT = (
    "Build a Python CLI whose main(argv) merges a base JSON object with an override JSON object "
    "recursively, writes the merged JSON to an output path, replaces lists instead of concatenating "
    "them, rejects a non-object root, and includes tests."
)

JSON_ARRAY_SORT_REQUIREMENT = (
    "Build a Python CLI whose main(argv) reads a JSON array of objects containing unique string "
    "id and numeric score fields, stably orders records by descending score and then ascending id, "
    "writes the ordered array as JSON to an output path, rejects duplicate ids or malformed records "
    "with ValueError, and includes behavioral tests."
)

SENSOR_PIPELINE_REQUIREMENT = (
    "Build a Python JSON Lines data pipeline exposing run(input_path: str, output_path: str) -> int. "
    "Each valid event contains a non-empty sensor_id and a numeric value. Skip malformed events, "
    "write JSON containing valid_count, malformed_count, and a sensors object with count, min, max, "
    "and mean for each sensor, return 0 on success, produce deterministic keys, and include "
    "end-to-end tests."
)


def test_cli_entrypoint_is_structural_evidence_without_framework_tokens():
    spec = RequirementCompiler().compile(
        "Build a Python CLI utility that reads one value and includes tests."
    )
    interfaces = [PlanInterface(name="main", interface_type="cli_entrypoint")]

    assert structurally_evidences(
        "cli_entrypoint",
        "def main(argv=None):\n    return 0\n",
        interfaces,
    )
    assert spec.requirement_atoms[0].evidence_terms == ["cli_entrypoint"]


def test_forbidden_interfaces_require_structural_absence_and_behavioral_assertions():
    interfaces = [PlanInterface(name="rotate_fields", interface_type="function")]
    library_source = "def rotate_fields(rows, field_order, shift=1):\n    return list(rows)\n"

    assert structurally_evidences("no_cli_entrypoint", library_source, interfaces)
    assert structurally_evidences("no_service_interface", library_source, interfaces)
    assert not structurally_evidences(
        "no_cli_entrypoint",
        library_source + "\ndef main(argv=None):\n    return 0\n",
        interfaces,
    )
    assert not structurally_evidences(
        "no_service_interface",
        "from fastapi import FastAPI\napp = FastAPI()\n",
        interfaces,
    )
    assert behaviorally_evidences(
        "no_cli_entrypoint",
        "import rowrotate\n\ndef test_public_surface():\n    assert not hasattr(rowrotate, 'main')\n",
        {"rotate_fields"},
    )
    assert not behaviorally_evidences(
        "no_cli_entrypoint",
        "def test_label_only():\n    no_cli_entrypoint = True\n    assert no_cli_entrypoint\n",
        {"rotate_fields"},
    )


def test_missing_field_none_and_iterated_forbidden_symbols_are_executable_evidence():
    interfaces = [PlanInterface(name="rotate_fields", interface_type="function")]
    source = (
        "def rotate_fields(rows, field_order):\n"
        "    return [{field: row.get(field, None) for field in field_order} for row in rows]\n"
    )
    test_source = '''import rowrotate


def test_missing_and_public_surface():
    result = rowrotate.rotate_fields([{}], ["missing"])
    assert result[0]["missing"] is None
    forbidden = ["main", "cli", "app", "router"]
    for symbol in forbidden:
        assert not hasattr(rowrotate, symbol)
'''

    assert structurally_evidences("missing_fields", source, interfaces)
    assert behaviorally_evidences("missing_fields", test_source, {"rotate_fields"})
    assert behaviorally_evidences("no_cli_entrypoint", test_source, {"rotate_fields"})
    assert behaviorally_evidences("no_service_interface", test_source, {"rotate_fields"})


def test_custom_cli_requires_candidate_generation_capability(tmp_path):
    spec = RequirementCompiler().compile(
        "Build a Python CLI utility that reads newline-delimited words from standard input, "
        "emits the first repeated word, and includes behavioral tests."
    )
    plan = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    ).plan(spec)

    assert isinstance(plan, FeasiblePlan)
    required = AdapterCapabilityContractChecker().required_capabilities(plan)
    assert "candidate_generation_required" in required


def test_declared_public_module_is_checked_as_interface_contract(tmp_path):
    spec = RequirementCompiler().compile(
        "Create a codec module exposing def encode_stream(stream: bytes) -> str "
        "that returns a deterministic digest and includes tests."
    )
    plan = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    ).plan(spec)
    assert isinstance(plan, FeasiblePlan)
    wrong_module = tmp_path / "other.py"
    wrong_module.write_text("def encode_stream(stream):\n    return 'digest'\n", encoding="utf-8")

    mismatches = AdversarialValidationLayer()._detect_interface_contract_mismatches(
        plan,
        {"src/other.py": wrong_module},
    )

    assert "src/codec.py:encode_stream:missing_public_module" in mismatches


def test_cli_public_import_contract_rejects_main_exported_from_wrong_module(tmp_path):
    spec = RequirementCompiler().compile(
        "Implement a CLI and importable function with signature main(argv: list[str] | None = None) -> int. "
        "Public import contract: from forge_blind_v7.cli_dedupe_adjacent import main."
    )
    plan = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    ).plan(spec)
    assert isinstance(plan, FeasiblePlan)
    wrong_module = tmp_path / "cli.py"
    wrong_module.write_text("def main(argv=None):\n    return 0\n", encoding="utf-8")

    mismatches = AdversarialValidationLayer()._detect_interface_contract_mismatches(
        plan,
        {"src/cli.py": wrong_module},
    )

    expected = "src/forge_blind_v7/cli_dedupe_adjacent.py:main:missing_public_module"
    assert expected in mismatches

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


@pytest.fixture(scope="module")
def pipeline_forge_pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("forge_validator_pipeline")
    spec = RequirementCompiler().compile(PIPELINE_REQUIREMENT)
    planner = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(root / "forge_audit.json"),
        memory_file=str(root / "forge_memory.json"),
        gene_pool_file=str(root / "forge_gene_pool.json"),
    )
    plan = planner.plan(spec)
    assert isinstance(plan, FeasiblePlan)
    artifact = CoderStage().generate(plan)
    return {
        "build_spec": spec,
        "plan": plan,
        "artifact": artifact,
        "validator": ValidatorStage(),
    }


@pytest.fixture(scope="module")
def production_service_forge_pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("forge_validator_service_capabilities")
    spec = RequirementCompiler().compile(PRODUCTION_SERVICE_REQUIREMENT)
    planner = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(root / "forge_audit.json"),
        memory_file=str(root / "forge_memory.json"),
        gene_pool_file=str(root / "forge_gene_pool.json"),
    )
    plan = planner.plan(spec)
    assert isinstance(plan, FeasiblePlan)
    artifact = CoderStage().generate(plan)
    return {"build_spec": spec, "plan": plan, "artifact": artifact}


def _find_file(artifact: CodeArtifact, path: str):
    for generated in artifact.files:
        if generated.path == path:
            return generated
    return None


def _materialize_artifact(artifact: CodeArtifact, root: Path):
    materialized = {}
    for generated in artifact.files:
        target = root / generated.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(generated.content, encoding="utf-8")
        materialized[generated.path] = target
    return materialized


def test_validator_stage_is_a_thin_composition_root():
    validator = ValidatorStage()

    assert isinstance(validator.runtime_layer, RuntimeValidationLayer)
    assert isinstance(validator.obligation_layer, ObligationValidationLayer)
    assert isinstance(validator.adversarial_layer, AdversarialValidationLayer)


def test_validator_workspace_cleanup_errors_cannot_abort_validation(
    monkeypatch,
    forge_pipeline,
):
    original_temporary_directory = validator_stage_module.tempfile.TemporaryDirectory
    calls = []

    def tracked_temporary_directory(*args, **kwargs):
        calls.append(dict(kwargs))
        return original_temporary_directory(*args, **kwargs)

    monkeypatch.setattr(
        validator_stage_module.tempfile,
        "TemporaryDirectory",
        tracked_temporary_directory,
    )

    result = forge_pipeline["validator"].validate(
        forge_pipeline["artifact"],
        forge_pipeline["plan"],
        forge_pipeline["build_spec"],
    )

    assert result.passed is True
    validator_workspace_calls = [
        call for call in calls if call.get("prefix") == "forge_validator_"
    ]
    assert validator_workspace_calls
    assert validator_workspace_calls[0]["ignore_cleanup_errors"] is True


def test_capability_contract_matches_composed_service(production_service_forge_pipeline, tmp_path):
    checker = CapabilityContractChecker()
    artifact = production_service_forge_pipeline["artifact"]
    plan = production_service_forge_pipeline["plan"]
    materialized = _materialize_artifact(artifact, tmp_path)

    failures, signatures, evidence = checker.check(artifact, plan, materialized)

    assert failures == []
    assert signatures == []
    assert evidence["passed"] is True
    assert set(evidence["capabilities"]) == {
        "cap_service_api",
        "cap_domain",
        "cap_storage",
        "cap_auth",
        "cap_rate_limit",
        "cap_audit",
        "cap_observability",
    }


def test_csv_adapter_capabilities_match_requirement_contract(forge_pipeline):
    checker = AdapterCapabilityContractChecker()

    failures, signatures, evidence = checker.check(
        forge_pipeline["artifact"],
        forge_pipeline["plan"],
    )

    assert failures == []
    assert signatures == []
    assert evidence["passed"] is True
    assert evidence["selected_adapter"] == "cli"
    assert "csv_input" in evidence["provided_capabilities"]
    assert "expiration_flagging" in evidence["required_capabilities"]


def test_json_merge_cli_is_certified_by_json_merge_profile(tmp_path):
    spec = RequirementCompiler().compile(JSON_MERGE_REQUIREMENT)
    planner = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "forge_audit.json"),
        memory_file=str(tmp_path / "forge_memory.json"),
        gene_pool_file=str(tmp_path / "forge_gene_pool.json"),
    )
    plan = planner.plan(spec)
    assert isinstance(plan, FeasiblePlan)
    artifact = CoderStage().generate(plan)

    result = ValidatorStage().validate(artifact, plan, spec)

    assert result.passed is True
    assert result.failure_signatures == []
    checks = result.layer2_result.evidence["adapter_capability_checks"]
    assert checks["selected_adapter"] == "cli"
    assert {
        "recursive_json_merge",
        "json_list_replacement",
        "json_object_root_validation",
    }.issubset(set(checks["provided_capabilities"]))
    assert checks["missing_capabilities"] == []
    cli_source = next(
        generated.content
        for generated in artifact.files
        if generated.path == "src/cli.py"
    )
    assert "contracts_csv" not in cli_source


def test_adapter_capability_manifest_cannot_self_certify(forge_pipeline):
    artifact = copy.deepcopy(forge_pipeline["artifact"])
    artifact.artifact_manifest["metadata"]["adapter_capabilities"].append(
        "recursive_json_merge"
    )

    failures, signatures, evidence = AdapterCapabilityContractChecker().check(
        artifact,
        forge_pipeline["plan"],
    )

    assert failures
    assert "adapter_capability_manifest_mismatch" in signatures
    assert evidence["unexpected_declared_capabilities"] == ["recursive_json_merge"]


def test_missing_capability_module_fails_closed(production_service_forge_pipeline, tmp_path):
    checker = CapabilityContractChecker()
    artifact = copy.deepcopy(production_service_forge_pipeline["artifact"])
    plan = production_service_forge_pipeline["plan"]
    artifact.files = [generated for generated in artifact.files if generated.path != "src/auth.py"]
    materialized = _materialize_artifact(artifact, tmp_path)

    failures, signatures, evidence = checker.check(artifact, plan, materialized)

    assert failures
    assert "missing_capability" in signatures
    assert evidence["capabilities"]["cap_auth"]["file_exists"] is False


def test_capability_provenance_mismatch_fails_closed(production_service_forge_pipeline, tmp_path):
    checker = CapabilityContractChecker()
    artifact = copy.deepcopy(production_service_forge_pipeline["artifact"])
    plan = production_service_forge_pipeline["plan"]
    auth_file = _find_file(artifact, "src/auth.py")
    assert auth_file is not None
    auth_file.generated_from_plan_sections = [
        token
        for token in auth_file.generated_from_plan_sections
        if token != "capability:cap_auth"
    ]
    materialized = _materialize_artifact(artifact, tmp_path)

    failures, signatures, evidence = checker.check(artifact, plan, materialized)

    assert failures
    assert "capability_contract_violation" in signatures
    assert evidence["capabilities"]["cap_auth"]["provenance_matches"] is False


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


def test_universal_category_fails_closed_even_when_strength_is_hard(forge_pipeline):
    validator: ValidatorStage = forge_pipeline["validator"]
    artifact: CodeArtifact = copy.deepcopy(forge_pipeline["artifact"])
    plan: FeasiblePlan = forge_pipeline["plan"]
    build_spec = copy.deepcopy(forge_pipeline["build_spec"])
    atom = next(item for item in build_spec.requirement_atoms if item.category != "ambiguity")
    atom.category = "universal_constraint"
    atom.strength = "hard"

    result = validator.validate(artifact, plan, build_spec)

    assert result.passed is False
    assert "universal_constraint_unproven" in result.failure_signatures


def test_callability_and_return_type_only_do_not_count_as_semantic_coverage(forge_pipeline):
    validator: ValidatorStage = forge_pipeline["validator"]
    artifact: CodeArtifact = copy.deepcopy(forge_pipeline["artifact"])
    plan: FeasiblePlan = forge_pipeline["plan"]
    build_spec = forge_pipeline["build_spec"]
    required_path = f"tests/{plan.required_tests[0].test_name}.py"
    generated = _find_file(artifact, required_path)
    assert generated is not None
    generated.content = (
        "def test_invokes_target_but_proves_no_behavior():\n"
        "    target = lambda: 0\n"
        "    assert callable(target)\n"
        "    result = target()\n"
        "    assert isinstance(result, int)\n"
    )

    result = validator.validate(artifact, plan, build_spec)

    assert result.passed is False
    assert "non_semantic_test" in result.failure_signatures
    assert "fake_acceptance_coverage" in result.failure_signatures
    assert required_path in result.layer3_result.evidence["non_semantic_tests"]


def test_tautological_assertion_does_not_count_as_acceptance_coverage(forge_pipeline):
    validator: ValidatorStage = forge_pipeline["validator"]
    artifact: CodeArtifact = copy.deepcopy(forge_pipeline["artifact"])
    plan: FeasiblePlan = forge_pipeline["plan"]
    build_spec = forge_pipeline["build_spec"]
    required_path = f"tests/{plan.required_tests[0].test_name}.py"
    generated = _find_file(artifact, required_path)
    assert generated is not None
    generated.content = (
        "def test_exit_status_contract():\n"
        "    target = lambda: 0\n"
        "    result = target()\n"
        "    assert isinstance(result, int)\n"
        "    assert result == 0 or result != 0\n"
    )

    result = validator.validate(artifact, plan, build_spec)

    assert result.passed is False
    assert "non_semantic_test" in result.failure_signatures
    assert "fake_acceptance_coverage" in result.failure_signatures
    assert required_path in result.layer3_result.evidence["non_semantic_tests"]


def test_disconnected_assertion_does_not_count_as_acceptance_coverage(forge_pipeline):
    validator: ValidatorStage = forge_pipeline["validator"]
    artifact: CodeArtifact = copy.deepcopy(forge_pipeline["artifact"])
    plan: FeasiblePlan = forge_pipeline["plan"]
    build_spec = forge_pipeline["build_spec"]
    required_path = f"tests/{plan.required_tests[0].test_name}.py"
    generated = _find_file(artifact, required_path)
    assert generated is not None
    generated.content = (
        "import cli\n"
        "\n"
        "def test_disconnected_contract(monkeypatch):\n"
        "    monkeypatch.setattr(cli, 'main', lambda argv: 0)\n"
        "    cli.main([])\n"
        "    unrelated = 2\n"
        "    assert unrelated == 2\n"
    )

    result = validator.validate(artifact, plan, build_spec)

    assert result.passed is False
    assert "non_semantic_test" in result.failure_signatures
    assert "fake_acceptance_coverage" in result.failure_signatures
    assert "disconnected_assertion" in result.failure_signatures
    assert result.layer3_result.evidence["non_semantic_test_reasons"][required_path] == [
        "disconnected_assertion"
    ]


def test_requirement_term_must_share_function_with_causal_assertion(forge_pipeline):
    validator: ValidatorStage = forge_pipeline["validator"]
    artifact: CodeArtifact = copy.deepcopy(forge_pipeline["artifact"])
    plan: FeasiblePlan = forge_pipeline["plan"]
    build_spec = forge_pipeline["build_spec"]
    atom = next(
        item
        for item in build_spec.requirement_atoms
        if "input_csv" in item.evidence_terms
    )
    required_path = f"tests/{plan.requirement_coverage[atom.requirement_id]['tests'][0]}.py"
    generated = _find_file(artifact, required_path)
    assert generated is not None
    generated.content = (
        "import cli\n"
        "\n"
        "def test_input_csv_label_only():\n"
        "    input_csv = 'fixture.csv'\n"
        "    assert input_csv.endswith('.csv')\n"
        "\n"
        "def test_unrelated_cli_behavior(monkeypatch):\n"
        "    monkeypatch.setattr(cli, 'main', lambda argv: 0)\n"
        "    result = cli.main([])\n"
        "    assert result == 0\n"
    )

    result = validator.validate(artifact, plan, build_spec)

    assert result.passed is False
    assert "missing_requirement_assertion_evidence" in result.failure_signatures
    assert "fake_acceptance_coverage" in result.failure_signatures
    layer2 = result.layer2_result.evidence["requirement_semantic_checks"]
    assertion_evidence = layer2["requirements"][atom.requirement_id][
        "assertion_evidence"
    ]
    assert assertion_evidence["passed"] is False
    assert assertion_evidence["missing_terms"] == ["input_csv"]
    assert assertion_evidence["failure_reason"] == "missing_requirement_assertion_evidence"


def test_every_declared_test_path_is_attacked_for_semantic_content(forge_pipeline):
    validator: ValidatorStage = forge_pipeline["validator"]
    artifact: CodeArtifact = copy.deepcopy(forge_pipeline["artifact"])
    plan: FeasiblePlan = forge_pipeline["plan"]
    build_spec = forge_pipeline["build_spec"]
    required_paths = {f"tests/{test.test_name}.py" for test in plan.required_tests if test.required}
    extra_path = next(path for path in artifact.test_paths if path not in required_paths)
    generated = _find_file(artifact, extra_path)
    assert generated is not None
    generated.content = "def helper_only():\n    return 0\n"

    result = validator.validate(artifact, plan, build_spec)

    assert result.passed is False
    assert "non_semantic_test" in result.failure_signatures
    assert extra_path in result.layer3_result.evidence["non_semantic_tests"]


def test_every_declared_test_path_is_executed(forge_pipeline):
    artifact = copy.deepcopy(forge_pipeline["artifact"])
    planned_test = next(item for item in artifact.files if item.path == "tests/test_cli_flow.py")
    planned_test.content = "def test_planned_regression():\n    assert False\n"

    result = forge_pipeline["validator"].validate(
        artifact,
        forge_pipeline["plan"],
        forge_pipeline["build_spec"],
    )

    assert result.passed is False
    assert "test_execution_failure" in result.failure_signatures
    executed = result.layer2_result.evidence["test_execution"]["tests"]
    assert "tests/test_cli_flow.py" in executed


def test_validator_uses_invoice_smoke_input_for_invoice_specs():
    validator = ValidatorStage()
    invoice_spec = RequirementCompiler().compile(INVOICE_REQUIREMENT)

    sample = validator._sample_input_csv_content(invoice_spec)

    assert "invoice_id,due_date,amount,customer_name" in sample
    assert "INV-1,2026-01-15,100.00,Acme" in sample


def test_runtime_smoke_uses_json_array_contract_for_json_cli(tmp_path):
    build_spec = RequirementCompiler().compile(JSON_ARRAY_SORT_REQUIREMENT)
    plan = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    ).plan(build_spec)
    assert isinstance(plan, FeasiblePlan)
    artifact = CoderStage().generate(plan)
    cli_file = _find_file(artifact, "src/cli.py")
    assert cli_file is not None
    cli_file.content = '''import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json")
    parser.add_argument("output_json")
    args = parser.parse_args(argv)
    records = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    ordered = sorted(records, key=lambda row: (-row["score"], row["id"]))
    Path(args.output_json).write_text(json.dumps(ordered), encoding="utf-8")
    return 0
'''
    materialized = _materialize_artifact(artifact, tmp_path / "workspace")
    layer = RuntimeValidationLayer("python", timeout_seconds=30)

    result = layer.validate(
        artifact,
        plan,
        build_spec,
        materialized,
        tmp_path / "workspace",
    )

    assert result.passed is True
    entrypoint = result.evidence["entrypoint_results"]["src/cli.py"]
    assert entrypoint["executed"] is True
    assert entrypoint["smoke_contract"]["input_format"] == "json"
    assert entrypoint["smoke_contract"]["output_format"] == "json"


def test_runtime_smoke_invokes_run_from_implemented_signature(tmp_path):
    build_spec = RequirementCompiler().compile(SENSOR_PIPELINE_REQUIREMENT)
    plan = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    ).plan(build_spec)
    assert isinstance(plan, FeasiblePlan)
    artifact = CoderStage().generate(plan)
    pipeline_file = _find_file(artifact, "src/pipeline.py")
    assert pipeline_file is not None
    pipeline_file.content = '''import json
from pathlib import Path


def run(input_path: str, output_path: str) -> int:
    valid = [json.loads(line) for line in Path(input_path).read_text(encoding="utf-8").splitlines()]
    Path(output_path).write_text(json.dumps({"valid_count": len(valid)}), encoding="utf-8")
    return 0
'''
    materialized = _materialize_artifact(artifact, tmp_path / "workspace")
    layer = RuntimeValidationLayer("python", timeout_seconds=30)

    result = layer.validate(
        artifact,
        plan,
        build_spec,
        materialized,
        tmp_path / "workspace",
    )

    assert result.passed is True
    entrypoint = result.evidence["entrypoint_results"]["src/pipeline.py"]
    assert entrypoint["executed"] is True
    assert entrypoint["smoke_contract"]["argument_count"] == 2
    assert entrypoint["smoke_contract"]["output_format"] == "json"


def test_end_to_end_quality_evidence_is_behavioral_not_name_based():
    content = '''def test_complete_flow(tmp_path):
    source = tmp_path / "events.jsonl"
    output = tmp_path / "summary.json"
    source.write_text('{"sensor_id":"a","value":1}\\n', encoding="utf-8")
    assert pipeline.run(str(source), str(output)) == 0
    result = output.read_text(encoding="utf-8")
    assert "sensor_id" in result
'''

    assert has_end_to_end_file_workflow_test(content, {"run"})
    assert not has_end_to_end_file_workflow_test(
        "def test_named_only():\n    assert True\n",
        {"run"},
    )


def test_runtime_module_names_are_package_aware():
    validator = ValidatorStage()

    assert validator.runtime_layer._module_name_for_src_path("src/library/__init__.py") == "library"
    assert validator.runtime_layer._module_name_for_src_path("src/library/core.py") == "library.core"


def test_materially_underspecified_requirement_fails_closed(forge_pipeline):
    build_spec = copy.deepcopy(forge_pipeline["build_spec"])
    plan = copy.deepcopy(forge_pipeline["plan"])
    artifact = copy.deepcopy(forge_pipeline["artifact"])
    build_spec.ambiguity_flags.append("Risk classification criteria are materially unspecified.")
    plan.build_spec = build_spec

    result = forge_pipeline["validator"].validate(artifact, plan, build_spec)

    assert result.passed is False
    assert "underspecified_requirement" in result.failure_signatures
    assert result.layer2_result.evidence["material_ambiguities"]


def test_exact_output_contract_mismatch_fails_closed(forge_pipeline):
    build_spec = copy.deepcopy(forge_pipeline["build_spec"])
    plan = copy.deepcopy(forge_pipeline["plan"])
    artifact = copy.deepcopy(forge_pipeline["artifact"])
    build_spec.normalized_requirement += (
        " If input is invalid, the tool outputs exactly 'error: invalid input' "
        "to stderr and exits with code 1."
    )
    plan.build_spec = build_spec
    entrypoint = _find_file(artifact, artifact.runnable_entrypoints[0])
    assert entrypoint is not None
    entrypoint.content += (
        "\nimport sys\n"
        "def main(argv=None):\n"
        "    if argv == ['invalid']:\n"
        "        sys.stderr.write('error: invalid input\\n')\n"
        "        return 1\n"
        "    return 0\n"
    )

    result = forge_pipeline["validator"].validate(artifact, plan, build_spec)

    assert result.passed is False
    assert "exact_output_mismatch" in result.failure_signatures
    checks = result.layer2_result.evidence["exact_output_contract_checks"]
    assert checks[0]["expected"] == "error: invalid input"
    assert checks[0]["observed"] == ["error: invalid input\n"]


def test_unspecified_seeded_prng_contract_fails_closed_after_compilation(tmp_path):
    spec = RequirementCompiler().compile(
        "Define a CLI command 'random_walk' that accepts integer arguments steps and seed and "
        "outputs a reproducible pseudo-random walk seeded by seed. Include behavioral tests."
    )
    plan = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    ).plan(spec)
    assert isinstance(plan, FeasiblePlan)
    artifact = CoderStage().generate(plan)

    result = ValidatorStage().validate(artifact, plan, spec)

    assert result.passed is False
    assert "underspecified_requirement" in result.failure_signatures
    assert any(
        "pseudo-random algorithm is materially unspecified" in flag.lower()
        for flag in result.layer2_result.evidence["material_ambiguities"]
    )


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
    auth_file = _find_file(artifact, "src/auth.py")
    assert auth_file is not None
    auth_file.content = (
        "import sqlite3\n"
        "from storage import init_db\n"
        "def register_user(username, api_key, db_path='service.db'):\n"
        "    with sqlite3.connect(db_path) as conn:\n"
        "        conn.execute('INSERT OR REPLACE INTO users(username, api_key_hash) VALUES (?, ?)', (username, api_key))\n"
        "        conn.commit()\n"
        "def authenticate(api_key, db_path='service.db'):\n"
        "    return None\n"
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


def test_pipeline_artifact_avoids_cli_import_failure_and_superficial_stub(pipeline_forge_pipeline):
    result = pipeline_forge_pipeline["validator"].validate(
        pipeline_forge_pipeline["artifact"],
        pipeline_forge_pipeline["plan"],
        pipeline_forge_pipeline["build_spec"],
    )

    assert result.passed is True
    assert "import_failure" not in result.failure_signatures
    assert "superficial_stub" not in result.failure_signatures


def test_jsonl_telemetry_artifact_passes_all_validation_layers(tmp_path):
    requirement = (
        "Build a Python CLI that reads JSON Lines telemetry events with fields device_id, "
        "timestamp, and temperature_c, rejects malformed records, missing fields, and invalid "
        "timestamps into a quarantine JSONL file, computes per-device minimum, maximum, and "
        "average temperature, writes a summary CSV, and includes behavioral tests for parsing, "
        "quarantine handling, aggregation, and the complete CLI flow."
    )
    build_spec = RequirementCompiler().compile(requirement)
    plan = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    ).plan(build_spec)
    assert isinstance(plan, FeasiblePlan)
    artifact = CoderStage().generate(plan)

    result = ValidatorStage().validate(artifact, plan, build_spec)

    assert result.passed is True
    assert result.failure_signatures == []
    assert result.layer1_result.passed is True
    assert result.layer2_result.passed is True
    assert result.layer3_result.passed is True
    assert result.evidence["executed_tests"]["returncode"] == 0
    assert result.layer1_result is not None
    entrypoint = result.layer1_result.evidence["entrypoint_results"]["src/pipeline.py"]
    assert entrypoint["executed"] is True
    quality_checks = result.layer2_result.evidence["quality_contract_checks"]
    assert quality_checks["passed"] is True


@pytest.mark.parametrize(
    "requirement",
    [
        (
            "Build a Python REST service module with API-key authentication and SQLite persistence exposing "
            "create_event(api_key: str, event_id: str, payload: dict, db_path: str) -> tuple[int, dict]. "
            "Repeating the same event_id must be idempotent and must not insert a duplicate row. "
            "Invalid keys return 401. Include integration tests."
        ),
        (
            "Build a Python JSON Lines data pipeline exposing "
            "run(input_path: str, quarantine_path: str, summary_json_path: str) -> int. "
            "Each valid sales event has customer_id and amount. Write malformed events to quarantine and "
            "write per-customer transaction_count and total_amount to summary JSON. Include end-to-end tests."
        ),
    ],
)
def test_contract_specific_artifacts_pass_all_validation_layers(tmp_path, requirement):
    build_spec = RequirementCompiler().compile(requirement)
    plan = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    ).plan(build_spec)
    assert isinstance(plan, FeasiblePlan)
    artifact = CoderStage().generate(plan)

    result = ValidatorStage().validate(artifact, plan, build_spec)

    assert result.passed is True
    assert result.failure_signatures == []
    assert result.layer1_result.passed is True
    assert result.layer2_result.passed is True
    assert result.layer3_result.passed is True
    capability_checks = result.layer2_result.evidence["adapter_capability_checks"]
    assert capability_checks["missing_capabilities"] == []


@pytest.mark.parametrize(
    "requirement",
    [
        (
            "Build a Python CLI whose main(argv) reads a JSON Lines application log with level and "
            "message fields, skips malformed lines, and writes a JSON report containing total_valid, "
            "malformed_count, and counts_by_level. Include behavioral tests."
        ),
        (
            "Build a Python CLI whose main(argv) merges a base JSON object with an override JSON object "
            "recursively, writes the merged JSON to an output path, replaces lists instead of "
            "concatenating them, and rejects a non-object root. Include tests."
        ),
    ],
)
def test_json_cli_profiles_pass_runtime_obligations_and_adversarial_layers(tmp_path, requirement):
    build_spec = RequirementCompiler().compile(requirement)
    plan = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    ).plan(build_spec)
    assert isinstance(plan, FeasiblePlan)
    artifact = CoderStage().generate(plan)

    result = ValidatorStage().validate(artifact, plan, build_spec)

    assert result.passed is True
    assert result.failure_signatures == []
    assert result.layer1_result.evidence["entrypoint_results"]["src/cli.py"]["executed"] is True
    assert result.layer2_result.evidence["adapter_capability_checks"]["missing_capabilities"] == []


@pytest.mark.parametrize(
    ("path", "old", "new", "expected_signature"),
    [
        (
            "src/pipeline.py",
            "INSERT INTO audit_events",
            "INSERT INTO removed_audit_events",
            "quality_contract_violation",
        ),
        (
            "src/pipeline.py",
            "@app.get('/health')",
            "@app.get('/status')",
            "quality_contract_violation",
        ),
        (
            "src/quarantine.py",
            "def quarantine_row(",
            "def disabled_quarantine_row(",
            "import_failure",
        ),
        (
            "src/pipeline.py",
            "def run(",
            "def disabled_run(",
            "missing_entrypoint",
        ),
    ],
)
def test_pipeline_mutations_fail_closed(
    pipeline_forge_pipeline,
    path,
    old,
    new,
    expected_signature,
):
    artifact = copy.deepcopy(pipeline_forge_pipeline["artifact"])
    generated = _find_file(artifact, path)
    assert generated is not None
    assert old in generated.content
    generated.content = generated.content.replace(old, new, 1)

    result = pipeline_forge_pipeline["validator"].validate(
        artifact,
        pipeline_forge_pipeline["plan"],
        pipeline_forge_pipeline["build_spec"],
    )

    assert result.passed is False
    assert expected_signature in result.failure_signatures


def test_non_cli_workflow_cannot_be_replaced_by_click_command(forge_pipeline):
    artifact = copy.deepcopy(forge_pipeline["artifact"])
    plan = copy.deepcopy(forge_pipeline["plan"])
    target_interface = next(interface for interface in plan.interfaces if interface.name == "main")
    target_interface.interface_type = "function"
    cli_file = next(item for item in artifact.files if item.path == "src/cli.py")
    cli_file.content = cli_file.content.replace(
        "def main(",
        "@click.command()\ndef main(",
        1,
    )
    cli_file.content = "import click\n" + cli_file.content

    result = forge_pipeline["validator"].validate(
        artifact,
        plan,
        forge_pipeline["build_spec"],
    )

    assert result.passed is False
    assert "interface_contract_mismatch" in result.failure_signatures


def _validate_runtime_cli_source(tmp_path, requirement, source):
    build_spec = RequirementCompiler().compile(requirement)
    plan = FeasiblePlan(
        plan_id="plan-runtime-cli",
        build_spec=build_spec,
        architecture_summary="Python CLI with an importable main entrypoint.",
        interfaces=[
            PlanInterface(
                name="main",
                interface_type="cli_entrypoint",
                signature="main(argv: list[str] | None = None) -> int",
                module_path="cli",
            )
        ],
    )
    artifact = CodeArtifact(
        artifact_id="artifact-runtime-cli",
        plan_id=plan.plan_id,
        files=[GeneratedFile("src/cli.py", source, "python_module")],
        runnable_entrypoints=["src/cli.py"],
    )
    workspace = tmp_path / "workspace"
    materialized = _materialize_artifact(artifact, workspace)
    return RuntimeValidationLayer("python", timeout_seconds=30).validate(
        artifact,
        plan,
        build_spec,
        materialized,
        workspace,
    )


def test_runtime_smoke_respects_optional_single_file_cli_contract(tmp_path):
    result = _validate_runtime_cli_source(
        tmp_path,
        (
            "Build a Python CLI exposing main(argv) that reads text from exactly one positional "
            "filename or from stdin when omitted, writes transformed text to stdout, "
            "and includes tests."
        ),
        '''import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="?")
    args = parser.parse_args(argv)
    if args.filename:
        print(Path(args.filename).read_text(encoding="utf-8"), end="")
    return 0
''',
    )

    assert result.passed is True
    entrypoint = result.evidence["entrypoint_results"]["src/cli.py"]
    smoke = entrypoint["smoke_contract"]
    assert smoke["source"] == "argparse_ast"
    assert smoke["cli_arguments"] == ["validator_input.csv"]
    assert smoke["cli_argument_count"] == 1
    assert "validator_output.csv" not in smoke["cli_arguments"]


def test_runtime_smoke_preserves_explicit_input_output_cli_contract(tmp_path):
    result = _validate_runtime_cli_source(
        tmp_path,
        (
            "Build a Python CLI exposing main(argv) that reads an input CSV path and writes an "
            "output CSV path, and includes tests."
        ),
        '''import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    args = parser.parse_args(argv)
    Path(args.output_path).write_text(
        Path(args.input_path).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return 0
''',
    )

    assert result.passed is True
    smoke = result.evidence["entrypoint_results"]["src/cli.py"]["smoke_contract"]
    assert smoke["cli_arguments"] == [
        "validator_input.csv",
        "validator_output.csv",
    ]
    assert smoke["cli_argument_count"] == 2


def test_runtime_entrypoint_exception_is_not_mislabeled_as_import_failure(tmp_path):
    result = _validate_runtime_cli_source(
        tmp_path,
        "Build a Python CLI exposing main(argv) that includes tests.",
        '''def main(argv=None):
    raise RuntimeError("workflow failed")
''',
    )

    assert result.passed is False
    assert result.evidence["failure_signatures"] == [
        "entrypoint_execution_failure"
    ]
    entrypoint = result.evidence["entrypoint_results"]["src/cli.py"]
    assert entrypoint["failure_phase"] == "execution"
    assert entrypoint["execution_status"]["error_type"] == "RuntimeError"


def test_runtime_real_import_error_retains_import_failure_signature(tmp_path):
    result = _validate_runtime_cli_source(
        tmp_path,
        "Build a Python CLI exposing main(argv) that includes tests.",
        '''import dependency_that_does_not_exist


def main(argv=None):
    return 0
''',
    )

    assert result.passed is False
    assert "import_failure" in result.evidence["failure_signatures"]
    assert "entrypoint_execution_failure" not in result.evidence["failure_signatures"]
    entrypoint = result.evidence["entrypoint_results"]["src/cli.py"]
    assert entrypoint["failure_phase"] == "import"
    assert entrypoint["execution_status"]["error_type"] == "ModuleNotFoundError"


def test_runtime_clean_system_exit_counts_as_completed_execution(tmp_path):
    result = _validate_runtime_cli_source(
        tmp_path,
        "Build a Python CLI exposing main(argv) that includes tests.",
        '''import sys


def main(argv=None):
    sys.exit(0)
''',
    )

    assert result.passed is True
    entrypoint = result.evidence["entrypoint_results"]["src/cli.py"]
    assert entrypoint["executed"] is True
    assert entrypoint["failure_phase"] == ""
    assert entrypoint["execution_status"] == {
        "phase": "completed",
        "result": 0,
    }


def test_runtime_nonzero_system_exit_remains_execution_failure(tmp_path):
    result = _validate_runtime_cli_source(
        tmp_path,
        "Build a Python CLI exposing main(argv) that includes tests.",
        '''raise_code = 2


def main(argv=None):
    raise SystemExit(raise_code)
''',
    )

    assert result.passed is False
    assert result.evidence["failure_signatures"] == [
        "entrypoint_execution_failure"
    ]
    entrypoint = result.evidence["entrypoint_results"]["src/cli.py"]
    assert entrypoint["failure_phase"] == "execution"
    assert entrypoint["execution_status"]["error_type"] == "SystemExit"


def test_runtime_smoke_materializes_declared_directory_arguments(tmp_path):
    result = _validate_runtime_cli_source(
        tmp_path,
        (
            "Build a Python CLI exposing main(argv) that processes an input directory "
            "and writes results to an output directory, and includes tests."
        ),
        '''import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args(argv)
    assert Path(args.input_dir).is_dir()
    assert Path(args.output_dir).is_dir()
    return 0
''',
    )

    assert result.passed is True
    smoke = result.evidence["entrypoint_results"]["src/cli.py"]["smoke_contract"]
    assert smoke["cli_arguments"] == [
        "validator_input_dir",
        "validator_output_dir",
    ]
    assert smoke["cli_argument_count"] == 2
