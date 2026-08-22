import copy
import json
from pathlib import Path

import pytest

from core.forge.coder_stage import CoderStage
from core.forge.contracts import (
    AcceptanceContract,
    ArtifactTargetType,
    BuildSpec,
    CodeArtifact,
    FeasiblePlan,
    ForgeRoute,
    GeneratedFile,
    ObligationContract,
    PlanFile,
    PlanInterface,
    PlanTest,
    ValidationArtifact,
    ValidationStrategy,
)
from core.forge.packaging_stage import PackagingStage
from core.forge.planner_stage import PlannerStage
from core.forge.repair import RepairPolicy
from core.forge.requirement_compiler import RequirementCompiler
from core.forge.validator_stage import ValidatorStage
from forge import TERMINAL_VERIFIED, run_forge
from forge import TERMINAL_VALIDATION_FAILED


REQUIREMENT = (
    "Build a Python CLI that reads a CSV of contracts, extracts expiration dates, "
    "flags contracts expiring in less than 90 days, writes a summary CSV, and includes tests."
)


@pytest.fixture(scope="module")
def repair_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("forge_repair")
    spec = RequirementCompiler().compile(REQUIREMENT)
    planner = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(root / "audit.json"),
        memory_file=str(root / "memory.json"),
        gene_pool_file=str(root / "genes.json"),
    )
    plan = planner.plan(spec)
    assert isinstance(plan, FeasiblePlan)
    coder = CoderStage()
    return {
        "spec": spec,
        "plan": plan,
        "coder": coder,
        "artifact": coder.generate(plan),
        "validator": ValidatorStage(),
    }


def _file(artifact, path):
    return next(generated for generated in artifact.files if generated.path == path)


def test_import_failure_targets_package_source_paths(repair_case):
    artifact = CodeArtifact(
        artifact_id="code-library",
        plan_id=repair_case["plan"].plan_id,
        files=[
            GeneratedFile("src/library/__init__.py", "", "python_module"),
            GeneratedFile("src/library/core.py", "", "python_module"),
        ],
    )
    validation = ValidationArtifact(
        passed=False,
        failures=["Import failed."],
        failure_signatures=["import_failure"],
        evidence={
            "layer1": {
                "import_results": {
                    "modules": {
                        "library": {"ok": False},
                        "library.core": {"ok": False},
                    }
                }
            }
        },
    )

    directive = RepairPolicy().compile(
        validation,
        repair_case["plan"],
        artifact,
        attempt=2,
    )

    assert set(directive.target_paths) == {
        "src/library/__init__.py",
        "src/library/core.py",
    }


def test_validator_evidence_drives_a_real_repair(repair_case):
    artifact = copy.deepcopy(repair_case["artifact"])
    _file(artifact, "src/expiration_rules.py").content = "def parse_expiration_date(:\n    pass\n"
    validation = repair_case["validator"].validate(
        artifact,
        repair_case["plan"],
        repair_case["spec"],
    )

    assert validation.passed is False
    assert "syntax_error" in validation.failure_signatures
    directive = RepairPolicy().compile(
        validation,
        repair_case["plan"],
        artifact,
        attempt=2,
    )
    result = repair_case["coder"].repair(
        repair_case["plan"],
        artifact,
        validation,
        directive,
    )

    assert directive.repairable is True
    assert "src/expiration_rules.py" in directive.target_paths
    assert result.changed is True
    assert "src/expiration_rules.py" in result.changed_paths
    assert result.previous_digest != result.repaired_digest
    assert result.artifact.revision == 2
    assert result.artifact.parent_artifact_id == artifact.artifact_id
    assert result.artifact.repair_history[-1]["repair_id"] == directive.repair_id

    repaired_validation = repair_case["validator"].validate(
        result.artifact,
        repair_case["plan"],
        repair_case["spec"],
    )
    assert repaired_validation.passed is True


def test_canonical_artifact_cannot_fake_a_repair(repair_case):
    validation = ValidationArtifact(
        passed=False,
        failures=["Synthetic test failure."],
        failure_signatures=["test_execution_failure"],
        evidence={"layer2": {"test_execution": {"returncode": 1, "tests": []}}},
    )
    directive = RepairPolicy().compile(
        validation,
        repair_case["plan"],
        repair_case["artifact"],
        attempt=2,
    )
    result = repair_case["coder"].repair(
        repair_case["plan"],
        repair_case["artifact"],
        validation,
        directive,
    )

    assert directive.repairable is True
    assert result.changed is False
    assert result.changed_paths == []
    assert result.previous_digest == result.repaired_digest


def test_non_semantic_repair_targets_only_validator_identified_tests(repair_case):
    artifact = repair_case["artifact"]
    failing_test = artifact.test_paths[0]
    validation = ValidationArtifact(
        passed=False,
        failures=["Non-semantic test."],
        failure_signatures=["non_semantic_test", "fake_acceptance_coverage"],
        evidence={"layer3": {"non_semantic_tests": [failing_test]}},
    )

    directive = RepairPolicy().compile(
        validation,
        repair_case["plan"],
        artifact,
        attempt=2,
    )

    assert directive.target_paths == [failing_test]


def test_adapter_mismatch_compiles_one_complete_artifact_transaction(repair_case):
    artifact = repair_case["artifact"]
    validation = ValidationArtifact(
        passed=False,
        failures=["Selected adapter lacks required capabilities."],
        failure_signatures=["adapter_capability_mismatch"],
        evidence={
            "layer2": {
                "adapter_capability_checks": {
                    "missing_capabilities": ["unimplemented_capability"]
                }
            }
        },
    )

    directive = RepairPolicy().compile(
        validation,
        repair_case["plan"],
        artifact,
        attempt=2,
    )

    assert directive.operations == ["compile_uncovered_capabilities"]
    assert set(directive.target_paths) == {
        generated.path
        for generated in artifact.files
        if generated.path not in artifact.manifest_paths
    }
    assert directive.evidence_refs == ["layer2.adapter_capability_checks"]


def test_candidate_followup_repair_remains_a_complete_transaction(repair_case):
    artifact = copy.deepcopy(repair_case["artifact"])
    artifact.artifact_manifest["metadata"]["generator"] = "forge_candidate_compiler"
    validation = ValidationArtifact(
        passed=False,
        failures=["Mapped test does not invoke the CLI entrypoint."],
        failure_signatures=["missing_semantic_requirement_coverage"],
        evidence={
            "layer2": {
                "requirement_semantic_checks": {
                    "semantic_content_mismatches": [
                        {
                            "requirement_id": "R001",
                            "source_paths": ["src/cli.py"],
                            "test_paths": [artifact.test_paths[0]],
                        }
                    ]
                }
            }
        },
    )

    directive = RepairPolicy().compile(
        validation,
        repair_case["plan"],
        artifact,
        attempt=3,
    )

    assert "recompile_candidate_transaction" in directive.operations
    assert set(directive.target_paths) == {
        generated.path
        for generated in artifact.files
        if generated.path not in artifact.manifest_paths
    }
    assert "artifact_manifest.metadata.candidate_compilation" in directive.evidence_refs
    assert directive.requirement_ids == ["R001"]
    assert "main" in directive.target_symbols


def test_execution_repair_targets_only_pytest_failed_paths(repair_case):
    artifact = repair_case["artifact"]
    failing_test = artifact.test_paths[0]
    passing_test = artifact.test_paths[1]
    validation = ValidationArtifact(
        passed=False,
        failures=["Generated tests failed."],
        failure_signatures=["test_execution_failure"],
        evidence={
            "layer2": {
                "test_execution": {
                    "returncode": 1,
                    "tests": [failing_test, passing_test],
                    "stdout": f"FAILED {failing_test}::test_behavior - AssertionError",
                    "stderr": "",
                }
            }
        },
    )

    directive = RepairPolicy().compile(
        validation,
        repair_case["plan"],
        artifact,
        attempt=3,
    )

    assert directive.target_paths == [failing_test]


def test_revision_two_uses_backend_instead_of_reverting_to_canonical(repair_case):
    artifact = copy.deepcopy(repair_case["artifact"])
    artifact.revision = 2
    artifact.artifact_id = f"{artifact.artifact_id}-r02"
    source = _file(artifact, "src/expiration_rules.py")
    source.content += "\nREPAIRED_BEHAVIOR = True\n"
    failing_test = artifact.test_paths[0]
    validation = ValidationArtifact(
        passed=False,
        failures=["One repaired test still fails."],
        failure_signatures=["test_execution_failure"],
        evidence={
            "layer2": {
                "test_execution": {
                    "returncode": 1,
                    "tests": [failing_test],
                    "stdout": f"FAILED {failing_test}::test_behavior - AssertionError",
                    "stderr": "",
                }
            }
        },
    )
    directive = RepairPolicy().compile(
        validation,
        repair_case["plan"],
        artifact,
        attempt=3,
    )

    result = CoderStage().repair(
        repair_case["plan"],
        artifact,
        validation,
        directive,
    )

    assert result.changed is False
    assert result.artifact is artifact
    assert "REPAIRED_BEHAVIOR = True" in _file(result.artifact, "src/expiration_rules.py").content


def test_semantic_content_mismatch_targets_mapped_sources_and_tests(repair_case):
    artifact = repair_case["artifact"]
    source_path = "src/expiration_rules.py"
    test_path = artifact.test_paths[0]
    validation = ValidationArtifact(
        passed=False,
        failures=["Requirement semantics are absent."],
        failure_signatures=["semantic_omission", "semantic_content_mismatch"],
        evidence={
            "layer2": {
                "requirement_semantic_checks": {
                    "semantic_content_mismatches": [
                        {
                            "requirement_id": "R002",
                            "source_paths": [source_path],
                            "test_paths": [test_path],
                            "missing_source_terms": ["summary_csv"],
                            "missing_test_terms": ["summary_csv"],
                        }
                    ]
                }
            }
        },
    )

    directive = RepairPolicy().compile(
        validation,
        repair_case["plan"],
        artifact,
        attempt=2,
    )

    assert directive.repairable is True
    assert directive.operations == ["implement_missing_requirement_semantics"]
    assert directive.target_paths == [source_path, test_path]
    assert directive.evidence_refs == ["layer2.requirement_semantic_checks"]


def test_requirement_assertion_failure_compiles_exact_test_repair(repair_case):
    artifact = copy.deepcopy(repair_case["artifact"])
    artifact.artifact_manifest["metadata"]["generator"] = "forge_candidate_compiler"
    test_path = artifact.test_paths[0]
    assertion_evidence = {
        "mapped_test_paths": [test_path],
        "existing_test_paths": [test_path],
        "required_terms": ["csv", "contracts"],
        "covered_terms": ["csv"],
        "missing_terms": ["contracts"],
        "causal_functions": [
            {
                "path": test_path,
                "function": "test_reads_contracts_csv",
                "matched_terms": ["csv"],
            }
        ],
        "assertions": [
            {
                "path": test_path,
                "function": "test_reads_contracts_csv",
                "line": 18,
                "kind": "assert",
                "expression": "len(rows) == 1",
                "evidence_terms": ["csv"],
            }
        ],
        "passed": False,
        "failure_reason": "missing_requirement_assertion_evidence",
    }
    validation = ValidationArtifact(
        passed=False,
        failures=["Requirement R001 lacks a contract-specific assertion."],
        failure_signatures=[
            "missing_requirement_assertion_evidence",
            "fake_acceptance_coverage",
        ],
        evidence={
            "layer2": {
                "requirement_semantic_checks": {
                    "requirement_assertion_mismatches": [
                        {
                            "requirement_id": "R001",
                            "test_paths": [test_path],
                            "assertion_evidence": assertion_evidence,
                        }
                    ]
                }
            },
            "layer3": {
                "semantic_requirement_test_coverage": {
                    "missing_requirement_assertion_evidence": ["R001"],
                    "requirements": {
                        "R001": {
                            "mapped_tests": [test_path],
                            "assertion_evidence": assertion_evidence,
                        }
                    },
                }
            },
        },
    )

    directive = RepairPolicy().compile(
        validation,
        repair_case["plan"],
        artifact,
        attempt=2,
    )

    assert directive.operations == [
        "repair_requirement_assertions",
        "rerender_semantic_tests",
    ]
    assert directive.target_paths == [test_path]
    assert "recompile_candidate_transaction" not in directive.operations
    assert directive.requirement_ids == ["R001"]
    assert directive.target_symbols == ["test_reads_contracts_csv"]
    assert directive.evidence_targets["R001"]["missing_terms"] == ["contracts"]
    assert directive.evidence_targets["R001"]["assertions"][0]["line"] == 18
    assert directive.evidence_targets["R001"]["evidence_refs"] == [
        "layer2.requirement_semantic_checks.requirements:R001.assertion_evidence",
        "layer3.semantic_requirement_test_coverage.requirements:R001.assertion_evidence",
    ]


def test_real_validator_assertion_evidence_routes_to_exact_repair(repair_case):
    artifact = copy.deepcopy(repair_case["artifact"])
    plan = repair_case["plan"]
    atom = next(
        item
        for item in repair_case["spec"].requirement_atoms
        if "input_csv" in item.evidence_terms
    )
    test_name = plan.requirement_coverage[atom.requirement_id]["tests"][0]
    test_path = f"tests/{test_name}.py"
    _file(artifact, test_path).content = (
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
    validation = repair_case["validator"].validate(
        artifact,
        plan,
        repair_case["spec"],
    )

    directive = RepairPolicy().compile(validation, plan, artifact, attempt=2)

    assert "missing_requirement_assertion_evidence" in validation.failure_signatures
    assert directive.target_paths == [test_path]
    assert directive.requirement_ids == [atom.requirement_id]
    assert directive.target_symbols == ["test_unrelated_cli_behavior"]
    assert directive.evidence_targets[atom.requirement_id]["missing_terms"] == [
        "input_csv"
    ]


def test_repair_contract_is_domain_neutral():
    build_spec = BuildSpec(
        build_id="build-generic",
        raw_requirement="Build an executable Python component.",
        normalized_requirement="Build an executable Python component.",
        functional_goals=["Execute a workflow."],
        acceptance_contract=AcceptanceContract(),
        obligation_contract=ObligationContract(mode="software_build"),
        target_artifact_type=ArtifactTargetType.UNKNOWN,
    )
    plan = FeasiblePlan(
        plan_id="plan-generic",
        build_spec=build_spec,
        architecture_summary="Generic executable component.",
        file_tree_plan=[PlanFile(path="src/component.py", purpose="Workflow implementation.")],
        interfaces=[PlanInterface(name="run", interface_type="entrypoint")],
        required_tests=[PlanTest(test_name="test_component", objective="Execute the workflow.")],
        validation_strategy=ValidationStrategy(),
        packaging_target="python_package",
    )
    coder = CoderStage()
    artifact = coder.generate(plan)
    _file(artifact, "src/component.py").content = "def run(:\n"
    validation = ValidationArtifact(
        passed=False,
        failures=["Syntax error."],
        failure_signatures=["syntax_error"],
        evidence={"layer1": {"parse_errors": [{"path": "src/component.py"}]}},
    )
    directive = RepairPolicy().compile(validation, plan, artifact, attempt=2)
    result = coder.repair(plan, artifact, validation, directive)

    assert directive.route == ForgeRoute.TO_CODER
    assert result.changed is True
    assert "src/component.py" in result.changed_paths
    assert "def run() -> int:" in _file(result.artifact, "src/component.py").content


class _StaticCompiler:
    def __init__(self, spec):
        self.spec = spec

    def compile(self, requirement):
        return self.spec


class _StaticPlanner:
    def __init__(self, plan):
        self.plan_output = plan

    def plan(self, build_spec):
        return self.plan_output


class _CorruptOnceCoder(CoderStage):
    def __init__(self):
        super().__init__()
        self.generate_calls = 0

    def generate(self, plan):
        self.generate_calls += 1
        artifact = super().generate(plan)
        if self.generate_calls == 1:
            _file(artifact, "src/expiration_rules.py").content = "def broken(:\n"
        return artifact


class _AlwaysFailValidator:
    def __init__(self):
        self.calls = 0

    def validate(self, code_artifact, plan, build_spec):
        self.calls += 1
        return ValidationArtifact(
            passed=False,
            failures=["Synthetic execution failure."],
            failure_signatures=["test_execution_failure"],
            evidence={
                "layer2": {
                    "test_execution": {
                        "returncode": 1,
                        "tests": list(code_artifact.test_paths),
                    }
                }
            },
        )


def test_orchestrator_persists_grounded_repair_trace(repair_case, tmp_path):
    output_root = tmp_path / "runs"
    coder = _CorruptOnceCoder()
    result = run_forge(
        requirement=REQUIREMENT,
        output_root=str(output_root),
        packaging_output_root=str(tmp_path / "packages"),
        requirement_compiler=_StaticCompiler(repair_case["spec"]),
        planner_stage=_StaticPlanner(repair_case["plan"]),
        coder_stage=coder,
        validator_stage=ValidatorStage(),
        packaging_stage=PackagingStage(output_root=str(tmp_path / "packages")),
        max_coder_attempts=2,
    )

    assert result.terminal_status == TERMINAL_VERIFIED
    assert coder.generate_calls == 2
    metadata_paths = list(output_root.glob("*/run_metadata.json"))
    assert len(metadata_paths) == 1
    metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    assert metadata["coder_attempts_used"] == 2
    assert len(metadata["attempt_trace"]) == 2
    repair = metadata["attempt_trace"][1]["repair"]
    assert repair["changed"] is True
    assert "src/expiration_rules.py" in repair["changed_paths"]
    package_root = Path(result.artifact_path)
    assert package_root.exists()
    package_manifest = json.loads(
        (package_root / "forge_package_manifest.json").read_text(encoding="utf-8")
    )
    assert package_manifest["artifact_revision"] == 2
    assert package_manifest["parent_artifact_id"]
    assert len(package_manifest["repair_history"]) == 1


def test_orchestrator_stops_when_repair_changes_nothing(repair_case, tmp_path):
    validator = _AlwaysFailValidator()
    result = run_forge(
        requirement=REQUIREMENT,
        output_root=str(tmp_path / "runs"),
        packaging_output_root=str(tmp_path / "packages"),
        requirement_compiler=_StaticCompiler(repair_case["spec"]),
        planner_stage=_StaticPlanner(repair_case["plan"]),
        coder_stage=CoderStage(),
        validator_stage=validator,
        packaging_stage=PackagingStage(output_root=str(tmp_path / "packages")),
        max_coder_attempts=2,
    )

    assert result.terminal_status == TERMINAL_VALIDATION_FAILED
    assert result.validation is not None
    assert "repair_no_change" in result.validation.failure_signatures
    assert validator.calls == 1
    assert not (tmp_path / "packages").exists()
    metadata_path = next((tmp_path / "runs").glob("*/run_metadata.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["attempt_trace"][1]["repair"]["changed"] is False


def test_entrypoint_execution_failure_targets_interface_implementation(repair_case):
    artifact = repair_case["artifact"]
    validation = ValidationArtifact(
        passed=False,
        failures=["Entrypoint execution failed for src/cli.py."],
        failure_signatures=["entrypoint_execution_failure"],
        evidence={
            "layer1": {
                "entrypoint_results": {
                    "src/cli.py": {
                        "exists": True,
                        "function_present": True,
                        "executed": False,
                        "failure_phase": "execution",
                    }
                }
            }
        },
    )

    directive = RepairPolicy().compile(
        validation,
        repair_case["plan"],
        artifact,
        attempt=2,
    )

    assert directive.repairable is True
    assert directive.operations == ["rerender_interface_implementation"]
    assert directive.target_paths == ["src/cli.py"]
    assert directive.evidence_refs == ["layer1.entrypoint_results:src/cli.py"]
