import copy
from types import SimpleNamespace

import pytest

from core.forge.candidate_compiler import SubstrateCandidateCompiler
from core.forge.candidate_preflight import run_semantic_preflight
from core.forge.coder_stage import CoderStage
from core.forge.contracts import FeasiblePlan
from core.forge.planner_stage import PlannerStage
from core.forge.repair import RepairPolicy
from core.forge.repair_support import test_generation_contracts as build_test_generation_contracts
from core.forge.requirement_compiler import RequirementCompiler
from core.forge.semantic_contracts import (
    has_json_lines_processing,
    interface_parameter_is_exercised,
)
from core.forge.validator_stage import ValidatorStage
from core.forge.validation.adapter_capabilities import AdapterCapabilityContractChecker


JSON_MERGE_REQUIREMENT = (
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


@pytest.fixture(scope="module")
def json_merge_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("candidate_compiler")
    spec = RequirementCompiler().compile(JSON_MERGE_REQUIREMENT)
    plan = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(root / "audit.json"),
        memory_file=str(root / "memory.json"),
        gene_pool_file=str(root / "genes.json"),
    ).plan(spec)
    assert isinstance(plan, FeasiblePlan)
    artifact = CoderStage().generate(plan)
    artifact.artifact_manifest["metadata"]["domain_adapter"] = "generic"
    artifact.artifact_manifest["metadata"]["adapter_capabilities"] = []
    validation = ValidatorStage().validate(artifact, plan, spec)
    assert "adapter_capability_mismatch" in validation.failure_signatures
    return spec, plan, artifact, validation


class _StaticSubstrate:
    def decompose(self, problem):
        assert "complete software candidate" in problem
        return [SimpleNamespace(lens_name="Formal Logic")]


class _JsonMergeKernel:
    use_live_model = True

    def propose_code_revision(self, repair_context, target_files, lens_framings):
        assert repair_context["candidate_transaction_required"] is True
        assert set(repair_context["current_target_paths"]) == set(target_files)
        files = {}
        for path in target_files:
            files[path] = _source() if path == "src/cli.py" else _test_source(path)
        return {"status": "candidate", "files": files}


class _UnavailableKernel:
    use_live_model = False


class _ExtraPathKernel(_JsonMergeKernel):
    def propose_code_revision(self, repair_context, target_files, lens_framings):
        payload = super().propose_code_revision(
            repair_context,
            target_files,
            lens_framings,
        )
        payload["files"]["src/unplanned.py"] = "VALUE = 1\n"
        return payload


class _MonotonicCorrectionKernel(_JsonMergeKernel):
    def __init__(self):
        self.target_history = []
        self.context_history = []

    def propose_code_revision(self, repair_context, target_files, lens_framings):
        self.target_history.append(sorted(target_files))
        self.context_history.append(copy.deepcopy(repair_context))
        return super().propose_code_revision(
            repair_context,
            target_files,
            lens_framings,
        )


def _source():
    return '''import argparse
import json
from pathlib import Path


def validate_object_root(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object root")
    return value


def recursive_json_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = recursive_json_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def replace_lists(base: dict, override: dict) -> dict:
    return recursive_json_merge(base, override)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recursively merge JSON objects")
    parser.add_argument("base_json")
    parser.add_argument("override_json")
    parser.add_argument("output_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = validate_object_root(json.loads(Path(args.base_json).read_text(encoding="utf-8")), "base")
    override = validate_object_root(json.loads(Path(args.override_json).read_text(encoding="utf-8")), "override")
    merged = recursive_json_merge(base, override)
    Path(args.output_json).write_text(json.dumps(merged, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _test_source(path):
    function_name = path.rsplit("/", 1)[-1].removesuffix(".py")
    if "rejects_non_object" in function_name:
        body = '''def test_rejects_non_object_root(tmp_path):
    base = tmp_path / "base.json"
    override = tmp_path / "override.json"
    output = tmp_path / "merged.json"
    base.write_text("[]", encoding="utf-8")
    override.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        cli.main([str(base), str(override), str(output)])
'''
    else:
        body = f'''def {function_name}(tmp_path):
    base = tmp_path / "base.json"
    override = tmp_path / "override.json"
    output = tmp_path / "merged.json"
    base.write_text('{{"db":{{"host":"localhost","ports":[80]}},"tags":["base"]}}', encoding="utf-8")
    override.write_text('{{"db":{{"ports":[443]}},"tags":["override"]}}', encoding="utf-8")
    assert cli.main([str(base), str(override), str(output)]) == 0
    merged = json.loads(output.read_text(encoding="utf-8"))
    assert merged["db"] == {{"host": "localhost", "ports": [443]}}
    assert merged["tags"] == ["override"]
'''
    return f'''import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cli


{body}'''


def _directive(plan, artifact, validation):
    return RepairPolicy().compile(validation, plan, artifact, attempt=2)


def test_json_merge_plan_is_neutral_and_preserves_semantic_contract(json_merge_case):
    spec, plan, _, _ = json_merge_case

    assert {item.path for item in plan.file_tree_plan} == {
        "src/cli.py",
        "tests/test_cli_flow.py",
    }
    assert [interface.name for interface in plan.interfaces] == ["main"]
    evidence_terms = {
        term
        for atom in spec.requirement_atoms
        for term in atom.evidence_terms
    }
    assert {
        "recursive_json_merge",
        "json_list_replacement",
        "json_object_root_validation",
    }.issubset(evidence_terms)
    assert {
        "test_recursive_json_merge",
        "test_replaces_json_lists",
        "test_rejects_non_object_json_root",
    }.issubset({test.test_name for test in plan.required_tests})


def test_candidate_compiler_produces_complete_validated_transaction(json_merge_case):
    spec, plan, artifact, validation = json_merge_case
    compiler = SubstrateCandidateCompiler(
        substrate=_StaticSubstrate(),
        kernel=_JsonMergeKernel(),
        max_preflight_corrections=0,
    )
    coder = CoderStage(candidate_compiler=compiler)

    result = coder.repair(
        plan,
        artifact,
        validation,
        _directive(plan, artifact, validation),
    )

    assert result.changed is True
    assert result.backend_name == "candidate_compiler"
    assert result.backend_evidence["complete_transaction"] is True
    assert result.backend_evidence["preflight_passed"] is True
    metadata = result.artifact.artifact_manifest["metadata"]
    assert metadata["generator"] == "forge_candidate_compiler"
    assert metadata["domain_adapter"] == "candidate"
    assert set(metadata["candidate_compilation"]["transaction_paths"]) == {
        generated.path
        for generated in result.artifact.files
        if generated.path != "forge_artifact_manifest.json"
    }

    validated = ValidatorStage().validate(result.artifact, plan, spec)

    assert validated.passed is True
    capability_evidence = validated.layer2_result.evidence["adapter_capability_checks"]
    assert capability_evidence["compiler_contract_valid"] is True


def test_candidate_manifest_digest_cannot_hide_source_tampering(json_merge_case):
    spec, plan, artifact, validation = json_merge_case
    coder = CoderStage(
        candidate_compiler=SubstrateCandidateCompiler(
            substrate=_StaticSubstrate(),
            kernel=_JsonMergeKernel(),
            max_preflight_corrections=0,
        )
    )
    repaired = coder.repair(plan, artifact, validation, _directive(plan, artifact, validation)).artifact
    corrupted = copy.deepcopy(repaired)
    source = next(item for item in corrupted.files if item.path == "src/cli.py")
    source.content += "\nCORRUPTED = True\n"

    failures, signatures, evidence = AdapterCapabilityContractChecker().check(corrupted, plan)

    assert failures
    assert "adapter_capability_manifest_mismatch" in signatures
    assert evidence["digest_mismatches"] == ["src/cli.py"]


def test_candidate_compiler_rejects_unplanned_files(json_merge_case):
    _, plan, artifact, validation = json_merge_case
    compiler = SubstrateCandidateCompiler(
        substrate=_StaticSubstrate(),
        kernel=_ExtraPathKernel(),
        max_preflight_corrections=0,
    )

    candidate = compiler.propose(
        plan,
        artifact,
        validation,
        _directive(plan, artifact, validation),
    )

    assert candidate.files == {}
    assert candidate.rejected_paths == ["src/unplanned.py"]
    assert candidate.evidence["complete_transaction"] is False


def test_candidate_compiler_fails_closed_without_live_kernel(json_merge_case):
    _, plan, artifact, validation = json_merge_case
    compiler = SubstrateCandidateCompiler(
        substrate=_StaticSubstrate(),
        kernel=_UnavailableKernel(),
    )

    candidate = compiler.propose(
        plan,
        artifact,
        validation,
        _directive(plan, artifact, validation),
    )

    assert candidate.available is False
    assert candidate.files == {}
    assert "unavailable" in candidate.stop_reason.lower()


def test_candidate_correction_preserves_passing_files(json_merge_case):
    _, plan, artifact, validation = json_merge_case
    kernel = _MonotonicCorrectionKernel()
    preflight_calls = []

    def preflight(files, tests):
        preflight_calls.append(dict(files))
        if len(preflight_calls) == 1:
            return {
                "ran": True,
                "passed": False,
                "phase": "semantic_contract",
                "failed_paths": ["tests/test_replaces_json_lists.py"],
                "source_failed_paths": [],
                "test_failed_paths": ["tests/test_replaces_json_lists.py"],
                "failures": [],
                "correction_requirements": ["Strengthen the list replacement assertion."],
            }
        return {
            "ran": True,
            "passed": True,
            "phase": "tests",
            "failed_paths": [],
            "source_failed_paths": [],
            "test_failed_paths": [],
            "failures": [],
        }

    compiler = SubstrateCandidateCompiler(
        substrate=_StaticSubstrate(),
        kernel=kernel,
        max_preflight_corrections=1,
        test_preflight_runner=preflight,
    )

    candidate = compiler.propose(
        plan,
        artifact,
        validation,
        _directive(plan, artifact, validation),
    )

    assert candidate.files
    assert set(kernel.target_history[0]) == set(candidate.files)
    assert kernel.target_history[1] == ["tests/test_replaces_json_lists.py"]
    assert candidate.files["src/cli.py"] == preflight_calls[0]["src/cli.py"]
    second_attempt = candidate.evidence["candidate_attempts"][1]
    assert "src/cli.py" in second_attempt["preserved_paths"]
    preserved = kernel.context_history[1]["preserved_candidate_files"]
    assert preserved["src/cli.py"] == preflight_calls[0]["src/cli.py"]
    assert "immutable context" in kernel.context_history[1]["preservation_contract"]


def test_executable_test_failure_expands_correction_to_imported_source(json_merge_case):
    _, plan, artifact, validation = json_merge_case
    kernel = _MonotonicCorrectionKernel()
    calls = []

    def preflight(files, tests):
        calls.append(dict(files))
        if len(calls) == 1:
            return {
                "ran": True,
                "passed": False,
                "phase": "tests",
                "failed_paths": ["tests/test_replaces_json_lists.py"],
                "source_failed_paths": [],
                "test_failed_paths": ["tests/test_replaces_json_lists.py"],
                "failures": [],
                "stdout": "assert merged['items'] == [3]",
                "stderr": "",
            }
        return {
            "ran": True,
            "passed": True,
            "phase": "tests",
            "failed_paths": [],
            "source_failed_paths": [],
            "test_failed_paths": [],
            "failures": [],
        }

    compiler = SubstrateCandidateCompiler(
        substrate=_StaticSubstrate(),
        kernel=kernel,
        max_preflight_corrections=1,
        test_preflight_runner=preflight,
    )

    candidate = compiler.propose(
        plan,
        artifact,
        validation,
        _directive(plan, artifact, validation),
    )

    assert candidate.files
    assert kernel.target_history[1] == [
        "src/cli.py",
        "tests/test_replaces_json_lists.py",
    ]
    first_preflight = candidate.evidence["candidate_attempts"][0]["preflight"]
    assert first_preflight["impact_expanded_paths"] == ["src/cli.py"]


def test_candidate_correction_receives_structured_pytest_failure(json_merge_case):
    _, plan, artifact, validation = json_merge_case
    kernel = _MonotonicCorrectionKernel()
    calls = 0

    def preflight(files, tests):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "ran": True,
                "passed": False,
                "phase": "tests",
                "failed_paths": ["tests/test_replaces_json_lists.py"],
                "source_failed_paths": [],
                "test_failed_paths": ["tests/test_replaces_json_lists.py"],
                "failure_details": [
                    {
                        "path": "tests/test_replaces_json_lists.py",
                        "node_id": "tests/test_replaces_json_lists.py::test_replacement",
                        "message": "assert [1, 2, 3] == [3]",
                    }
                ],
            }
        return {
            "ran": True,
            "passed": True,
            "phase": "tests",
            "failed_paths": [],
            "source_failed_paths": [],
            "test_failed_paths": [],
        }

    compiler = SubstrateCandidateCompiler(
        substrate=_StaticSubstrate(),
        kernel=kernel,
        max_preflight_corrections=1,
        test_preflight_runner=preflight,
    )

    candidate = compiler.propose(
        plan,
        artifact,
        validation,
        _directive(plan, artifact, validation),
    )

    assert candidate.files
    requirements = kernel.context_history[1]["candidate_correction_requirements"]
    assert any("test_replacement" in item for item in requirements)
    assert any("assert [1, 2, 3] == [3]" in item for item in requirements)


def test_semantic_preflight_uses_validator_anti_stub_contract(json_merge_case):
    _, plan, artifact, _ = json_merge_case
    files = {
        generated.path: generated.content
        for generated in artifact.files
        if generated.path != "forge_artifact_manifest.json"
    }
    files["src/cli.py"] = _source()
    test_paths = sorted(path for path in files if path.startswith("tests/"))
    for path in test_paths:
        files[path] = _test_source(path)
    target_path = test_paths[0]
    files[target_path] = '''import cli


def test_public_api_exists():
    assert callable(cli.main)
'''
    contracts = build_test_generation_contracts(test_paths, plan, artifact)

    result = run_semantic_preflight(
        files,
        plan,
        contracts,
        {"ran": True, "passed": True, "phase": "tests"},
    )

    assert result["passed"] is False
    assert result["phase"] == "semantic_contract"
    assert any(
        failure.get("kind") == "non_semantic_test"
        and failure.get("path") == target_path
        for failure in result["failures"]
    )
    assert any(
        target_path in requirement and "observable behavioral result" in requirement
        for requirement in result["correction_requirements"]
    )


def test_semantic_preflight_rejects_tautological_acceptance_assertion(json_merge_case):
    _, plan, artifact, _ = json_merge_case
    files = {
        generated.path: generated.content
        for generated in artifact.files
        if generated.path != "forge_artifact_manifest.json"
    }
    files["src/cli.py"] = _source()
    test_paths = sorted(path for path in files if path.startswith("tests/"))
    for path in test_paths:
        files[path] = _test_source(path)
    target_path = test_paths[0]
    files[target_path] = '''def test_exit_status_contract():
    target = lambda: 0
    result = target()
    assert isinstance(result, int)
    assert result == 0 or result != 0
'''
    contracts = build_test_generation_contracts(test_paths, plan, artifact)

    result = run_semantic_preflight(
        files,
        plan,
        contracts,
        {"ran": True, "passed": True, "phase": "tests"},
    )

    assert result["passed"] is False
    assert result["phase"] == "semantic_contract"
    assert any(
        failure.get("kind") == "non_semantic_test"
        and failure.get("path") == target_path
        for failure in result["failures"]
    )


def test_imported_source_expansion_resolves_nested_module_basename():
    files = {
        "src/library/core.py": "def merge_intervals(values):\n    return values\n",
        "src/library/__init__.py": "from .core import merge_intervals\n",
        "tests/test_library.py": "import core\n\ndef test_merge():\n    assert core.merge_intervals([]) == []\n",
    }

    impacted = SubstrateCandidateCompiler._imported_source_paths(
        files,
        ["tests/test_library.py"],
        list(files),
    )

    assert impacted == ["src/library/core.py"]


def test_semantic_preflight_rejects_return_code_only_rejection_test(json_merge_case):
    _, plan, artifact, _ = json_merge_case
    files = {
        generated.path: generated.content
        for generated in artifact.files
        if generated.path != "forge_artifact_manifest.json"
    }
    files["src/cli.py"] = _source()
    files["tests/test_recursive_json_merge.py"] = _test_source(
        "tests/test_recursive_json_merge.py"
    )
    files["tests/test_rejects_non_object_json_root.py"] = '''import cli


def test_rejects_non_object_root(tmp_path):
    result = cli.main([str(tmp_path / "base.json"), str(tmp_path / "override.json"), str(tmp_path / "out.json")])
    assert result != 0
'''
    contracts = build_test_generation_contracts(
        sorted(path for path in files if path.startswith("tests/")),
        plan,
        artifact,
    )

    result = run_semantic_preflight(
        files,
        plan,
        contracts,
        {"ran": True, "passed": True, "phase": "tests"},
    )

    assert result["passed"] is False
    assert result["phase"] == "semantic_contract"
    rejection_failure = next(
        item
        for item in result["failures"]
        if item.get("requirement_id") == "R003"
    )
    assert rejection_failure["expected_exception_missing"] is True
    assert any(
        "pytest.raises((ValueError, TypeError, SystemExit))" in requirement
        for requirement in result["correction_requirements"]
    )


def test_semantic_preflight_accepts_behavioral_list_replacement_and_try_except(json_merge_case):
    _, plan, artifact, _ = json_merge_case
    files = {
        generated.path: generated.content
        for generated in artifact.files
        if generated.path != "forge_artifact_manifest.json"
    }
    files["src/cli.py"] = _source()
    files["tests/test_recursive_json_merge.py"] = _test_source(
        "tests/test_recursive_json_merge.py"
    )
    files["tests/test_replaces_json_lists.py"] = '''import json
import cli


def test_override_list_replaces_base_list(tmp_path):
    base = tmp_path / "base.json"
    override = tmp_path / "override.json"
    output = tmp_path / "output.json"
    base.write_text('{"items":[1,2]}', encoding="utf-8")
    override.write_text('{"items":[3]}', encoding="utf-8")
    assert cli.main([str(base), str(override), str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["items"] == [3]
'''
    files["tests/test_rejects_non_object_json_root.py"] = '''import cli


def test_rejects_non_object_root(tmp_path):
    base = tmp_path / "base.json"
    override = tmp_path / "override.json"
    output = tmp_path / "output.json"
    base.write_text('[]', encoding="utf-8")
    override.write_text('{}', encoding="utf-8")
    try:
        cli.main([str(base), str(override), str(output)])
    except ValueError:
        return
    raise AssertionError("expected ValueError")
'''
    contracts = build_test_generation_contracts(
        sorted(path for path in files if path.startswith("tests/")),
        plan,
        artifact,
    )

    result = run_semantic_preflight(
        files,
        plan,
        contracts,
        {"ran": True, "passed": True, "phase": "tests"},
    )

    assert result["passed"] is True


def test_semantic_preflight_rejects_helper_only_non_object_rejection(json_merge_case):
    _, plan, artifact, _ = json_merge_case
    files = {
        generated.path: generated.content
        for generated in artifact.files
        if generated.path != "forge_artifact_manifest.json"
    }
    files["src/cli.py"] = _source()
    files["tests/test_rejects_non_object_json_root.py"] = '''import pytest
import cli


def test_rejects_non_object_root(tmp_path):
    base = tmp_path / "base.json"
    base.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        cli.load_json_file(str(base))
    result = cli.main([str(base), str(tmp_path / "override.json"), str(tmp_path / "out.json")])
    assert result != 0
'''
    contracts = build_test_generation_contracts(
        sorted(path for path in files if path.startswith("tests/")),
        plan,
        artifact,
    )

    result = run_semantic_preflight(
        files,
        plan,
        contracts,
        {"ran": True, "passed": True, "phase": "tests"},
    )

    assert result["passed"] is False
    rejection_failure = next(
        item for item in result["failures"] if item.get("requirement_id") == "R003"
    )
    assert rejection_failure["expected_exception_missing"] is True


def test_semantic_preflight_rejects_noncanonical_deduplication_result(tmp_path):
    spec = RequirementCompiler().compile(EMAIL_LIBRARY_REQUIREMENT)
    plan = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    ).plan(spec)
    assert isinstance(plan, FeasiblePlan)
    artifact = CoderStage().generate(plan)
    files = {
        generated.path: generated.content
        for generated in artifact.files
        if generated.path != "forge_artifact_manifest.json"
    }
    files["src/library/core.py"] = '''def canonicalize_email(value: str) -> str:
    return value.strip().lower()


def deduplicate_emails(values: list[str]) -> list[str]:
    return list(dict.fromkeys(canonicalize_email(value) for value in values))
'''
    files["src/library/__init__.py"] = (
        "from .core import canonicalize_email, deduplicate_emails\n"
    )
    deduplication_atom = next(
        atom
        for atom in spec.requirement_atoms
        if "deduplication must preserve" in atom.text.lower()
    )
    mapped_test = next(
        test
        for test in plan.required_tests
        if deduplication_atom.requirement_id in test.requirement_ids
    )
    mapped_path = f"tests/{mapped_test.test_name}.py"
    files[mapped_path] = '''from library import deduplicate_emails


def test_deduplicate_preserves_original_first_value():
    values = [" Alice@Example.com ", "alice@example.COM", "BOB@example.com"]
    result = deduplicate_emails(values)
    assert result == [" Alice@Example.com ", "BOB@example.com"]
'''
    contracts = build_test_generation_contracts(
        sorted(path for path in files if path.startswith("tests/")),
        plan,
        artifact,
    )

    result = run_semantic_preflight(
        files,
        plan,
        contracts,
        {"ran": True, "passed": True, "phase": "tests"},
    )

    assert result["passed"] is False
    failure = next(
        item
        for item in result["failures"]
        if item.get("requirement_id") == deduplication_atom.requirement_id
    )
    assert failure["canonicalized_deduplication_missing"] is True


def test_semantic_preflight_accepts_canonicalized_deduplication_result(tmp_path):
    spec = RequirementCompiler().compile(EMAIL_LIBRARY_REQUIREMENT)
    plan = PlannerStage(
        execution_mode="local-only",
        audit_log_file=str(tmp_path / "audit.json"),
        memory_file=str(tmp_path / "memory.json"),
        gene_pool_file=str(tmp_path / "genes.json"),
    ).plan(spec)
    assert isinstance(plan, FeasiblePlan)
    artifact = CoderStage().generate(plan)
    files = {
        generated.path: generated.content
        for generated in artifact.files
        if generated.path != "forge_artifact_manifest.json"
    }
    files["src/library/core.py"] = '''def canonicalize_email(value: str) -> str:
    return value.strip().lower()


def deduplicate_emails(values: list[str]) -> list[str]:
    return list(dict.fromkeys(canonicalize_email(value) for value in values))
'''
    files["src/library/__init__.py"] = (
        "from .core import canonicalize_email, deduplicate_emails\n"
    )
    mapped_path = "tests/test_implement_functional_goal_build_a_python.py"
    files[mapped_path] = '''from library import canonicalize_email, deduplicate_emails


def test_canonicalized_deduplication():
    values = [" Alice@Example.com ", "alice@example.COM", "BOB@example.com"]
    result = deduplicate_emails(values)
    assert canonicalize_email(" Alice@Example.com ") == "alice@example.com"
    assert result == ["alice@example.com", "bob@example.com"]
'''
    contracts = build_test_generation_contracts(
        sorted(path for path in files if path.startswith("tests/")),
        plan,
        artifact,
    )

    result = run_semantic_preflight(
        files,
        plan,
        contracts,
        {"ran": True, "passed": True, "phase": "tests"},
    )

    assert result["passed"] is True


def test_interface_parameter_evidence_requires_public_invocation_and_assertion():
    source = '''def run(input_path: str, output_path: str) -> int:
    return 0
'''
    behavioral_test = '''def test_pipeline(tmp_path):
    source = tmp_path / "events.jsonl"
    output = tmp_path / "summary.json"
    assert pipeline.run(str(source), str(output)) == 0
    assert output.exists()
'''

    assert interface_parameter_is_exercised(
        "input_path",
        behavioral_test,
        source,
        {"run"},
    )
    assert interface_parameter_is_exercised(
        "output_path",
        behavioral_test,
        source,
        {"run"},
    )
    assert not interface_parameter_is_exercised(
        "sensor_id",
        behavioral_test,
        source,
        {"run"},
    )
    assert not interface_parameter_is_exercised(
        "input_path",
        "def test_no_assertion():\n    pipeline.run('in', 'out')\n",
        source,
        {"run"},
    )


def test_json_lines_evidence_requires_per_record_json_decoding():
    jsonl_source = '''import json


def read_events(path):
    events = []
    for raw_line in path.read_text().splitlines():
        events.append(json.loads(raw_line))
    return events
'''
    document_source = '''import json


def read_document(path):
    return json.loads(path.read_text())
'''

    assert has_json_lines_processing(jsonl_source)
    assert not has_json_lines_processing(document_source)
