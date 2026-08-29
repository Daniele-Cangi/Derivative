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


class _LiveUnavailableKernel:
    use_live_model = True

    def propose_code_revision(self, repair_context, target_files, lens_framings):
        return {
            "status": "unavailable",
            "files": [],
            "reason": "Live revision failed: BadRequestError: invalid schema",
        }


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


_BROKEN_SOURCE = "def broken(:\n"


class _SyntaxRegressionKernel:
    use_live_model = True

    def propose_code_revision(self, repair_context, target_files, lens_framings):
        return {
            "status": "candidate",
            "files": {path: _BROKEN_SOURCE for path in target_files},
        }


class _RegressingThenRecoveringKernel:
    use_live_model = True

    def __init__(self):
        self.calls = 0
        self.context_history = []
        self.target_history = []

    def propose_code_revision(self, repair_context, target_files, lens_framings):
        self.calls += 1
        self.context_history.append(copy.deepcopy(repair_context))
        self.target_history.append(sorted(target_files))
        if self.calls == 1:
            files = {
                path: "# regressed source state\n"
                if path == "src/cli.py"
                else _test_source(path)
                for path in target_files
            }
        else:
            files = {path: _test_source(path) for path in target_files}
        return {"status": "candidate", "files": files}


class _OmissionRecoveryKernel(_JsonMergeKernel):
    def __init__(self):
        self.calls = 0
        self.target_history = []

    def propose_code_revision(self, repair_context, target_files, lens_framings):
        self.calls += 1
        self.target_history.append(sorted(target_files))
        if self.calls == 2:
            return {"status": "candidate", "files": {}}
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


def test_complete_failed_preflight_candidate_is_preserved_for_outer_repair(json_merge_case):
    _, plan, artifact, validation = json_merge_case

    def failing_preflight(files, tests):
        return {
            "ran": True,
            "passed": False,
            "phase": "tests",
            "failed_paths": ["tests/test_replaces_json_lists.py"],
            "source_failed_paths": [],
            "test_failed_paths": ["tests/test_replaces_json_lists.py"],
            "failures": [
                {
                    "kind": "test_failure",
                    "path": "tests/test_replaces_json_lists.py",
                    "message": "one residual assertion failed",
                }
            ],
        }

    compiler = SubstrateCandidateCompiler(
        substrate=_StaticSubstrate(),
        kernel=_JsonMergeKernel(),
        max_preflight_corrections=0,
        test_preflight_runner=failing_preflight,
    )

    candidate = compiler.propose(
        plan,
        artifact,
        validation,
        _directive(plan, artifact, validation),
    )

    assert set(candidate.files) == set(candidate.evidence["planned_paths"])
    assert candidate.evidence["complete_transaction"] is True
    assert candidate.evidence["preflight_passed"] is False
    assert candidate.evidence["handoff_status"] == "validator_repair_required"
    assert "validator-guided repair" in candidate.stop_reason

    repaired = CoderStage(candidate_compiler=compiler).repair(
        plan,
        artifact,
        validation,
        _directive(plan, artifact, validation),
    )
    assert repaired.changed is True
    compilation = repaired.artifact.artifact_manifest["metadata"]["candidate_compilation"]
    assert compilation["preflight_passed"] is False

    revalidated = ValidatorStage().validate(repaired.artifact, plan, plan.build_spec)
    assert revalidated.passed is False
    assert "adapter_capability_mismatch" in revalidated.failure_signatures


def test_candidate_compiler_rejects_syntax_regression_from_semantic_baseline(
    json_merge_case,
):
    _, plan, artifact, validation = json_merge_case

    def staged_preflight(files, tests):
        broken_paths = sorted(
            path for path, content in files.items() if content == _BROKEN_SOURCE
        )
        if broken_paths:
            return {
                "ran": False,
                "passed": False,
                "phase": "syntax",
                "failed_paths": broken_paths,
                "source_failed_paths": [
                    path for path in broken_paths if path.startswith("src/")
                ],
                "test_failed_paths": [
                    path for path in broken_paths if path.startswith("tests/")
                ],
                "failures": [
                    {
                        "path": path,
                        "kind": "syntax_error",
                        "line": 1,
                        "message": "invalid syntax",
                    }
                    for path in broken_paths
                ],
            }
        failed_path = artifact.test_paths[0]
        return {
            "ran": True,
            "passed": False,
            "phase": "semantic_contract",
            "failed_paths": [failed_path],
            "source_failed_paths": [],
            "test_failed_paths": [failed_path],
            "failures": [
                {
                    "path": failed_path,
                    "kind": "semantic_contract_failure",
                    "message": "baseline semantic evidence is incomplete",
                }
            ],
        }

    compiler = SubstrateCandidateCompiler(
        substrate=_StaticSubstrate(),
        kernel=_SyntaxRegressionKernel(),
        max_preflight_corrections=1,
        test_preflight_runner=staged_preflight,
    )

    candidate = compiler.propose(
        plan,
        artifact,
        validation,
        _directive(plan, artifact, validation),
    )

    assert candidate.available is True
    assert candidate.files == {}
    assert candidate.evidence["complete_transaction"] is False
    assert candidate.evidence["regression_rejected"] is True
    assert candidate.evidence["regression_rejected_attempts"] == [1, 2]
    assert candidate.evidence["baseline_preflight_quality"]["phase"] == (
        "semantic_contract"
    )
    assert all(
        attempt["preflight_quality"]["phase"] == "syntax"
        and attempt["regresses_from_baseline"] is True
        for attempt in candidate.evidence["candidate_attempts"]
    )
    assert "regressed from the current artifact preflight" in candidate.stop_reason

    repaired = CoderStage(candidate_compiler=compiler).repair(
        plan,
        artifact,
        validation,
        _directive(plan, artifact, validation),
    )

    assert repaired.changed is False
    assert {
        generated.path: generated.content
        for generated in repaired.artifact.files
    } == {
        generated.path: generated.content
        for generated in artifact.files
    }


def test_candidate_compiler_restores_safe_state_before_later_correction(
    json_merge_case,
):
    _, plan, artifact, validation = json_merge_case
    kernel = _RegressingThenRecoveringKernel()
    calls = 0
    baseline_failed_path = artifact.test_paths[0]
    rejected_failed_path = artifact.test_paths[1]
    assert baseline_failed_path != rejected_failed_path

    def semantic_failure(path, message):
        return {
            "ran": True,
            "passed": False,
            "phase": "semantic_contract",
            "failed_paths": [path],
            "source_failed_paths": [],
            "test_failed_paths": [path],
            "failures": [
                {
                    "path": path,
                    "kind": "semantic_contract_failure",
                    "message": message,
                }
            ],
        }

    def staged_preflight(files, tests):
        nonlocal calls
        calls += 1
        if calls == 1:
            return semantic_failure(
                baseline_failed_path,
                "baseline semantic evidence is incomplete",
            )
        if calls == 2:
            result = semantic_failure(
                rejected_failed_path,
                "candidate semantic evidence regressed",
            )
            result["failures"].append(
                {
                    "path": rejected_failed_path,
                    "kind": "requirement_assertion_evidence_failure",
                    "message": "candidate introduced a second semantic failure",
                }
            )
            return result
        return semantic_failure(
            baseline_failed_path,
            "recovered baseline-quality semantic evidence",
        )

    compiler = SubstrateCandidateCompiler(
        substrate=_StaticSubstrate(),
        kernel=kernel,
        max_preflight_corrections=1,
        test_preflight_runner=staged_preflight,
    )

    candidate = compiler.propose(
        plan,
        artifact,
        validation,
        _directive(plan, artifact, validation),
    )

    baseline_files = {
        generated.path: generated.content
        for generated in artifact.files
        if generated.path != "forge_artifact_manifest.json"
    }
    first_attempt, second_attempt = candidate.evidence["candidate_attempts"]
    assert first_attempt["regresses_from_baseline"] is True
    assert first_attempt["working_state_restored"] is True
    assert second_attempt["regresses_from_baseline"] is False
    assert second_attempt["selected_for_handoff"] is True
    assert kernel.target_history[1] == [baseline_failed_path]
    assert kernel.context_history[1]["preflight_test_execution"]["failed_paths"] == [
        baseline_failed_path
    ]
    assert candidate.files["src/cli.py"] == baseline_files["src/cli.py"]
    assert candidate.files["src/cli.py"] != "# regressed source state\n"
    assert kernel.context_history[1]["preserved_candidate_files"]["src/cli.py"] == (
        baseline_files["src/cli.py"]
    )


def test_incomplete_correction_restores_selected_safe_state(json_merge_case):
    _, plan, artifact, validation = json_merge_case
    kernel = _OmissionRecoveryKernel()
    calls = 0
    baseline_failed_path = artifact.test_paths[0]
    selected_failed_path = artifact.test_paths[1]

    def semantic_failure(path, message):
        return {
            "ran": True,
            "passed": False,
            "phase": "semantic_contract",
            "failed_paths": [path],
            "source_failed_paths": [],
            "test_failed_paths": [path],
            "failures": [
                {
                    "path": path,
                    "kind": "semantic_contract_failure",
                    "message": message,
                }
            ],
        }

    def staged_preflight(files, tests):
        nonlocal calls
        calls += 1
        if calls == 1:
            return semantic_failure(
                baseline_failed_path,
                "baseline semantic evidence is incomplete",
            )
        if calls == 2:
            return semantic_failure(
                selected_failed_path,
                "selected candidate requires one correction",
            )
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
        max_preflight_corrections=2,
        test_preflight_runner=staged_preflight,
    )

    candidate = compiler.propose(
        plan,
        artifact,
        validation,
        _directive(plan, artifact, validation),
    )

    first_attempt, omitted_attempt, recovered_attempt = candidate.evidence[
        "candidate_attempts"
    ]
    assert first_attempt["selected_for_handoff"] is True
    assert omitted_attempt["preflight"]["phase"] == "candidate_completeness"
    assert omitted_attempt["working_state_restored"] is True
    assert omitted_attempt["routing_active_paths"] == [selected_failed_path]
    assert recovered_attempt["preflight"]["passed"] is True
    assert kernel.target_history[1:] == [
        [selected_failed_path],
        [selected_failed_path],
    ]
    assert candidate.evidence["preflight_passed"] is True
    assert candidate.files["src/cli.py"] == _source()


def test_candidate_compiler_preserves_sanitized_backend_failure_reason(json_merge_case):
    _, plan, artifact, validation = json_merge_case
    compiler = SubstrateCandidateCompiler(
        substrate=_StaticSubstrate(),
        kernel=_LiveUnavailableKernel(),
    )

    candidate = compiler.propose(
        plan,
        artifact,
        validation,
        _directive(plan, artifact, validation),
    )

    expected = "Live revision failed: BadRequestError: invalid schema"
    assert candidate.available is False
    assert candidate.files == {}
    assert candidate.stop_reason == expected
    assert candidate.evidence["backend_reason"] == expected
    assert candidate.evidence["candidate_attempts"][0]["reason"] == expected


def test_candidate_correction_preserves_passing_files(json_merge_case):
    _, plan, artifact, validation = json_merge_case
    kernel = _MonotonicCorrectionKernel()
    preflight_calls = []

    def preflight(files, tests):
        preflight_calls.append(dict(files))
        if len(preflight_calls) <= 2:
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
    assert candidate.files["src/cli.py"] == preflight_calls[1]["src/cli.py"]
    second_attempt = candidate.evidence["candidate_attempts"][1]
    assert "src/cli.py" in second_attempt["preserved_paths"]
    preserved = kernel.context_history[1]["preserved_candidate_files"]
    assert preserved["src/cli.py"] == preflight_calls[1]["src/cli.py"]
    assert "immutable context" in kernel.context_history[1]["preservation_contract"]


def test_candidate_gets_one_extra_correction_on_new_semantic_phase(json_merge_case):
    _, plan, artifact, validation = json_merge_case
    kernel = _MonotonicCorrectionKernel()
    failed_test = artifact.test_paths[0]
    calls = 0

    def staged_preflight(files, tests):
        nonlocal calls
        calls += 1
        if calls <= 2:
            return {
                "ran": True,
                "passed": False,
                "phase": "tests",
                "failed_paths": [failed_test],
                "source_failed_paths": [],
                "test_failed_paths": [failed_test],
                "failures": [],
            }
        if calls == 3:
            return {
                "ran": True,
                "passed": False,
                "phase": "semantic_contract",
                "failed_paths": [failed_test, "src/cli.py"],
                "source_failed_paths": ["src/cli.py"],
                "test_failed_paths": [failed_test],
                "failures": [
                    {
                        "path": failed_test,
                        "kind": "non_semantic_test",
                        "reasons": ["ambiguous_exit_status_assertion"],
                    }
                ],
                "correction_requirements": ["Assert one exact exit status."],
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
        test_preflight_runner=staged_preflight,
    )

    candidate = compiler.propose(
        plan,
        artifact,
        validation,
        _directive(plan, artifact, validation),
    )

    assert candidate.evidence["preflight_passed"] is True
    assert len(kernel.target_history) == 3
    assert candidate.evidence["initial_candidate_attempt_limit"] == 2
    assert candidate.evidence["semantic_phase_extension_used"] is True
    assert candidate.evidence["final_candidate_attempt_limit"] == 3
    assert candidate.evidence["candidate_attempts"][1][
        "semantic_phase_extension_granted"
    ] is True


def test_executable_test_failure_expands_correction_to_imported_source(json_merge_case):
    _, plan, artifact, validation = json_merge_case
    kernel = _MonotonicCorrectionKernel()
    calls = []

    def preflight(files, tests):
        calls.append(dict(files))
        if len(calls) <= 2:
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


def test_non_semantic_test_failure_reopens_imported_source(json_merge_case):
    _, _, artifact, _ = json_merge_case
    files = {
        generated.path: generated.content
        for generated in artifact.files
        if generated.path != "forge_artifact_manifest.json"
    }
    failed_test = next(
        path
        for path in artifact.test_paths
        if "import cli" in files[path]
    )
    preflight = {
        "ran": True,
        "passed": False,
        "phase": "semantic_contract",
        "failed_paths": [failed_test],
        "source_failed_paths": [],
        "test_failed_paths": [failed_test],
        "failures": [
            {
                "path": failed_test,
                "kind": "non_semantic_test",
                "failure_reason": "missing_causal_assertion",
            }
        ],
    }

    active_paths = SubstrateCandidateCompiler._preflight_active_paths(
        files,
        preflight,
        sorted(files),
    )

    assert failed_test in active_paths
    assert "src/cli.py" in active_paths
    assert preflight["impact_expanded_paths"] == ["src/cli.py"]


def test_candidate_correction_receives_structured_pytest_failure(json_merge_case):
    _, plan, artifact, validation = json_merge_case
    kernel = _MonotonicCorrectionKernel()
    calls = 0

    def preflight(files, tests):
        nonlocal calls
        calls += 1
        if calls <= 2:
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


def test_candidate_correction_targets_lossy_newline_observation():
    requirements = SubstrateCandidateCompiler._correction_requirements(
        {
            "phase": "tests",
            "passed": False,
            "failed_paths": ["tests/test_cli_flow.py"],
            "stdout": (
                "out = output_path.read_text(encoding='utf-8')\n"
                "assert out.endswith('\\r\\n')\n"
            ),
            "stderr": "",
        }
    )

    exact_observation = next(
        requirement
        for requirement in requirements
        if "read_bytes" in requirement
    )
    assert "tests/test_cli_flow.py" in exact_observation
    assert "normalizes CRLF" in exact_observation


def test_candidate_correction_reports_structural_and_unicode_fixture_failures():
    requirements = SubstrateCandidateCompiler._correction_requirements(
        {
            "phase": "syntax",
            "failures": [
                {
                    "path": "tests/test_cli_flow.py",
                    "kind": "syntax_error",
                    "line": 7,
                    "message": "expected an indented block",
                }
            ],
            "stdout": (
                'expected = value.encode("utf-8").decode("unicode_escape")\n'
            ),
            "stderr": "",
        }
    )

    assert any(
        "tests/test_cli_flow.py" in requirement
        and "behaviorally unvalidated" in requirement
        for requirement in requirements
    )
    assert any(
        "already hold decoded Unicode" in requirement
        for requirement in requirements
    )


def test_structural_preflight_does_not_freeze_unexecuted_paths(json_merge_case):
    _, plan, artifact, validation = json_merge_case
    kernel = _MonotonicCorrectionKernel()
    calls = 0

    def preflight(files, tests):
        nonlocal calls
        calls += 1
        if calls <= 2:
            failed_path = artifact.test_paths[0]
            return {
                "ran": False,
                "passed": False,
                "phase": "syntax",
                "failed_paths": [failed_path],
                "source_failed_paths": [],
                "test_failed_paths": [failed_path],
                "failures": [
                    {
                        "path": failed_path,
                        "kind": "syntax_error",
                        "line": 3,
                        "message": "invalid syntax",
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
    assert kernel.target_history[1] == kernel.target_history[0]
    assert kernel.context_history[1]["preserve_passing_paths"] == []
    first_preflight = candidate.evidence["candidate_attempts"][0]["preflight"]
    assert set(first_preflight["behaviorally_unvalidated_paths"]) == set(
        kernel.target_history[0]
    )


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


def test_semantic_preflight_rejects_fixture_that_contradicts_explicit_regex():
    requirement = (
        "Implement a CLI whose lines must match the regex: "
        "'^[A-Z_][A-Z0-9_]*=[^\\n]*$'."
    )
    path = "tests/test_explicit_pattern.py"
    plan = SimpleNamespace(
        build_spec=SimpleNamespace(
            normalized_requirement=requirement,
            requirement_atoms=[],
        ),
        required_tests=[SimpleNamespace(test_name="test_explicit_pattern", required=True)],
        interfaces=[],
        requirement_coverage={},
    )
    content = (
        "INVALID_LINES = ['FOO=bar extra\\n']\n"
        "\n"
        "def test_explicit_pattern():\n"
        "    assert len(INVALID_LINES) == 1\n"
    )

    result = run_semantic_preflight(
        {path: content},
        plan,
        {path: {"requirements": []}},
        {"phase": "tests", "passed": True, "failures": []},
    )

    mismatch = next(
        failure
        for failure in result["failures"]
        if failure.get("kind") == "explicit_pattern_fixture_mismatch"
    )
    assert mismatch["sample"] == "FOO=bar extra\n"
    assert mismatch["oracle_classification"] == "invalid"
    assert mismatch["derived_classification"] == "valid"
    assert any(
        "Do not invent stricter whitespace or delimiter rules" in instruction
        for instruction in result["correction_requirements"]
    )


def test_semantic_preflight_rejects_forbidden_cli_in_library_source():
    spec = RequirementCompiler().compile(
        "Implement a Python module called 'rowrotate' exposing a single function "
        "rotate_fields(rows: list[dict], field_order: list[str], shift: int = 1) -> list[dict]. "
        "Only rotate_fields is public, and the module has no CLI or service interface."
    )
    restriction = next(
        atom for atom in spec.requirement_atoms if "no CLI or service" in atom.text
    )
    plan = SimpleNamespace(
        build_spec=SimpleNamespace(
            requirement_atoms=[restriction],
            normalized_requirement=spec.normalized_requirement,
        ),
        required_tests=[],
        interfaces=[SimpleNamespace(name="rotate_fields", interface_type="function")],
        requirement_coverage={
            restriction.requirement_id: {
                "files": ["src/rowrotate.py"],
                "tests": [],
                "acceptance_criteria": [],
            }
        },
    )
    files = {
        "src/rowrotate.py": (
            "def rotate_fields(rows, field_order, shift=1):\n"
            "    return list(rows)\n\n"
            "def main(argv=None):\n"
            "    return 0\n"
        )
    }

    result = run_semantic_preflight(
        files,
        plan,
        {},
        {"phase": "tests", "passed": True, "failures": []},
    )

    assert result["passed"] is False
    failure = next(
        item for item in result["failures"] if item["kind"] == "source_semantic_contract_failure"
    )
    assert failure["missing_evidence_terms"] == ["no_cli_entrypoint"]


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


def test_semantic_preflight_rejects_exact_output_with_added_newline():
    plan = SimpleNamespace(
        build_spec=SimpleNamespace(
            normalized_requirement=(
                "If input is invalid, the tool outputs exactly 'error: invalid input' "
                "to stderr and exits with code 1."
            ),
            requirement_atoms=[],
        ),
        required_tests=[],
        interfaces=[],
        requirement_coverage={},
    )

    result = run_semantic_preflight(
        {
            "src/tool.py": (
                "import sys\n"
                "def main(argv=None):\n"
                "    if argv == ['invalid']:\n"
                "        sys.stderr.write('error: invalid input\\n')\n"
                "        return 1\n"
                "    return 0\n"
            )
        },
        plan,
        {},
        {"ran": True, "passed": True, "phase": "tests", "failures": []},
    )

    assert result["passed"] is False
    failure = next(
        item
        for item in result["failures"]
        if item["kind"] == "exact_output_contract_failure"
    )
    assert failure["expected"] == "error: invalid input"
    assert failure["observed"] == ["error: invalid input\n"]


def test_semantic_preflight_ignores_exact_output_literal_in_test_source():
    plan = SimpleNamespace(
        build_spec=SimpleNamespace(
            normalized_requirement=(
                "If input is invalid, the tool outputs exactly 'error: invalid input' "
                "to stderr and exits with code 1."
            ),
            requirement_atoms=[],
        ),
        required_tests=[],
        interfaces=[],
        requirement_coverage={},
    )

    result = run_semantic_preflight(
        {
            "src/tool.py": (
                "import sys\n"
                "def fail():\n"
                "    sys.stderr.write('wrong')\n"
            ),
            "tests/test_tool.py": (
                "import sys\n"
                "def test_output():\n"
                "    sys.stderr.write('error: invalid input')\n"
            ),
        },
        plan,
        {},
        {"ran": True, "passed": True, "phase": "tests", "failures": []},
    )

    assert result["passed"] is False
    failure = next(
        item
        for item in result["failures"]
        if item["kind"] == "exact_output_contract_failure"
    )
    assert failure["observed"] == ["wrong"]


def test_semantic_preflight_accepts_module_path_and_local_target_wrapper():
    atom = SimpleNamespace(
        requirement_id="R001",
        category="functional",
        evidence_terms=["cli_entrypoint", "word_freq_stats"],
        text="Define the word_freq_stats CLI entrypoint.",
    )
    plan = SimpleNamespace(
        build_spec=SimpleNamespace(
            normalized_requirement="Define the word_freq_stats CLI entrypoint.",
            requirement_atoms=[atom],
        ),
        required_tests=[SimpleNamespace(test_name="test_word_freq_stats", required=True)],
        interfaces=[SimpleNamespace(name="main", interface_type="cli_entrypoint")],
        requirement_coverage={
            "R001": {
                "files": ["src/word_freq_stats.py"],
                "tests": ["test_word_freq_stats"],
                "acceptance_criteria": ["AC001"],
            }
        },
    )
    test_path = "tests/test_word_freq_stats.py"
    files = {
        "src/word_freq_stats.py": "def main(argv=None):\n    return 0\n",
        test_path: (
            "import word_freq_stats as cli\n\n"
            "def run_cli(argv):\n"
            "    return cli.main(argv)\n\n"
            "def test_word_freq_stats():\n"
            "    result = run_cli([])\n"
            "    assert result == 0\n"
        ),
    }
    contracts = {
        test_path: {
            "requirements": [
                {
                    "id": "R001",
                    "evidence_terms": ["cli_entrypoint", "word_freq_stats"],
                }
            ]
        }
    }

    result = run_semantic_preflight(
        files,
        plan,
        contracts,
        {"ran": True, "passed": True, "phase": "tests", "failures": []},
    )

    assert result["passed"] is True, result


def test_semantic_preflight_rejects_assertion_disconnected_from_target(json_merge_case):
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


def test_disconnected_contract():
    cli.main([])
    unrelated = 2
    assert unrelated == 2
'''
    contracts = build_test_generation_contracts(test_paths, plan, artifact)

    result = run_semantic_preflight(
        files,
        plan,
        contracts,
        {"ran": True, "passed": True, "phase": "tests"},
    )

    failure = next(
        item
        for item in result["failures"]
        if item.get("kind") == "non_semantic_test"
        and item.get("path") == target_path
    )
    assert result["passed"] is False
    assert failure["reasons"] == ["disconnected_assertion"]
    assert any(
        "assertions must observe values returned" in requirement
        for requirement in result["correction_requirements"]
    )


def test_semantic_preflight_requires_assertion_evidence_per_requirement(json_merge_case):
    spec, plan, artifact, _ = json_merge_case
    files = {
        generated.path: generated.content
        for generated in artifact.files
        if generated.path != "forge_artifact_manifest.json"
    }
    files["src/cli.py"] = _source()
    test_paths = sorted(path for path in files if path.startswith("tests/"))
    for path in test_paths:
        files[path] = _test_source(path)
    atom = next(
        item
        for item in spec.requirement_atoms
        if "recursive_json_merge" in item.evidence_terms
    )
    target_path = f"tests/{plan.requirement_coverage[atom.requirement_id]['tests'][0]}.py"
    files[target_path] = '''import cli


def test_recursive_json_merge_label_only():
    recursive_json_merge = "declared"
    assert recursive_json_merge == "declared"


def test_unrelated_generated_behavior():
    result = cli.validate_object_root({})
    assert result == {}
'''
    contracts = build_test_generation_contracts(test_paths, plan, artifact)

    result = run_semantic_preflight(
        files,
        plan,
        contracts,
        {"ran": True, "passed": True, "phase": "tests"},
    )

    failure = next(
        item
        for item in result["failures"]
        if item.get("kind") == "requirement_assertion_evidence_failure"
        and item.get("requirement_id") == atom.requirement_id
    )
    assert result["passed"] is False
    assert failure["failure_reason"] == "missing_requirement_assertion_evidence"
    assert "recursive_json_merge" in failure["missing_evidence_terms"]
    assert any(
        "target-dependent assertion in the same test function" in requirement
        for requirement in result["correction_requirements"]
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


@pytest.mark.parametrize(
    "import_statement",
    [
        "import component",
        "from component import run",
        "from src.component import run",
        "from src import component",
    ],
)
def test_imported_source_expansion_resolves_src_qualified_imports(import_statement):
    files = {
        "src/component.py": "def run():\n    return 0\n",
        "tests/test_component.py": (
            f"{import_statement}\n\n"
            "def test_component():\n"
            "    assert True\n"
        ),
    }

    impacted = SubstrateCandidateCompiler._imported_source_paths(
        files,
        ["tests/test_component.py"],
        list(files),
    )

    assert impacted == ["src/component.py"]


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


def test_semantic_preflight_rejects_lossy_line_ending_observation(json_merge_case):
    _, original_plan, _, _ = json_merge_case
    plan = copy.deepcopy(original_plan)
    plan.build_spec.normalized_requirement = (
        "Write transformed text with original line endings preserved, including CRLF, LF, or CR."
    )
    mapped_test = plan.required_tests[0]
    requirement_id = mapped_test.requirement_ids[0]
    atom = next(
        item
        for item in plan.build_spec.requirement_atoms
        if item.requirement_id == requirement_id
    )
    atom.text = "Preserve original line endings including CRLF, LF, or CR."
    atom.evidence_terms = ["line endings"]
    test_path = f"tests/{mapped_test.test_name}.py"
    files = {
        "src/cli.py": "def main(argv=None):\n    return 0\n",
        test_path: '''import cli


def test_preserves_line_endings(tmp_path):
    output = tmp_path / "out.txt"
    assert cli.main(["input.txt", str(output)]) == 0
    observed = output.read_text(encoding="utf-8")
    assert observed.endswith("\\r\\n")
''',
    }
    contracts = {
        test_path: {
            "requirements": [
                {
                    "id": requirement_id,
                    "evidence_terms": ["line endings"],
                }
            ]
        }
    }

    result = run_semantic_preflight(
        files,
        plan,
        contracts,
        {"ran": True, "passed": True, "phase": "tests"},
    )

    exact_failure = next(
        failure
        for failure in result["failures"]
        if failure.get("kind") == "lossy_observation_api"
    )
    assert exact_failure["requirement_id"] == requirement_id
    assert exact_failure["required_observation"] == "byte_exact"
    assert any(
        "Path.read_bytes()" in requirement
        for requirement in result["correction_requirements"]
    )
