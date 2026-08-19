import json
from types import SimpleNamespace

from core.forge.coder_stage import CoderStage
from core.forge.contracts import (
    AcceptanceContract,
    ArtifactTargetType,
    BuildSpec,
    FeasiblePlan,
    GeneratedFile,
    ObligationContract,
    PlanFile,
    PlanInterface,
    PlanTest,
    RequirementAtom,
    RepairPatchCandidate,
    ValidationArtifact,
    ValidationStrategy,
)
from core.forge.repair import RepairPolicy
from core.forge.repair_backend import SubstrateRepairBackend
from core.kernel import ReasoningKernel


def _generic_plan() -> FeasiblePlan:
    spec = BuildSpec(
        build_id="build-repair-backend",
        raw_requirement="Build an executable Python component with tests.",
        normalized_requirement="Build an executable Python component with tests.",
        functional_goals=["Execute a workflow."],
        requirement_atoms=[
            RequirementAtom(
                requirement_id="R001",
                text="Execute a workflow.",
                category="functional",
                strength="hard",
                source_fragment="executable Python component",
                evidence_terms=["workflow"],
            )
        ],
        acceptance_contract=AcceptanceContract(),
        obligation_contract=ObligationContract(mode="software_build"),
        target_artifact_type=ArtifactTargetType.UNKNOWN,
    )
    return FeasiblePlan(
        plan_id="plan-repair-backend",
        build_spec=spec,
        architecture_summary="Executable Python component.",
        file_tree_plan=[PlanFile(path="src/component.py", purpose="Workflow implementation.")],
        interfaces=[PlanInterface(name="run", interface_type="entrypoint")],
        required_tests=[
            PlanTest(
                test_name="test_component",
                objective="Execute the workflow.",
                requirement_ids=["R001"],
            )
        ],
        validation_strategy=ValidationStrategy(),
        packaging_target="python_package",
    )


class _UnavailableKernel:
    use_live_model = False


class _FailIfUsedSubstrate:
    def decompose(self, problem):
        raise AssertionError("Local unavailable backend must not invoke the substrate.")


def test_substrate_backend_fails_closed_when_live_kernel_is_unavailable():
    plan = _generic_plan()
    artifact = CoderStage().generate(plan)
    validation = ValidationArtifact(
        passed=False,
        failures=["Execution failed."],
        failure_signatures=["test_execution_failure"],
        evidence={"layer2": {"test_execution": {"returncode": 1, "tests": artifact.test_paths}}},
    )
    directive = RepairPolicy().compile(validation, plan, artifact, attempt=2)
    backend = SubstrateRepairBackend(
        substrate=_FailIfUsedSubstrate(),
        kernel=_UnavailableKernel(),
    )

    candidate = backend.propose(plan, artifact, validation, directive)

    assert candidate.available is False
    assert candidate.files == {}
    assert "unavailable" in candidate.stop_reason.lower()


class _StaticSubstrate:
    def decompose(self, problem):
        assert "Validation failures" in problem
        return [SimpleNamespace(lens_name="Formal Logic")]


def _passing_preflight(candidate_files, test_paths):
    return {
        "ran": True,
        "passed": True,
        "returncode": 0,
        "tests": list(test_paths),
        "stdout": "1 passed",
        "stderr": "",
    }


class _StaticKernel:
    use_live_model = True

    def __init__(self, files):
        self.files = files
        self.received_targets = None

    def propose_code_revision(self, repair_context, target_files, lens_framings):
        self.received_targets = target_files
        assert repair_context["failure_signatures"] == ["syntax_error"]
        assert repair_context["file_tree_plan"] == [
            {
                "path": "src/component.py",
                "purpose": "Workflow implementation.",
                "requirement_ids": [],
            }
        ]
        assert lens_framings[0].lens_name == "Formal Logic"
        return {"status": "candidate", "files": self.files}


class _UnavailableResponseKernel:
    use_live_model = True

    def propose_code_revision(self, repair_context, target_files, lens_framings):
        return {
            "status": "unavailable",
            "files": [],
            "reason": "Live revision failed: APIConnectionError",
        }


def test_substrate_backend_rejects_files_outside_validator_target_allowlist():
    plan = _generic_plan()
    artifact = CoderStage().generate(plan)
    plan.required_tests = []
    validation = ValidationArtifact(
        passed=False,
        failures=["Syntax error."],
        failure_signatures=["syntax_error"],
        evidence={"layer1": {"parse_errors": [{"path": "src/component.py"}]}},
    )
    directive = RepairPolicy().compile(validation, plan, artifact, attempt=2)
    kernel = _StaticKernel(
        {
            "src/component.py": "def run() -> int:\n    return 1\n",
            "src/unplanned.py": "raise RuntimeError('not allowed')\n",
            "forge_artifact_manifest.json": "{}",
        }
    )
    backend = SubstrateRepairBackend(substrate=_StaticSubstrate(), kernel=kernel)

    candidate = backend.propose(plan, artifact, validation, directive)

    assert sorted(kernel.received_targets) == ["src/component.py"]
    assert list(candidate.files) == ["src/component.py"]
    assert candidate.rejected_paths == [
        "forge_artifact_manifest.json",
        "src/unplanned.py",
    ]
    assert candidate.evidence["accepted_paths"] == ["src/component.py"]


def test_substrate_backend_preserves_live_kernel_unavailable_reason():
    plan = _generic_plan()
    artifact = CoderStage().generate(plan)
    validation = ValidationArtifact(
        passed=False,
        failures=["Syntax error."],
        failure_signatures=["syntax_error"],
        evidence={"layer1": {"parse_errors": [{"path": "src/component.py"}]}},
    )
    directive = RepairPolicy().compile(validation, plan, artifact, attempt=2)
    backend = SubstrateRepairBackend(
        substrate=_StaticSubstrate(),
        kernel=_UnavailableResponseKernel(),
    )

    candidate = backend.propose(plan, artifact, validation, directive)

    assert candidate.available is False
    assert candidate.files == {}
    assert candidate.evidence["kernel_status"] == "unavailable"
    assert candidate.evidence["kernel_reason"] == "Live revision failed: APIConnectionError"
    assert candidate.stop_reason == "Live revision failed: APIConnectionError"


class _StaticRepairBackend:
    def propose(self, plan, artifact, validation, directive):
        source = next(file.content for file in artifact.files if file.path == "src/component.py")
        return RepairPatchCandidate(
            backend_name="test_grounded_backend",
            files={
                "src/component.py": f"{source}\nREPAIR_MARKER = True\n",
                "tests/test_component.py": "def test_bypass():\n    assert True\n",
            },
            evidence={"failure_signatures": list(validation.failure_signatures)},
        )


class _ImpactExpandedRepairBackend:
    def propose(self, plan, artifact, validation, directive):
        source = next(file.content for file in artifact.files if file.path == "src/component.py")
        test_path = artifact.test_paths[0]
        test_content = next(file.content for file in artifact.files if file.path == test_path)
        return RepairPatchCandidate(
            backend_name="impact_expanded_backend",
            files={
                "src/component.py": f"{source}\nSOURCE_REPAIR = True\n",
                test_path: f"{test_content}\n# impact regression repair\n",
            },
            evidence={"impact_expanded_paths": [test_path]},
        )


class _PerTargetKernel:
    use_live_model = True

    def __init__(self):
        self.calls = []

    def propose_code_revision(self, repair_context, target_files, lens_framings):
        target_paths = list(target_files)
        target_path = repair_context["current_target_path"]
        self.calls.append(
            {
                "target_path": target_path,
                "target_paths": target_paths,
                "target_count": len(target_files),
                "related_sources": dict(repair_context["related_repaired_source_files"]),
                "source_api_contracts": dict(repair_context.get("source_api_contracts", {})),
                "test_generation_contracts": dict(
                    repair_context.get("test_generation_contracts", {})
                ),
                "repair_phase": repair_context.get("repair_phase"),
            }
        )
        return {
            "status": "candidate",
            "files": {
                path: f"{content}\n# revised:{path}\n"
                for path, content in target_files.items()
            },
        }


class _OmittingAtomicKernel:
    use_live_model = True

    def __init__(self):
        self.calls = []

    def propose_code_revision(self, repair_context, target_files, lens_framings):
        self.calls.append(list(target_files))
        path, content = next(iter(target_files.items()))
        return {"status": "candidate", "files": {path: content + "\n# partial\n"}}


class _AssertionEvidenceKernel:
    use_live_model = True

    def __init__(self):
        self.context = None

    def propose_code_revision(self, repair_context, target_files, lens_framings):
        self.context = repair_context
        return {
            "status": "candidate",
            "files": {
                path: f"{content}\n# requirement assertion revised\n"
                for path, content in target_files.items()
            },
        }


def test_substrate_backend_revises_each_grounded_target_separately():
    plan = _generic_plan()
    artifact = CoderStage().generate(plan)
    test_path = artifact.test_paths[0]
    validation = ValidationArtifact(
        passed=False,
        failures=["Superficial implementation and test."],
        failure_signatures=["superficial_stub", "non_semantic_test"],
        evidence={
            "layer3": {
                "superficial_interfaces": ["run"],
                "non_semantic_tests": [test_path],
            }
        },
    )
    directive = RepairPolicy().compile(validation, plan, artifact, attempt=2)
    kernel = _PerTargetKernel()
    backend = SubstrateRepairBackend(
        substrate=_StaticSubstrate(),
        kernel=kernel,
        test_preflight_runner=_passing_preflight,
    )

    candidate = backend.propose(plan, artifact, validation, directive)

    assert set(candidate.files) == {"src/component.py", test_path}
    assert candidate.evidence["omitted_paths"] == []
    assert candidate.evidence["kernel_status"] == "candidate"
    assert len(kernel.calls) == 2
    assert all(call["target_count"] == 1 for call in kernel.calls)
    source_call, test_call = kernel.calls
    assert source_call["target_path"] == "src/component.py"
    assert source_call["related_sources"] == {}
    assert test_call["target_path"] == test_path
    assert "src/component.py" in test_call["related_sources"]


def test_substrate_backend_receives_requirement_assertion_repair_contract():
    plan = _generic_plan()
    artifact = CoderStage().generate(plan)
    test_path = artifact.test_paths[0]
    assertion_evidence = {
        "mapped_test_paths": [test_path],
        "existing_test_paths": [test_path],
        "required_terms": ["workflow"],
        "covered_terms": [],
        "missing_terms": ["workflow"],
        "causal_functions": [
            {
                "path": test_path,
                "function": "test_component",
                "matched_terms": [],
            }
        ],
        "assertions": [],
        "passed": False,
        "failure_reason": "missing_requirement_assertion_evidence",
    }
    validation = ValidationArtifact(
        passed=False,
        failures=["Requirement R001 lacks a causal assertion."],
        failure_signatures=["missing_requirement_assertion_evidence"],
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
            }
        },
    )
    directive = RepairPolicy().compile(validation, plan, artifact, attempt=2)
    kernel = _AssertionEvidenceKernel()
    backend = SubstrateRepairBackend(
        substrate=_StaticSubstrate(),
        kernel=kernel,
        test_preflight_runner=_passing_preflight,
    )

    candidate = backend.propose(plan, artifact, validation, directive)

    assert list(candidate.files) == [test_path]
    assert kernel.context["repair_requirement_ids"] == ["R001"]
    assert kernel.context["repair_target_symbols"] == ["test_component"]
    target = kernel.context["repair_evidence_targets"]["R001"]
    assert target["test_paths"] == [test_path]
    assert target["missing_terms"] == ["workflow"]
    assert target["causal_functions"][0]["function"] == "test_component"


def test_repair_context_compacts_validator_evidence_without_losing_execution_signals():
    oversized_output = "failure detail " * 2000
    evidence = {
        "layer_status": {"layer1": False, "layer2": False, "layer3": False},
        "validated_entrypoints": {"src/cli.py": False},
        "manifest_provenance_checks": {"passed": True},
        "layer1": {
            "failure_signatures": ["import_failure"],
            "import_results": {"cli": {"stderr": oversized_output}},
            "entrypoint_results": {"src/cli.py": {"stderr": oversized_output}},
        },
        "layer2": {
            "failure_signatures": ["test_execution_failure"],
            "test_execution": {
                "returncode": 1,
                "stdout": oversized_output,
                "stderr": oversized_output,
            },
            "requirement_semantic_checks": {"duplicated": oversized_output},
        },
        "layer3": {
            "failure_signatures": ["non_semantic_test"],
            "non_semantic_test_reasons": {
                "tests/test_cli.py": ["missing_target_invocation"]
            },
            "semantic_requirement_test_coverage": {"duplicated": oversized_output},
        },
        "obligation_acceptance_checks": {"duplicated": oversized_output},
    }

    compact = SubstrateRepairBackend._compact_validator_evidence(evidence)
    serialized = json.dumps(compact, sort_keys=True)

    assert len(serialized) < 12_000
    assert compact["layer2"]["test_execution"]["returncode"] == 1
    assert compact["layer3"]["non_semantic_test_reasons"] == {
        "tests/test_cli.py": ["missing_target_invocation"]
    }
    assert "requirement_semantic_checks" not in compact["layer2"]
    assert "semantic_requirement_test_coverage" not in compact["layer3"]
    assert "obligation_acceptance_checks" not in compact


def test_substrate_backend_repairs_sources_atomically_then_shares_them_with_tests():
    plan = _generic_plan()
    artifact = CoderStage().generate(plan)
    artifact.files.append(
        GeneratedFile(
            path="src/helper.py",
            content="def transform(value):\n    return value\n",
            kind="python_module",
        )
    )
    test_path = artifact.test_paths[0]
    validation = ValidationArtifact(
        passed=False,
        failures=["Cross-file behavior is inconsistent."],
        failure_signatures=["semantic_content_mismatch"],
        evidence={"layer2": {}},
    )
    directive = RepairPolicy().compile(validation, plan, artifact, attempt=2)
    directive.target_paths = ["src/component.py", test_path, "src/helper.py"]
    kernel = _PerTargetKernel()
    backend = SubstrateRepairBackend(
        substrate=_StaticSubstrate(),
        kernel=kernel,
        test_preflight_runner=_passing_preflight,
    )

    candidate = backend.propose(plan, artifact, validation, directive)

    assert [call["target_path"] for call in kernel.calls] == ["source_transaction", test_path]
    assert kernel.calls[0]["target_paths"] == ["src/component.py", "src/helper.py"]
    assert kernel.calls[0]["related_sources"] == {}
    assert set(kernel.calls[1]["related_sources"]) == {"src/component.py", "src/helper.py"}
    assert "# revised:src/component.py" in kernel.calls[1]["related_sources"]["src/component.py"]
    assert "# revised:src/helper.py" in kernel.calls[1]["related_sources"]["src/helper.py"]
    assert "src/component.py" in kernel.calls[1]["source_api_contracts"]
    assert test_path in kernel.calls[1]["test_generation_contracts"]
    assert kernel.calls[1]["repair_phase"] == "test_suite_generation"
    assert set(candidate.files) == {"src/component.py", "src/helper.py", test_path}
    assert candidate.evidence["test_preflight_attempts"][0]["passed"] is True


def test_source_repair_expands_to_every_required_acceptance_test():
    plan = _generic_plan()
    plan.file_tree_plan.append(
        PlanFile(
            path="tests/test_plan_smoke.py",
            purpose="Planned end-to-end regression test.",
            source_requirement_refs=["R001"],
        )
    )
    plan.required_tests.append(
        PlanTest(
            test_name="test_regression",
            objective="Preserve existing workflow behavior.",
            requirement_ids=["R001"],
        )
    )
    artifact = CoderStage().generate(plan)
    primary_test = "tests/test_component.py"
    regression_test = "tests/test_regression.py"
    planned_test = "tests/test_plan_smoke.py"
    validation = ValidationArtifact(
        passed=False,
        failures=["Source behavior is incomplete."],
        failure_signatures=["semantic_content_mismatch"],
        evidence={"layer2": {}},
    )
    directive = RepairPolicy().compile(validation, plan, artifact, attempt=2)
    directive.target_paths = ["src/component.py", primary_test]
    kernel = _PerTargetKernel()
    backend = SubstrateRepairBackend(
        substrate=_StaticSubstrate(),
        kernel=kernel,
        test_preflight_runner=_passing_preflight,
    )

    candidate = backend.propose(plan, artifact, validation, directive)

    assert [call["target_path"] for call in kernel.calls] == [
        "src/component.py",
        "test_suite_transaction",
    ]
    assert set(kernel.calls[1]["target_paths"]) == {
        primary_test,
        planned_test,
        regression_test,
    }
    assert candidate.evidence["impact_expanded_paths"] == [planned_test, regression_test]
    assert set(candidate.files) == {
        "src/component.py",
        primary_test,
        planned_test,
        regression_test,
    }


def test_atomic_source_repair_rejects_partial_revision_and_skips_tests():
    plan = _generic_plan()
    artifact = CoderStage().generate(plan)
    artifact.files.append(
        GeneratedFile(path="src/helper.py", content="VALUE = 1\n", kind="python_module")
    )
    test_path = artifact.test_paths[0]
    validation = ValidationArtifact(
        passed=False,
        failures=["Semantic mismatch."],
        failure_signatures=["semantic_content_mismatch"],
        evidence={"layer2": {}},
    )
    directive = RepairPolicy().compile(validation, plan, artifact, attempt=2)
    directive.target_paths = ["src/component.py", "src/helper.py", test_path]
    kernel = _OmittingAtomicKernel()
    backend = SubstrateRepairBackend(substrate=_StaticSubstrate(), kernel=kernel)

    candidate = backend.propose(plan, artifact, validation, directive)

    assert len(kernel.calls) == 1
    assert set(kernel.calls[0]) == {"src/component.py", "src/helper.py"}
    assert candidate.files == {}
    assert candidate.available is True
    assert set(candidate.evidence["omitted_paths"]) == {"src/component.py", "src/helper.py", test_path}
    assert "Atomic revision omitted required targets" in candidate.stop_reason


def test_substrate_backend_generates_multiple_tests_as_one_atomic_suite():
    plan = _generic_plan()
    artifact = CoderStage().generate(plan)
    second_path = "tests/test_secondary.py"
    artifact.files.append(
        GeneratedFile(
            path=second_path,
            content="def test_secondary():\n    assert True\n",
            kind="python_test",
            generated_from_plan_sections=["requirement:R001"],
        )
    )
    artifact.test_paths.append(second_path)
    artifact.traceability[second_path] = ["requirement:R001"]
    validation = ValidationArtifact(
        passed=False,
        failures=["Tests are non-semantic."],
        failure_signatures=["non_semantic_test"],
        evidence={
            "layer3": {
                "non_semantic_tests": list(artifact.test_paths),
            }
        },
    )
    directive = RepairPolicy().compile(validation, plan, artifact, attempt=2)
    kernel = _PerTargetKernel()
    backend = SubstrateRepairBackend(
        substrate=_StaticSubstrate(),
        kernel=kernel,
        test_preflight_runner=_passing_preflight,
    )

    candidate = backend.propose(plan, artifact, validation, directive)

    assert len(kernel.calls) == 1
    assert set(kernel.calls[0]["target_paths"]) == set(artifact.test_paths)
    assert kernel.calls[0]["target_count"] == 2
    assert kernel.calls[0]["target_path"] == "test_suite_transaction"
    assert kernel.calls[0]["repair_phase"] == "test_suite_generation"
    assert set(kernel.calls[0]["test_generation_contracts"]) == set(artifact.test_paths)
    primary_contract = kernel.calls[0]["test_generation_contracts"][artifact.test_paths[0]]
    assert primary_contract["requirements"] == [
        {
            "id": "R001",
            "text": "Execute a workflow.",
            "evidence_terms": ["workflow"],
        }
    ]
    assert primary_contract["forbidden_unrequested_behaviors"] == [
        "SQLite or database persistence assertions",
        "records, audit, or schema table assertions",
    ]
    assert primary_contract["declared_plan_interfaces"][0]["name"] == "run"
    assert set(candidate.files) == set(artifact.test_paths)


class _PreflightCorrectionKernel:
    use_live_model = True

    def __init__(self):
        self.calls = []

    def propose_code_revision(self, repair_context, target_files, lens_framings):
        self.calls.append(
            {
                "phase": repair_context.get("repair_phase"),
                "preflight": repair_context.get("preflight_test_execution"),
                "targets": list(target_files),
            }
        )
        path = next(iter(target_files))
        if repair_context.get("repair_phase") == "test_suite_correction":
            content = (
                "from pathlib import Path\n"
                "import sys\n"
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                "from component import run\n\n"
                "def test_component():\n"
                "    assert run() == 0\n"
            )
        else:
            content = "def test_component():\n    assert 1 == 2\n"
        return {"status": "candidate", "files": {path: content}}


class _SourcePreflightCorrectionKernel:
    use_live_model = True

    def __init__(self):
        self.calls = []

    def propose_code_revision(self, repair_context, target_files, lens_framings):
        phase = repair_context.get("repair_phase", "file_revision")
        self.calls.append(
            {
                "phase": phase,
                "targets": list(target_files),
                "preflight": repair_context.get("preflight_test_execution"),
                "candidate_tests": repair_context.get("candidate_test_suite"),
            }
        )
        files = {}
        for path, content in target_files.items():
            if path.startswith("src/"):
                return_value = 0 if phase == "source_preflight_correction" else 1
                files[path] = f"def run() -> int:\n    return {return_value}\n"
            else:
                files[path] = (
                    "from pathlib import Path\n"
                    "import sys\n"
                    "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
                    "from component import run\n\n"
                    "def test_component():\n"
                    "    assert run() == 0\n"
                )
        return {"status": "candidate", "files": files}


def test_substrate_backend_uses_real_preflight_failure_to_correct_test_suite():
    plan = _generic_plan()
    artifact = CoderStage().generate(plan)
    test_path = artifact.test_paths[0]
    validation = ValidationArtifact(
        passed=False,
        failures=["Generated test is non-semantic."],
        failure_signatures=["non_semantic_test"],
        evidence={"layer3": {"non_semantic_tests": [test_path]}},
    )
    directive = RepairPolicy().compile(validation, plan, artifact, attempt=2)
    kernel = _PreflightCorrectionKernel()
    backend = SubstrateRepairBackend(
        substrate=_StaticSubstrate(),
        kernel=kernel,
        preflight_timeout_seconds=20,
    )

    candidate = backend.propose(plan, artifact, validation, directive)

    assert [call["phase"] for call in kernel.calls] == [
        "test_suite_generation",
        "test_suite_correction",
    ]
    assert kernel.calls[1]["preflight"]["returncode"] == 1
    assert "1 failed" in kernel.calls[1]["preflight"]["stdout"]
    assert "from component import run" in candidate.files[test_path]
    assert [attempt["passed"] for attempt in candidate.evidence["test_preflight_attempts"]] == [
        False,
        True,
    ]


def test_substrate_backend_routes_application_preflight_failure_back_to_sources():
    plan = _generic_plan()
    artifact = CoderStage().generate(plan)
    test_path = artifact.test_paths[0]
    validation = ValidationArtifact(
        passed=False,
        failures=["Source semantics are incomplete."],
        failure_signatures=["semantic_content_mismatch"],
        evidence={"layer2": {}},
    )
    directive = RepairPolicy().compile(validation, plan, artifact, attempt=2)
    directive.target_paths = ["src/component.py", test_path]
    kernel = _SourcePreflightCorrectionKernel()
    backend = SubstrateRepairBackend(
        substrate=_StaticSubstrate(),
        kernel=kernel,
        preflight_timeout_seconds=20,
    )

    candidate = backend.propose(plan, artifact, validation, directive)

    assert [call["phase"] for call in kernel.calls] == [
        "file_revision",
        "test_suite_generation",
        "source_preflight_correction",
    ]
    source_correction = kernel.calls[2]
    assert source_correction["preflight"]["returncode"] == 1
    assert test_path in source_correction["candidate_tests"]
    assert "return 0" in candidate.files["src/component.py"]
    assert "assert run() == 0" in candidate.files[test_path]
    assert [attempt["passed"] for attempt in candidate.evidence["test_preflight_attempts"]] == [
        False,
        True,
    ]


def test_substrate_backend_retries_source_when_source_correction_introduces_new_error():
    plan = _generic_plan()
    artifact = CoderStage().generate(plan)
    test_path = artifact.test_paths[0]
    validation = ValidationArtifact(
        passed=False,
        failures=["Source semantics are incomplete."],
        failure_signatures=["semantic_content_mismatch"],
        evidence={"layer2": {}},
    )
    directive = RepairPolicy().compile(validation, plan, artifact, attempt=2)
    directive.target_paths = ["src/component.py", test_path]
    kernel = _PerTargetKernel()
    preflight_results = iter(
        [
            {
                "ran": True,
                "passed": False,
                "returncode": 1,
                "tests": [test_path],
                "stdout": "src/component.py:2: TypeError",
                "stderr": "",
            },
            {
                "ran": True,
                "passed": False,
                "returncode": 2,
                "tests": [test_path],
                "stdout": "C:\\Temp\\run\\src\\component.py:1: NameError: Any",
                "stderr": "",
            },
            {
                "ran": True,
                "passed": True,
                "returncode": 0,
                "tests": [test_path],
                "stdout": "1 passed",
                "stderr": "",
            },
        ]
    )
    backend = SubstrateRepairBackend(
        substrate=_StaticSubstrate(),
        kernel=kernel,
        test_preflight_runner=lambda files, paths: next(preflight_results),
    )

    candidate = backend.propose(plan, artifact, validation, directive)

    assert [call["repair_phase"] for call in kernel.calls] == [
        "file_revision",
        "test_suite_generation",
        "source_preflight_correction",
        "source_preflight_correction",
    ]
    assert [attempt["passed"] for attempt in candidate.evidence["test_preflight_attempts"]] == [
        False,
        False,
        True,
    ]
    assert set(candidate.files) == {"src/component.py", test_path}


def test_substrate_backend_returns_to_source_after_test_correction_exposes_source_failure():
    plan = _generic_plan()
    artifact = CoderStage().generate(plan)
    test_path = artifact.test_paths[0]
    validation = ValidationArtifact(
        passed=False,
        failures=["Source and tests require semantic repair."],
        failure_signatures=["semantic_content_mismatch"],
        evidence={"layer2": {}},
    )
    directive = RepairPolicy().compile(validation, plan, artifact, attempt=2)
    directive.target_paths = ["src/component.py", test_path]
    kernel = _PerTargetKernel()
    preflight_results = iter(
        [
            {
                "ran": True,
                "passed": False,
                "returncode": 2,
                "tests": [test_path],
                "stdout": "tests/test_component.py:1: SyntaxError",
                "stderr": "",
            },
            {
                "ran": True,
                "passed": False,
                "returncode": 2,
                "tests": [test_path],
                "stdout": "tests/test_component.py:1: SyntaxError",
                "stderr": "",
            },
            {
                "ran": True,
                "passed": False,
                "returncode": 1,
                "tests": [test_path],
                "stdout": "src/component.py:2: AttributeError",
                "stderr": "",
            },
            {
                "ran": True,
                "passed": True,
                "returncode": 0,
                "tests": [test_path],
                "stdout": "1 passed",
                "stderr": "",
            },
        ]
    )
    backend = SubstrateRepairBackend(
        substrate=_StaticSubstrate(),
        kernel=kernel,
        test_preflight_runner=lambda files, paths: next(preflight_results),
    )

    candidate = backend.propose(plan, artifact, validation, directive)

    assert [call["repair_phase"] for call in kernel.calls] == [
        "file_revision",
        "test_suite_generation",
        "source_preflight_correction",
        "test_suite_correction",
        "source_preflight_correction",
    ]
    assert [attempt["passed"] for attempt in candidate.evidence["test_preflight_attempts"]] == [
        False,
        False,
        False,
        True,
    ]
    assert set(candidate.files) == {"src/component.py", test_path}


def test_substrate_backend_corrects_only_the_test_identified_by_candidate_gate():
    plan = _generic_plan()
    plan.required_tests.append(
        PlanTest(
            test_name="test_other",
            objective="Verify another workflow behavior.",
            requirement_ids=["R001"],
        )
    )
    artifact = CoderStage().generate(plan)
    first_test, second_test = sorted(artifact.test_paths)
    validation = ValidationArtifact(
        passed=False,
        failures=["Generated tests require correction."],
        failure_signatures=["semantic_content_mismatch"],
        evidence={"layer2": {}},
    )
    directive = RepairPolicy().compile(validation, plan, artifact, attempt=2)
    directive.target_paths = ["src/component.py", first_test]
    kernel = _PerTargetKernel()
    preflight_results = iter(
        [
            {
                "phase": "syntax",
                "ran": False,
                "passed": False,
                "returncode": None,
                "tests": [first_test, second_test],
                "stdout": "",
                "stderr": "",
                "failed_paths": [first_test],
                "source_failed_paths": [],
                "test_failed_paths": [first_test],
            },
            {
                "phase": "tests",
                "ran": True,
                "passed": True,
                "returncode": 0,
                "tests": [first_test, second_test],
                "stdout": "2 passed",
                "stderr": "",
                "failed_paths": [],
                "source_failed_paths": [],
                "test_failed_paths": [],
            },
        ]
    )
    backend = SubstrateRepairBackend(
        substrate=_StaticSubstrate(),
        kernel=kernel,
        test_preflight_runner=lambda files, paths: next(preflight_results),
    )

    candidate = backend.propose(plan, artifact, validation, directive)

    assert [call["repair_phase"] for call in kernel.calls] == [
        "file_revision",
        "test_suite_generation",
        "test_suite_correction",
    ]
    assert kernel.calls[1]["target_paths"] == [first_test, second_test]
    assert kernel.calls[2]["target_paths"] == [first_test]
    assert set(candidate.files) == {"src/component.py", first_test, second_test}


def test_substrate_backend_discards_test_suite_when_preflight_correction_still_fails():
    plan = _generic_plan()
    artifact = CoderStage().generate(plan)
    test_path = artifact.test_paths[0]
    validation = ValidationArtifact(
        passed=False,
        failures=["Generated test is non-semantic."],
        failure_signatures=["non_semantic_test"],
        evidence={"layer3": {"non_semantic_tests": [test_path]}},
    )
    directive = RepairPolicy().compile(validation, plan, artifact, attempt=2)
    kernel = _PerTargetKernel()

    def failing_preflight(candidate_files, test_paths):
        return {
            "ran": True,
            "passed": False,
            "returncode": 1,
            "tests": list(test_paths),
            "stdout": "1 failed",
            "stderr": "",
        }

    backend = SubstrateRepairBackend(
        substrate=_StaticSubstrate(),
        kernel=kernel,
        test_preflight_runner=failing_preflight,
    )

    candidate = backend.propose(plan, artifact, validation, directive)

    assert candidate.files == {}
    assert candidate.evidence["omitted_paths"] == [test_path]
    assert len(candidate.evidence["test_preflight_attempts"]) == 4
    assert "failed executable preflight" in candidate.stop_reason


def test_coder_applies_grounded_candidate_with_revision_and_lineage():
    plan = _generic_plan()
    original = CoderStage().generate(plan)
    validation = ValidationArtifact(
        passed=False,
        failures=["Implementation-level syntax failure."],
        failure_signatures=["syntax_error"],
        evidence={"layer1": {"parse_errors": [{"path": "src/component.py"}]}},
    )
    directive = RepairPolicy().compile(validation, plan, original, attempt=2)
    coder = CoderStage(repair_backend=_StaticRepairBackend())

    result = coder.repair(plan, original, validation, directive)

    assert result.changed is True
    assert result.backend_name == "test_grounded_backend"
    assert result.changed_paths == ["forge_artifact_manifest.json", "src/component.py"]
    assert result.artifact.revision == 2
    assert result.artifact.parent_artifact_id == original.artifact_id
    assert result.artifact.repair_history[-1]["backend_name"] == "test_grounded_backend"
    repaired_source = next(
        file.content for file in result.artifact.files if file.path == "src/component.py"
    )
    assert "REPAIR_MARKER = True" in repaired_source
    repaired_test = next(
        file.content for file in result.artifact.files if file.path == "tests/test_component.py"
    )
    original_test = next(
        file.content for file in original.files if file.path == "tests/test_component.py"
    )
    assert repaired_test == original_test


def test_coder_applies_preflighted_impact_expanded_tests_with_source_repair():
    plan = _generic_plan()
    original = CoderStage().generate(plan)
    test_path = original.test_paths[0]
    validation = ValidationArtifact(
        passed=False,
        failures=["Source behavior requires repair."],
        failure_signatures=["semantic_content_mismatch"],
        evidence={"layer2": {}},
    )
    directive = RepairPolicy().compile(validation, plan, original, attempt=2)
    directive.target_paths = ["src/component.py"]

    result = CoderStage(repair_backend=_ImpactExpandedRepairBackend()).repair(
        plan,
        original,
        validation,
        directive,
    )

    assert result.changed is True
    assert "src/component.py" in result.changed_paths
    assert test_path in result.changed_paths
    repaired_test = next(
        file.content for file in result.artifact.files if file.path == test_path
    )
    assert "impact regression repair" in repaired_test


class _RaisingRepairBackend:
    def propose(self, plan, artifact, validation, directive):
        raise RuntimeError("backend unavailable")


def test_coder_fails_closed_when_grounded_backend_raises():
    plan = _generic_plan()
    original = CoderStage().generate(plan)
    validation = ValidationArtifact(
        passed=False,
        failures=["Syntax error."],
        failure_signatures=["syntax_error"],
        evidence={"layer1": {"parse_errors": [{"path": "src/component.py"}]}},
    )
    directive = RepairPolicy().compile(validation, plan, original, attempt=2)

    result = CoderStage(repair_backend=_RaisingRepairBackend()).repair(
        plan,
        original,
        validation,
        directive,
    )

    assert result.changed is False
    assert result.artifact is original
    assert result.backend_name == "_RaisingRepairBackend"
    assert result.backend_evidence == {"error_type": "RuntimeError"}
    assert "failed" in result.stop_reason.lower()


class _Responses:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            output_text=(
                '{"status":"candidate","files":'
                '[{"path":"src/component.py","content":"def run():\\n    return 2\\n"}]}'
            )
        )


def test_reasoning_kernel_returns_typed_revision_payload_without_execution_claims():
    kernel = ReasoningKernel(api_key="dummy_key_for_testing", execution_mode="local-only")
    responses = _Responses()
    kernel.use_live_model = True
    kernel.client = SimpleNamespace(responses=responses)
    framing = SimpleNamespace(
        lens_name="Formal Logic",
        framing="Preserve interfaces.",
        constraints=["Return valid Python."],
        blind_spots=[],
        operator_primitives=["verify"],
    )

    payload = kernel.propose_code_revision(
        repair_context={"failure_signatures": ["syntax_error"]},
        target_files={"src/component.py": "def run(:\n"},
        lens_framings=[framing],
    )

    assert payload["status"] == "candidate"
    assert payload["files"][0]["path"] == "src/component.py"
    assert "complete replacements" in responses.request["instructions"]
    assert responses.request["max_output_tokens"] == 12000
    assert responses.request["model"] == "gpt-4.1-mini"
    revision_schema = responses.request["text"]["format"]["schema"]
    assert revision_schema["properties"]["files"]["required"] == ["src/component.py"]
    assert revision_schema["properties"]["files"]["additionalProperties"] is False
    assert "one replacement for every supplied target path" in responses.request["instructions"]
    assert "flat src/ layout" in responses.request["instructions"]
    assert "related_repaired_source_files" in responses.request["instructions"]
    assert "must not use skip" in " ".join(responses.request["instructions"].split())
    assert "source_api_contracts" in responses.request["instructions"]
    assert "preflight_test_execution" in responses.request["instructions"]
    assert "Path.read_text() performs universal-newline translation" in responses.request[
        "instructions"
    ]


def test_reasoning_kernel_preserves_sanitized_live_revision_error_detail():
    kernel = ReasoningKernel(api_key="dummy_key_for_testing", execution_mode="local-only")

    class FailingResponses:
        @staticmethod
        def create(**kwargs):
            raise ValueError("response incomplete: max_output_tokens")

    kernel.use_live_model = True
    kernel.client = SimpleNamespace(responses=FailingResponses())

    payload = kernel.propose_code_revision(
        repair_context={"failure_signatures": ["semantic_content_mismatch"]},
        target_files={"src/component.py": "def run():\n    return 1\n"},
        lens_framings=[],
    )

    assert payload["status"] == "unavailable"
    assert payload["reason"] == (
        "Live revision failed: ValueError: response incomplete: max_output_tokens"
    )
