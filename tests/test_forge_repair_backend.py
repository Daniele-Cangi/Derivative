from types import SimpleNamespace

from core.forge.coder_stage import CoderStage
from core.forge.contracts import (
    AcceptanceContract,
    ArtifactTargetType,
    BuildSpec,
    FeasiblePlan,
    ObligationContract,
    PlanFile,
    PlanInterface,
    PlanTest,
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
        required_tests=[PlanTest(test_name="test_component", objective="Execute the workflow.")],
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


class _StaticKernel:
    use_live_model = True

    def __init__(self, files):
        self.files = files
        self.received_targets = None

    def propose_code_revision(self, repair_context, target_files, lens_framings):
        self.received_targets = target_files
        assert repair_context["failure_signatures"] == ["syntax_error"]
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


class _Messages:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    text=(
                        '{"status":"candidate","files":'
                        '[{"path":"src/component.py","content":"def run():\\n    return 2\\n"}]}'
                    )
                )
            ]
        )


def test_reasoning_kernel_returns_typed_revision_payload_without_execution_claims():
    kernel = ReasoningKernel(api_key="dummy_key_for_testing", execution_mode="local-only")
    messages = _Messages()
    kernel.use_live_model = True
    kernel.client = SimpleNamespace(messages=messages)
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
    assert "complete replacements" in messages.request["system"]
    assert messages.request["max_tokens"] == 4000
