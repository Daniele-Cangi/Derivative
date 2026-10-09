from pathlib import Path
from types import SimpleNamespace

import pytest

from core.forge.artifact_files import ArtifactPathError, artifact_file_targets
from core.forge.contracts import (
    CodeArtifact, FailureCategory, FeasiblePlan, ForgeRoute, GeneratedFile,
    ValidationArtifact, ValidationLayerResult,
)
from core.forge.evidence_integrity import artifact_validation_seal, validation_artifact_seal
from core.forge.packaging_stage import PackagingRefusedError, PackagingStage
from core.forge.repair_support import behavioral_contract_seal
from core.forge.requirement_compiler import RequirementCompiler
from core.forge.validator_stage import ValidatorStage
from forge import _retry_route_for_validation


@pytest.fixture
def context():
    spec = RequirementCompiler().compile("Build a Python library exposing run() and include tests.")
    plan = FeasiblePlan("plan-file-boundary", spec, "Unit-test file boundary")
    artifact = CodeArtifact("artifact-file-boundary", plan.plan_id, files=[
        GeneratedFile("src/worker.py", "def run():\n    return 1\n", "python"),
        GeneratedFile("tests/test_worker.py", "def test_worker():\n    assert 1 == 1\n", "python"),
    ])
    return spec, plan, artifact


def simulated_validator(action=None, action_index=0):
    calls = []
    stage = ValidatorStage()
    def layer(index):
        def validate(artifact, plan, spec, materialized, workspace):
            calls.append(index)
            if action is not None and index == action_index:
                action(artifact, materialized, workspace)
            return ValidationLayerResult(f"simulated-{index}", True)
        return SimpleNamespace(validate=validate)
    stage.runtime_layer, stage.obligation_layer, stage.adversarial_layer = [layer(i) for i in range(3)]
    return stage, calls


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("change", ["source", "test", "missing", "artifact", "metadata"])
def test_validator_refuses_changed_artifact_before_next_layer(context, tmp_path, index, change):
    spec, plan, artifact = context
    def action(artifact, materialized, workspace):
        if change in {"source", "test"}:
            path = "src/worker.py" if change == "source" else "tests/test_worker.py"
            materialized[path].write_bytes(b"different temporary content\n")
        elif change == "missing":
            materialized["src/worker.py"].unlink()
        elif change == "artifact":
            artifact.files[0].content += "# changed object\n"
        else:
            artifact.artifact_manifest["changed"] = True
    stage, calls = simulated_validator(action, index)
    result = stage.validate(artifact, plan, spec)
    assert not result.passed
    assert result.failure_signatures == ["artifact_integrity_violation"]
    assert calls == list(range(index + 1))
    assert not result.evidence["workspace_integrity"]["checkpoints"][-1]["passed"]
    assert _retry_route_for_validation(result) == ForgeRoute.TERMINAL_VALIDATION_FAILED
    output = tmp_path / "packages"
    with pytest.raises(PackagingRefusedError):
        PackagingStage(str(output)).package(spec, plan, artifact, result)
    assert not output.exists()


def test_validator_allows_outputs_but_preserves_declared_file_bytes(context, tmp_path):
    spec, plan, artifact = context
    # Exercise platform newline handling without normalizing away file changes.
    artifact.files[0].content = "def run():\r\n    return 1\r\n"
    def output_only(artifact, materialized, workspace):
        (workspace / "summary.csv").write_text("result\n1\n", encoding="utf-8")
    stage, calls = simulated_validator(output_only)
    result = stage.validate(artifact, plan, spec)
    assert result.passed
    assert calls == [0, 1, 2]
    assert all(item["passed"] for item in result.evidence["workspace_integrity"]["checkpoints"])
    package = PackagingStage(str(tmp_path / "packages")).package(spec, plan, artifact, result)
    assert (Path(package.package_root) / "src/worker.py").exists()


@pytest.mark.parametrize("path", ["../other.txt", "/other.txt", "C:/other.txt", "C:other.txt",
    "\\\\server\\share\\other.txt", "src/../other.txt", "src//other.txt", "src\\other.txt",
    "src/./other.txt", "", "src/data:stream", "src/worker.py.", "src/worker.py ",
    "src/NUL.py", "src/CON", "src/a?.py", "src/a*.py", "src/file|name.py",
    "src/com1.txt", "src/LPT².txt", "src/data\x01.txt"])
def test_validator_checks_all_paths_before_writing_or_running(context, tmp_path, path):
    spec, plan, artifact = context
    artifact.files.append(GeneratedFile(path, "unit-test data", "text"))
    stage, calls = simulated_validator()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ArtifactPathError):
        stage._materialize_workspace(artifact, workspace)
    assert list(workspace.iterdir()) == []
    result = stage.validate(artifact, plan, spec)
    assert not result.passed
    assert result.failure_signatures == ["artifact_path_violation"]
    assert calls == []
    assert _retry_route_for_validation(result) == ForgeRoute.TERMINAL_VALIDATION_FAILED


@pytest.mark.parametrize("paths", [
    ["src/file.py", "src/file.py"], ["src/File.py", "src/file.py"],
    ["src", "src/file.py"],
])
def test_conflicting_artifact_paths_are_rejected(tmp_path, paths):
    with pytest.raises(ArtifactPathError):
        artifact_file_targets([GeneratedFile(path, "data", "text") for path in paths], tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("path", ["../other.txt", "/other.txt", "C:/other.txt",
    "forge_package_manifest.json", "validation_evidence.json", "code_artifact_manifest_dump.json",
    "validation_evidence.json/child.txt"])
def test_packaging_checks_paths_even_with_matching_stage_seals(context, tmp_path, path):
    spec, plan, artifact = context
    stage, _ = simulated_validator()
    # Seal a test fixture without materializing an invalid path. This isolates
    # Packaging's independent path check, not a production validation bypass.
    artifact.files.append(GeneratedFile(path, "unit-test data", "text"))
    result = stage._validation_refusal({}, "fixture", "fixture")
    result.passed = True
    result.evidence["behavioral_contract_seal"] = behavioral_contract_seal(plan)
    result.evidence["validated_artifact_seal"] = artifact_validation_seal(artifact)
    result.integrity_seal = validation_artifact_seal(result)
    output = tmp_path / "packages"
    with pytest.raises(PackagingRefusedError):
        PackagingStage(str(output)).package(spec, plan, artifact, result)
    assert not output.exists()


def test_file_targets_refuse_symbolic_link_components(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    redirect = tmp_path / "redirect"
    try:
        redirect.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("Symbolic link creation is not available on this host")
    with pytest.raises(ArtifactPathError):
        artifact_file_targets([GeneratedFile("redirect/file.txt", "data", "text")], tmp_path)
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("signature", ["artifact_path_violation", "artifact_integrity_violation"])
@pytest.mark.parametrize("other", ["semantic_content_mismatch", "semantic_omission", "syntax_error"])
def test_file_boundary_refusal_takes_priority_over_repair_routes(signature, other):
    validation = ValidationArtifact(
        passed=False, failure_signatures=[other, signature],
        failure_category=FailureCategory.ARCHITECTURAL,
    )
    assert _retry_route_for_validation(validation) == ForgeRoute.TERMINAL_VALIDATION_FAILED
