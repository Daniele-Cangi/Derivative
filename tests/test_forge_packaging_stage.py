import copy
import json
from pathlib import Path

import pytest

from core.forge.coder_stage import CoderStage
from core.forge.contracts import FeasiblePlan, PackagedArtifact
from core.forge.packaging_stage import PackagingRefusedError, PackagingStage
from core.forge.planner_stage import PlannerStage
from core.forge.requirement_compiler import RequirementCompiler
from core.forge.validator_stage import ValidatorStage


FEASIBLE_REQUIREMENT = (
    "Build a Python CLI that reads a CSV of contracts, extracts expiration dates, "
    "flags contracts expiring in less than 90 days, writes a summary CSV, and includes tests."
)


@pytest.fixture(scope="module")
def forge_packaging_context(tmp_path_factory):
    root = tmp_path_factory.mktemp("forge_packaging_stage")
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

    artifact = CoderStage().generate(planned)
    validator = ValidatorStage()
    passing_validation = validator.validate(artifact, planned, spec)

    broken_artifact = copy.deepcopy(artifact)
    broken_artifact.files = [file for file in broken_artifact.files if file.path != "src/summary_writer.py"]
    broken_artifact.traceability.pop("src/summary_writer.py", None)
    failing_validation = validator.validate(broken_artifact, planned, spec)

    return {
        "root": root,
        "build_spec": spec,
        "plan": planned,
        "artifact": artifact,
        "passing_validation": passing_validation,
        "failing_validation": failing_validation,
    }


def test_packaging_succeeds_only_for_passed_validation(forge_packaging_context):
    stage = PackagingStage(output_root=str(forge_packaging_context["root"] / "packages"))
    packaged = stage.package(
        forge_packaging_context["build_spec"],
        forge_packaging_context["plan"],
        forge_packaging_context["artifact"],
        forge_packaging_context["passing_validation"],
    )

    assert isinstance(packaged, PackagedArtifact)
    assert Path(packaged.package_root).exists()
    assert Path(packaged.manifest_path).exists()
    assert "forge_package_manifest.json" in packaged.packaged_files
    assert packaged.verification_metadata.get("terminal_status") == "verified"


def test_packaging_refuses_failed_validation(forge_packaging_context):
    stage = PackagingStage(output_root=str(forge_packaging_context["root"] / "packages_refuse"))

    with pytest.raises(PackagingRefusedError):
        stage.package(
            forge_packaging_context["build_spec"],
            forge_packaging_context["plan"],
            forge_packaging_context["artifact"],
            forge_packaging_context["failing_validation"],
        )


def test_package_manifest_includes_ids_and_validation_summary(forge_packaging_context):
    stage = PackagingStage(output_root=str(forge_packaging_context["root"] / "packages_manifest"))
    packaged = stage.package(
        forge_packaging_context["build_spec"],
        forge_packaging_context["plan"],
        forge_packaging_context["artifact"],
        forge_packaging_context["passing_validation"],
    )

    manifest_path = Path(packaged.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["terminal_status"] == "verified"
    assert manifest["build_id"] == forge_packaging_context["build_spec"].build_id
    assert manifest["plan_id"] == forge_packaging_context["plan"].plan_id
    assert manifest["artifact_id"] == forge_packaging_context["artifact"].artifact_id
    assert manifest["validation_summary"]["passed"] is True
    assert "evidence_refs" in manifest
    assert "manifest_paths" in manifest

    package_root = Path(packaged.package_root)
    assert (package_root / manifest["evidence_refs"]["validation_evidence"]).exists()
    assert (package_root / manifest["evidence_refs"]["artifact_manifest_dump"]).exists()
