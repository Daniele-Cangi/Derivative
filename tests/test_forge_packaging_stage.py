import copy
import hashlib
import json
from pathlib import Path

import pytest

from core.forge.coder_stage import CoderStage
from core.forge.contracts import FeasiblePlan, PackagedArtifact
from core.forge.evidence_integrity import (
    artifact_validation_seal,
    validation_artifact_seal,
)
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


def test_packaging_preserves_binary_validation_evidence(forge_packaging_context):
    validation = copy.deepcopy(forge_packaging_context["passing_validation"])
    validation.evidence["binary_probe"] = b"\x00\xff"
    validation.integrity_seal = validation_artifact_seal(validation)
    stage = PackagingStage(
        output_root=str(forge_packaging_context["root"] / "packages_binary_evidence")
    )

    packaged = stage.package(
        forge_packaging_context["build_spec"],
        forge_packaging_context["plan"],
        forge_packaging_context["artifact"],
        validation,
    )

    manifest = json.loads(Path(packaged.manifest_path).read_text(encoding="utf-8"))
    validation_path = (
        Path(packaged.package_root)
        / manifest["evidence_refs"]["validation_evidence"]
    )
    document = json.loads(validation_path.read_text(encoding="utf-8"))
    assert document["validation_artifact"]["evidence"]["binary_probe"] == {
        "__forge_scalar__": "bytes",
        "hex": "00ff",
    }
    assert manifest["validation_receipt"]["sha256"] == hashlib.sha256(
        validation_path.read_bytes()
    ).hexdigest()


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
    assert manifest["artifact_revision"] == 1
    assert manifest["parent_artifact_id"] is None
    assert manifest["repair_history"] == []
    assert manifest["package_id"].startswith("pkg-")
    assert manifest["package_base_id"].startswith("pkg-")
    assert manifest["package_run_id"].startswith("pkg-")
    assert isinstance(manifest["code_artifact_digest"], str) and len(manifest["code_artifact_digest"]) == 64
    assert manifest["behavioral_contract_seal"] == (
        forge_packaging_context["passing_validation"].evidence[
            "behavioral_contract_seal"
        ]
    )
    assert packaged.verification_metadata["behavioral_contract_seal"] == (
        manifest["behavioral_contract_seal"]
    )
    assert manifest["validated_artifact_seal"] == artifact_validation_seal(
        forge_packaging_context["artifact"]
    )
    assert packaged.verification_metadata["validated_artifact_seal"] == (
        manifest["validated_artifact_seal"]
    )
    assert manifest["validation_artifact_seal"] == validation_artifact_seal(
        forge_packaging_context["passing_validation"]
    )
    assert packaged.verification_metadata["validation_artifact_seal"] == (
        manifest["validation_artifact_seal"]
    )
    assert manifest["validation_summary"]["passed"] is True
    assert "evidence_refs" in manifest
    assert "manifest_paths" in manifest

    package_root = Path(packaged.package_root)
    validation_path = package_root / manifest["evidence_refs"]["validation_evidence"]
    artifact_manifest_path = (
        package_root / manifest["evidence_refs"]["artifact_manifest_dump"]
    )
    assert validation_path.exists()
    assert artifact_manifest_path.exists()

    receipt = manifest["validation_receipt"]
    assert receipt["schema_version"] == 1
    assert receipt["digest_mode"] == "canonical_json_utf8_v1"
    assert receipt["sha256"] == hashlib.sha256(
        validation_path.read_bytes()
    ).hexdigest()
    assert receipt["artifact_manifest_sha256"] == hashlib.sha256(
        artifact_manifest_path.read_bytes()
    ).hexdigest()
    assert receipt["package_id"] == manifest["package_id"]
    assert receipt["validated_artifact_sha256"] == manifest[
        "validated_artifact_seal"
    ]["sha256"]
    assert receipt["validation_artifact_sha256"] == manifest[
        "validation_artifact_seal"
    ]["sha256"]
    assert packaged.verification_metadata["validation_receipt"] == receipt

    validation_document = json.loads(validation_path.read_text(encoding="utf-8"))
    context = validation_document["receipt_context"]
    assert context["package_id"] == manifest["package_id"]
    assert context["code_artifact_digest"] == manifest["code_artifact_digest"]
    assert context["validated_artifact_seal"] == manifest[
        "validated_artifact_seal"
    ]
    assert context["validation_artifact_seal"] == manifest[
        "validation_artifact_seal"
    ]
    assert validation_document["validation_artifact"]["passed"] is True


@pytest.mark.parametrize("mutation", ["missing", "mismatched"])
def test_packaging_refuses_invalid_behavioral_contract_seal(
    forge_packaging_context,
    mutation,
):
    validation = copy.deepcopy(forge_packaging_context["passing_validation"])
    if mutation == "missing":
        validation.evidence.pop("behavioral_contract_seal")
    else:
        validation.evidence["behavioral_contract_seal"]["sha256"] = "0" * 64
    output_root = forge_packaging_context["root"] / f"packages_bad_contract_{mutation}"
    stage = PackagingStage(output_root=str(output_root))

    with pytest.raises(PackagingRefusedError, match="behavioral contract seal"):
        stage.package(
            forge_packaging_context["build_spec"],
            forge_packaging_context["plan"],
            forge_packaging_context["artifact"],
            validation,
        )

    assert not output_root.exists()


def test_packaging_refuses_missing_validated_artifact_seal(forge_packaging_context):
    validation = copy.deepcopy(forge_packaging_context["passing_validation"])
    validation.evidence.pop("validated_artifact_seal")
    output_root = forge_packaging_context["root"] / "packages_missing_artifact_seal"
    stage = PackagingStage(output_root=str(output_root))

    with pytest.raises(PackagingRefusedError, match="validated artifact seal"):
        stage.package(
            forge_packaging_context["build_spec"],
            forge_packaging_context["plan"],
            forge_packaging_context["artifact"],
            validation,
        )

    assert not output_root.exists()


@pytest.mark.parametrize("mutation", ["missing", "mismatched"])
def test_packaging_refuses_invalid_validation_artifact_seal(
    forge_packaging_context,
    mutation,
):
    validation = copy.deepcopy(forge_packaging_context["passing_validation"])
    if mutation == "missing":
        validation.integrity_seal = {}
    else:
        validation.integrity_seal["sha256"] = "0" * 64
    output_root = forge_packaging_context["root"] / f"packages_bad_validation_{mutation}"
    stage = PackagingStage(output_root=str(output_root))

    with pytest.raises(PackagingRefusedError, match="validation artifact seal"):
        stage.package(
            forge_packaging_context["build_spec"],
            forge_packaging_context["plan"],
            forge_packaging_context["artifact"],
            validation,
        )

    assert not output_root.exists()


def test_packaging_refuses_build_not_bound_to_plan(forge_packaging_context):
    build_spec = copy.deepcopy(forge_packaging_context["build_spec"])
    build_spec.build_id = "build-not-validated-by-plan"
    output_root = forge_packaging_context["root"] / "packages_mismatched_build"
    stage = PackagingStage(output_root=str(output_root))

    with pytest.raises(PackagingRefusedError, match="build and plan identities"):
        stage.package(
            build_spec,
            forge_packaging_context["plan"],
            forge_packaging_context["artifact"],
            forge_packaging_context["passing_validation"],
        )

    assert not output_root.exists()


def test_packaging_refuses_artifact_changed_after_validation(forge_packaging_context):
    changed_artifact = copy.deepcopy(forge_packaging_context["artifact"])
    changed_artifact.files[0].content += "\n# changed after validation\n"
    output_root = forge_packaging_context["root"] / "packages_changed_artifact"
    stage = PackagingStage(output_root=str(output_root))

    with pytest.raises(PackagingRefusedError, match="validated artifact seal"):
        stage.package(
            forge_packaging_context["build_spec"],
            forge_packaging_context["plan"],
            changed_artifact,
            forge_packaging_context["passing_validation"],
        )

    assert not output_root.exists()


def test_packaging_refuses_validation_evidence_changed_after_validation(
    forge_packaging_context,
):
    changed_validation = copy.deepcopy(
        forge_packaging_context["passing_validation"]
    )
    changed_validation.evidence["receipt_probe"] = "changed"
    output_root = forge_packaging_context["root"] / "packages_changed_validation"
    stage = PackagingStage(output_root=str(output_root))

    assert changed_validation.integrity_seal != validation_artifact_seal(
        changed_validation
    )
    with pytest.raises(PackagingRefusedError, match="validation artifact seal"):
        stage.package(
            forge_packaging_context["build_spec"],
            forge_packaging_context["plan"],
            forge_packaging_context["artifact"],
            changed_validation,
        )

    assert not output_root.exists()


def test_packaging_creates_new_revision_directory_without_overwriting(forge_packaging_context):
    stage = PackagingStage(output_root=str(forge_packaging_context["root"] / "packages_revision"))

    first = stage.package(
        forge_packaging_context["build_spec"],
        forge_packaging_context["plan"],
        forge_packaging_context["artifact"],
        forge_packaging_context["passing_validation"],
    )
    second = stage.package(
        forge_packaging_context["build_spec"],
        forge_packaging_context["plan"],
        forge_packaging_context["artifact"],
        forge_packaging_context["passing_validation"],
    )

    first_root = Path(first.package_root)
    second_root = Path(second.package_root)

    assert first_root.exists()
    assert second_root.exists()
    assert first_root != second_root
    assert first.package_id != second.package_id
    assert second.package_id.endswith("-r02")
