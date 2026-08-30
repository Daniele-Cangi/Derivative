import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from core.forge.contracts import (
    BuildSpec,
    CodeArtifact,
    FeasiblePlan,
    PackagedArtifact,
    ValidationArtifact,
)
from core.forge.evidence_integrity import (
    CANONICAL_JSON_DIGEST_MODE,
    artifact_validation_seal,
    canonical_json_bytes,
    to_jsonable,
    validation_artifact_seal,
)
from core.forge.repair_support import behavioral_contract_seal


VALIDATION_RECEIPT_SCHEMA_VERSION = 1


class PackagingStageError(Exception):
    """Base error for Forge packaging stage."""


class PackagingRefusedError(PackagingStageError):
    """Raised when packaging is attempted for a non-verified artifact."""


class PackagingStage:
    def __init__(self, output_root: str = "generated_artifacts/forge_packages"):
        self.output_root = Path(output_root)

    def package(
        self,
        build_spec: BuildSpec,
        plan: FeasiblePlan,
        code_artifact: CodeArtifact,
        validation: ValidationArtifact,
    ) -> PackagedArtifact:
        if not validation.passed:
            raise PackagingRefusedError("Packaging requires a passed ValidationArtifact.")
        if build_spec.build_id != plan.build_spec.build_id:
            raise PackagingRefusedError(
                "Packaging requires the build and plan identities to match."
            )

        expected_contract_seal = behavioral_contract_seal(plan)
        declared_contract_seal = validation.evidence.get(
            "behavioral_contract_seal"
        )
        if declared_contract_seal != expected_contract_seal:
            raise PackagingRefusedError(
                "Packaging requires the validated behavioral contract seal."
            )

        expected_artifact_seal = artifact_validation_seal(code_artifact)
        declared_artifact_seal = validation.evidence.get(
            "validated_artifact_seal"
        )
        if declared_artifact_seal != expected_artifact_seal:
            raise PackagingRefusedError(
                "Packaging requires the matching validated artifact seal."
            )
        expected_validation_artifact_seal = validation_artifact_seal(validation)
        if validation.integrity_seal != expected_validation_artifact_seal:
            raise PackagingRefusedError(
                "Packaging requires the matching validation artifact seal."
            )
        if code_artifact.plan_id != plan.plan_id:
            raise PackagingRefusedError(
                "Packaging requires the artifact and plan identities to match."
            )

        base_package_id = self._package_id(build_spec.build_id, plan.plan_id, code_artifact.artifact_id)
        package_id, package_root = self._resolve_package_root(base_package_id)
        code_artifact_digest = self._artifact_content_digest(code_artifact)
        artifact_manifest_bytes = canonical_json_bytes(
            self._to_jsonable(code_artifact.artifact_manifest)
        )
        artifact_manifest_sha256 = hashlib.sha256(
            artifact_manifest_bytes
        ).hexdigest()
        validation_payload = self._build_validation_payload(validation)
        validation_document = {
            "receipt_context": {
                "schema_version": VALIDATION_RECEIPT_SCHEMA_VERSION,
                "build_id": build_spec.build_id,
                "plan_id": plan.plan_id,
                "artifact_id": code_artifact.artifact_id,
                "package_id": package_id,
                "package_base_id": base_package_id,
                "code_artifact_digest": code_artifact_digest,
                "artifact_manifest_sha256": artifact_manifest_sha256,
                "behavioral_contract_seal": expected_contract_seal,
                "validated_artifact_seal": expected_artifact_seal,
                "validation_artifact_seal": expected_validation_artifact_seal,
            },
            "validation_artifact": validation_payload,
        }
        validation_evidence_bytes = canonical_json_bytes(validation_document)
        validation_receipt = {
            "schema_version": VALIDATION_RECEIPT_SCHEMA_VERSION,
            "digest_mode": CANONICAL_JSON_DIGEST_MODE,
            "sha256": hashlib.sha256(validation_evidence_bytes).hexdigest(),
            "evidence_path": "validation_evidence.json",
            "build_id": build_spec.build_id,
            "plan_id": plan.plan_id,
            "artifact_id": code_artifact.artifact_id,
            "package_id": package_id,
            "code_artifact_digest": code_artifact_digest,
            "artifact_manifest_sha256": artifact_manifest_sha256,
            "behavioral_contract_sha256": expected_contract_seal["sha256"],
            "validated_artifact_sha256": expected_artifact_seal["sha256"],
            "validation_artifact_sha256": expected_validation_artifact_seal[
                "sha256"
            ],
        }

        package_root.mkdir(parents=True, exist_ok=True)

        packaged_files = self._write_code_artifact_files(package_root, code_artifact)
        validation_evidence_path = package_root / "validation_evidence.json"
        artifact_manifest_dump_path = package_root / "code_artifact_manifest_dump.json"
        package_manifest_path = package_root / "forge_package_manifest.json"

        validation_evidence_path.write_bytes(validation_evidence_bytes)
        artifact_manifest_dump_path.write_bytes(artifact_manifest_bytes)
        packaged_files.extend(
            sorted(
                [
                    "validation_evidence.json",
                    "code_artifact_manifest_dump.json",
                ]
            )
        )

        evidence_refs = {
            "validation_evidence": "validation_evidence.json",
            "artifact_manifest_dump": "code_artifact_manifest_dump.json",
        }
        manifest_paths = list(code_artifact.manifest_paths)
        package_manifest = {
            "terminal_status": "verified",
            "build_id": build_spec.build_id,
            "plan_id": plan.plan_id,
            "artifact_id": code_artifact.artifact_id,
            "artifact_revision": code_artifact.revision,
            "parent_artifact_id": code_artifact.parent_artifact_id or None,
            "repair_history": self._to_jsonable(code_artifact.repair_history),
            "package_id": package_id,
            "package_base_id": base_package_id,
            "package_run_id": package_id,
            "code_artifact_digest": code_artifact_digest,
            "behavioral_contract_seal": self._to_jsonable(expected_contract_seal),
            "validated_artifact_seal": self._to_jsonable(expected_artifact_seal),
            "validation_artifact_seal": self._to_jsonable(
                expected_validation_artifact_seal
            ),
            "validation_receipt": validation_receipt,
            "validation_summary": {
                "passed": validation.passed,
                "failure_count": len(validation.failures),
                "failure_signature_count": len(validation.failure_signatures),
                "failure_signatures": list(validation.failure_signatures),
                "passed_layers": validation.metrics.get("passed_layers", {}),
            },
            "evidence_refs": evidence_refs,
            "manifest_paths": manifest_paths,
            "packaged_files": sorted(packaged_files + ["forge_package_manifest.json"]),
        }
        package_manifest_path.write_text(
            json.dumps(package_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        return PackagedArtifact(
            package_id=package_id,
            package_root=str(package_root.resolve()),
            manifest_path=str(package_manifest_path.resolve()),
            packaged_files=sorted(packaged_files + ["forge_package_manifest.json"]),
            evidence_paths=evidence_refs,
            verification_metadata={
                "terminal_status": "verified",
                "build_id": build_spec.build_id,
                "plan_id": plan.plan_id,
                "artifact_id": code_artifact.artifact_id,
                "artifact_revision": code_artifact.revision,
                "parent_artifact_id": code_artifact.parent_artifact_id or None,
                "repair_count": len(code_artifact.repair_history),
                "package_base_id": base_package_id,
                "package_run_id": package_id,
                "code_artifact_digest": code_artifact_digest,
                "behavioral_contract_seal": self._to_jsonable(expected_contract_seal),
                "validated_artifact_seal": self._to_jsonable(expected_artifact_seal),
                "validation_artifact_seal": self._to_jsonable(
                    expected_validation_artifact_seal
                ),
                "validation_receipt": validation_receipt,
                "passed_layers": validation.metrics.get("passed_layers", {}),
            },
        )

    def _write_code_artifact_files(self, package_root: Path, code_artifact: CodeArtifact) -> List[str]:
        packaged_files: List[str] = []
        for generated_file in sorted(code_artifact.files, key=lambda item: item.path):
            target = package_root / generated_file.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(generated_file.content, encoding="utf-8")
            packaged_files.append(generated_file.path)
        return packaged_files

    def _build_validation_payload(self, validation: ValidationArtifact) -> Dict[str, Any]:
        payload = {
            "passed": validation.passed,
            "failures": list(validation.failures),
            "failure_signatures": list(validation.failure_signatures),
            "failure_category": validation.failure_category.value if validation.failure_category else None,
            "metrics": self._to_jsonable(validation.metrics),
            "evidence": self._to_jsonable(validation.evidence),
            "layer1_result": self._to_jsonable(validation.layer1_result),
            "layer2_result": self._to_jsonable(validation.layer2_result),
            "layer3_result": self._to_jsonable(validation.layer3_result),
            "next_route": self._to_jsonable(validation.next_route),
            "integrity_seal": self._to_jsonable(validation.integrity_seal),
        }
        return payload

    def _package_id(self, build_id: str, plan_id: str, artifact_id: str) -> str:
        digest = hashlib.sha256(f"{build_id}:{plan_id}:{artifact_id}".encode("utf-8")).hexdigest()[:12]
        return f"pkg-{digest}"

    def _resolve_package_root(self, base_package_id: str) -> tuple[str, Path]:
        base_root = self.output_root / base_package_id
        if not base_root.exists() or not any(base_root.iterdir()):
            return base_package_id, base_root

        revision = 2
        while True:
            candidate_id = f"{base_package_id}-r{revision:02d}"
            candidate_root = self.output_root / candidate_id
            if not candidate_root.exists():
                return candidate_id, candidate_root
            revision += 1

    def _artifact_content_digest(self, code_artifact: CodeArtifact) -> str:
        hasher = hashlib.sha256()
        for generated_file in sorted(code_artifact.files, key=lambda item: item.path):
            hasher.update(generated_file.path.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(generated_file.content.encode("utf-8"))
            hasher.update(b"\0")
        return hasher.hexdigest()

    def _to_jsonable(self, value: Any) -> Any:
        return to_jsonable(value)
