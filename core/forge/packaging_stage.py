import hashlib
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

from core.forge.contracts import (
    BuildSpec,
    CodeArtifact,
    FeasiblePlan,
    PackagedArtifact,
    ValidationArtifact,
)


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

        package_id = self._package_id(build_spec.build_id, plan.plan_id, code_artifact.artifact_id)
        package_root = self.output_root / package_id
        package_root.mkdir(parents=True, exist_ok=True)

        packaged_files = self._write_code_artifact_files(package_root, code_artifact)
        validation_evidence_path = package_root / "validation_evidence.json"
        artifact_manifest_dump_path = package_root / "code_artifact_manifest_dump.json"
        package_manifest_path = package_root / "forge_package_manifest.json"

        validation_payload = self._build_validation_payload(validation)
        validation_evidence_path.write_text(
            json.dumps(validation_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        artifact_manifest_dump_path.write_text(
            json.dumps(code_artifact.artifact_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
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
            "package_id": package_id,
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
        }
        return payload

    def _package_id(self, build_id: str, plan_id: str, artifact_id: str) -> str:
        digest = hashlib.sha256(f"{build_id}:{plan_id}:{artifact_id}".encode("utf-8")).hexdigest()[:12]
        return f"pkg-{digest}"

    def _to_jsonable(self, value: Any) -> Any:
        if is_dataclass(value):
            return self._to_jsonable(asdict(value))
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {str(key): self._to_jsonable(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._to_jsonable(item) for item in value]
        return value
