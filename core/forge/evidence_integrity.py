import hashlib
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from core.forge.contracts import CodeArtifact, ValidationArtifact


CANONICAL_JSON_DIGEST_MODE = "canonical_json_utf8_v1"
ARTIFACT_VALIDATION_SCHEMA_VERSION = 1
VALIDATION_ARTIFACT_SCHEMA_VERSION = 1


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value with the Forge canonical encoding."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def artifact_validation_seal(code_artifact: CodeArtifact) -> dict[str, Any]:
    """Return the canonical identity of the complete artifact seen by validation."""

    payload = {
        "schema_version": ARTIFACT_VALIDATION_SCHEMA_VERSION,
        "artifact_id": code_artifact.artifact_id,
        "plan_id": code_artifact.plan_id,
        "revision": code_artifact.revision,
        "parent_artifact_id": code_artifact.parent_artifact_id,
        "files": [
            {
                "path": generated.path,
                "content": generated.content,
                "kind": generated.kind,
                "generated_from_plan_sections": list(
                    generated.generated_from_plan_sections
                ),
            }
            for generated in sorted(code_artifact.files, key=lambda item: item.path)
        ],
        "test_paths": sorted(code_artifact.test_paths),
        "manifest_paths": sorted(code_artifact.manifest_paths),
        "runnable_entrypoints": sorted(code_artifact.runnable_entrypoints),
        "artifact_manifest": code_artifact.artifact_manifest,
        "traceability": code_artifact.traceability,
        "repair_history": code_artifact.repair_history,
    }
    return {
        "schema_version": ARTIFACT_VALIDATION_SCHEMA_VERSION,
        "digest_mode": CANONICAL_JSON_DIGEST_MODE,
        "sha256": canonical_json_sha256(payload),
        "artifact_id": code_artifact.artifact_id,
        "plan_id": code_artifact.plan_id,
        "revision": code_artifact.revision,
    }


def validation_artifact_seal(
    validation: ValidationArtifact,
) -> dict[str, Any]:
    """Seal validator output while excluding the seal field itself."""

    payload = {
        "schema_version": VALIDATION_ARTIFACT_SCHEMA_VERSION,
        "passed": validation.passed,
        "failures": list(validation.failures),
        "failure_signatures": list(validation.failure_signatures),
        "evidence": to_jsonable(validation.evidence),
        "metrics": to_jsonable(validation.metrics),
        "layer1_result": to_jsonable(validation.layer1_result),
        "layer2_result": to_jsonable(validation.layer2_result),
        "layer3_result": to_jsonable(validation.layer3_result),
        "failure_category": to_jsonable(validation.failure_category),
        "next_route": to_jsonable(validation.next_route),
    }
    return {
        "schema_version": VALIDATION_ARTIFACT_SCHEMA_VERSION,
        "digest_mode": CANONICAL_JSON_DIGEST_MODE,
        "sha256": canonical_json_sha256(payload),
        "passed": validation.passed,
        "failure_count": len(validation.failures),
        "failure_signature_count": len(validation.failure_signatures),
    }
