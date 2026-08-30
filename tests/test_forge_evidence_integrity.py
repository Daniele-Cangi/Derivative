import json

from core.forge.contracts import ValidationArtifact
from core.forge.evidence_integrity import (
    canonical_json_bytes,
    to_jsonable,
    validation_artifact_seal,
)


def test_json_safe_payload_keeps_the_existing_canonical_encoding():
    payload = {"z": [1, True, None], "a": "å"}

    assert canonical_json_bytes(payload) == b'{"a":"\xc3\xa5","z":[1,true,null]}'


def test_binary_evidence_is_canonicalized_losslessly_and_deterministically():
    payload = {
        "raw": b"\x00\xff\n",
        "nested": [bytearray(b"\x10\x20")],
    }

    normalized = to_jsonable(payload)

    assert normalized == {
        "raw": {
            "__forge_scalar__": "bytes",
            "hex": "00ff0a",
        },
        "nested": [
            {
                "__forge_scalar__": "bytearray",
                "hex": "1020",
            }
        ],
    }
    encoded = canonical_json_bytes(payload)
    assert json.loads(encoded) == normalized
    assert encoded == canonical_json_bytes(payload)


def test_validation_artifact_seal_accepts_binary_evidence():
    validation = ValidationArtifact(
        passed=False,
        failures=["Binary observation did not match."],
        failure_signatures=["semantic_content_mismatch"],
        evidence={"observed": b"\xff\xfe"},
    )

    seal = validation_artifact_seal(validation)

    assert seal["digest_mode"] == "canonical_json_utf8_v1"
    assert len(seal["sha256"]) == 64
