import json
from pathlib import Path

import pytest

from core.forge.blind_producer import BlindProducerConfig, produce_and_freeze_blind_bundle


def _case_payload() -> dict:
    return {
        "cases": [
            {
                "requirement": (
                    "Build a Python library module code_policy exposing classify_code(code: str) -> str. "
                    "Return 'internal' for codes beginning INT-, return 'external' for EXT-, reject empty "
                    "or unknown codes with ValueError, preserve deterministic behavior, and include tests."
                ),
                "expected_terminal_status": "verified",
                "tags": ["library", "classification"],
            },
            {
                "requirement": (
                    "Build a production tool that identifies every meaningful risk in arbitrary customer "
                    "documents with perfect accuracy, without defining risk, document formats, or acceptance "
                    "criteria, and include exhaustive proof that every result is correct."
                ),
                "expected_terminal_status": "validation_failed",
                "tags": ["ambiguity", "universal"],
            },
            {
                "requirement": (
                    "Build a reversible encoder mapping every possible 24-bit input to one 8-bit output, "
                    "recover every original input exactly, and use no metadata, external state, rejection, "
                    "randomness, or additional storage under any circumstances."
                ),
                "expected_terminal_status": "infeasible_proven",
                "tags": ["contradiction", "information"],
            },
        ]
    }


def _oracle_payload() -> dict:
    return {
        "oracle_py": (
            "from code_policy import classify_code\n\n"
            "def test_internal_code():\n"
            "    assert classify_code('INT-42') == 'internal'\n\n"
            "def test_external_code():\n"
            "    assert classify_code('EXT-9') == 'external'\n\n"
            "def test_unknown_code_rejected():\n"
            "    try:\n"
            "        classify_code('OTHER')\n"
            "    except ValueError as exc:\n"
            "        assert str(exc)\n"
            "    else:\n"
            "        assert False, 'ValueError was not raised'\n"
        )
    }


class _RecordingGenerator:
    def __init__(self, oracle_payload: dict | None = None):
        self.calls = []
        self.oracle_payload = oracle_payload or _oracle_payload()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["output_schema_name"] == "forge_blind_v3_requirements":
            return json.dumps(_case_payload())
        return json.dumps(self.oracle_payload)


def test_one_shot_producer_separates_generation_and_freezes_before_publication(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "blind-v3-external"
    generator = _RecordingGenerator()

    bundle = produce_and_freeze_blind_bundle(
        output_root=output_root,
        repository_root=repository_root,
        config=BlindProducerConfig(
            bundle_id="blind-v3-test",
            verified_cases=1,
            validation_failed_cases=1,
            infeasible_cases=1,
        ),
        text_generator=generator,
        model="external-test-model",
    )

    assert output_root.is_dir()
    assert Path(bundle.manifest_path).is_file()
    assert len(bundle.cases) == 3
    assert b"\r\n" not in (output_root / "cases.json").read_bytes()
    assert b"\r\n" not in (
        output_root / "oracles" / "V3-001" / "oracle.py"
    ).read_bytes()
    assert [call["output_schema_name"] for call in generator.calls] == [
        "forge_blind_v3_requirements",
        "forge_blind_v3_oracle",
    ]
    oracle_request = generator.calls[1]["input_text"]
    assert _case_payload()["cases"][0]["requirement"] in oracle_request
    assert "core/forge" not in oracle_request
    assert "candidate implementation" not in oracle_request.lower()

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["independent_of_forge"] is True
    assert manifest["provenance"]["sealed_before_first_execution"] is True
    assert manifest["forge_baseline"]["sha256"] == bundle.baseline_sha256
    assert manifest["forge_baseline"]["digest_mode"] == "canonical_lf_v1"
    assert set(manifest["oracle_sha256"]) == {"V3-001"}


def test_one_shot_producer_refuses_existing_destination_without_model_call(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    generator = _RecordingGenerator()

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        produce_and_freeze_blind_bundle(
            output_root=destination,
            repository_root=Path(__file__).resolve().parents[1],
            config=BlindProducerConfig(
                bundle_id="blind-v3-test",
                verified_cases=1,
                validation_failed_cases=1,
                infeasible_cases=1,
            ),
            text_generator=generator,
            model="external-test-model",
        )

    assert generator.calls == []


def test_one_shot_producer_rejects_superficial_oracle_and_publishes_nothing(tmp_path):
    destination = tmp_path / "rejected"
    generator = _RecordingGenerator(
        oracle_payload={
            "oracle_py": (
                "def test_one():\n    assert True\n\n"
                "def test_two():\n    assert True\n\n"
                "def test_three():\n    assert True\n"
            )
        }
    )

    with pytest.raises(ValueError, match="Oracle producer failed validation"):
        produce_and_freeze_blind_bundle(
            output_root=destination,
            repository_root=Path(__file__).resolve().parents[1],
            config=BlindProducerConfig(
                bundle_id="blind-v3-test",
                verified_cases=1,
                validation_failed_cases=1,
                infeasible_cases=1,
                max_generation_attempts=2,
            ),
            text_generator=generator,
            model="external-test-model",
        )

    assert destination.exists() is False
    assert [call["output_schema_name"] for call in generator.calls].count(
        "forge_blind_v3_oracle"
    ) == 2


def test_frozen_bundle_transport_disables_git_text_normalization():
    repository_root = Path(__file__).resolve().parents[1]
    attributes = (repository_root / ".gitattributes").read_text(encoding="utf-8")

    assert "benchmarks/blind_v3/external_*/** -text" in attributes
