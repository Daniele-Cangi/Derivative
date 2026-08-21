import json
from dataclasses import replace

import pytest

from core.forge.blind_adjudication import (
    AdjudicationReviewer,
    adjudicate_blind_bundle,
)
from core.forge.blind_benchmark import BlindBenchmarkBundle
from core.forge.heldout_benchmark import HeldoutBenchmarkCase


def _bundle() -> BlindBenchmarkBundle:
    return BlindBenchmarkBundle(
        bundle_id="blind-adjudication-test",
        schema_version=2,
        frozen_at="2026-08-21T00:00:00Z",
        manifest_path="manifest.json",
        manifest_sha256="manifest-sha",
        dataset_path="cases.json",
        dataset_sha256="dataset-sha",
        baseline_sha256="baseline-sha",
        observed_baseline_sha256="baseline-sha",
        baseline_file_count=1,
        observed_baseline_file_count=1,
        baseline_verified=True,
        cases=[
            HeldoutBenchmarkCase(
                case_id="A-001",
                requirement="Provide add(a: int, b: int) -> int returning a + b.",
                expected_terminal_status="verified",
            ),
            HeldoutBenchmarkCase(
                case_id="A-002",
                requirement="Return every possible answer with perfect accuracy without defining answer.",
                expected_terminal_status="validation_failed",
            ),
        ],
    )


def _review_payload(second_status: str = "validation_failed") -> dict:
    second_basis = (
        "material_underspecification"
        if second_status == "validation_failed"
        else "objective_satisfiable_contract"
    )
    return {
        "reviews": [
            {
                "case_id": "A-001",
                "claimed_status": "verified",
                "label_valid": True,
                "defensible_status": "verified",
                "classification_basis": "objective_satisfiable_contract",
                "findings": ["The return value is objective and executable."],
            },
            {
                "case_id": "A-002",
                "claimed_status": "validation_failed",
                "label_valid": second_status == "validation_failed",
                "defensible_status": second_status,
                "classification_basis": second_basis,
                "findings": ["The meaning of answer is not defined."],
            },
        ]
    }


def _reviewers():
    return (
        AdjudicationReviewer("reviewer-a", 2.0, 12.0),
        AdjudicationReviewer("reviewer-b", 1.25, 10.0),
    )


def test_adjudication_reaches_consensus_without_forge_results():
    calls = []

    def generator(**kwargs):
        calls.append(kwargs)
        return json.dumps(_review_payload())

    receipt = adjudicate_blind_bundle(
        bundle=_bundle(),
        reviewers=_reviewers(),
        text_generator=generator,
        created_at="2026-08-21T01:00:00Z",
    )

    assert receipt["summary"]["label_valid"] == 2
    assert receipt["summary"]["label_invalid"] == 0
    assert receipt["summary"]["unresolved"] == 0
    assert receipt["method"]["blind_to_forge_results"] is True
    assert receipt["method"]["forge_results_supplied"] is False
    assert [call["model"] for call in calls] == ["reviewer-a", "reviewer-b"]
    for call in calls:
        assert "observed_terminal_status" not in call["input_text"]
        assert "failure_signatures" not in call["input_text"]
        assert "baseline_result" not in call["input_text"]


def test_adjudication_marks_cross_model_disagreement_unresolved():
    payloads = [_review_payload(), _review_payload(second_status="verified")]

    def generator(**_kwargs):
        return json.dumps(payloads.pop(0))

    receipt = adjudicate_blind_bundle(
        bundle=_bundle(),
        reviewers=_reviewers(),
        text_generator=generator,
    )

    second = receipt["consensus"][1]
    assert second["verdict"] == "unresolved"
    assert second["adjudicated_status"] is None
    assert receipt["summary"]["unresolved"] == 1


def test_adjudication_records_consensus_that_frozen_label_is_invalid():
    payload = _review_payload(second_status="verified")

    def generator(**_kwargs):
        return json.dumps(payload)

    receipt = adjudicate_blind_bundle(
        bundle=_bundle(),
        reviewers=_reviewers(),
        text_generator=generator,
    )

    second = receipt["consensus"][1]
    assert second["verdict"] == "label_invalid"
    assert second["adjudicated_status"] == "verified"
    assert receipt["summary"]["label_invalid"] == 1


def test_adjudication_rejects_internally_inconsistent_review_set():
    invalid = _review_payload()
    invalid["reviews"][0]["label_valid"] = False

    def generator(**_kwargs):
        return json.dumps(invalid)

    with pytest.raises(ValueError, match="failed deterministic validation"):
        adjudicate_blind_bundle(
            bundle=_bundle(),
            reviewers=_reviewers(),
            text_generator=generator,
        )


def test_adjudication_requires_distinct_reviewers_and_sealed_baseline():
    duplicate = (
        AdjudicationReviewer("same", 1.0, 1.0),
        AdjudicationReviewer("same", 1.0, 1.0),
    )

    with pytest.raises(ValueError, match="must be distinct"):
        adjudicate_blind_bundle(
            bundle=_bundle(),
            reviewers=duplicate,
            text_generator=lambda **_kwargs: "{}",
        )

    bundle = replace(_bundle(), baseline_verified=False)
    with pytest.raises(ValueError, match="exact sealed Forge baseline"):
        adjudicate_blind_bundle(
            bundle=bundle,
            reviewers=_reviewers(),
            text_generator=lambda **_kwargs: "{}",
        )
