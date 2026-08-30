from dataclasses import replace

import pytest

from core.forge.blind_benchmark import BlindBenchmarkBundle
from core.forge.blind_metrics import derive_adjudicated_metrics
from core.forge.heldout_benchmark import HeldoutBenchmarkCase, OracleSpec
from core.forge.public_contract import PublicImportContract


def _bundle() -> BlindBenchmarkBundle:
    return BlindBenchmarkBundle(
        bundle_id="metrics-test",
        schema_version=3,
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
                requirement=(
                    "Return the integer unchanged. Public import contract: "
                    "from identity import identity."
                ),
                expected_terminal_status="verified",
                oracle=OracleSpec("oracle.py"),
                public_contract=PublicImportContract(
                    "identity",
                    "identity",
                    "function",
                ),
            ),
            HeldoutBenchmarkCase(
                case_id="A-002",
                requirement=(
                    "Return the normalized value. Public import contract: "
                    "from normalize import normalize."
                ),
                expected_terminal_status="validation_failed",
            ),
            HeldoutBenchmarkCase(
                case_id="A-003",
                requirement="Return a bounded deterministic sequence.",
                expected_terminal_status="infeasible_proven",
            ),
            HeldoutBenchmarkCase(
                case_id="A-004",
                requirement=(
                    "Return a list of the same length as input. Behavioral test: "
                    "[1, 1, 2] returns [1, 2]."
                ),
                expected_terminal_status="validation_failed",
            ),
            HeldoutBenchmarkCase(
                case_id="A-005",
                requirement=(
                    "Build a lossless encoder that maps every possible two-byte input to "
                    "exactly one output byte and a decoder that reconstructs every original "
                    "input without external state, metadata, or a side channel."
                ),
                expected_terminal_status="infeasible_proven",
            ),
        ],
    )


def _result(
    case_id: str,
    expected: str,
    observed: str,
    *,
    oracle_passed: bool = False,
    verified_at_1: bool = False,
    repaired: bool = False,
) -> dict:
    return {
        "case_id": case_id,
        "expected_terminal_status": expected,
        "observed_terminal_status": observed,
        "execution_time_seconds": 2.0,
        "repair_count": 1 if repaired else 0,
        "model_request_count": 2,
        "model_total_tokens": 100,
        "estimated_model_cost_usd": 0.25,
        "verified_at_1": verified_at_1,
        "success_after_repair": repaired,
        "oracle_result": (
            {"executed": True, "passed": oracle_passed, "valid": True}
            if oracle_passed
            else None
        ),
    }


def _baseline() -> dict:
    return {
        "schema_version": 3,
        "bundle_id": "metrics-test",
        "manifest_sha256": "manifest-sha",
        "dataset_sha256": "dataset-sha",
        "baseline_sha256": "baseline-sha",
        "baseline_verified": True,
        "baseline_file_count": 1,
        "observed_baseline_sha256": "baseline-sha",
        "observed_baseline_file_count": 1,
        "execution_kind": "sealed_baseline",
        "summary": {
            "case_results": [
                _result(
                    "A-001",
                    "verified",
                    "verified",
                    oracle_passed=True,
                    repaired=True,
                ),
                _result("A-002", "validation_failed", "validation_failed"),
                _result("A-003", "infeasible_proven", "validation_failed"),
                _result("A-004", "validation_failed", "validation_failed"),
                _result("A-005", "infeasible_proven", "infeasible_proven"),
            ]
        },
    }


def _adjudication() -> dict:
    return {
        "schema_version": 1,
        "bundle": {
            "bundle_id": "metrics-test",
            "manifest_sha256": "manifest-sha",
            "dataset_sha256": "dataset-sha",
            "baseline_sha256": "baseline-sha",
        },
        "method": {"raw_benchmark_metrics_modified": False},
        "consensus": [
            {
                "case_id": "A-001",
                "claimed_status": "verified",
                "verdict": "label_valid",
                "adjudicated_status": "verified",
            },
            {
                "case_id": "A-002",
                "claimed_status": "validation_failed",
                "verdict": "label_invalid",
                "adjudicated_status": "verified",
            },
            {
                "case_id": "A-003",
                "claimed_status": "infeasible_proven",
                "verdict": "unresolved",
                "adjudicated_status": None,
            },
            {
                "case_id": "A-004",
                "claimed_status": "validation_failed",
                "verdict": "label_invalid",
                "adjudicated_status": "verified",
            },
            {
                "case_id": "A-005",
                "claimed_status": "infeasible_proven",
                "verdict": "label_valid",
                "adjudicated_status": "infeasible_proven",
            },
        ],
    }


def test_adjudicated_metrics_exclude_unresolved_and_invalid_consensus():
    receipt = derive_adjudicated_metrics(
        bundle=_bundle(),
        baseline_report=_baseline(),
        adjudication_receipt=_adjudication(),
        baseline_report_sha256="report-sha",
        adjudication_sha256="adjudication-sha",
        created_at="2026-08-21T01:00:00Z",
    )

    assert receipt["summary"]["definitive_status_cases"] == 3
    assert receipt["summary"]["excluded_cases"] == 2
    assert receipt["summary"]["unresolved_cases"] == 1
    assert receipt["summary"]["invalid_adjudication_cases"] == 1
    assert receipt["summary"]["corrected_label_cases"] == 2
    assert receipt["metrics"]["status_accuracy"] == {
        "numerator": 2,
        "denominator": 3,
        "value": 2 / 3,
    }
    assert receipt["metrics"]["external_verified_at_1"]["denominator"] == 1
    assert receipt["metrics"]["external_success_after_repair"]["value"] == 1.0
    assert receipt["metrics"]["infeasible_detection_rate"]["value"] == 1.0
    decisions = {item["case_id"]: item for item in receipt["case_decisions"]}
    assert decisions["A-003"]["exclusion_reason"] == "reviewer_disagreement"
    assert decisions["A-004"]["exclusion_reason"] == "deterministic_contract_conflict"
    assert "3-item input returns a 2-item list" in decisions["A-004"][
        "deterministic_evidence"
    ]


def test_legacy_verified_case_is_not_used_for_external_metrics():
    bundle = _bundle()
    legacy_case = HeldoutBenchmarkCase(
        case_id="L-001",
        requirement="Return the value unchanged.",
        expected_terminal_status="verified",
        oracle=OracleSpec("legacy-oracle.py"),
    )
    bundle = replace(bundle, schema_version=2, cases=[legacy_case])
    baseline = _baseline()
    baseline["schema_version"] = 2
    baseline["summary"]["case_results"] = [
        _result("L-001", "verified", "verified", oracle_passed=True, verified_at_1=True)
    ]
    adjudication = _adjudication()
    adjudication["consensus"] = [
        {
            "case_id": "L-001",
            "claimed_status": "verified",
            "verdict": "label_valid",
            "adjudicated_status": "verified",
        }
    ]

    receipt = derive_adjudicated_metrics(
        bundle=bundle,
        baseline_report=baseline,
        adjudication_receipt=adjudication,
        baseline_report_sha256="report-sha",
        adjudication_sha256="adjudication-sha",
    )

    assert receipt["metrics"]["status_accuracy"]["value"] == 1.0
    assert receipt["metrics"]["external_verified_at_1"]["value"] is None
    assert receipt["metrics"]["external_false_verified_rate"]["value"] is None
    assert receipt["summary"]["verified_cases_without_definitive_external_contract"] == 1
    assert receipt["summary"]["observed_verified_without_definitive_external_judge"] == 1


def test_adjudicated_metrics_fail_closed_on_source_mismatch():
    baseline = _baseline()
    baseline["dataset_sha256"] = "different"

    with pytest.raises(ValueError, match="dataset_sha256"):
        derive_adjudicated_metrics(
            bundle=_bundle(),
            baseline_report=baseline,
            adjudication_receipt=_adjudication(),
            baseline_report_sha256="report-sha",
            adjudication_sha256="adjudication-sha",
        )


def test_adjudicated_replay_metrics_have_distinct_terminal_identity():
    replay = _baseline()
    replay.update(
        execution_kind="post_fix_replay",
        baseline_verified=False,
        observed_baseline_sha256="post-fix-sha",
    )
    receipt = derive_adjudicated_metrics(
        bundle=_bundle(),
        baseline_report=replay,
        adjudication_receipt=_adjudication(),
        baseline_report_sha256="replay-sha",
        adjudication_sha256="adjudication-sha",
        execution_kind="post_fix_replay",
        receipt_id="metrics-test-post-fix-replay-001",
    )

    assert receipt["receipt_id"] == "metrics-test-post-fix-replay-001"
    assert receipt["execution_kind"] == "post_fix_replay"

    with pytest.raises(ValueError, match="explicit receipt_id"):
        derive_adjudicated_metrics(
            bundle=_bundle(),
            baseline_report=replay,
            adjudication_receipt=_adjudication(),
            baseline_report_sha256="replay-sha",
            adjudication_sha256="adjudication-sha",
            execution_kind="post_fix_replay",
        )

    incomplete = replay.copy()
    incomplete.pop("observed_baseline_sha256")
    with pytest.raises(ValueError, match="incomplete baseline observations"):
        derive_adjudicated_metrics(
            bundle=_bundle(),
            baseline_report=incomplete,
            adjudication_receipt=_adjudication(),
            baseline_report_sha256="replay-sha",
            adjudication_sha256="adjudication-sha",
            execution_kind="post_fix_replay",
            receipt_id="incomplete-replay",
        )

    tampered = replay.copy()
    tampered["baseline_file_count"] = 2
    with pytest.raises(ValueError, match="incomplete baseline observations"):
        derive_adjudicated_metrics(
            bundle=_bundle(),
            baseline_report=tampered,
            adjudication_receipt=_adjudication(),
            baseline_report_sha256="replay-sha",
            adjudication_sha256="adjudication-sha",
            execution_kind="post_fix_replay",
            receipt_id="tampered-replay",
        )

    with pytest.raises(ValueError, match="execution kind"):
        derive_adjudicated_metrics(
            bundle=_bundle(),
            baseline_report=_baseline(),
            adjudication_receipt=_adjudication(),
            baseline_report_sha256="replay-sha",
            adjudication_sha256="adjudication-sha",
            execution_kind="unknown",
        )
