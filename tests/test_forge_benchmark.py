import json
from pathlib import Path

import pytest

from core.forge.benchmark import (
    BenchmarkThresholds,
    TERMINAL_INFEASIBLE_PROVEN,
    TERMINAL_VALIDATION_FAILED,
    TERMINAL_VERIFIED,
    BenchmarkCase,
    bundled_dataset_path,
    default_forge_benchmark_cases,
    evaluate_benchmark_thresholds,
    extended_forge_benchmark_cases,
    load_benchmark_cases,
    persist_benchmark_summary,
    render_benchmark_summary,
    run_benchmark_cases,
)
from core.forge.contracts import ForgeResult, ForgeRoute, ValidationArtifact


def _forge_result(status: str) -> ForgeResult:
    validation = None
    if status == TERMINAL_VALIDATION_FAILED:
        validation = ValidationArtifact(
            passed=False,
            failures=["validation failed"],
            failure_signatures=["missing_requirement_coverage"],
            evidence={"validated_entrypoints": {}, "executed_tests": {}},
            metrics={},
        )
    return ForgeResult(
        route={
            TERMINAL_VERIFIED: ForgeRoute.TERMINAL_VERIFIED,
            TERMINAL_INFEASIBLE_PROVEN: ForgeRoute.TERMINAL_INFEASIBLE,
            TERMINAL_VALIDATION_FAILED: ForgeRoute.TERMINAL_VALIDATION_FAILED,
        }[status],
        terminal_status=status,
        summary="stub",
        validation=validation,
        packaged_artifact=None,
        infeasibility_certificate=None,
        artifact_path=f"C:/tmp/{status}",
        execution_time_seconds=0.1,
    )


def test_run_benchmark_cases_computes_metrics():
    cases = [
        BenchmarkCase("C1", "req1", TERMINAL_VERIFIED),
        BenchmarkCase("C2", "req2", TERMINAL_VERIFIED),
        BenchmarkCase("C3", "req3", TERMINAL_INFEASIBLE_PROVEN),
        BenchmarkCase("C4", "req4", TERMINAL_VALIDATION_FAILED),
    ]
    observed = {
        "req1": TERMINAL_VERIFIED,
        "req2": TERMINAL_VALIDATION_FAILED,
        "req3": TERMINAL_INFEASIBLE_PROVEN,
        "req4": TERMINAL_VERIFIED,
    }

    summary = run_benchmark_cases(cases, run_case=lambda req: _forge_result(observed[req]))

    assert summary.total_cases == 4
    assert summary.passed_cases == 2
    assert summary.failed_cases == 2
    assert summary.status_accuracy == pytest.approx(0.5, rel=1e-6)
    assert summary.verified_at_1 == pytest.approx(0.5, rel=1e-6)
    assert summary.false_verified_rate == pytest.approx(0.5, rel=1e-6)
    assert summary.infeasible_detection_rate == pytest.approx(1.0, rel=1e-6)
    assert len(summary.case_results) == 4


def test_default_cases_have_supported_terminal_statuses():
    cases = default_forge_benchmark_cases()
    assert cases
    assert {case.expected_terminal_status for case in cases}.issubset(
        {TERMINAL_VERIFIED, TERMINAL_INFEASIBLE_PROVEN, TERMINAL_VALIDATION_FAILED}
    )


def test_extended_cases_load_from_bundled_dataset():
    dataset_path = bundled_dataset_path("forge_extended_benchmark.json")
    assert Path(dataset_path).exists()

    cases = extended_forge_benchmark_cases()
    assert len(cases) >= 10
    statuses = {case.expected_terminal_status for case in cases}
    assert TERMINAL_VERIFIED in statuses
    assert TERMINAL_VALIDATION_FAILED in statuses
    assert TERMINAL_INFEASIBLE_PROVEN in statuses


def test_load_benchmark_cases_from_json(tmp_path):
    dataset = [
        {
            "case_id": "D001",
            "requirement": "Build a CLI.",
            "expected_terminal_status": TERMINAL_VERIFIED,
            "tags": ["cli"],
        },
        {
            "case_id": "D002",
            "requirement": "Contradictory constraints.",
            "expected_terminal_status": TERMINAL_INFEASIBLE_PROVEN,
        },
    ]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    loaded = load_benchmark_cases(str(path))

    assert [case.case_id for case in loaded] == ["D001", "D002"]
    assert loaded[0].tags == ["cli"]
    assert loaded[1].expected_terminal_status == TERMINAL_INFEASIBLE_PROVEN


def test_persist_and_render_summary(tmp_path):
    cases = [BenchmarkCase("C1", "req1", TERMINAL_VERIFIED)]
    summary = run_benchmark_cases(cases, run_case=lambda req: _forge_result(TERMINAL_VERIFIED))

    output_path = persist_benchmark_summary(summary, str(tmp_path))
    assert Path(output_path).exists()

    rendered = render_benchmark_summary(summary, output_path)
    assert "Forge Benchmark" in rendered
    assert "Status accuracy" in rendered
    assert output_path in rendered


def test_evaluate_benchmark_thresholds_passes_when_metrics_meet_limits():
    cases = [BenchmarkCase("C1", "req1", TERMINAL_VERIFIED)]
    summary = run_benchmark_cases(cases, run_case=lambda req: _forge_result(TERMINAL_VERIFIED))

    failures = evaluate_benchmark_thresholds(
        summary,
        BenchmarkThresholds(
            min_status_accuracy=0.9,
            min_verified_at_1=0.9,
            max_false_verified_rate=0.0,
            min_infeasible_detection_rate=0.0,
        ),
    )

    assert failures == []


def test_evaluate_benchmark_thresholds_reports_violations():
    cases = [
        BenchmarkCase("C1", "req1", TERMINAL_VERIFIED),
        BenchmarkCase("C2", "req2", TERMINAL_VERIFIED),
        BenchmarkCase("C3", "req3", TERMINAL_INFEASIBLE_PROVEN),
    ]
    observed = {
        "req1": TERMINAL_VALIDATION_FAILED,
        "req2": TERMINAL_VALIDATION_FAILED,
        "req3": TERMINAL_VALIDATION_FAILED,
    }
    summary = run_benchmark_cases(cases, run_case=lambda req: _forge_result(observed[req]))

    failures = evaluate_benchmark_thresholds(
        summary,
        BenchmarkThresholds(
            min_status_accuracy=0.8,
            min_verified_at_1=0.8,
            max_false_verified_rate=0.0,
            min_infeasible_detection_rate=1.0,
        ),
    )

    assert any(item.startswith("status_accuracy_below_threshold:") for item in failures)
    assert any(item.startswith("verified_at_1_below_threshold:") for item in failures)
    assert any(item.startswith("infeasible_detection_rate_below_threshold:") for item in failures)
