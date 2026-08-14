import json
from pathlib import Path

import pytest

from core.forge.benchmark import (
    TERMINAL_INFEASIBLE_PROVEN,
    TERMINAL_VALIDATION_FAILED,
    TERMINAL_VERIFIED,
)
from core.forge.contracts import ForgeResult, ForgeRoute
from core.forge.execution import ExecutionPolicy, SandboxProcessResult
from core.forge.heldout_benchmark import (
    HeldoutBenchmarkCase,
    HeldoutThresholds,
    OracleResult,
    OracleSpec,
    bundled_heldout_dataset_path,
    evaluate_heldout_thresholds,
    execute_pytest_oracle,
    load_heldout_cases,
    persist_heldout_summary,
    render_heldout_summary,
    run_heldout_cases,
)


class _OracleRecordingExecutor:
    def __init__(self):
        self.policy = ExecutionPolicy(backend="docker")
        self.request = None
        self.staged_files = []

    def run(self, request):
        self.request = request
        self.staged_files = sorted(
            path.relative_to(request.workspace).as_posix()
            for path in request.workspace.rglob("*")
            if path.is_file()
        )
        return SandboxProcessResult(
            returncode=0,
            stdout="1 passed\n",
            stderr="",
            backend="docker",
            execution_time_seconds=0.02,
            isolation=self.policy.evidence(),
        )


def _forge_result(status: str, artifact_path: str = "") -> ForgeResult:
    routes = {
        TERMINAL_VERIFIED: ForgeRoute.TERMINAL_VERIFIED,
        TERMINAL_VALIDATION_FAILED: ForgeRoute.TERMINAL_VALIDATION_FAILED,
        TERMINAL_INFEASIBLE_PROVEN: ForgeRoute.TERMINAL_INFEASIBLE,
    }
    return ForgeResult(
        route=routes[status],
        terminal_status=status,
        summary=status,
        artifact_path=artifact_path,
    )


def test_bundled_heldout_dataset_is_frozen_and_oracle_complete():
    cases = load_heldout_cases(bundled_heldout_dataset_path())

    assert len(cases) == 15
    assert len({case.case_id for case in cases}) == 15
    verified = [case for case in cases if case.expected_terminal_status == TERMINAL_VERIFIED]
    assert len(verified) == 8
    assert all(case.oracle is not None for case in verified)
    assert all(Path(case.oracle.path).is_file() for case in verified if case.oracle)


def test_loader_rejects_verified_case_without_oracle(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "case_id": "X001",
                    "requirement": "Build a feasible library.",
                    "expected_terminal_status": TERMINAL_VERIFIED,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires an external oracle"):
        load_heldout_cases(str(dataset))


def test_loader_rejects_oracle_outside_benchmark_directory(tmp_path):
    dataset_root = tmp_path / "benchmark"
    dataset_root.mkdir()
    outside = tmp_path / "outside_oracle.py"
    outside.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    dataset = dataset_root / "dataset.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "case_id": "X001",
                    "requirement": "Build a feasible library.",
                    "expected_terminal_status": TERMINAL_VERIFIED,
                    "oracle": {"path": "../outside_oracle.py"},
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes the benchmark directory"):
        load_heldout_cases(str(dataset))


def test_pytest_oracle_executes_against_packaged_source(tmp_path):
    package_root = tmp_path / "package"
    source_root = package_root / "src"
    source_root.mkdir(parents=True)
    (source_root / "calculator.py").write_text(
        "def add(left, right):\n    return left + right\n",
        encoding="utf-8",
    )
    oracle_path = tmp_path / "oracle.py"
    oracle_path.write_text(
        "from calculator import add\n\n"
        "def test_external_contract():\n"
        "    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )

    result = execute_pytest_oracle(
        OracleSpec(path=str(oracle_path), timeout_seconds=20),
        str(package_root),
    )

    assert result.executed is True
    assert result.passed is True
    assert result.exit_code == 0
    assert "1 passed" in result.stdout


def test_pytest_oracle_failure_is_external_evidence(tmp_path):
    package_root = tmp_path / "package"
    source_root = package_root / "src"
    source_root.mkdir(parents=True)
    (source_root / "calculator.py").write_text(
        "def add(left, right):\n    return left - right\n",
        encoding="utf-8",
    )
    oracle_path = tmp_path / "oracle.py"
    oracle_path.write_text(
        "from calculator import add\n\n"
        "def test_external_contract():\n"
        "    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )

    result = execute_pytest_oracle(
        OracleSpec(path=str(oracle_path), timeout_seconds=20),
        str(package_root),
    )

    assert result.executed is True
    assert result.passed is False
    assert result.exit_code == 1
    assert "1 failed" in result.stdout


def test_pytest_oracle_stages_only_package_and_oracle_for_executor(tmp_path):
    package_root = tmp_path / "original-package"
    source_root = package_root / "src"
    source_root.mkdir(parents=True)
    (source_root / "calculator.py").write_text(
        "def add(left, right):\n    return left + right\n",
        encoding="utf-8",
    )
    oracle_path = tmp_path / "private-oracle.py"
    oracle_path.write_text("def test_external():\n    assert True\n", encoding="utf-8")
    executor = _OracleRecordingExecutor()

    result = execute_pytest_oracle(
        OracleSpec(path=str(oracle_path), timeout_seconds=20),
        str(package_root),
        executor=executor,
    )

    assert result.passed is True
    assert result.backend == "docker"
    assert result.isolation["isolated"] is True
    assert executor.request is not None
    assert executor.request.working_directory == "package"
    assert "../oracle.py" in executor.request.command
    assert executor.request.environment["PYTHONPATH"] == "src"
    assert executor.staged_files == ["oracle.py", "package/src/calculator.py"]
    assert str(package_root) not in " ".join(executor.request.command)
    assert str(oracle_path) not in " ".join(executor.request.command)


def test_heldout_metrics_require_external_oracle_success():
    oracle = OracleSpec(path="oracle.py")
    cases = [
        HeldoutBenchmarkCase("A", "verified-pass", TERMINAL_VERIFIED, oracle=oracle),
        HeldoutBenchmarkCase("B", "verified-missed", TERMINAL_VERIFIED, oracle=oracle),
        HeldoutBenchmarkCase("C", "must-fail", TERMINAL_VALIDATION_FAILED),
        HeldoutBenchmarkCase("D", "impossible", TERMINAL_INFEASIBLE_PROVEN),
    ]
    observed = {
        "verified-pass": _forge_result(TERMINAL_VERIFIED, "pkg-a"),
        "verified-missed": _forge_result(TERMINAL_VALIDATION_FAILED),
        "must-fail": _forge_result(TERMINAL_VERIFIED, "pkg-c"),
        "impossible": _forge_result(TERMINAL_INFEASIBLE_PROVEN),
    }

    summary = run_heldout_cases(
        cases,
        run_case=lambda requirement: observed[requirement],
        run_oracle=lambda spec, package: OracleResult(
            executed=True,
            passed=package == "pkg-a",
            exit_code=0 if package == "pkg-a" else 1,
        ),
    )

    assert summary.total_cases == 4
    assert summary.passed_cases == 2
    assert summary.status_accuracy == 0.5
    assert summary.external_verified_at_1 == 0.5
    assert summary.oracle_pass_rate == 1.0
    assert summary.external_false_verified_rate == 0.5
    assert summary.infeasible_detection_rate == 1.0


def test_oracle_failure_turns_matching_verified_status_into_failed_case():
    oracle = OracleSpec(path="oracle.py")
    summary = run_heldout_cases(
        [HeldoutBenchmarkCase("A", "candidate", TERMINAL_VERIFIED, oracle=oracle)],
        run_case=lambda requirement: _forge_result(TERMINAL_VERIFIED, "pkg"),
        run_oracle=lambda spec, package: OracleResult(executed=True, passed=False, exit_code=1),
    )

    result = summary.case_results[0]
    assert result.status_matched is True
    assert result.passed is False
    assert summary.status_accuracy == 1.0
    assert summary.external_verified_at_1 == 0.0
    assert summary.external_false_verified_rate == 1.0


def test_heldout_summary_persistence_and_thresholds(tmp_path):
    oracle = OracleSpec(path="oracle.py")
    summary = run_heldout_cases(
        [HeldoutBenchmarkCase("A", "candidate", TERMINAL_VERIFIED, oracle=oracle)],
        run_case=lambda requirement: _forge_result(TERMINAL_VERIFIED, "pkg"),
        run_oracle=lambda spec, package: OracleResult(executed=True, passed=True, exit_code=0),
    )

    report_path = persist_heldout_summary(summary, str(tmp_path))
    rendered = render_heldout_summary(summary, report_path)
    failures = evaluate_heldout_thresholds(
        summary,
        HeldoutThresholds(
            min_status_accuracy=1.0,
            min_external_verified_at_1=1.0,
            max_external_false_verified_rate=0.0,
            min_infeasible_detection_rate=0.0,
        ),
    )

    assert Path(report_path).is_file()
    assert "External Verified@1: 1.000" in rendered
    assert failures == []
