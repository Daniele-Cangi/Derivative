import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List

from core.forge.benchmark import (
    SUPPORTED_TERMINAL_STATUSES,
    TERMINAL_INFEASIBLE_PROVEN,
    TERMINAL_VERIFIED,
)
from core.forge.contracts import ForgeResult


@dataclass(frozen=True)
class OracleSpec:
    path: str
    timeout_seconds: int = 30


@dataclass(frozen=True)
class HeldoutBenchmarkCase:
    case_id: str
    requirement: str
    expected_terminal_status: str
    tags: List[str] = field(default_factory=list)
    oracle: OracleSpec | None = None


@dataclass
class OracleResult:
    executed: bool
    passed: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    execution_time_seconds: float = 0.0


@dataclass
class HeldoutCaseResult:
    case_id: str
    expected_terminal_status: str
    observed_terminal_status: str
    status_matched: bool
    passed: bool
    execution_time_seconds: float
    artifact_path: str
    failure_signatures: List[str] = field(default_factory=list)
    oracle_result: OracleResult | None = None
    error: str | None = None


@dataclass
class HeldoutBenchmarkSummary:
    benchmark_id: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    status_accuracy: float
    external_verified_at_1: float
    oracle_pass_rate: float
    external_false_verified_rate: float
    infeasible_detection_rate: float
    avg_case_runtime_seconds: float
    total_runtime_seconds: float
    case_results: List[HeldoutCaseResult] = field(default_factory=list)


@dataclass(frozen=True)
class HeldoutThresholds:
    min_status_accuracy: float = 0.0
    min_external_verified_at_1: float = 0.0
    max_external_false_verified_rate: float = 0.0
    min_infeasible_detection_rate: float = 0.0


def bundled_heldout_dataset_path() -> str:
    root = Path(__file__).resolve().parents[2]
    return str((root / "benchmarks" / "forge_heldout_benchmark.json").resolve())


def load_heldout_cases(path: str) -> List[HeldoutBenchmarkCase]:
    dataset_path = Path(path).resolve()
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Held-out benchmark dataset must be a JSON list.")

    cases: List[HeldoutBenchmarkCase] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Held-out case at index {index} is not an object.")
        case_id = str(item.get("case_id", "")).strip() or f"H{index:03d}"
        if case_id in seen_ids:
            raise ValueError(f"Duplicate held-out case_id '{case_id}'.")
        seen_ids.add(case_id)
        requirement = str(item.get("requirement", "")).strip()
        expected = str(item.get("expected_terminal_status", "")).strip()
        tags_raw = item.get("tags", [])
        tags = [str(tag) for tag in tags_raw] if isinstance(tags_raw, list) else []
        if not requirement:
            raise ValueError(f"Held-out case '{case_id}' has empty requirement.")
        if expected not in SUPPORTED_TERMINAL_STATUSES:
            raise ValueError(
                f"Held-out case '{case_id}' has unsupported expected_terminal_status '{expected}'."
            )

        oracle = _load_oracle_spec(dataset_path, case_id, item.get("oracle"))
        if expected == TERMINAL_VERIFIED and oracle is None:
            raise ValueError(f"Held-out verified case '{case_id}' requires an external oracle.")
        cases.append(
            HeldoutBenchmarkCase(
                case_id=case_id,
                requirement=requirement,
                expected_terminal_status=expected,
                tags=tags,
                oracle=oracle,
            )
        )
    if not cases:
        raise ValueError("Held-out benchmark requires at least one case.")
    return cases


def _load_oracle_spec(
    dataset_path: Path,
    case_id: str,
    payload: object,
) -> OracleSpec | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"Held-out case '{case_id}' oracle must be an object.")
    raw_path = str(payload.get("path", "")).strip()
    if not raw_path:
        raise ValueError(f"Held-out case '{case_id}' oracle path is empty.")
    oracle_path = (dataset_path.parent / raw_path).resolve()
    if not oracle_path.is_relative_to(dataset_path.parent):
        raise ValueError(f"Held-out case '{case_id}' oracle escapes the benchmark directory.")
    if not oracle_path.is_file():
        raise ValueError(f"Held-out case '{case_id}' oracle does not exist: {oracle_path}")
    timeout_seconds = int(payload.get("timeout_seconds", 30))
    if timeout_seconds < 1 or timeout_seconds > 300:
        raise ValueError(f"Held-out case '{case_id}' oracle timeout must be between 1 and 300 seconds.")
    return OracleSpec(path=str(oracle_path), timeout_seconds=timeout_seconds)


def execute_pytest_oracle(
    oracle: OracleSpec,
    package_root: str,
    python_executable: str = sys.executable,
) -> OracleResult:
    started = time.perf_counter()
    package_path = Path(package_root).resolve()
    if not package_path.is_dir():
        return OracleResult(
            executed=False,
            passed=False,
            error=f"Package root does not exist: {package_path}",
            execution_time_seconds=time.perf_counter() - started,
        )

    env = os.environ.copy()
    src_path = package_path / "src"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(src_path), existing_pythonpath) if part
    )
    env["FORGE_PACKAGE_ROOT"] = str(package_path)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    try:
        with tempfile.TemporaryDirectory(prefix="forge-heldout-oracle-") as temp_root:
            completed = subprocess.run(
                [
                    python_executable,
                    "-B",
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    oracle.path,
                    f"--basetemp={Path(temp_root) / 'pytest'}",
                ],
                cwd=str(package_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=oracle.timeout_seconds,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        return OracleResult(
            executed=True,
            passed=False,
            stdout=_coerce_output(exc.stdout),
            stderr=_coerce_output(exc.stderr),
            error=f"Oracle timed out after {oracle.timeout_seconds}s.",
            execution_time_seconds=time.perf_counter() - started,
        )
    except OSError as exc:
        return OracleResult(
            executed=False,
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
            execution_time_seconds=time.perf_counter() - started,
        )

    return OracleResult(
        executed=True,
        passed=completed.returncode == 0,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        execution_time_seconds=time.perf_counter() - started,
    )


def _coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_heldout_cases(
    cases: List[HeldoutBenchmarkCase],
    run_case: Callable[[str], ForgeResult],
    run_oracle: Callable[[OracleSpec, str], OracleResult] = execute_pytest_oracle,
) -> HeldoutBenchmarkSummary:
    if not cases:
        raise ValueError("Held-out benchmark requires at least one case.")

    run_started = time.perf_counter()
    results: List[HeldoutCaseResult] = []
    for case in cases:
        case_started = time.perf_counter()
        oracle_result: OracleResult | None = None
        try:
            forge_result = run_case(case.requirement)
            observed = forge_result.terminal_status
            failure_signatures = (
                list(forge_result.validation.failure_signatures)
                if forge_result.validation is not None
                else []
            )
            artifact_path = forge_result.artifact_path
            error = None
            if observed == TERMINAL_VERIFIED and case.oracle is not None:
                oracle_result = run_oracle(case.oracle, artifact_path)
        except Exception as exc:  # pragma: no cover - safety net only
            observed = "exception"
            failure_signatures = []
            artifact_path = ""
            error = f"{type(exc).__name__}: {exc}"

        status_matched = observed == case.expected_terminal_status
        oracle_gate_passed = (
            case.expected_terminal_status != TERMINAL_VERIFIED
            or (
                observed == TERMINAL_VERIFIED
                and oracle_result is not None
                and oracle_result.passed
            )
        )
        results.append(
            HeldoutCaseResult(
                case_id=case.case_id,
                expected_terminal_status=case.expected_terminal_status,
                observed_terminal_status=observed,
                status_matched=status_matched,
                passed=status_matched and oracle_gate_passed,
                execution_time_seconds=time.perf_counter() - case_started,
                artifact_path=artifact_path,
                failure_signatures=failure_signatures,
                oracle_result=oracle_result,
                error=error,
            )
        )

    return _summarize_results(results, time.perf_counter() - run_started)


def _summarize_results(
    results: List[HeldoutCaseResult],
    total_runtime: float,
) -> HeldoutBenchmarkSummary:
    total_cases = len(results)
    expected_verified = [item for item in results if item.expected_terminal_status == TERMINAL_VERIFIED]
    oracle_executed = [
        item for item in results if item.oracle_result is not None and item.oracle_result.executed
    ]
    externally_verified = [
        item
        for item in expected_verified
        if item.observed_terminal_status == TERMINAL_VERIFIED
        and item.oracle_result is not None
        and item.oracle_result.passed
    ]
    observed_verified = [item for item in results if item.observed_terminal_status == TERMINAL_VERIFIED]
    false_verified = [
        item
        for item in observed_verified
        if item.expected_terminal_status != TERMINAL_VERIFIED
        or item.oracle_result is None
        or not item.oracle_result.passed
    ]
    expected_infeasible = [
        item
        for item in results
        if item.expected_terminal_status == TERMINAL_INFEASIBLE_PROVEN
    ]
    correct_infeasible = [
        item
        for item in expected_infeasible
        if item.observed_terminal_status == TERMINAL_INFEASIBLE_PROVEN
    ]
    passed_cases = sum(1 for item in results if item.passed)
    benchmark_id = f"forge-heldout-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    return HeldoutBenchmarkSummary(
        benchmark_id=benchmark_id,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=total_cases - passed_cases,
        status_accuracy=sum(1 for item in results if item.status_matched) / total_cases,
        external_verified_at_1=(
            len(externally_verified) / len(expected_verified) if expected_verified else 0.0
        ),
        oracle_pass_rate=(
            sum(1 for item in oracle_executed if item.oracle_result and item.oracle_result.passed)
            / len(oracle_executed)
            if oracle_executed
            else 0.0
        ),
        external_false_verified_rate=(
            len(false_verified) / len(observed_verified) if observed_verified else 0.0
        ),
        infeasible_detection_rate=(
            len(correct_infeasible) / len(expected_infeasible) if expected_infeasible else 0.0
        ),
        avg_case_runtime_seconds=statistics.mean(
            item.execution_time_seconds for item in results
        ),
        total_runtime_seconds=total_runtime,
        case_results=results,
    )


def evaluate_heldout_thresholds(
    summary: HeldoutBenchmarkSummary,
    thresholds: HeldoutThresholds,
) -> List[str]:
    failures: List[str] = []
    if summary.status_accuracy < thresholds.min_status_accuracy:
        failures.append(
            "status_accuracy_below_threshold:"
            f" actual={summary.status_accuracy:.3f} required>={thresholds.min_status_accuracy:.3f}"
        )
    if summary.external_verified_at_1 < thresholds.min_external_verified_at_1:
        failures.append(
            "external_verified_at_1_below_threshold:"
            f" actual={summary.external_verified_at_1:.3f}"
            f" required>={thresholds.min_external_verified_at_1:.3f}"
        )
    if summary.external_false_verified_rate > thresholds.max_external_false_verified_rate:
        failures.append(
            "external_false_verified_rate_above_threshold:"
            f" actual={summary.external_false_verified_rate:.3f}"
            f" required<={thresholds.max_external_false_verified_rate:.3f}"
        )
    if summary.infeasible_detection_rate < thresholds.min_infeasible_detection_rate:
        failures.append(
            "infeasible_detection_rate_below_threshold:"
            f" actual={summary.infeasible_detection_rate:.3f}"
            f" required>={thresholds.min_infeasible_detection_rate:.3f}"
        )
    return failures


def persist_heldout_summary(summary: HeldoutBenchmarkSummary, output_root: str) -> str:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / f"{summary.benchmark_id}.json"
    output_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8")
    return str(output_path.resolve())


def render_heldout_summary(summary: HeldoutBenchmarkSummary, output_path: str) -> str:
    return "\n".join(
        [
            "Forge Held-out Benchmark",
            f"Benchmark id: {summary.benchmark_id}",
            f"Cases: {summary.total_cases}",
            f"Passed: {summary.passed_cases}",
            f"Failed: {summary.failed_cases}",
            f"Status accuracy: {summary.status_accuracy:.3f}",
            f"External Verified@1: {summary.external_verified_at_1:.3f}",
            f"Oracle pass rate: {summary.oracle_pass_rate:.3f}",
            f"External false-verified rate: {summary.external_false_verified_rate:.3f}",
            f"Infeasible detection rate: {summary.infeasible_detection_rate:.3f}",
            f"Average case runtime: {summary.avg_case_runtime_seconds:.2f}s",
            f"Total runtime: {summary.total_runtime_seconds:.2f}s",
            f"Report: {output_path}",
        ]
    )
