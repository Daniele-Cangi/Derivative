import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List

from core.forge.contracts import ForgeResult


TERMINAL_VERIFIED = "verified"
TERMINAL_INFEASIBLE_PROVEN = "infeasible_proven"
TERMINAL_VALIDATION_FAILED = "validation_failed"
SUPPORTED_TERMINAL_STATUSES = {
    TERMINAL_VERIFIED,
    TERMINAL_INFEASIBLE_PROVEN,
    TERMINAL_VALIDATION_FAILED,
}


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    requirement: str
    expected_terminal_status: str
    tags: List[str] = field(default_factory=list)


@dataclass
class BenchmarkCaseResult:
    case_id: str
    expected_terminal_status: str
    observed_terminal_status: str
    passed: bool
    execution_time_seconds: float
    artifact_path: str
    failure_signatures: List[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class BenchmarkSummary:
    benchmark_id: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    status_accuracy: float
    verified_at_1: float
    false_verified_rate: float
    infeasible_detection_rate: float
    avg_case_runtime_seconds: float
    total_runtime_seconds: float
    case_results: List[BenchmarkCaseResult] = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkThresholds:
    min_status_accuracy: float = 0.95
    min_verified_at_1: float = 0.90
    max_false_verified_rate: float = 0.00
    min_infeasible_detection_rate: float = 1.00


def default_forge_benchmark_cases() -> List[BenchmarkCase]:
    return [
        BenchmarkCase(
            case_id="F001",
            requirement=(
                "Build a Python CLI that reads a CSV of contracts, extracts expiration dates, "
                "flags contracts expiring in less than 90 days, writes a summary CSV, and includes tests."
            ),
            expected_terminal_status=TERMINAL_VERIFIED,
            tags=["cli", "csv", "expiration", "baseline_feasible"],
        ),
        BenchmarkCase(
            case_id="F002",
            requirement=(
                "Build a Python CLI that reads a CSV of invoices with columns invoice_id, due_date, amount, "
                "customer_name, flags overdue invoices, writes a summary CSV with totals and counts, and "
                "includes tests for malformed rows and invalid dates."
            ),
            expected_terminal_status=TERMINAL_VERIFIED,
            tags=["cli", "business", "validation", "invoice"],
        ),
        BenchmarkCase(
            case_id="F003",
            requirement=(
                "Build a Python CLI that reads a CSV of contracts, extracts expiration dates, "
                "flags contracts expiring in less than 90 days, writes a summary CSV, includes tests, "
                "and guarantees support for every possible date format."
            ),
            expected_terminal_status=TERMINAL_VALIDATION_FAILED,
            tags=["universal_constraint", "truth_gate"],
        ),
        BenchmarkCase(
            case_id="F004",
            requirement=(
                "Design a network on exactly 4 nodes such that every pair of nodes is directly connected, "
                "the network diameter is strictly greater than 2, vertex connectivity is at least 3, and "
                "the total number of edges does not exceed 3."
            ),
            expected_terminal_status=TERMINAL_INFEASIBLE_PROVEN,
            tags=["infeasible", "contradiction"],
        ),
    ]


def bundled_dataset_path(filename: str) -> str:
    root = Path(__file__).resolve().parents[2]
    return str((root / "benchmarks" / filename).resolve())


def extended_forge_benchmark_cases() -> List[BenchmarkCase]:
    return load_benchmark_cases(bundled_dataset_path("forge_extended_benchmark.json"))


def load_benchmark_cases(path: str) -> List[BenchmarkCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Benchmark dataset must be a JSON list.")
    cases: List[BenchmarkCase] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Benchmark case at index {index} is not an object.")
        case_id = str(item.get("case_id", "")).strip() or f"C{index:03d}"
        requirement = str(item.get("requirement", "")).strip()
        expected = str(item.get("expected_terminal_status", "")).strip()
        tags_raw = item.get("tags", [])
        tags = [str(tag) for tag in tags_raw] if isinstance(tags_raw, list) else []
        if not requirement:
            raise ValueError(f"Benchmark case '{case_id}' has empty requirement.")
        if expected not in SUPPORTED_TERMINAL_STATUSES:
            raise ValueError(
                f"Benchmark case '{case_id}' has unsupported expected_terminal_status '{expected}'."
            )
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                requirement=requirement,
                expected_terminal_status=expected,
                tags=tags,
            )
        )
    return cases


def run_benchmark_cases(
    cases: List[BenchmarkCase],
    run_case: Callable[[str], ForgeResult],
) -> BenchmarkSummary:
    if not cases:
        raise ValueError("Benchmark requires at least one case.")

    run_started = time.perf_counter()
    results: List[BenchmarkCaseResult] = []
    for case in cases:
        case_started = time.perf_counter()
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
        except Exception as exc:  # pragma: no cover - safety net only
            observed = "exception"
            failure_signatures = []
            artifact_path = ""
            error = f"{type(exc).__name__}: {exc}"
        case_runtime = time.perf_counter() - case_started
        passed = observed == case.expected_terminal_status
        results.append(
            BenchmarkCaseResult(
                case_id=case.case_id,
                expected_terminal_status=case.expected_terminal_status,
                observed_terminal_status=observed,
                passed=passed,
                execution_time_seconds=case_runtime,
                artifact_path=artifact_path,
                failure_signatures=failure_signatures,
                error=error,
            )
        )

    total_runtime = time.perf_counter() - run_started
    total_cases = len(results)
    passed_cases = sum(1 for result in results if result.passed)
    failed_cases = total_cases - passed_cases

    expected_verified = [result for result in results if result.expected_terminal_status == TERMINAL_VERIFIED]
    correct_verified = [result for result in expected_verified if result.observed_terminal_status == TERMINAL_VERIFIED]
    observed_verified = [result for result in results if result.observed_terminal_status == TERMINAL_VERIFIED]
    false_verified = [
        result
        for result in observed_verified
        if result.expected_terminal_status != TERMINAL_VERIFIED
    ]
    expected_infeasible = [
        result
        for result in results
        if result.expected_terminal_status == TERMINAL_INFEASIBLE_PROVEN
    ]
    correct_infeasible = [
        result
        for result in expected_infeasible
        if result.observed_terminal_status == TERMINAL_INFEASIBLE_PROVEN
    ]
    avg_case_runtime = statistics.mean(result.execution_time_seconds for result in results)

    benchmark_id = f"forge-benchmark-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    return BenchmarkSummary(
        benchmark_id=benchmark_id,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        status_accuracy=passed_cases / total_cases,
        verified_at_1=(len(correct_verified) / len(expected_verified)) if expected_verified else 0.0,
        false_verified_rate=(len(false_verified) / len(observed_verified)) if observed_verified else 0.0,
        infeasible_detection_rate=(
            len(correct_infeasible) / len(expected_infeasible)
        )
        if expected_infeasible
        else 0.0,
        avg_case_runtime_seconds=avg_case_runtime,
        total_runtime_seconds=total_runtime,
        case_results=results,
    )


def persist_benchmark_summary(summary: BenchmarkSummary, output_root: str) -> str:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / f"{summary.benchmark_id}.json"
    output_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8")
    return str(output_path.resolve())


def evaluate_benchmark_thresholds(
    summary: BenchmarkSummary,
    thresholds: BenchmarkThresholds,
) -> List[str]:
    failures: List[str] = []
    if summary.status_accuracy < thresholds.min_status_accuracy:
        failures.append(
            "status_accuracy_below_threshold:"
            f" actual={summary.status_accuracy:.3f} required>={thresholds.min_status_accuracy:.3f}"
        )
    if summary.verified_at_1 < thresholds.min_verified_at_1:
        failures.append(
            "verified_at_1_below_threshold:"
            f" actual={summary.verified_at_1:.3f} required>={thresholds.min_verified_at_1:.3f}"
        )
    if summary.false_verified_rate > thresholds.max_false_verified_rate:
        failures.append(
            "false_verified_rate_above_threshold:"
            f" actual={summary.false_verified_rate:.3f} required<={thresholds.max_false_verified_rate:.3f}"
        )
    if summary.infeasible_detection_rate < thresholds.min_infeasible_detection_rate:
        failures.append(
            "infeasible_detection_rate_below_threshold:"
            f" actual={summary.infeasible_detection_rate:.3f} required>={thresholds.min_infeasible_detection_rate:.3f}"
        )
    return failures


def render_benchmark_summary(summary: BenchmarkSummary, output_path: str) -> str:
    return "\n".join(
        [
            "Forge Benchmark",
            f"Benchmark id: {summary.benchmark_id}",
            f"Cases: {summary.total_cases}",
            f"Passed: {summary.passed_cases}",
            f"Failed: {summary.failed_cases}",
            f"Status accuracy: {summary.status_accuracy:.3f}",
            f"Verified@1: {summary.verified_at_1:.3f}",
            f"False-verified rate: {summary.false_verified_rate:.3f}",
            f"Infeasible detection rate: {summary.infeasible_detection_rate:.3f}",
            f"Average case runtime: {summary.avg_case_runtime_seconds:.2f}s",
            f"Total runtime: {summary.total_runtime_seconds:.2f}s",
            f"Report: {output_path}",
        ]
    )
