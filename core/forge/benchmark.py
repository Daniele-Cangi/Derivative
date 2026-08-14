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
    verified_at_1: bool = False
    success_after_repair: bool = False
    repair_count: int = 0
    validation_attempts: int = 0
    model_request_count: int = 0
    model_input_tokens: int = 0
    model_output_tokens: int = 0
    model_total_tokens: int = 0
    estimated_model_cost_usd: float | None = 0.0
    model_cost_pricing_source: str = "no_model_calls"


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
    median_case_runtime_seconds: float
    p95_case_runtime_seconds: float
    total_runtime_seconds: float
    success_after_repair_rate: float | None
    total_repairs: int
    avg_repairs_per_case: float
    total_model_requests: int
    total_model_input_tokens: int
    total_model_output_tokens: int
    total_model_tokens: int
    total_estimated_model_cost_usd: float | None
    model_cost_coverage_rate: float
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
            run_metrics = forge_result.run_metrics
            verified_at_1 = run_metrics.verified_at_1
            success_after_repair = run_metrics.success_after_repair
            repair_count = run_metrics.repair_count
            validation_attempts = run_metrics.validation_attempts
            model_request_count = run_metrics.model_request_count
            model_input_tokens = run_metrics.model_input_tokens
            model_output_tokens = run_metrics.model_output_tokens
            model_total_tokens = run_metrics.model_total_tokens
            estimated_model_cost_usd = run_metrics.estimated_model_cost_usd
            model_cost_pricing_source = run_metrics.model_cost_pricing_source
            error = None
        except Exception as exc:  # pragma: no cover - safety net only
            observed = "exception"
            failure_signatures = []
            artifact_path = ""
            verified_at_1 = False
            success_after_repair = False
            repair_count = 0
            validation_attempts = 0
            model_request_count = 0
            model_input_tokens = 0
            model_output_tokens = 0
            model_total_tokens = 0
            estimated_model_cost_usd = None
            model_cost_pricing_source = "unavailable"
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
                verified_at_1=verified_at_1,
                success_after_repair=success_after_repair,
                repair_count=repair_count,
                validation_attempts=validation_attempts,
                model_request_count=model_request_count,
                model_input_tokens=model_input_tokens,
                model_output_tokens=model_output_tokens,
                model_total_tokens=model_total_tokens,
                estimated_model_cost_usd=estimated_model_cost_usd,
                model_cost_pricing_source=model_cost_pricing_source,
            )
        )

    total_runtime = time.perf_counter() - run_started
    total_cases = len(results)
    passed_cases = sum(1 for result in results if result.passed)
    failed_cases = total_cases - passed_cases

    expected_verified = [result for result in results if result.expected_terminal_status == TERMINAL_VERIFIED]
    first_pass_verified = [result for result in expected_verified if result.verified_at_1]
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
    runtimes = sorted(result.execution_time_seconds for result in results)
    repair_eligible = [result for result in expected_verified if not result.verified_at_1]
    repaired_successes = [
        result
        for result in repair_eligible
        if result.observed_terminal_status == TERMINAL_VERIFIED
        and result.success_after_repair
    ]
    costed_results = [
        result for result in results if result.estimated_model_cost_usd is not None
    ]

    benchmark_id = f"forge-benchmark-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    return BenchmarkSummary(
        benchmark_id=benchmark_id,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        status_accuracy=passed_cases / total_cases,
        verified_at_1=(len(first_pass_verified) / len(expected_verified)) if expected_verified else 0.0,
        false_verified_rate=(len(false_verified) / len(observed_verified)) if observed_verified else 0.0,
        infeasible_detection_rate=(
            len(correct_infeasible) / len(expected_infeasible)
        )
        if expected_infeasible
        else 0.0,
        avg_case_runtime_seconds=avg_case_runtime,
        median_case_runtime_seconds=statistics.median(runtimes),
        p95_case_runtime_seconds=_nearest_rank_percentile(runtimes, 0.95),
        total_runtime_seconds=total_runtime,
        success_after_repair_rate=(
            len(repaired_successes) / len(repair_eligible)
            if repair_eligible
            else None
        ),
        total_repairs=sum(result.repair_count for result in results),
        avg_repairs_per_case=statistics.mean(result.repair_count for result in results),
        total_model_requests=sum(result.model_request_count for result in results),
        total_model_input_tokens=sum(result.model_input_tokens for result in results),
        total_model_output_tokens=sum(result.model_output_tokens for result in results),
        total_model_tokens=sum(result.model_total_tokens for result in results),
        total_estimated_model_cost_usd=(
            round(sum(result.estimated_model_cost_usd or 0.0 for result in results), 8)
            if len(costed_results) == total_cases
            else None
        ),
        model_cost_coverage_rate=len(costed_results) / total_cases,
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
            "Success after repair: " + _format_optional_rate(summary.success_after_repair_rate),
            f"False-verified rate: {summary.false_verified_rate:.3f}",
            f"Infeasible detection rate: {summary.infeasible_detection_rate:.3f}",
            f"Repairs: {summary.total_repairs} total, {summary.avg_repairs_per_case:.2f} per case",
            f"Average case runtime: {summary.avg_case_runtime_seconds:.2f}s",
            f"Median case runtime: {summary.median_case_runtime_seconds:.2f}s",
            f"P95 case runtime: {summary.p95_case_runtime_seconds:.2f}s",
            f"Total runtime: {summary.total_runtime_seconds:.2f}s",
            f"Model tokens: {summary.total_model_tokens}",
            "Estimated model cost: " + _format_optional_cost(summary.total_estimated_model_cost_usd),
            f"Model cost coverage: {summary.model_cost_coverage_rate:.3f}",
            f"Report: {output_path}",
        ]
    )


def _nearest_rank_percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, int((len(values) * percentile) + 0.999999) - 1))
    return values[index]


def _format_optional_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _format_optional_cost(value: float | None) -> str:
    return "unavailable" if value is None else f"${value:.8f}"
