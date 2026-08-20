import json
import shutil
import statistics
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
from core.constraint_witnesses import finite_witness_contradictions
from core.forge.contracts import ForgeResult
from core.forge.execution import (
    LocalProcessExecutor,
    ProcessExecutor,
    SandboxProcessRequest,
)
from core.forge.fixture_oracle import fixture_oracle_mismatches
from core.forge.oracle_contract import oracle_contract_mismatches


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
    valid: bool = True
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    execution_time_seconds: float = 0.0
    backend: str = ""
    timed_out: bool = False
    isolation: dict[str, object] = field(default_factory=dict)
    sanity_failures: List[dict[str, object]] = field(default_factory=list)


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
    forge_execution_time_seconds: float = 0.0
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
    adjudicated_cases: int = 0
    invalid_oracle_cases: int = 0
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
    executor: ProcessExecutor | None = None,
) -> OracleResult:
    started = time.perf_counter()
    process_executor = executor or LocalProcessExecutor(
        python_executable=python_executable,
    )
    package_path = Path(package_root).resolve()
    if not package_path.is_dir():
        return OracleResult(
            executed=False,
            passed=False,
            error=f"Package root does not exist: {package_path}",
            execution_time_seconds=time.perf_counter() - started,
        )

    try:
        with tempfile.TemporaryDirectory(prefix="forge-heldout-oracle-") as temp_root:
            staging_root = Path(temp_root)
            staged_package = staging_root / "package"
            _copy_package_for_oracle(package_path, staged_package)
            staged_oracle = staging_root / "oracle.py"
            staged_oracle.write_bytes(Path(oracle.path).read_bytes())
            completed = process_executor.run(
                SandboxProcessRequest(
                    command=[
                        "python",
                        "-B",
                        "-m",
                        "pytest",
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        "../oracle.py",
                        "--basetemp=../pytest",
                    ],
                    workspace=staging_root,
                    working_directory="package",
                    environment={
                        "PYTHONPATH": "src",
                        "FORGE_PACKAGE_ROOT": ".",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                    },
                    timeout_seconds=oracle.timeout_seconds,
                )
            )
    except OSError as exc:
        return OracleResult(
            executed=False,
            passed=False,
            error=f"{type(exc).__name__}: {exc}",
            execution_time_seconds=time.perf_counter() - started,
        )

    return OracleResult(
        executed=completed.launch_error is None,
        passed=completed.returncode == 0,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        error=(
            f"Oracle timed out after {oracle.timeout_seconds}s."
            if completed.timed_out
            else completed.launch_error
        ),
        execution_time_seconds=time.perf_counter() - started,
        backend=completed.backend,
        timed_out=completed.timed_out,
        isolation=completed.isolation,
    )


def _copy_package_for_oracle(source: Path, destination: Path) -> None:
    symlinks = [path for path in source.rglob("*") if path.is_symlink()]
    if symlinks:
        raise OSError(
            "Oracle package staging refuses symbolic links: "
            + ", ".join(str(path.relative_to(source)) for path in symlinks[:5])
        )
    shutil.copytree(source, destination)


def inspect_oracle_sanity(case: HeldoutBenchmarkCase) -> OracleResult | None:
    if case.oracle is None:
        return None
    witness_contradictions = finite_witness_contradictions(case.requirement)
    if (
        witness_contradictions
        and case.expected_terminal_status != TERMINAL_INFEASIBLE_PROVEN
    ):
        return OracleResult(
            executed=False,
            passed=False,
            valid=False,
            error=(
                "oracle_invalid: expected terminal status contradicts a finite "
                "requirement witness"
            ),
            sanity_failures=[
                contradiction.to_evidence()
                for contradiction in witness_contradictions
            ],
        )
    try:
        source = Path(case.oracle.path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    mismatches = fixture_oracle_mismatches(source, case.requirement)
    if mismatches:
        return OracleResult(
            executed=False,
            passed=False,
            valid=False,
            error="oracle_invalid: fixture expectations contradict the requirement",
            sanity_failures=[mismatch.to_evidence() for mismatch in mismatches],
        )
    contract_mismatches = oracle_contract_mismatches(source, case.requirement)
    if contract_mismatches:
        return OracleResult(
            executed=False,
            passed=False,
            valid=False,
            error="oracle_invalid: invocation contract contradicts the requirement",
            sanity_failures=[
                mismatch.to_evidence() for mismatch in contract_mismatches
            ],
        )
    return None


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
        oracle_result = inspect_oracle_sanity(case)
        if oracle_result is not None and not oracle_result.valid:
            results.append(
                HeldoutCaseResult(
                    case_id=case.case_id,
                    expected_terminal_status=case.expected_terminal_status,
                    observed_terminal_status="oracle_invalid",
                    status_matched=False,
                    passed=False,
                    execution_time_seconds=time.perf_counter() - case_started,
                    artifact_path="",
                    oracle_result=oracle_result,
                    error=oracle_result.error,
                    model_cost_pricing_source="not_executed_invalid_oracle",
                )
            )
            continue
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
            forge_execution_time_seconds = forge_result.execution_time_seconds
            model_request_count = run_metrics.model_request_count
            model_input_tokens = run_metrics.model_input_tokens
            model_output_tokens = run_metrics.model_output_tokens
            model_total_tokens = run_metrics.model_total_tokens
            estimated_model_cost_usd = run_metrics.estimated_model_cost_usd
            model_cost_pricing_source = run_metrics.model_cost_pricing_source
            error = None
            if observed == TERMINAL_VERIFIED and case.oracle is not None:
                oracle_result = run_oracle(case.oracle, artifact_path)
        except Exception as exc:  # pragma: no cover - safety net only
            observed = "exception"
            failure_signatures = []
            artifact_path = ""
            forge_execution_time_seconds = 0.0
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
                forge_execution_time_seconds=forge_execution_time_seconds,
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

    return _summarize_results(results, time.perf_counter() - run_started)


def _summarize_results(
    results: List[HeldoutCaseResult],
    total_runtime: float,
) -> HeldoutBenchmarkSummary:
    total_cases = len(results)
    invalid_oracle = [
        item
        for item in results
        if item.oracle_result is not None and not item.oracle_result.valid
    ]
    adjudicable = [
        item
        for item in results
        if item.oracle_result is None or item.oracle_result.valid
    ]
    expected_verified = [
        item
        for item in adjudicable
        if item.expected_terminal_status == TERMINAL_VERIFIED
    ]
    oracle_executed = [
        item for item in results if item.oracle_result is not None and item.oracle_result.executed
    ]
    externally_verified = [
        item
        for item in expected_verified
        if item.verified_at_1
        and item.oracle_result is not None
        and item.oracle_result.passed
    ]
    observed_verified = [
        item
        for item in adjudicable
        if item.observed_terminal_status == TERMINAL_VERIFIED
    ]
    false_verified = [
        item
        for item in observed_verified
        if item.expected_terminal_status != TERMINAL_VERIFIED
        or item.oracle_result is None
        or not item.oracle_result.passed
    ]
    expected_infeasible = [
        item
        for item in adjudicable
        if item.expected_terminal_status == TERMINAL_INFEASIBLE_PROVEN
    ]
    correct_infeasible = [
        item
        for item in expected_infeasible
        if item.observed_terminal_status == TERMINAL_INFEASIBLE_PROVEN
    ]
    passed_cases = sum(1 for item in results if item.passed)
    repair_eligible = [item for item in expected_verified if not item.verified_at_1]
    repaired_successes = [
        item
        for item in repair_eligible
        if item.observed_terminal_status == TERMINAL_VERIFIED
        and item.success_after_repair
        and item.oracle_result is not None
        and item.oracle_result.passed
    ]
    runtimes = sorted(item.execution_time_seconds for item in results)
    costed_results = [
        item for item in results if item.estimated_model_cost_usd is not None
    ]
    benchmark_id = f"forge-heldout-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    return HeldoutBenchmarkSummary(
        benchmark_id=benchmark_id,
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=total_cases - passed_cases,
        status_accuracy=(
            sum(1 for item in adjudicable if item.status_matched) / len(adjudicable)
            if adjudicable
            else 0.0
        ),
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
        median_case_runtime_seconds=statistics.median(runtimes),
        p95_case_runtime_seconds=_nearest_rank_percentile(runtimes, 0.95),
        total_runtime_seconds=total_runtime,
        success_after_repair_rate=(
            len(repaired_successes) / len(repair_eligible)
            if repair_eligible
            else None
        ),
        total_repairs=sum(item.repair_count for item in results),
        avg_repairs_per_case=statistics.mean(item.repair_count for item in results),
        total_model_requests=sum(item.model_request_count for item in results),
        total_model_input_tokens=sum(item.model_input_tokens for item in results),
        total_model_output_tokens=sum(item.model_output_tokens for item in results),
        total_model_tokens=sum(item.model_total_tokens for item in results),
        total_estimated_model_cost_usd=(
            round(sum(item.estimated_model_cost_usd or 0.0 for item in results), 8)
            if len(costed_results) == total_cases
            else None
        ),
        model_cost_coverage_rate=len(costed_results) / total_cases,
        adjudicated_cases=len(adjudicable),
        invalid_oracle_cases=len(invalid_oracle),
        case_results=results,
    )


def evaluate_heldout_thresholds(
    summary: HeldoutBenchmarkSummary,
    thresholds: HeldoutThresholds,
) -> List[str]:
    failures: List[str] = []
    if summary.invalid_oracle_cases:
        failures.append(
            f"invalid_oracle_cases: actual={summary.invalid_oracle_cases} required=0"
        )
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
            f"Adjudicated cases: {summary.adjudicated_cases}",
            f"Passed: {summary.passed_cases}",
            f"Failed: {summary.failed_cases}",
            f"Status accuracy: {summary.status_accuracy:.3f}",
            f"External Verified@1: {summary.external_verified_at_1:.3f}",
            "External success after repair: "
            + _format_optional_rate(summary.success_after_repair_rate),
            f"Oracle pass rate: {summary.oracle_pass_rate:.3f}",
            f"Invalid oracle cases: {summary.invalid_oracle_cases}",
            f"External false-verified rate: {summary.external_false_verified_rate:.3f}",
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
