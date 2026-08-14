import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

from core.forge.contracts import ForgeResult
from core.forge.heldout_benchmark import (
    HeldoutBenchmarkCase,
    HeldoutBenchmarkSummary,
    HeldoutThresholds,
    OracleResult,
    OracleSpec,
    evaluate_heldout_thresholds,
    execute_pytest_oracle,
    load_heldout_cases,
    run_heldout_cases,
)


BLIND_BENCHMARK_SCHEMA_VERSION = 1
_EXCLUDED_BASELINE_FILES = {
    "core/forge/benchmark.py",
    "core/forge/blind_benchmark.py",
    "core/forge/heldout_benchmark.py",
}


@dataclass(frozen=True)
class BlindBenchmarkBundle:
    bundle_id: str
    schema_version: int
    frozen_at: str
    manifest_path: str
    manifest_sha256: str
    dataset_path: str
    dataset_sha256: str
    baseline_sha256: str
    baseline_file_count: int
    source_urls: List[str] = field(default_factory=list)
    oracle_sha256: Dict[str, str] = field(default_factory=dict)
    cases: List[HeldoutBenchmarkCase] = field(default_factory=list)


@dataclass
class BlindBenchmarkReport:
    report_id: str
    bundle_id: str
    schema_version: int
    frozen_at: str
    manifest_sha256: str
    dataset_sha256: str
    baseline_sha256: str
    baseline_file_count: int
    baseline_verified: bool
    source_urls: List[str]
    oracle_sha256: Dict[str, str]
    summary: HeldoutBenchmarkSummary


def bundled_blind_manifest_path() -> str:
    root = Path(__file__).resolve().parents[2]
    return str((root / "benchmarks" / "blind_v2" / "manifest.json").resolve())


def compute_forge_baseline_digest(repository_root: str | Path) -> tuple[str, int]:
    root = Path(repository_root).resolve()
    candidates = {
        root / "forge.py",
        root / "core" / "execution_loop.py",
        root / "core" / "kernel.py",
        root / "core" / "obligation_compiler.py",
    }
    candidates.update((root / "core" / "forge").rglob("*.py"))
    protected = [
        path
        for path in candidates
        if path.is_file()
        and path.relative_to(root).as_posix() not in _EXCLUDED_BASELINE_FILES
    ]
    if not protected:
        raise ValueError(f"No Forge baseline files found under {root}.")

    digest = hashlib.sha256()
    for path in sorted(protected, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(protected)


def load_blind_bundle(
    manifest_path: str,
    repository_root: str | Path | None = None,
) -> BlindBenchmarkBundle:
    path = Path(manifest_path).resolve()
    payload_bytes = path.read_bytes()
    payload = json.loads(payload_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Blind benchmark manifest must be a JSON object.")

    schema_version = int(payload.get("schema_version", 0))
    if schema_version != BLIND_BENCHMARK_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported blind benchmark schema_version: "
            f"{schema_version}; expected {BLIND_BENCHMARK_SCHEMA_VERSION}."
        )
    bundle_id = str(payload.get("bundle_id", "")).strip()
    frozen_at = str(payload.get("frozen_at", "")).strip()
    if not bundle_id or not frozen_at:
        raise ValueError("Blind benchmark manifest requires bundle_id and frozen_at.")

    dataset_spec = _required_mapping(payload, "dataset")
    dataset_path = _resolve_bundle_file(path.parent, dataset_spec.get("path"), "dataset")
    expected_dataset_sha256 = _required_sha256(dataset_spec.get("sha256"), "dataset")
    actual_dataset_sha256 = _sha256_file(dataset_path)
    if actual_dataset_sha256 != expected_dataset_sha256:
        raise ValueError(
            "Blind benchmark dataset digest mismatch: "
            f"expected={expected_dataset_sha256}, actual={actual_dataset_sha256}."
        )

    cases = load_heldout_cases(str(dataset_path))
    expected_oracles = {
        case.case_id: case
        for case in cases
        if case.oracle is not None
    }
    oracle_digests_raw = _required_mapping(payload, "oracle_sha256")
    if set(oracle_digests_raw) != set(expected_oracles):
        raise ValueError(
            "Blind benchmark oracle digest ids do not match oracle-backed cases: "
            f"expected={sorted(expected_oracles)}, actual={sorted(oracle_digests_raw)}."
        )
    oracle_sha256: Dict[str, str] = {}
    for case_id, case in sorted(expected_oracles.items()):
        expected_digest = _required_sha256(oracle_digests_raw[case_id], f"oracle {case_id}")
        actual_digest = _sha256_file(Path(case.oracle.path))
        if actual_digest != expected_digest:
            raise ValueError(
                f"Blind benchmark oracle digest mismatch for {case_id}: "
                f"expected={expected_digest}, actual={actual_digest}."
            )
        oracle_sha256[case_id] = actual_digest

    baseline_spec = _required_mapping(payload, "forge_baseline")
    expected_baseline_sha256 = _required_sha256(
        baseline_spec.get("sha256"),
        "forge baseline",
    )
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    actual_baseline_sha256, baseline_file_count = compute_forge_baseline_digest(root)
    if actual_baseline_sha256 != expected_baseline_sha256:
        raise ValueError(
            "Forge baseline digest mismatch. The blind bundle is locked to an earlier "
            f"implementation: expected={expected_baseline_sha256}, "
            f"actual={actual_baseline_sha256}."
        )
    expected_file_count = int(baseline_spec.get("file_count", 0))
    if expected_file_count != baseline_file_count:
        raise ValueError(
            "Forge baseline file count mismatch: "
            f"expected={expected_file_count}, actual={baseline_file_count}."
        )

    sources_raw = payload.get("source_urls", [])
    if not isinstance(sources_raw, list) or not all(
        isinstance(item, str) and item.startswith("https://") for item in sources_raw
    ):
        raise ValueError("Blind benchmark source_urls must be a list of HTTPS URLs.")

    return BlindBenchmarkBundle(
        bundle_id=bundle_id,
        schema_version=schema_version,
        frozen_at=frozen_at,
        manifest_path=str(path),
        manifest_sha256=hashlib.sha256(payload_bytes).hexdigest(),
        dataset_path=str(dataset_path),
        dataset_sha256=actual_dataset_sha256,
        baseline_sha256=actual_baseline_sha256,
        baseline_file_count=baseline_file_count,
        source_urls=list(sources_raw),
        oracle_sha256=oracle_sha256,
        cases=cases,
    )


def run_blind_bundle(
    bundle: BlindBenchmarkBundle,
    run_case: Callable[[str], ForgeResult],
    run_oracle: Callable[[OracleSpec, str], OracleResult] = execute_pytest_oracle,
) -> BlindBenchmarkReport:
    summary = run_heldout_cases(
        bundle.cases,
        run_case=run_case,
        run_oracle=run_oracle,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return BlindBenchmarkReport(
        report_id=f"{bundle.bundle_id}-{timestamp}",
        bundle_id=bundle.bundle_id,
        schema_version=bundle.schema_version,
        frozen_at=bundle.frozen_at,
        manifest_sha256=bundle.manifest_sha256,
        dataset_sha256=bundle.dataset_sha256,
        baseline_sha256=bundle.baseline_sha256,
        baseline_file_count=bundle.baseline_file_count,
        baseline_verified=True,
        source_urls=list(bundle.source_urls),
        oracle_sha256=dict(bundle.oracle_sha256),
        summary=summary,
    )


def evaluate_blind_thresholds(
    report: BlindBenchmarkReport,
    thresholds: HeldoutThresholds,
) -> List[str]:
    failures = evaluate_heldout_thresholds(report.summary, thresholds)
    if not report.baseline_verified:
        failures.insert(0, "forge_baseline_not_verified")
    return failures


def persist_blind_report(report: BlindBenchmarkReport, output_root: str) -> str:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / f"{report.report_id}.json"
    output_path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(output_path.resolve())


def render_blind_report(report: BlindBenchmarkReport, output_path: str) -> str:
    summary = report.summary
    return "\n".join(
        [
            "Forge Blind Benchmark v2",
            f"Report id: {report.report_id}",
            f"Bundle: {report.bundle_id}",
            f"Baseline verified: {str(report.baseline_verified).lower()}",
            f"Baseline SHA-256: {report.baseline_sha256}",
            f"Cases: {summary.total_cases}",
            f"Passed: {summary.passed_cases}",
            f"Failed: {summary.failed_cases}",
            f"Status accuracy: {summary.status_accuracy:.3f}",
            f"External Verified@1: {summary.external_verified_at_1:.3f}",
            f"Oracle pass rate: {summary.oracle_pass_rate:.3f}",
            f"External false-verified rate: {summary.external_false_verified_rate:.3f}",
            f"Infeasible detection rate: {summary.infeasible_detection_rate:.3f}",
            f"Total runtime: {summary.total_runtime_seconds:.2f}s",
            f"Report: {output_path}",
        ]
    )


def _required_mapping(payload: Dict[str, object], key: str) -> Dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Blind benchmark manifest field '{key}' must be an object.")
    return value


def _resolve_bundle_file(bundle_root: Path, raw_path: object, label: str) -> Path:
    relative = str(raw_path or "").strip()
    if not relative:
        raise ValueError(f"Blind benchmark {label} path is empty.")
    resolved = (bundle_root / relative).resolve()
    if not resolved.is_relative_to(bundle_root):
        raise ValueError(f"Blind benchmark {label} escapes the bundle directory.")
    if not resolved.is_file():
        raise ValueError(f"Blind benchmark {label} does not exist: {resolved}")
    return resolved


def _required_sha256(value: object, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"Blind benchmark {label} sha256 must contain 64 hex characters.")
    return digest


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
