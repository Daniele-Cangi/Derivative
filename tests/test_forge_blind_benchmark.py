import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from core.forge.benchmark import TERMINAL_VERIFIED
from core.forge.blind_benchmark import (
    bundled_blind_manifest_path,
    compute_forge_baseline_digest,
    load_blind_bundle,
    persist_blind_report,
    render_blind_report,
    run_blind_bundle,
)
from core.forge.contracts import ForgeResult, ForgeRoute, ForgeRunMetrics
from core.forge.heldout_benchmark import OracleResult


def _write_bundle(tmp_path: Path, repository_root: Path) -> Path:
    oracle_root = tmp_path / "oracles" / "B001"
    oracle_root.mkdir(parents=True)
    oracle = oracle_root / "oracle.py"
    oracle.write_text(
        "from library.core import identity\n\n"
        "def test_identity():\n"
        "    assert identity(3) == 3\n",
        encoding="utf-8",
    )
    cases = [
        {
            "case_id": "B001",
            "requirement": "Build a Python library exposing identity(value). Include tests.",
            "expected_terminal_status": "verified",
            "tags": ["blind-v2", "library"],
            "oracle": {"path": "oracles/B001/oracle.py", "timeout_seconds": 20},
        }
    ]
    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    baseline_digest, file_count = compute_forge_baseline_digest(repository_root)
    manifest = {
        "schema_version": 1,
        "bundle_id": "test-blind-v2",
        "frozen_at": "2026-08-14T00:00:00Z",
        "dataset": {
            "path": "cases.json",
            "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        },
        "forge_baseline": {
            "sha256": baseline_digest,
            "file_count": file_count,
        },
        "oracle_sha256": {
            "B001": hashlib.sha256(oracle.read_bytes()).hexdigest(),
        },
        "source_urls": ["https://example.com/specification"],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def test_blind_bundle_loads_only_when_dataset_oracles_and_baseline_match(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    manifest_path = _write_bundle(tmp_path, repository_root)

    bundle = load_blind_bundle(str(manifest_path), repository_root=repository_root)

    assert bundle.bundle_id == "test-blind-v2"
    assert bundle.baseline_file_count > 0
    assert len(bundle.cases) == 1
    assert bundle.oracle_sha256["B001"]


def test_forge_baseline_digest_is_independent_of_source_line_endings(tmp_path):
    paths = [
        tmp_path / "forge.py",
        tmp_path / "core" / "execution_loop.py",
        tmp_path / "core" / "kernel.py",
        tmp_path / "core" / "obligation_compiler.py",
        tmp_path / "core" / "forge" / "module.py",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"def value():\n    return 1\n")

    lf_digest = compute_forge_baseline_digest(tmp_path)
    for path in paths:
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

    assert compute_forge_baseline_digest(tmp_path) == lf_digest


def test_bundled_blind_v2_is_frozen_and_complete():
    bundle = load_blind_bundle(
        bundled_blind_manifest_path(),
        verify_baseline=False,
    )
    baseline_path = Path(bundle.manifest_path).with_name("baseline_result.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    assert bundle.bundle_id == "forge-blind-v2-20260814"
    assert len(bundle.cases) == 10
    assert len(bundle.oracle_sha256) == 6
    assert all("blind-v2" in case.tags for case in bundle.cases)
    assert bundle.baseline_file_count == 48
    assert bundle.observed_baseline_file_count >= bundle.baseline_file_count
    assert bundle.baseline_verified is False
    assert baseline["baseline_sha256"] == bundle.baseline_sha256
    assert baseline["metrics"]["external_false_verified_rate"] == 0.0


def test_blind_bundle_rejects_tampered_dataset(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    manifest_path = _write_bundle(tmp_path, repository_root)
    (tmp_path / "cases.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="dataset digest mismatch"):
        load_blind_bundle(str(manifest_path), repository_root=repository_root)


def test_blind_bundle_rejects_tampered_oracle(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    manifest_path = _write_bundle(tmp_path, repository_root)
    oracle = tmp_path / "oracles" / "B001" / "oracle.py"
    oracle.write_text("def test_tampered():\n    assert True\n", encoding="utf-8")

    with pytest.raises(ValueError, match="oracle digest mismatch"):
        load_blind_bundle(str(manifest_path), repository_root=repository_root)


def test_blind_bundle_rejects_changed_forge_baseline(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    manifest_path = _write_bundle(tmp_path, repository_root)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["forge_baseline"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="baseline digest mismatch"):
        load_blind_bundle(str(manifest_path), repository_root=repository_root)


def test_blind_bundle_rejects_unknown_baseline_digest_mode(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    manifest_path = _write_bundle(tmp_path, repository_root)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["forge_baseline"]["digest_mode"] = "platform_bytes_v0"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported Forge baseline digest mode"):
        load_blind_bundle(str(manifest_path), repository_root=repository_root)


def test_blind_runner_exposes_requirement_only_and_persists_seal(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    manifest_path = _write_bundle(tmp_path, repository_root)
    bundle = load_blind_bundle(str(manifest_path), repository_root=repository_root)
    observed_requirements = []

    def run_case(requirement: str) -> ForgeResult:
        observed_requirements.append(requirement)
        assert "test_identity" not in requirement
        return ForgeResult(
            route=ForgeRoute.TERMINAL_VERIFIED,
            terminal_status=TERMINAL_VERIFIED,
            summary="verified",
            artifact_path="package",
            run_metrics=ForgeRunMetrics(
                validation_attempts=1,
                verified_at_1=True,
            ),
        )

    report = run_blind_bundle(
        bundle,
        run_case=run_case,
        run_oracle=lambda oracle, package: OracleResult(
            executed=True,
            passed=True,
            exit_code=0,
        ),
    )
    output_path = persist_blind_report(report, str(tmp_path / "reports"))
    rendered = render_blind_report(report, output_path)
    payload = json.loads(Path(output_path).read_text(encoding="utf-8"))

    assert observed_requirements == [bundle.cases[0].requirement]
    assert report.summary.passed_cases == 1
    assert payload["baseline_verified"] is True
    assert payload["manifest_sha256"] == bundle.manifest_sha256
    assert "Baseline verified: true" in rendered
    assert "External Verified@1: 1.000" in rendered
    assert "External success after repair: n/a" in rendered
    assert "P95 case runtime:" in rendered
    assert "Estimated model cost: $0.00000000" in rendered


def test_post_fix_replay_is_explicit_and_cannot_be_reported_as_sealed_baseline(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    manifest_path = _write_bundle(tmp_path, repository_root)
    sealed = load_blind_bundle(str(manifest_path), repository_root=repository_root)
    changed = replace(
        sealed,
        baseline_verified=False,
        observed_baseline_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="exact sealed Forge baseline"):
        run_blind_bundle(changed, run_case=lambda requirement: None)

    report = run_blind_bundle(
        changed,
        run_case=lambda requirement: ForgeResult(
            route=ForgeRoute.TERMINAL_VERIFIED,
            terminal_status=TERMINAL_VERIFIED,
            summary="verified",
            artifact_path="package",
            run_metrics=ForgeRunMetrics(validation_attempts=1, verified_at_1=True),
        ),
        run_oracle=lambda oracle, package: OracleResult(
            executed=True,
            passed=True,
            exit_code=0,
        ),
        post_fix_replay=True,
    )

    assert report.execution_kind == "post_fix_replay"
    assert report.baseline_verified is False
    rendered = render_blind_report(report, "report.json")
    assert "Execution kind: post_fix_replay" in rendered
    assert "Baseline verified: false" in rendered


def test_post_fix_replay_can_select_frozen_cases_without_mutating_bundle(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    manifest_path = _write_bundle(tmp_path, repository_root)
    sealed = load_blind_bundle(str(manifest_path), repository_root=repository_root)
    second = replace(
        sealed.cases[0],
        case_id="B002",
        requirement="Build a second deterministic component.",
        expected_terminal_status="validation_failed",
        oracle=None,
    )
    changed = replace(
        sealed,
        baseline_verified=False,
        observed_baseline_sha256="0" * 64,
        cases=[sealed.cases[0], second],
    )
    observed = []

    report = run_blind_bundle(
        changed,
        run_case=lambda requirement: (
            observed.append(requirement)
            or ForgeResult(
                route=ForgeRoute.TERMINAL_VALIDATION_FAILED,
                terminal_status="validation_failed",
                summary="validation failed",
                artifact_path="artifact",
            )
        ),
        post_fix_replay=True,
        case_ids=["B002"],
    )

    assert observed == [second.requirement]
    assert report.summary.total_cases == 1
    assert report.summary.case_results[0].case_id == "B002"
    assert [case.case_id for case in changed.cases] == ["B001", "B002"]


def test_blind_case_selection_rejects_unknown_ids(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    manifest_path = _write_bundle(tmp_path, repository_root)
    bundle = load_blind_bundle(str(manifest_path), repository_root=repository_root)

    with pytest.raises(ValueError, match="Unknown blind benchmark case ids: B999"):
        run_blind_bundle(
            bundle,
            run_case=lambda requirement: None,
            case_ids=["B999"],
        )
