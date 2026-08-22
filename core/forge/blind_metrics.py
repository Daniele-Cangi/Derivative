import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.forge.benchmark import (
    SUPPORTED_TERMINAL_STATUSES,
    TERMINAL_INFEASIBLE_PROVEN,
    TERMINAL_VERIFIED,
)
from core.forge.blind_benchmark import BlindBenchmarkBundle, load_blind_bundle
from core.forge.blind_requirement import requirement_preflight_error


_SUPPORTED_EXECUTION_KINDS = frozenset({"sealed_baseline", "post_fix_replay"})


def derive_adjudicated_metrics_from_files(
    *,
    manifest_path: str,
    baseline_report_path: str,
    adjudication_path: str,
    output_path: str,
    repository_root: str | Path = ".",
    created_at: str | None = None,
    execution_kind: str = "sealed_baseline",
    receipt_id: str | None = None,
) -> dict[str, Any]:
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Adjudicated metrics receipt already exists: {destination}"
        )
    baseline_path = Path(baseline_report_path).resolve()
    adjudication_receipt_path = Path(adjudication_path).resolve()
    baseline_bytes, baseline_report = _read_json_object(
        baseline_path,
        "baseline report",
    )
    adjudication_bytes, adjudication_receipt = _read_json_object(
        adjudication_receipt_path,
        "adjudication receipt",
    )
    bundle = load_blind_bundle(
        manifest_path,
        repository_root=repository_root,
        verify_baseline=False,
    )
    receipt = derive_adjudicated_metrics(
        bundle=bundle,
        baseline_report=baseline_report,
        adjudication_receipt=adjudication_receipt,
        baseline_report_sha256=hashlib.sha256(baseline_bytes).hexdigest(),
        adjudication_sha256=hashlib.sha256(adjudication_bytes).hexdigest(),
        created_at=created_at,
        execution_kind=execution_kind,
        receipt_id=receipt_id,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return receipt


def derive_adjudicated_metrics(
    *,
    bundle: BlindBenchmarkBundle,
    baseline_report: dict[str, Any],
    adjudication_receipt: dict[str, Any],
    baseline_report_sha256: str,
    adjudication_sha256: str,
    created_at: str | None = None,
    execution_kind: str = "sealed_baseline",
    receipt_id: str | None = None,
) -> dict[str, Any]:
    if execution_kind not in _SUPPORTED_EXECUTION_KINDS:
        raise ValueError(
            f"Unsupported adjudicated metrics execution kind: {execution_kind}."
        )
    if receipt_id is not None and not receipt_id.strip():
        raise ValueError("Adjudicated metrics receipt_id cannot be blank.")
    _validate_source_links(
        bundle,
        baseline_report,
        adjudication_receipt,
        execution_kind,
    )
    cases = {case.case_id: case for case in bundle.cases}
    results = _indexed_objects(
        _required_mapping(baseline_report, "summary").get("case_results"),
        "baseline case_results",
    )
    consensus = _indexed_objects(
        adjudication_receipt.get("consensus"),
        "adjudication consensus",
    )
    if set(results) != set(cases) or set(consensus) != set(cases):
        raise ValueError(
            "Bundle, baseline report, and adjudication receipt case ids must match exactly."
        )

    decisions: list[dict[str, Any]] = []
    included: list[tuple[dict[str, Any], dict[str, Any], Any]] = []
    for case_id, case in cases.items():
        result = results[case_id]
        review = consensus[case_id]
        decision = _case_decision(case, result, review)
        decisions.append(decision)
        if decision["status_metric_included"]:
            included.append((decision, result, case))

    expected_verified = [
        item for item in included if item[0]["effective_expected_status"] == TERMINAL_VERIFIED
    ]
    externally_evaluable = [
        item
        for item in expected_verified
        if item[2].oracle is not None and item[2].public_contract is not None
    ]
    externally_accepted = [
        item
        for item in externally_evaluable
        if item[0]["observed_terminal_status"] == TERMINAL_VERIFIED
        and _oracle_passed(item[1])
    ]
    verified_at_1 = [
        item
        for item in externally_evaluable
        if bool(item[1].get("verified_at_1")) and _oracle_passed(item[1])
    ]
    repair_eligible = [
        item for item in externally_evaluable if not bool(item[1].get("verified_at_1"))
    ]
    repaired_successes = [
        item
        for item in repair_eligible
        if item[0]["observed_terminal_status"] == TERMINAL_VERIFIED
        and bool(item[1].get("success_after_repair"))
        and _oracle_passed(item[1])
    ]
    observed_verified = [
        item for item in included if item[0]["observed_terminal_status"] == TERMINAL_VERIFIED
    ]
    false_verified_evaluable = [
        item
        for item in observed_verified
        if item[0]["effective_expected_status"] != TERMINAL_VERIFIED
        or (item[2].oracle is not None and item[2].public_contract is not None)
    ]
    false_verified = [
        item
        for item in false_verified_evaluable
        if item[0]["effective_expected_status"] != TERMINAL_VERIFIED
        or not _oracle_passed(item[1])
    ]
    expected_infeasible = [
        item
        for item in included
        if item[0]["effective_expected_status"] == TERMINAL_INFEASIBLE_PROVEN
    ]
    correct_infeasible = [
        item
        for item in expected_infeasible
        if item[0]["observed_terminal_status"] == TERMINAL_INFEASIBLE_PROVEN
    ]
    executed_oracles = [
        item
        for item in externally_evaluable
        if isinstance(item[1].get("oracle_result"), dict)
        and item[1]["oracle_result"].get("executed") is True
    ]

    resources = _resource_metrics(included, len(externally_accepted))
    summary = {
        "raw_total_cases": len(cases),
        "definitive_status_cases": len(included),
        "excluded_cases": len(cases) - len(included),
        "unresolved_cases": sum(
            decision["exclusion_reason"] == "reviewer_disagreement"
            for decision in decisions
        ),
        "invalid_adjudication_cases": sum(
            decision["exclusion_reason"] == "deterministic_contract_conflict"
            for decision in decisions
        ),
        "invalid_oracle_cases": sum(
            decision["exclusion_reason"] == "invalid_oracle"
            for decision in decisions
        ),
        "corrected_label_cases": sum(
            decision["verdict"] == "label_invalid" for decision in decisions
        ),
        "verified_cases_without_definitive_external_contract": (
            len(expected_verified) - len(externally_evaluable)
        ),
        "observed_verified_without_definitive_external_judge": (
            len(observed_verified) - len(false_verified_evaluable)
        ),
        "externally_accepted_artifacts": len(externally_accepted),
    }
    metrics = {
        "status_accuracy": _fraction(
            sum(decision["status_matched"] for decision, _, _ in included),
            len(included),
        ),
        "external_verified_at_1": _fraction(
            len(verified_at_1),
            len(externally_evaluable),
        ),
        "external_success_after_repair": _fraction(
            len(repaired_successes),
            len(repair_eligible),
        ),
        "oracle_pass_rate": _fraction(
            sum(_oracle_passed(item[1]) for item in executed_oracles),
            len(executed_oracles),
        ),
        "external_false_verified_rate": _fraction(
            len(false_verified),
            len(false_verified_evaluable),
        ),
        "infeasible_detection_rate": _fraction(
            len(correct_infeasible),
            len(expected_infeasible),
        ),
    }
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    resolved_receipt_id = (
        receipt_id or f"{bundle.bundle_id}-adjudicated-definitive-metrics"
    )
    return {
        "schema_version": 1,
        "receipt_id": resolved_receipt_id,
        "created_at": timestamp,
        "execution_kind": execution_kind,
        "terminal_status": "adjudicated_metrics_derived",
        "sources": {
            "bundle_id": bundle.bundle_id,
            "manifest_sha256": bundle.manifest_sha256,
            "dataset_sha256": bundle.dataset_sha256,
            "forge_baseline_sha256": bundle.baseline_sha256,
            "baseline_report_sha256": baseline_report_sha256,
            "adjudication_receipt_sha256": adjudication_sha256,
        },
        "policy": {
            "raw_benchmark_metrics_modified": False,
            "unresolved_excluded": True,
            "deterministic_contract_conflicts_excluded": True,
            "consensus_label_corrections_applied": True,
            "external_metrics_require_frozen_oracle": True,
            "external_metrics_require_typed_public_contract": True,
            "undefined_rates_are_null": True,
        },
        "summary": summary,
        "metrics": metrics,
        "resource_metrics": resources,
        "case_decisions": decisions,
    }


def _case_decision(case: Any, result: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    claimed = case.expected_terminal_status
    if str(result.get("expected_terminal_status", "")) != claimed:
        raise ValueError(f"Baseline result changed the frozen status for {case.case_id}.")
    if str(review.get("claimed_status", "")) != claimed:
        raise ValueError(f"Adjudication changed the frozen status for {case.case_id}.")
    verdict = str(review.get("verdict", ""))
    adjudicated = review.get("adjudicated_status")
    observed = str(result.get("observed_terminal_status", ""))
    if observed not in SUPPORTED_TERMINAL_STATUSES and observed != "oracle_invalid":
        raise ValueError(
            f"Baseline result has an unsupported observed status for {case.case_id}."
        )
    if verdict == "unresolved":
        if adjudicated is not None:
            raise ValueError(f"Unresolved adjudication has a status for {case.case_id}.")
        return _excluded_decision(
            case.case_id,
            claimed,
            verdict,
            observed,
            "reviewer_disagreement",
        )
    if verdict not in {"label_valid", "label_invalid"}:
        raise ValueError(f"Unsupported adjudication verdict for {case.case_id}: {verdict!r}.")
    if adjudicated not in SUPPORTED_TERMINAL_STATUSES:
        raise ValueError(f"Resolved adjudication omitted a status for {case.case_id}.")
    if (verdict == "label_valid") != (adjudicated == claimed):
        raise ValueError(f"Adjudication verdict and status disagree for {case.case_id}.")

    preflight_error = requirement_preflight_error(case.requirement, adjudicated)
    if preflight_error is not None:
        decision = _excluded_decision(
            case.case_id,
            claimed,
            verdict,
            observed,
            "deterministic_contract_conflict",
        )
        decision["adjudicated_status"] = adjudicated
        decision["deterministic_evidence"] = preflight_error
        return decision
    oracle_result = result.get("oracle_result")
    if isinstance(oracle_result, dict) and oracle_result.get("valid") is False:
        decision = _excluded_decision(
            case.case_id,
            claimed,
            verdict,
            observed,
            "invalid_oracle",
        )
        decision["adjudicated_status"] = adjudicated
        return decision
    if observed == "oracle_invalid":
        raise ValueError(
            f"Baseline result reports oracle_invalid without invalid oracle evidence for "
            f"{case.case_id}."
        )
    return {
        "case_id": case.case_id,
        "claimed_status": claimed,
        "verdict": verdict,
        "adjudicated_status": adjudicated,
        "effective_expected_status": adjudicated,
        "observed_terminal_status": observed,
        "status_metric_included": True,
        "status_matched": observed == adjudicated,
        "exclusion_reason": None,
        "external_contract_ready": (
            adjudicated == TERMINAL_VERIFIED
            and case.oracle is not None
            and case.public_contract is not None
        ),
    }


def _excluded_decision(
    case_id: str,
    claimed: str,
    verdict: str,
    observed: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "claimed_status": claimed,
        "verdict": verdict,
        "adjudicated_status": None,
        "effective_expected_status": None,
        "observed_terminal_status": observed,
        "status_metric_included": False,
        "status_matched": False,
        "exclusion_reason": reason,
        "external_contract_ready": False,
    }


def _resource_metrics(
    included: list[tuple[dict[str, Any], dict[str, Any], Any]],
    accepted_artifacts: int,
) -> dict[str, Any]:
    results = [result for _, result, _ in included]
    runtimes = sorted(float(result.get("execution_time_seconds", 0.0)) for result in results)
    costs = [result.get("estimated_model_cost_usd") for result in results]
    total_cost = (
        round(sum(float(cost) for cost in costs), 8)
        if costs and all(cost is not None for cost in costs)
        else None
    )
    return {
        "included_case_runtime_seconds": round(sum(runtimes), 8),
        "median_case_runtime_seconds": statistics.median(runtimes) if runtimes else None,
        "p95_case_runtime_seconds": _nearest_rank(runtimes, 0.95),
        "total_repairs": sum(int(result.get("repair_count", 0)) for result in results),
        "total_model_requests": sum(
            int(result.get("model_request_count", 0)) for result in results
        ),
        "total_model_tokens": sum(int(result.get("model_total_tokens", 0)) for result in results),
        "total_estimated_model_cost_usd": total_cost,
        "cost_per_externally_accepted_artifact_usd": (
            round(total_cost / accepted_artifacts, 8)
            if total_cost is not None and accepted_artifacts
            else None
        ),
    }


def _validate_source_links(
    bundle: BlindBenchmarkBundle,
    baseline: dict[str, Any],
    adjudication: dict[str, Any],
    execution_kind: str,
) -> None:
    if baseline.get("schema_version") != bundle.schema_version:
        raise ValueError("Baseline report schema_version does not match the frozen bundle.")
    if adjudication.get("schema_version") != 1:
        raise ValueError("Unsupported adjudication receipt schema_version.")
    if baseline.get("execution_kind") != execution_kind:
        raise ValueError(
            "Baseline report execution_kind does not match the metrics receipt."
        )
    baseline_matches = (
        baseline.get("observed_baseline_sha256") == bundle.baseline_sha256
        and baseline.get("observed_baseline_file_count")
        == bundle.baseline_file_count
    )
    if baseline.get("baseline_verified") is not baseline_matches:
        raise ValueError(
            "Baseline report verification flag contradicts its observed baseline."
        )
    if execution_kind == "sealed_baseline" and not baseline_matches:
        raise ValueError("Adjudicated baseline metrics require a sealed baseline report.")
    adjudicated_bundle = _required_mapping(adjudication, "bundle")
    expected = {
        "bundle_id": bundle.bundle_id,
        "manifest_sha256": bundle.manifest_sha256,
        "dataset_sha256": bundle.dataset_sha256,
        "baseline_sha256": bundle.baseline_sha256,
    }
    for key, value in expected.items():
        baseline_key = "baseline_sha256" if key == "baseline_sha256" else key
        if baseline.get(baseline_key) != value:
            raise ValueError(f"Baseline report {key} does not match the frozen bundle.")
        if adjudicated_bundle.get(key) != value:
            raise ValueError(f"Adjudication receipt {key} does not match the frozen bundle.")
    method = _required_mapping(adjudication, "method")
    if method.get("raw_benchmark_metrics_modified") is not False:
        raise ValueError("Adjudication receipt must preserve raw benchmark metrics.")


def _indexed_objects(payload: object, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{label} must be a non-empty list.")
    indexed: dict[str, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"{label} contains a non-object item.")
        case_id = str(item.get("case_id", "")).strip()
        if not case_id or case_id in indexed:
            raise ValueError(f"{label} contains a missing or duplicate case_id.")
        indexed[case_id] = item
    return indexed


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Required object is missing: {key}.")
    return value


def _read_json_object(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        content = path.read_bytes()
        payload = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {label} as UTF-8 JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return content, payload


def _oracle_passed(result: dict[str, Any]) -> bool:
    oracle = result.get("oracle_result")
    return isinstance(oracle, dict) and oracle.get("passed") is True


def _fraction(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    index = max(0, min(len(values) - 1, int(len(values) * percentile + 0.999999) - 1))
    return values[index]
