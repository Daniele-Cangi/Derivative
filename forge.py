import json
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import typer
from dotenv import load_dotenv

from core.forge.coder_stage import CoderStage
from core.forge.candidate_compiler import SubstrateCandidateCompiler
from core.forge.contracts import (
    BuildSpec,
    CodeArtifact,
    FailureCategory,
    FeasiblePlan,
    ForgeResult,
    ForgeRoute,
    ForgeRunMetrics,
    InfeasibilityCertificate,
    PackagedArtifact,
    ValidationArtifact,
)
from core.forge.execution import (
    DEFAULT_SANDBOX_IMAGE,
    DOCKER_BACKEND,
    create_process_executor,
)
from core.forge.packaging_stage import PackagingStage
from core.forge.planner_stage import PlannerStage
from core.forge.repair import RepairPolicy
from core.forge.repair_backend import SubstrateRepairBackend
from core.forge.requirement_compiler import RequirementCompiler
from core.forge.telemetry import track_model_usage
from core.forge.validator_stage import ValidatorStage

load_dotenv(Path(__file__).resolve().with_name(".env"))


TERMINAL_VERIFIED = "verified"
TERMINAL_INFEASIBLE_PROVEN = "infeasible_proven"
TERMINAL_VALIDATION_FAILED = "validation_failed"

app = typer.Typer(
    help="Forge - execution-grounded software build orchestrator.",
    pretty_exceptions_show_locals=False,
)


def _capture_run_model_usage(function: Callable[..., ForgeResult]) -> Callable[..., ForgeResult]:
    @wraps(function)
    def wrapped(*args: object, **kwargs: object) -> ForgeResult:
        with track_model_usage() as usage:
            result = function(*args, **kwargs)
        estimated_cost, pricing_source = usage.estimated_cost()
        result.run_metrics.model_request_count = usage.request_count
        result.run_metrics.model_input_tokens = usage.input_tokens
        result.run_metrics.model_output_tokens = usage.output_tokens
        result.run_metrics.model_total_tokens = usage.total_tokens
        result.run_metrics.estimated_model_cost_usd = estimated_cost
        result.run_metrics.model_cost_pricing_source = pricing_source
        return result

    return wrapped


@_capture_run_model_usage
def run_forge(
    requirement: str,
    execution_mode: str = "local-only",
    output_root: str = "generated_artifacts/forge_runs",
    packaging_output_root: str = "generated_artifacts/forge_packages",
    max_planner_attempts: int = 1,
    max_coder_attempts: int = 3,
    execution_backend: str = DOCKER_BACKEND,
    sandbox_image: str = DEFAULT_SANDBOX_IMAGE,
    requirement_compiler: RequirementCompiler | None = None,
    planner_stage: PlannerStage | None = None,
    coder_stage: CoderStage | None = None,
    validator_stage: ValidatorStage | None = None,
    packaging_stage: PackagingStage | None = None,
    repair_policy: RepairPolicy | None = None,
) -> ForgeResult:
    started = time.perf_counter()
    normalized_planner_attempts = max(1, int(max_planner_attempts))
    normalized_coder_attempts = max(1, int(max_coder_attempts))
    compiler = requirement_compiler or RequirementCompiler()
    planner = planner_stage or PlannerStage(execution_mode=execution_mode)
    process_executor = create_process_executor(
        execution_backend,
        image=sandbox_image,
    )
    if coder_stage is not None:
        coder = coder_stage
    else:
        repair_backend = None
        candidate_compiler = None
        if execution_mode != "local-only":
            repair_backend = SubstrateRepairBackend(
                execution_mode=execution_mode,
                substrate=getattr(planner, "substrate", None),
                kernel=getattr(planner, "kernel", None),
                executor=process_executor,
            )
            candidate_compiler = SubstrateCandidateCompiler(
                execution_mode=execution_mode,
                substrate=getattr(planner, "substrate", None),
                kernel=getattr(planner, "kernel", None),
                executor=process_executor,
            )
        coder = CoderStage(
            repair_backend=repair_backend,
            candidate_compiler=candidate_compiler,
        )
    validator = validator_stage or ValidatorStage(
        executor=process_executor,
        require_isolation=True,
    )
    packager = packaging_stage or PackagingStage(output_root=packaging_output_root)
    repairs = repair_policy or RepairPolicy()

    build_spec = compiler.compile(requirement)
    attempt_trace: list[dict[str, object]] = []

    latest_plan: FeasiblePlan | InfeasibilityCertificate | None = None
    latest_code_artifact: CodeArtifact | None = None
    latest_validation: ValidationArtifact | None = None

    planner_attempt = 0
    while planner_attempt < normalized_planner_attempts:
        planner_attempt += 1
        planning_output = planner.plan(build_spec)
        latest_plan = planning_output

        if isinstance(planning_output, InfeasibilityCertificate):
            run_root = _persist_run_artifacts(
                output_root=output_root,
                build_spec=build_spec,
                terminal_status=TERMINAL_INFEASIBLE_PROVEN,
                plan_output=planning_output,
                code_artifact=None,
                validation=None,
                packaged_artifact=None,
                run_metadata={
                    "max_planner_attempts": normalized_planner_attempts,
                    "max_coder_attempts": normalized_coder_attempts,
                    "planner_attempts_used": planner_attempt,
                    "coder_attempts_used": 0,
                    "attempt_trace": attempt_trace,
                },
            )
            elapsed = time.perf_counter() - started
            return ForgeResult(
                route=ForgeRoute.TERMINAL_INFEASIBLE,
                terminal_status=TERMINAL_INFEASIBLE_PROVEN,
                summary=(
                    "Planning terminated with an infeasibility certificate grounded in execution evidence. "
                    "The stated constraints cannot be satisfied simultaneously."
                ),
                validation=None,
                packaged_artifact=None,
                infeasibility_certificate=planning_output,
                artifact_path=run_root,
                execution_time_seconds=elapsed,
                run_metrics=_build_run_metrics(
                    planner_attempts=planner_attempt,
                    attempt_trace=attempt_trace,
                    terminal_status=TERMINAL_INFEASIBLE_PROVEN,
                ),
            )

        if not isinstance(planning_output, FeasiblePlan):
            raise TypeError(f"PlannerStage returned unsupported output type: {type(planning_output)!r}")

        route_to_planner = False
        coder_attempt = 0
        code_artifact: CodeArtifact | None = None
        previous_validation: ValidationArtifact | None = None
        pending_directive = None
        while coder_attempt < normalized_coder_attempts:
            coder_attempt += 1
            repair_trace: dict[str, object] | None = None
            force_terminal = False

            if pending_directive is None:
                code_artifact = coder.generate(planning_output)
                validation = validator.validate(code_artifact, planning_output, build_spec)
            else:
                repair_method = getattr(coder, "repair", None)
                if not callable(repair_method) or code_artifact is None or previous_validation is None:
                    validation = previous_validation or ValidationArtifact(passed=False)
                    _mark_repair_terminal(
                        validation,
                        "repair_not_supported",
                        "CoderStage does not expose a grounded repair operation.",
                        {"repair_id": pending_directive.repair_id},
                    )
                    force_terminal = True
                else:
                    repair_result = repair_method(
                        planning_output,
                        code_artifact,
                        previous_validation,
                        pending_directive,
                    )
                    repair_trace = {
                        "repair_id": pending_directive.repair_id,
                        "operations": list(pending_directive.operations),
                        "target_paths": list(pending_directive.target_paths),
                        "changed": repair_result.changed,
                        "changed_paths": list(repair_result.changed_paths),
                        "previous_digest": repair_result.previous_digest,
                        "repaired_digest": repair_result.repaired_digest,
                        "backend_name": repair_result.backend_name,
                        "backend_evidence": repair_result.backend_evidence,
                        "stop_reason": repair_result.stop_reason,
                    }
                    if not repair_result.changed:
                        validation = previous_validation
                        repair_signature = "repair_no_change"
                        if repair_result.backend_evidence.get("error_type"):
                            repair_signature = "repair_backend_failure"
                        elif (
                            repair_result.backend_name != "canonical"
                            and repair_result.backend_evidence.get("available") is False
                        ):
                            repair_signature = "repair_backend_unavailable"
                        _mark_repair_terminal(
                            validation,
                            repair_signature,
                            repair_result.stop_reason
                            or "Failure-guided repair produced no source, test, manifest, or provenance changes.",
                            repair_trace,
                        )
                        force_terminal = True
                    else:
                        code_artifact = repair_result.artifact
                        validation = validator.validate(code_artifact, planning_output, build_spec)

            if repair_trace is not None and previous_validation is not None:
                previous_signatures = set(previous_validation.failure_signatures)
                current_signatures = set(validation.failure_signatures)
                resolved_signatures = sorted(previous_signatures - current_signatures)
                introduced_signatures = sorted(current_signatures - previous_signatures)
                validation_delta = {
                    "resolved_signatures": resolved_signatures,
                    "remaining_signatures": sorted(previous_signatures & current_signatures),
                    "introduced_signatures": introduced_signatures,
                    "failure_count_before": len(previous_validation.failures),
                    "failure_count_after": len(validation.failures),
                    "passed_after_repair": validation.passed,
                    "evidence_improved": bool(resolved_signatures) or validation.passed,
                }
                repair_trace["validation_delta"] = validation_delta
                validation.evidence["repair_effectiveness"] = dict(validation_delta)

            if code_artifact is None:
                raise RuntimeError("Coder attempt did not produce a CodeArtifact.")
            latest_code_artifact = code_artifact
            latest_validation = validation
            retry_route = (
                ForgeRoute.TERMINAL_VALIDATION_FAILED
                if force_terminal
                else _retry_route_for_validation(validation)
            )
            attempt_entry: dict[str, object] = {
                    "planner_attempt": planner_attempt,
                    "coder_attempt": coder_attempt,
                    "artifact_id": code_artifact.artifact_id,
                    "artifact_revision": code_artifact.revision,
                    "validation_passed": validation.passed,
                    "retry_route": retry_route.value,
                    "failure_category": (
                        validation.failure_category.value
                        if validation.failure_category is not None
                        else None
                    ),
                    "failure_signatures": list(validation.failure_signatures),
                }
            if repair_trace is not None:
                attempt_entry["repair"] = repair_trace
            attempt_trace.append(attempt_entry)

            if validation.passed:
                packaged_artifact = packager.package(build_spec, planning_output, code_artifact, validation)
                _persist_run_artifacts(
                    output_root=output_root,
                    build_spec=build_spec,
                    terminal_status=TERMINAL_VERIFIED,
                    plan_output=planning_output,
                    code_artifact=code_artifact,
                    validation=validation,
                    packaged_artifact=packaged_artifact,
                    run_metadata={
                        "max_planner_attempts": normalized_planner_attempts,
                        "max_coder_attempts": normalized_coder_attempts,
                        "planner_attempts_used": planner_attempt,
                        "coder_attempts_used": coder_attempt,
                        "attempt_trace": attempt_trace,
                    },
                )
                elapsed = time.perf_counter() - started
                return ForgeResult(
                    route=ForgeRoute.TERMINAL_VERIFIED,
                    terminal_status=TERMINAL_VERIFIED,
                    summary=(
                        "Requirement compiled into a feasible build plan. Code was generated, validated across "
                        "syntax/import/run, obligation and acceptance coverage, and adversarial checks, then "
                        "packaged successfully."
                    ),
                    validation=validation,
                    packaged_artifact=packaged_artifact,
                    infeasibility_certificate=None,
                    artifact_path=packaged_artifact.package_root,
                    execution_time_seconds=elapsed,
                    run_metrics=_build_run_metrics(
                        planner_attempts=planner_attempt,
                        attempt_trace=attempt_trace,
                        terminal_status=TERMINAL_VERIFIED,
                    ),
                )

            if retry_route == ForgeRoute.TO_CODER and coder_attempt < normalized_coder_attempts:
                pending_directive = repairs.compile(
                    validation=validation,
                    plan=planning_output,
                    artifact=code_artifact,
                    attempt=coder_attempt + 1,
                    route=retry_route,
                )
                attempt_entry["repair_directive"] = _to_jsonable(pending_directive)
                if pending_directive.repairable:
                    previous_validation = validation
                    continue
                _mark_repair_terminal(
                    validation,
                    "repair_not_applicable",
                    pending_directive.stop_reason,
                    {"repair_id": pending_directive.repair_id},
                )
                attempt_entry["failure_signatures"] = list(validation.failure_signatures)
                attempt_entry["repair_terminal"] = validation.evidence.get("repair_terminal", {})
            if retry_route == ForgeRoute.TO_PLANNER and planner_attempt < normalized_planner_attempts:
                route_to_planner = True
                break

            run_root = _persist_run_artifacts(
                output_root=output_root,
                build_spec=build_spec,
                terminal_status=TERMINAL_VALIDATION_FAILED,
                plan_output=planning_output,
                code_artifact=code_artifact,
                validation=validation,
                packaged_artifact=None,
                run_metadata={
                    "max_planner_attempts": normalized_planner_attempts,
                    "max_coder_attempts": normalized_coder_attempts,
                    "planner_attempts_used": planner_attempt,
                    "coder_attempts_used": coder_attempt,
                    "attempt_trace": attempt_trace,
                },
            )
            elapsed = time.perf_counter() - started
            return ForgeResult(
                route=ForgeRoute.TERMINAL_VALIDATION_FAILED,
                terminal_status=TERMINAL_VALIDATION_FAILED,
                summary="Planning and code generation completed, but validation did not pass. Packaging was not attempted.",
                validation=validation,
                packaged_artifact=None,
                infeasibility_certificate=None,
                artifact_path=run_root,
                execution_time_seconds=elapsed,
                run_metrics=_build_run_metrics(
                    planner_attempts=planner_attempt,
                    attempt_trace=attempt_trace,
                    terminal_status=TERMINAL_VALIDATION_FAILED,
                ),
            )

        if route_to_planner:
            continue
        break

    if not isinstance(latest_plan, FeasiblePlan):
        raise RuntimeError("Forge orchestration exhausted attempts without a feasible plan.")
    if latest_code_artifact is None or latest_validation is None:
        raise RuntimeError("Forge orchestration exhausted attempts without a validation artifact.")

    run_root = _persist_run_artifacts(
        output_root=output_root,
        build_spec=build_spec,
        terminal_status=TERMINAL_VALIDATION_FAILED,
        plan_output=latest_plan,
        code_artifact=latest_code_artifact,
        validation=latest_validation,
        packaged_artifact=None,
        run_metadata={
            "max_planner_attempts": normalized_planner_attempts,
            "max_coder_attempts": normalized_coder_attempts,
            "planner_attempts_used": planner_attempt,
            "coder_attempts_used": normalized_coder_attempts,
            "attempt_trace": attempt_trace,
        },
    )
    elapsed = time.perf_counter() - started
    return ForgeResult(
        route=ForgeRoute.TERMINAL_VALIDATION_FAILED,
        terminal_status=TERMINAL_VALIDATION_FAILED,
        summary="Planning and code generation completed, but validation did not pass. Packaging was not attempted.",
        validation=latest_validation,
        packaged_artifact=None,
        infeasibility_certificate=None,
        artifact_path=run_root,
        execution_time_seconds=elapsed,
        run_metrics=_build_run_metrics(
            planner_attempts=planner_attempt,
            attempt_trace=attempt_trace,
            terminal_status=TERMINAL_VALIDATION_FAILED,
        ),
    )


def render_cli_output(result: ForgeResult) -> str:
    lines = ["Forge", f"Status: {result.terminal_status}", ""]
    if result.terminal_status == TERMINAL_VERIFIED:
        lines.append(result.summary)
        lines.append("")
        lines.append(f"Packaged artifact: {result.artifact_path}")
        lines.append(f"Execution time: {result.execution_time_seconds:.2f}s")
        return "\n".join(lines)

    if result.terminal_status == TERMINAL_INFEASIBLE_PROVEN:
        lines.append(result.summary)
        lines.append("")
        lines.append(f"Certificate artifacts: {result.artifact_path}")
        lines.append(f"Execution time: {result.execution_time_seconds:.2f}s")
        return "\n".join(lines)

    if result.terminal_status == TERMINAL_VALIDATION_FAILED:
        lines.append(result.summary)
        lines.append(f"Validation failures: {_concise_validation_failures(result.validation)}")
        lines.append("")
        lines.append(f"Artifacts: {result.artifact_path}")
        lines.append(f"Execution time: {result.execution_time_seconds:.2f}s")
        return "\n".join(lines)

    lines.append(result.summary)
    lines.append("")
    lines.append(f"Artifacts: {result.artifact_path}")
    lines.append(f"Execution time: {result.execution_time_seconds:.2f}s")
    return "\n".join(lines)


@app.command()
def main(
    requirement: str = typer.Argument(..., help="Natural-language software requirement."),
    mode: str = typer.Option("local-only", "--mode", help="Execution mode for planner substrate."),
    output_root: str = typer.Option(
        "generated_artifacts/forge_runs",
        "--output-root",
        help="Directory for run-level typed artifacts.",
    ),
    packaging_root: str = typer.Option(
        "generated_artifacts/forge_packages",
        "--packaging-root",
        help="Directory for verified packaged artifacts.",
    ),
    max_planner_attempts: int = typer.Option(
        1,
        "--max-planner-attempts",
        min=1,
        help="Maximum planner attempts before terminal failure.",
    ),
    max_coder_attempts: int = typer.Option(
        3,
        "--max-coder-attempts",
        min=1,
        help="Maximum coder attempts per planner attempt.",
    ),
    execution_backend: str = typer.Option(
        DOCKER_BACKEND,
        "--execution-backend",
        help="Generated-code execution backend. Production verification requires docker.",
    ),
    sandbox_image: str = typer.Option(
        DEFAULT_SANDBOX_IMAGE,
        "--sandbox-image",
        help="Docker image used for isolated generated-code execution.",
    ),
) -> None:
    result = run_forge(
        requirement=requirement,
        execution_mode=mode,
        output_root=output_root,
        packaging_output_root=packaging_root,
        max_planner_attempts=max_planner_attempts,
        max_coder_attempts=max_coder_attempts,
        execution_backend=execution_backend,
        sandbox_image=sandbox_image,
    )
    typer.echo(render_cli_output(result))


def _persist_run_artifacts(
    output_root: str,
    build_spec: BuildSpec,
    terminal_status: str,
    plan_output: FeasiblePlan | InfeasibilityCertificate,
    code_artifact: CodeArtifact | None,
    validation: ValidationArtifact | None,
    packaged_artifact: PackagedArtifact | None,
    run_metadata: dict[str, Any] | None = None,
) -> str:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = root / f"{timestamp}_{build_spec.build_id}_{terminal_status}"
    run_root.mkdir(parents=True, exist_ok=True)

    _write_json(run_root / "build_spec.json", build_spec)
    if isinstance(plan_output, InfeasibilityCertificate):
        _write_json(run_root / "infeasibility_certificate.json", plan_output)
    else:
        _write_json(run_root / "feasible_plan.json", plan_output)
    if code_artifact is not None:
        _write_json(run_root / "code_artifact.json", code_artifact)
    if validation is not None:
        _write_json(run_root / "validation_artifact.json", validation)
    if packaged_artifact is not None:
        _write_json(run_root / "packaged_artifact.json", packaged_artifact)
    if run_metadata is not None:
        _write_json(run_root / "run_metadata.json", run_metadata)

    return str(run_root.resolve())


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _to_jsonable(payload: Any) -> Any:
    if is_dataclass(payload):
        return _to_jsonable(asdict(payload))
    if isinstance(payload, Enum):
        return payload.value
    if isinstance(payload, dict):
        return {str(key): _to_jsonable(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_to_jsonable(value) for value in payload]
    return payload


def _concise_validation_failures(validation: ValidationArtifact | None, limit: int = 5) -> str:
    if validation is None:
        return "none"
    failures = list(validation.failure_signatures or [])
    if not failures:
        failures = list(validation.failures or [])
    if not failures:
        return "none"
    trimmed = failures[:limit]
    if len(failures) > limit:
        trimmed.append("...")
    return ", ".join(trimmed)


def _retry_route_for_validation(validation: ValidationArtifact) -> ForgeRoute:
    if validation.passed:
        return ForgeRoute.TERMINAL_VERIFIED
    signatures = set(validation.failure_signatures or [])
    if "semantic_content_mismatch" in signatures:
        return ForgeRoute.TO_CODER
    planner_signatures = {
        "semantic_omission",
        "missing_requirement_coverage",
        "universal_constraint_unproven",
    }
    if signatures & planner_signatures:
        return ForgeRoute.TO_PLANNER

    category = validation.failure_category
    if category in {FailureCategory.ARCHITECTURAL, FailureCategory.CONTRADICTION, FailureCategory.UNDERSPECIFIED}:
        return ForgeRoute.TO_PLANNER
    return ForgeRoute.TO_CODER


def _mark_repair_terminal(
    validation: ValidationArtifact,
    signature: str,
    failure: str,
    evidence: dict[str, object],
) -> None:
    if signature not in validation.failure_signatures:
        validation.failure_signatures.append(signature)
    if failure and failure not in validation.failures:
        validation.failures.append(failure)
    validation.evidence["repair_terminal"] = dict(evidence)
    validation.evidence["failure_signatures"] = list(validation.failure_signatures)
    validation.metrics["failure_count"] = len(validation.failures)
    validation.metrics["failure_signature_count"] = len(validation.failure_signatures)


def _build_run_metrics(
    *,
    planner_attempts: int,
    attempt_trace: list[dict[str, object]],
    terminal_status: str,
) -> ForgeRunMetrics:
    repair_count = sum(1 for attempt in attempt_trace if "repair" in attempt)
    first_validation_passed = bool(
        attempt_trace and attempt_trace[0].get("validation_passed") is True
    )
    return ForgeRunMetrics(
        planner_attempts=planner_attempts,
        validation_attempts=len(attempt_trace),
        repair_count=repair_count,
        verified_at_1=(
            terminal_status == TERMINAL_VERIFIED and first_validation_passed
        ),
        success_after_repair=(
            terminal_status == TERMINAL_VERIFIED and repair_count > 0
        ),
    )


if __name__ == "__main__":
    app()
