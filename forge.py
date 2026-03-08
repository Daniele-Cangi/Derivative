import json
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import typer

from core.forge.coder_stage import CoderStage
from core.forge.contracts import (
    BuildSpec,
    CodeArtifact,
    FailureCategory,
    FeasiblePlan,
    ForgeResult,
    ForgeRoute,
    InfeasibilityCertificate,
    PackagedArtifact,
    ValidationArtifact,
)
from core.forge.packaging_stage import PackagingStage
from core.forge.planner_stage import PlannerStage
from core.forge.requirement_compiler import RequirementCompiler
from core.forge.validator_stage import ValidatorStage


TERMINAL_VERIFIED = "verified"
TERMINAL_INFEASIBLE_PROVEN = "infeasible_proven"
TERMINAL_VALIDATION_FAILED = "validation_failed"

app = typer.Typer(help="Forge - execution-grounded software build orchestrator.")


def run_forge(
    requirement: str,
    execution_mode: str = "local-only",
    output_root: str = "generated_artifacts/forge_runs",
    packaging_output_root: str = "generated_artifacts/forge_packages",
    max_planner_attempts: int = 1,
    max_coder_attempts: int = 1,
    requirement_compiler: RequirementCompiler | None = None,
    planner_stage: PlannerStage | None = None,
    coder_stage: CoderStage | None = None,
    validator_stage: ValidatorStage | None = None,
    packaging_stage: PackagingStage | None = None,
) -> ForgeResult:
    started = time.perf_counter()
    normalized_planner_attempts = max(1, int(max_planner_attempts))
    normalized_coder_attempts = max(1, int(max_coder_attempts))
    compiler = requirement_compiler or RequirementCompiler()
    planner = planner_stage or PlannerStage(execution_mode=execution_mode)
    coder = coder_stage or CoderStage()
    validator = validator_stage or ValidatorStage()
    packager = packaging_stage or PackagingStage(output_root=packaging_output_root)

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
            )

        if not isinstance(planning_output, FeasiblePlan):
            raise TypeError(f"PlannerStage returned unsupported output type: {type(planning_output)!r}")

        route_to_planner = False
        coder_attempt = 0
        while coder_attempt < normalized_coder_attempts:
            coder_attempt += 1
            code_artifact = coder.generate(planning_output)
            validation = validator.validate(code_artifact, planning_output, build_spec)
            latest_code_artifact = code_artifact
            latest_validation = validation
            retry_route = _retry_route_for_validation(validation)
            attempt_trace.append(
                {
                    "planner_attempt": planner_attempt,
                    "coder_attempt": coder_attempt,
                    "validation_passed": validation.passed,
                    "retry_route": retry_route.value,
                    "failure_category": (
                        validation.failure_category.value
                        if validation.failure_category is not None
                        else None
                    ),
                    "failure_signatures": list(validation.failure_signatures),
                }
            )

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
                )

            if retry_route == ForgeRoute.TO_CODER and coder_attempt < normalized_coder_attempts:
                continue
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
        1,
        "--max-coder-attempts",
        min=1,
        help="Maximum coder attempts per planner attempt.",
    ),
) -> None:
    result = run_forge(
        requirement=requirement,
        execution_mode=mode,
        output_root=output_root,
        packaging_output_root=packaging_root,
        max_planner_attempts=max_planner_attempts,
        max_coder_attempts=max_coder_attempts,
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
    signatures = set(validation.failure_signatures or [])
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


if __name__ == "__main__":
    app()
