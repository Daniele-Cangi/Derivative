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
    requirement_compiler: RequirementCompiler | None = None,
    planner_stage: PlannerStage | None = None,
    coder_stage: CoderStage | None = None,
    validator_stage: ValidatorStage | None = None,
    packaging_stage: PackagingStage | None = None,
) -> ForgeResult:
    started = time.perf_counter()
    compiler = requirement_compiler or RequirementCompiler()
    planner = planner_stage or PlannerStage(execution_mode=execution_mode)
    coder = coder_stage or CoderStage()
    validator = validator_stage or ValidatorStage()
    packager = packaging_stage or PackagingStage(output_root=packaging_output_root)

    build_spec = compiler.compile(requirement)
    planning_output = planner.plan(build_spec)

    if isinstance(planning_output, InfeasibilityCertificate):
        run_root = _persist_run_artifacts(
            output_root=output_root,
            build_spec=build_spec,
            terminal_status=TERMINAL_INFEASIBLE_PROVEN,
            plan_output=planning_output,
            code_artifact=None,
            validation=None,
            packaged_artifact=None,
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

    code_artifact = coder.generate(planning_output)
    validation = validator.validate(code_artifact, planning_output, build_spec)
    if not validation.passed:
        run_root = _persist_run_artifacts(
            output_root=output_root,
            build_spec=build_spec,
            terminal_status=TERMINAL_VALIDATION_FAILED,
            plan_output=planning_output,
            code_artifact=code_artifact,
            validation=validation,
            packaged_artifact=None,
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

    packaged_artifact = packager.package(build_spec, planning_output, code_artifact, validation)
    _persist_run_artifacts(
        output_root=output_root,
        build_spec=build_spec,
        terminal_status=TERMINAL_VERIFIED,
        plan_output=planning_output,
        code_artifact=code_artifact,
        validation=validation,
        packaged_artifact=packaged_artifact,
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
) -> None:
    result = run_forge(
        requirement=requirement,
        execution_mode=mode,
        output_root=output_root,
        packaging_output_root=packaging_root,
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


if __name__ == "__main__":
    app()
