from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from core.forge.contracts import ForgeResult, ValidationArtifact


VERIFIED = "verified"
INFEASIBLE_PROVEN = "infeasible_proven"
VALIDATION_FAILED = "validation_failed"


@dataclass(frozen=True)
class StageView:
    sequence: int
    name: str
    state: str
    detail: str


@dataclass(frozen=True)
class CliPresentation:
    status: str
    summary: str
    stages: tuple[StageView, ...]
    artifact_label: str
    artifact_path: str
    trace_seal: str
    attempts: str
    model_usage: str
    failures: str
    execution_time: str


def build_cli_presentation(result: ForgeResult) -> CliPresentation:
    failures = _concise_failures(result.validation)
    return CliPresentation(
        status=result.terminal_status,
        summary=result.summary,
        stages=_stage_views(result),
        artifact_label=_artifact_label(result.terminal_status),
        artifact_path=result.artifact_path,
        trace_seal=_trace_seal(result),
        attempts=(
            f"planner {result.run_metrics.planner_attempts} | "
            f"validation {result.run_metrics.validation_attempts} | "
            f"repairs {result.run_metrics.repair_count}"
        ),
        model_usage=_model_usage(result),
        failures=failures,
        execution_time=f"{result.execution_time_seconds:.2f}s",
    )


def render_cli_output(result: ForgeResult) -> str:
    presentation = build_cli_presentation(result)
    lines = [
        "+- FORGE // DERIVATIVE",
        "|  EXECUTION-GROUNDED BUILD",
        "+- Evidence precedes packaging.",
        "",
        "EVIDENCE RAIL",
    ]
    lines.extend(
        f"  {stage.sequence:02d} / {stage.name:<8} -> {stage.state:<6} / {stage.detail}"
        for stage in presentation.stages
    )
    lines.extend(
        [
            "",
            f"Status: {presentation.status}",
            presentation.summary,
        ]
    )
    if presentation.failures:
        lines.append(f"Validation failures: {presentation.failures}")
    lines.extend(
        [
            "",
            f"Trace seal: {presentation.trace_seal}",
            f"Attempts: {presentation.attempts}",
        ]
    )
    if presentation.model_usage:
        lines.append(f"Model usage: {presentation.model_usage}")
    lines.extend(
        [
            f"{presentation.artifact_label}: {presentation.artifact_path}",
            f"Execution time: {presentation.execution_time}",
        ]
    )
    return "\n".join(lines)


def print_cli_output(result: ForgeResult, console: Any | None = None) -> None:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text

    target = console or Console(highlight=False)
    presentation = build_cli_presentation(result)

    heading = Text()
    heading.append("FORGE", style="bold bright_cyan")
    heading.append(" // DERIVATIVE", style="dim cyan")
    heading.append("\nEXECUTION-GROUNDED BUILD", style="bold white")
    heading.append("\nEvidence precedes packaging.", style="dim")
    target.print(Panel(heading, box=box.ROUNDED, border_style="cyan", padding=(0, 2)))

    target.print(Text("EVIDENCE RAIL", style="bold cyan"))
    for stage in presentation.stages:
        line = Text()
        line.append(f" {stage.sequence:02d} ", style="dim cyan")
        line.append(f"{stage.name:<8}", style="bold white")
        line.append("  ")
        line.append(f"{stage.state:<6}", style=_state_style(stage.state))
        line.append(f"  {stage.detail}", style="dim")
        target.print(line)

    target.print(Rule(characters="-", style="grey37"))
    status_line = Text("Status: ", style="bold")
    status_line.append(presentation.status, style=_status_style(presentation.status))
    target.print(status_line)
    target.print(Text(presentation.summary))

    if presentation.failures:
        failure_line = Text("Validation failures: ", style="bold red")
        failure_line.append(presentation.failures, style="red")
        target.print(failure_line)

    target.print()
    _print_fact(target, "Trace seal", presentation.trace_seal, "bright_cyan")
    _print_fact(target, "Attempts", presentation.attempts)
    if presentation.model_usage:
        _print_fact(target, "Model usage", presentation.model_usage)
    _print_fact(target, presentation.artifact_label, presentation.artifact_path, "cyan")
    _print_fact(target, "Execution time", presentation.execution_time)


def _stage_views(result: ForgeResult) -> tuple[StageView, ...]:
    if result.terminal_status == VERIFIED:
        passed_layers = _passed_layer_count(result.validation)
        return (
            StageView(1, "COMPILE", "PASS", "requirement contract preserved"),
            StageView(2, "PLAN", "PASS", "feasible architecture grounded"),
            StageView(3, "GENERATE", "PASS", "candidate artifact emitted"),
            StageView(4, "VALIDATE", "PASS", f"{passed_layers}/3 evidence layers passed"),
            StageView(5, "PACKAGE", "SEALED", "verified artifact packaged"),
        )
    if result.terminal_status == INFEASIBLE_PROVEN:
        contradiction_count = len(
            result.infeasibility_certificate.contradictions
            if result.infeasibility_certificate is not None
            else []
        )
        return (
            StageView(1, "COMPILE", "PASS", "requirement contract preserved"),
            StageView(2, "PLAN", "PROVEN", f"{contradiction_count} contradiction(s) witnessed"),
            StageView(3, "GENERATE", "SKIP", "candidate generation not permitted"),
            StageView(4, "VALIDATE", "SKIP", "build validation not applicable"),
            StageView(5, "PACKAGE", "BLOCK", "infeasibility certificate retained"),
        )
    if result.terminal_status == VALIDATION_FAILED:
        failure_count = len(
            result.validation.failure_signatures
            if result.validation is not None
            else []
        )
        return (
            StageView(1, "COMPILE", "PASS", "requirement contract preserved"),
            StageView(2, "PLAN", "PASS", "feasible architecture grounded"),
            StageView(3, "GENERATE", "PASS", "candidate artifact emitted"),
            StageView(4, "VALIDATE", "FAIL", f"{failure_count} failure signature(s) retained"),
            StageView(5, "PACKAGE", "BLOCK", "unverified artifact not packaged"),
        )
    return (
        StageView(1, "COMPILE", "PASS", "terminal result available"),
        StageView(2, "PLAN", "UNKNOWN", "inspect persisted evidence"),
        StageView(3, "GENERATE", "UNKNOWN", "inspect persisted evidence"),
        StageView(4, "VALIDATE", "UNKNOWN", "inspect persisted evidence"),
        StageView(5, "PACKAGE", "BLOCK", "unknown terminal status"),
    )


def _passed_layer_count(validation: ValidationArtifact | None) -> int:
    if validation is None:
        return 0
    passed_layers = validation.metrics.get("passed_layers", {})
    if not isinstance(passed_layers, dict):
        return 0
    return sum(value is True for value in passed_layers.values())


def _artifact_label(status: str) -> str:
    if status == VERIFIED:
        return "Packaged artifact"
    if status == INFEASIBLE_PROVEN:
        return "Certificate artifacts"
    return "Artifacts"


def _trace_seal(result: ForgeResult) -> str:
    if result.terminal_status == VERIFIED and result.packaged_artifact is not None:
        digest = str(
            result.packaged_artifact.verification_metadata.get("code_artifact_digest")
            or result.packaged_artifact.package_id
        )
        return f"code:{digest[:16]}"
    if result.terminal_status == INFEASIBLE_PROVEN and result.infeasibility_certificate is not None:
        return f"certificate:{result.infeasibility_certificate.certificate_id}"

    validation = result.validation
    failure_material = []
    if validation is not None:
        failure_material = list(validation.failure_signatures or validation.failures)
    material = "\0".join(failure_material or [result.terminal_status])
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"failure:{digest}"


def _model_usage(result: ForgeResult) -> str:
    metrics = result.run_metrics
    if metrics.model_request_count <= 0:
        return ""
    return (
        f"{metrics.model_request_count} request(s) | "
        f"{metrics.model_total_tokens} token(s) | "
        f"cost {_format_cost(metrics.estimated_model_cost_usd)}"
    )


def _format_cost(cost: float | None) -> str:
    return "unconfigured" if cost is None else f"${cost:.6f}"


def _concise_failures(validation: ValidationArtifact | None, limit: int = 5) -> str:
    if validation is None:
        return ""
    failures = list(validation.failure_signatures or validation.failures)
    if not failures:
        return ""
    trimmed = failures[:limit]
    if len(failures) > limit:
        trimmed.append("...")
    return ", ".join(trimmed)


def _status_style(status: str) -> str:
    return {
        VERIFIED: "bold green",
        INFEASIBLE_PROVEN: "bold yellow",
        VALIDATION_FAILED: "bold red",
    }.get(status, "bold white")


def _state_style(state: str) -> str:
    return {
        "PASS": "bold green",
        "SEALED": "bold bright_cyan",
        "PROVEN": "bold yellow",
        "FAIL": "bold red",
        "BLOCK": "bold red",
        "SKIP": "dim yellow",
    }.get(state, "dim")


def _print_fact(console: Any, label: str, value: str, value_style: str = "white") -> None:
    from rich.text import Text

    line = Text()
    line.append(f"{label}: ", style="bold")
    line.append(value, style=value_style)
    console.print(line)
