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
        "       *",
        "  ===========",
        "  \\====*====/  FORGE // DERIVATIVE",
        "      ||       EXECUTION-GROUNDED BUILD",
        "     /__\\      Evidence precedes packaging.",
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
    from rich.table import Table
    from rich.text import Text

    target = console or Console(highlight=False)
    presentation = build_cli_presentation(result)
    unicode_capable = _supports_unicode(target)

    identity = Text()
    identity.append("F O R G E", style="bold bright_cyan")
    identity.append("\nDERIVATIVE / BUILD ENGINE", style="bold white")
    identity.append("\n\nREQUIREMENT", style="dim cyan")
    identity.append("  >  " if not unicode_capable else "  ›  ", style="bright_yellow")
    identity.append("EVIDENCE", style="bold bright_cyan")
    identity.append("  >  " if not unicode_capable else "  ›  ", style="bright_yellow")
    identity.append("ARTIFACT", style="dim cyan")

    header = Table.grid(expand=True, padding=(0, 2))
    header.add_column(width=14, justify="center", vertical="middle")
    header.add_column(ratio=1, vertical="middle")
    header.add_row(_brand_mark(unicode_capable), identity)
    target.print(
        Panel(
            header,
            box=box.ROUNDED,
            border_style="bright_cyan",
            padding=(1, 2),
            title=Text(" FORGE RUNTIME ", style="bold bright_cyan"),
            subtitle=Text(" EVIDENCE PRECEDES PACKAGING ", style="dim cyan"),
        )
    )

    target.print(Text("EVIDENCE RAIL", style="bold bright_cyan"), justify="center")
    track, labels = _stage_track(presentation.stages, unicode_capable)
    target.print(track, justify="center")
    target.print(labels, justify="center")
    target.print()

    evidence = Table.grid(expand=True, padding=(0, 1))
    evidence.add_column(width=4, justify="right", style="dim cyan")
    evidence.add_column(width=10, style="bold white")
    evidence.add_column(width=8)
    evidence.add_column(ratio=1, style="dim")
    for stage in presentation.stages:
        evidence.add_row(
            f"{stage.sequence:02d}",
            stage.name,
            Text(stage.state, style=_state_style(stage.state)),
            stage.detail,
        )
    target.print(evidence)

    terminal = Text()
    terminal.append("Status: ", style="bold")
    terminal.append(presentation.status, style=_status_style(presentation.status))
    terminal.append(f"\n{presentation.summary}")

    if presentation.failures:
        terminal.append("\n\nFailure signatures: ", style="bold red")
        terminal.append(presentation.failures, style="red")
    target.print(
        Panel(
            terminal,
            box=box.ROUNDED,
            border_style=_status_border(presentation.status),
            title=Text(" TERMINAL RESULT ", style="bold"),
            padding=(0, 1),
        )
    )

    receipt = Table.grid(expand=True, padding=(0, 1))
    receipt.add_column(width=19, style="bold")
    receipt.add_column(ratio=1)
    _add_receipt_row(receipt, "Trace seal", presentation.trace_seal, "bright_cyan")
    _add_receipt_row(receipt, "Attempts", presentation.attempts)
    if presentation.model_usage:
        _add_receipt_row(receipt, "Model usage", presentation.model_usage)
    _add_receipt_row(receipt, presentation.artifact_label, presentation.artifact_path, "cyan")
    _add_receipt_row(receipt, "Execution time", presentation.execution_time)
    target.print(
        Panel(
            receipt,
            box=box.SQUARE,
            border_style="grey37",
            title=Text(" RUN RECEIPT ", style="bold bright_cyan"),
            padding=(0, 1),
        )
    )


def _supports_unicode(console: Any) -> bool:
    encoding = str(getattr(console, "encoding", "") or "").lower().replace("-", "")
    return not encoding or "utf" in encoding


def _brand_mark(unicode_capable: bool) -> Any:
    from rich.text import Text

    mark = Text(justify="center")
    if unicode_capable:
        mark.append("·      ✦\n", style="yellow")
        mark.append("━━━━━━━━━━━\n", style="bright_yellow")
        mark.append("╲━━━━", style="bright_yellow")
        mark.append("◆", style="bold bright_yellow")
        mark.append("━━━━╱\n", style="bright_yellow")
        mark.append("    ┃┃\n", style="yellow")
        mark.append("   ╱━━╲", style="yellow")
        return mark

    mark.append(".      *\n", style="yellow")
    mark.append("===========\n", style="bright_yellow")
    mark.append("\\====", style="bright_yellow")
    mark.append("*", style="bold bright_yellow")
    mark.append("====/\n", style="bright_yellow")
    mark.append("    ||\n", style="yellow")
    mark.append("   /__\\", style="yellow")
    return mark


def _stage_track(
    stages: tuple[StageView, ...], unicode_capable: bool
) -> tuple[Any, Any]:
    from rich.text import Text

    track = Text()
    labels = Text()
    connector = "━━" if unicode_capable else "--"
    markers = {
        "PASS": "●" if unicode_capable else "+",
        "SEALED": "◆" if unicode_capable else "#",
        "PROVEN": "◆" if unicode_capable else "#",
        "FAIL": "×" if unicode_capable else "x",
        "BLOCK": "■" if unicode_capable else "!",
        "SKIP": "○" if unicode_capable else "-",
    }
    for index, stage in enumerate(stages):
        marker = markers.get(stage.state, "?")
        track.append(f"{marker:^9}", style=_state_style(stage.state))
        labels.append(f"{stage.name:^9}", style="dim white")
        if index < len(stages) - 1:
            track.append(connector, style="grey37")
            labels.append("  ")
    return track, labels


def _add_receipt_row(table: Any, label: str, value: str, style: str = "white") -> None:
    from rich.text import Text

    table.add_row(f"{label}:", Text(value, style=style, overflow="fold"))


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


def _status_border(status: str) -> str:
    return {
        VERIFIED: "green",
        INFEASIBLE_PROVEN: "yellow",
        VALIDATION_FAILED: "red",
    }.get(status, "white")
