import logging
import time
import uuid
from datetime import datetime, timezone

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from audit.trail import AuditEntry, AuditTrail
from core.executor import ControlledExecutor
from core.kernel import ReasoningKernel
from core.launcher import GeneratedArtifactLauncher
from core.substrate import CognitiveSubstrate
from core.runtime_mode import VALID_EXECUTION_MODES, normalize_execution_mode
from core.validator import AdversarialValidator
from core.workspace import ArtifactWorkspace
from memory.delta import DeltaMemory

load_dotenv()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

app = typer.Typer(help="Derivative - Next-generation cognitive architecture.")
console = Console()

memory = DeltaMemory()
audit = AuditTrail()


@app.command()
def main(
    problem: str = typer.Argument(None, help="The problem to reason about"),
    file: str = typer.Option(None, "--file", "-f", help="A file to augment the problem"),
    reasoning: str = typer.Option(None, "--reasoning", "-r", help="Reasoning to validate"),
    show_audit: bool = typer.Option(False, "--audit", help="Show full audit trail"),
    show_memory: bool = typer.Option(False, "--memory", help="Show reasoning history"),
    show_lenses: bool = typer.Option(False, "--lenses", help="Show all installed cognitive lenses"),
    mode: str = typer.Option(
        None,
        "--mode",
        help=f"Execution mode: {', '.join(VALID_EXECUTION_MODES)}.",
    ),
    run_generated: bool = typer.Option(
        False,
        "--run-generated",
        help="Launch generated Python artifacts after the workspace export.",
    ),
):
    try:
        execution_mode = normalize_execution_mode(mode)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return

    substrate = CognitiveSubstrate(execution_mode=execution_mode)
    kernel = ReasoningKernel(execution_mode=execution_mode)
    validator = AdversarialValidator(execution_mode=execution_mode)
    executor = ControlledExecutor()
    launcher = GeneratedArtifactLauncher()
    workspace = ArtifactWorkspace()

    if show_lenses:
        table = Table(title="Installed Cognitive Lenses")
        table.add_column("Lens")
        table.add_column("Library Focus")
        table.add_column("Epistemic Tag")
        table.add_column("Runtime")

        for lens in substrate.lenses:
            runtime_status = getattr(lens, "runtime_status", lambda: "n/a")()
            table.add_row(lens.lens_name, lens.library_focus, lens.epistemic_tag, runtime_status)
        console.print(table)
        return

    if show_audit:
        trace = audit.get_full_trace()
        if not trace:
            console.print("No audit trail found.")
            return
        for entry in trace:
            body = (
                f"Validation: {entry.validation_result}\n"
                f"Confidence: {entry.confidence:.2f}\n"
                f"Lenses: {', '.join(entry.lenses_applied)}"
            )
            if entry.generated_design_titles:
                body += f"\nDesigns: {', '.join(entry.generated_design_titles)}"
            if entry.generated_artifact_names:
                body += f"\nArtifacts: {', '.join(entry.generated_artifact_names[:4])}"
            if entry.execution_session_summaries:
                body += f"\nExecution: {', '.join(entry.execution_session_summaries[:3])}"
            if entry.launch_session_summaries:
                body += f"\nLaunch: {', '.join(entry.launch_session_summaries[:3])}"
            if entry.artifact_workspace_path:
                body += f"\nWorkspace: {entry.artifact_workspace_path}"
            console.print(
                Panel(
                    body,
                    title=f"Step {entry.step_id} - {entry.timestamp}",
                    subtitle=entry.problem[:50] + "..." if len(entry.problem) > 50 else entry.problem,
                )
            )
        return

    if show_memory:
        history = memory.get_reasoning_history()
        if not history:
            console.print("No memory history found.")
            return
        for delta in history:
            body = (
                f"Delta: {delta.reasoning_delta}\n"
                f"Confidence: {delta.confidence_score:.2f}\n"
                f"Confidence Shift: {delta.confidence_delta:.2f}"
            )
            if delta.top_design_titles:
                body += f"\nTop Designs: {', '.join(delta.top_design_titles)}"
            console.print(
                Panel(
                    body,
                    title=f"Memory Hash: {delta.problem_hash[:8]}",
                    subtitle=delta.timestamp,
                )
            )
        return

    if not problem and not reasoning:
        console.print("[red]Error: You must provide a problem to solve or use an option.[/red]")
        return

    start_time = time.time()
    full_problem = problem or f"Validate reasoning: {reasoning}"

    if file:
        try:
            with open(file, "r", encoding="utf-8", errors="replace") as handle:
                content = handle.read()
            full_problem += f"\n\nFile Content ({file}):\n{content}"
        except Exception as exc:
            console.print(f"[red]Error reading file {file}: {exc}[/red]")
            return

    step_id = str(uuid.uuid4())[:8]
    design_context = memory.retrieve_design_context(full_problem)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task1 = progress.add_task("[cyan]Decomposing problem through all cognitive lenses...", total=None)
            framings = substrate.decompose(full_problem)
            progress.update(task1, completed=True)

            task2 = progress.add_task("[magenta]Kernel synthesizing multi-frame reasoning...", total=None)
            result = kernel.synthesize(full_problem, framings, design_context=design_context)
            progress.update(task2, completed=True)

            task3 = progress.add_task("[yellow]Adversarial Validator running...", total=None)
            report = validator.validate(result, full_problem)
            progress.update(task3, completed=True)
    except Exception as exc:
        console.print(f"[red]Derivative failed before completing the reasoning pipeline: {exc}[/red]")
        return

    console.print("\n[bold]Derivative Analytics System[/bold]")
    console.print(f"[dim]Execution Mode: {execution_mode}[/dim]")
    if design_context:
        console.print(f"[dim]Evolutionary Seeds: {len(design_context)} prior lineage snapshot(s)[/dim]")

    if substrate.last_errors:
        console.print(
            f"[yellow]Recovered from {len(substrate.last_errors)} lens issue(s); local fallback was used where needed.[/yellow]"
        )

    color = "green" if report.confidence_adjusted > 0.8 else "yellow" if report.confidence_adjusted > 0.5 else "red"
    bar = f"{'#' * max(1, int(report.confidence_adjusted * 20)):<20}"
    console.print(f"Epistemic Confidence: [{color}]{bar}[/{color}] ({report.confidence_adjusted:.2f})")

    execution_sessions = executor.evaluate_designs(result.generated_designs) if result.generated_designs else []
    workspace_export = None
    launch_sessions = []
    if result.generated_designs:
        try:
            workspace_export = workspace.export(
                step_id=step_id,
                problem=full_problem,
                designs=result.generated_designs,
                execution_sessions=execution_sessions,
            )
        except OSError as exc:
            console.print(f"[yellow]Warning: unable to export artifact workspace: {exc}[/yellow]")
        if run_generated and workspace_export:
            launch_sessions = launcher.launch_designs(
                result.generated_designs,
                workspace_export.artifact_paths,
            )

    if not report.is_valid:
        details = [
            f"[bold red]Validation Failed: {report.recommendation}[/bold red]",
            "",
            f"Violations: {result.violated_constraints}",
            f"Attacks: {report.attacks}",
        ]
        if report.edge_cases:
            details.append(f"Edge Cases: {report.edge_cases}")
        console.print(Panel("\n".join(details), title="Validation Failure"))
    else:
        for index, step in enumerate(result.reasoning_chain, start=1):
            console.print(f"  [dim]{index}.[/dim] {step.description}")

        console.print(Panel(result.conclusion, title="Synthesized Conclusion"))
        if report.edge_cases:
            console.print(Panel("\n".join(report.edge_cases), title="Validator Edge Cases"))

    if result.topology_search:
        optimal = result.topology_search.optimal_topology
        summary_lines = [
            f"Evaluated Shapes: {result.topology_search.evaluated_topologies}",
            f"Satisfiable Topologies: {len(result.topology_search.satisfiable_topologies)}",
        ]
        if optimal:
            summary_lines.append(
                f"Optimal: {optimal.candidate_id} | score {optimal.composite_score:.2f} | "
                f"diameter {optimal.diameter} | fidelity {optimal.end_to_end_fidelity:.5f}"
            )
        if result.topology_search.assumptions:
            summary_lines.append("")
            summary_lines.append("Assumptions:")
            for assumption in result.topology_search.assumptions:
                summary_lines.append(f"- {assumption}")
        console.print(Panel("\n".join(summary_lines), title="Exact Topology Search"))

        if result.topology_search.satisfiable_topologies:
            topology_table = Table(title="Satisfiable Topologies")
            topology_table.add_column("ID")
            topology_table.add_column("Edges")
            topology_table.add_column("Diameter")
            topology_table.add_column("Latency")
            topology_table.add_column("Fidelity")
            topology_table.add_column("Overhead")
            topology_table.add_column("Score")
            topology_table.add_column("Edge Set")

            for candidate in result.topology_search.satisfiable_topologies:
                edge_preview = ", ".join(
                    f"({left},{right})" for left, right in candidate.canonical_edges
                )
                topology_table.add_row(
                    candidate.candidate_id,
                    str(candidate.edge_count),
                    str(candidate.diameter),
                    f"{candidate.worst_case_latency_ms:.0f}ms",
                    f"{candidate.end_to_end_fidelity:.5f}",
                    f"{candidate.entanglement_overhead_factor:.2f}x",
                    f"{candidate.composite_score:.2f}",
                    edge_preview,
                )
            console.print(topology_table)

    if execution_sessions:
        for session in execution_sessions:
            lines = [
                f"Status: {session.status}",
                f"Aggregate Score: {session.aggregate_score:.2f}",
                "Reports:",
            ]
            for report_item in session.reports[:3]:
                observation = report_item.observations[0] if report_item.observations else "No observations."
                lines.append(
                    f"- {report_item.filename}: {report_item.status} ({report_item.validation_score:.2f}) :: {observation}"
                )
            console.print(Panel("\n".join(lines), title=f"{session.session_id} - {session.design_title}"))

    if workspace_export:
        console.print(
            Panel(
                f"Workspace: {workspace_export.root_path}\n"
                f"Manifest: {workspace_export.manifest_path}\n"
                f"Files Exported: {len(workspace_export.exported_files)}",
                title="Artifact Workspace",
            )
        )

    if launch_sessions:
        for session in launch_sessions:
            lines = [
                f"Status: {session.status}",
                f"Launched: {session.successful_count}/{session.launched_count}",
                "Reports:",
            ]
            if not session.reports:
                lines.append("- No launchable Python artifacts were found.")
            for report_item in session.reports[:3]:
                lines.append(
                    f"- {report_item.filename}: {report_item.status} (exit {report_item.exit_code}) :: {report_item.result_summary}"
                )
            console.print(Panel("\n".join(lines), title=f"{session.session_id} - {session.design_title}"))

    if result.generated_designs:
        for design in result.generated_designs:
            body_lines = [
                design.premise,
                "",
                f"Composite Score: {design.composite_score:.2f}",
                f"Novelty: {design.novelty_score:.2f}",
                f"Feasibility: {design.feasibility_score:.2f}",
                f"Tags: {', '.join(design.composition_tags)}",
                f"Mutation: {design.mutation_strategy}",
                "",
                "Primitives:",
            ]
            for primitive in design.component_primitives[:3]:
                body_lines.append(f"- {primitive}")
            if design.lineage_titles:
                body_lines.append("")
                body_lines.append("Lineage:")
                for title in design.lineage_titles[:3]:
                    body_lines.append(f"- {title}")
            body_lines.append("")
            body_lines.append("Implementation:")
            for step in design.implementation_outline[:4]:
                body_lines.append(f"- {step}")
            if design.artifacts:
                body_lines.append("")
                body_lines.append("Artifacts:")
                for artifact in design.artifacts[:3]:
                    preview = artifact.content.splitlines()[0] if artifact.content else ""
                    body_lines.append(
                        f"- {artifact.filename} [{artifact.artifact_type}/{artifact.language}] :: {preview}"
                    )
            console.print(Panel("\n".join(body_lines), title=f"{design.design_id} - {design.title}"))

    try:
        memory.record(result, full_problem)
    except OSError as exc:
        console.print(f"[yellow]Warning: unable to persist memory: {exc}[/yellow]")

    try:
        audit.log(
            AuditEntry(
                step_id=step_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                problem=full_problem,
                lenses_applied=[framing.lens_name for framing in framings],
                reasoning_chain=[step.description for step in result.reasoning_chain],
                epistemic_tags=[framing.epistemic_tag for framing in framings],
                confidence=report.confidence_adjusted,
                validation_result=report.recommendation,
                generated_design_titles=[design.title for design in result.generated_designs],
                generated_artifact_names=[
                    artifact.filename
                    for design in result.generated_designs
                    for artifact in design.artifacts
                ],
                artifact_workspace_path=workspace_export.root_path if workspace_export else "",
                execution_session_summaries=[
                    f"{session.design_title}:{session.status}:{session.aggregate_score:.2f}"
                    for session in execution_sessions
                ],
                launch_session_summaries=[
                    f"{session.design_title}:{session.status}:{session.successful_count}/{session.launched_count}"
                    for session in launch_sessions
                ],
            )
        )
    except OSError as exc:
        console.print(f"[yellow]Warning: unable to persist audit trail: {exc}[/yellow]")

    console.print(f"[dim]Execution Time: {time.time() - start_time:.2f}s[/dim]")


if __name__ == "__main__":
    app()
