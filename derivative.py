import logging
import time
import uuid
from datetime import datetime, timezone

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from audit.trail import AuditEntry, AuditTrail
from core.execution_loop import MAX_EXECUTION_CYCLES
from core.executor import ControlledExecutor
from core.kernel import ReasoningKernel
from core.launcher import GeneratedArtifactLauncher
from core.substrate import CognitiveSubstrate
from core.runtime_mode import VALID_EXECUTION_MODES, normalize_execution_mode
from core.validator import AdversarialValidator
from core.workspace import ArtifactWorkspace
from memory.delta import DeltaMemory
from memory.gene_pool import DesignGenePool

load_dotenv()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

app = typer.Typer(help="Derivative - Next-generation cognitive architecture.")
console = Console()

memory = DeltaMemory()
audit = AuditTrail()
gene_pool = DesignGenePool()


def _render_lenses(substrate: CognitiveSubstrate) -> str:
    if not substrate.lenses:
        return "No cognitive lenses are installed."

    sentences = []
    for lens in substrate.lenses:
        runtime_status = getattr(lens, "runtime_status", lambda: "n/a")()
        sentences.append(
            f"{lens.lens_name} focuses on {lens.library_focus} with epistemic tag {lens.epistemic_tag} "
            f"and runtime status {runtime_status}."
        )
    return "Installed cognitive lenses: " + " ".join(sentences)


def _render_audit_entry(entry: AuditEntry) -> str:
    base = (
        f"Step {entry.step_id} ran at {entry.timestamp}. Validation was {entry.validation_result} at confidence "
        f"{entry.confidence:.2f}. The problem was {entry.problem!r}. The active lenses were "
        f"{', '.join(entry.lenses_applied) if entry.lenses_applied else 'none'}."
    )
    if entry.reasoning_chain:
        base += f" The recorded reasoning chain was: {' '.join(entry.reasoning_chain)}"
    if entry.generated_design_titles:
        base += f" Generated designs included {', '.join(entry.generated_design_titles[:3])}."
    if entry.generated_artifact_names:
        base += f" Generated artifacts included {', '.join(entry.generated_artifact_names[:4])}."
    if entry.execution_code:
        base += f" The execution loop recorded delta {entry.execution_delta:.2f} for this cycle."
    if entry.execution_prediction:
        base += f" The recorded prediction payload was {entry.execution_prediction}."
    if entry.execution_prediction or entry.execution_code:
        base += f" The residual margin was {entry.execution_residual:.4f}."
    if entry.execution_output:
        base += f" The structured execution output was {entry.execution_output}."
    if entry.artifact_workspace_path:
        base += f" The artifact workspace was {entry.artifact_workspace_path}."
    return base


def _render_memory_delta(delta) -> str:
    text = (
        f"Memory snapshot {delta.problem_hash[:8]} was recorded at {delta.timestamp}. The reasoning delta says "
        f"{delta.reasoning_delta} Confidence is now {delta.confidence_score:.2f}, with a shift of "
        f"{delta.confidence_delta:.2f}."
    )
    if delta.top_design_titles:
        text += f" Top designs were {', '.join(delta.top_design_titles[:3])}."
    if delta.execution_cycle_summaries:
        text += f" Execution history across cycles: {' '.join(delta.execution_cycle_summaries)}"
    if delta.verified_hypotheses:
        text += f" Verified hypotheses persisted from this run include {' '.join(delta.verified_hypotheses[:2])}."
    return text


def _summarize_execution_sessions(execution_sessions) -> str:
    if not execution_sessions:
        return ""
    average_score = sum(session.aggregate_score for session in execution_sessions) / max(1, len(execution_sessions))
    validated_count = sum(1 for session in execution_sessions if session.status == "validated")
    return (
        f"The artifact evaluator checked {len(execution_sessions)} design session(s), validated {validated_count}, "
        f"and produced an average aggregate score of {average_score:.2f}."
    )


def _summarize_launch_sessions(launch_sessions) -> str:
    if not launch_sessions:
        return ""
    launched = sum(session.launched_count for session in launch_sessions)
    successful = sum(session.successful_count for session in launch_sessions)
    return (
        f"The launcher executed {launched} Python artifact(s) and completed {successful} successfully."
    )


def _build_cli_narrative(
    result,
    report,
    design_context,
    substrate,
    execution_sessions,
    launch_sessions,
) -> str:
    sentences = []
    if design_context:
        sentences.append(
            f"I loaded {len(design_context)} evolutionary seed snapshot(s) before reasoning so the search could reuse prior verified design traits."
        )
    if substrate.last_errors:
        sentences.append(
            f"I recovered from {len(substrate.last_errors)} lens issue(s) by falling back to local framing where needed."
        )
    sentences.append(result.conclusion)

    execution_summary = _summarize_execution_sessions(execution_sessions)
    if execution_summary:
        sentences.append(execution_summary)

    launch_summary = _summarize_launch_sessions(launch_sessions)
    if launch_summary:
        sentences.append(launch_summary)

    if report.is_valid:
        sentences.append(
            f"The adversarial validator accepted this run with adjusted confidence {report.confidence_adjusted:.2f}."
        )
    else:
        sentences.append(
            f"The adversarial validator did not accept this run and returned {report.recommendation} with adjusted confidence {report.confidence_adjusted:.2f}."
        )
    if report.attacks:
        sentences.append("Its main attack surface was " + " ".join(report.attacks))
    if report.edge_cases:
        sentences.append("The validator also flagged these edge cases: " + " ".join(report.edge_cases))

    return " ".join(sentence.strip() for sentence in sentences if sentence and sentence.strip())


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
        console.print(_render_lenses(substrate))
        return

    if show_audit:
        trace = audit.get_full_trace()
        if not trace:
            console.print("No audit trail found.")
            return
        for entry in trace:
            console.print(_render_audit_entry(entry))
            console.print("")
        return

    if show_memory:
        history = memory.get_reasoning_history()
        if not history:
            console.print("No memory history found.")
            return
        for delta in history:
            console.print(_render_memory_delta(delta))
            console.print("")
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
            result = kernel.synthesize(full_problem, framings, design_context=design_context, audit=audit)
            progress.update(task2, completed=True)

            task3 = progress.add_task("[yellow]Adversarial Validator running...", total=None)
            report = validator.validate(result, full_problem)
            progress.update(task3, completed=True)
    except Exception as exc:
        console.print(f"[red]Derivative failed before completing the reasoning pipeline: {exc}[/red]")
        return

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

    execution_result = result.execution_result
    cycles_used = execution_result.cycles_used if execution_result is not None else 0
    status_text = "converged" if execution_result and execution_result.converged else "not converged"
    narrative = _build_cli_narrative(
        result=result,
        report=report,
        design_context=design_context,
        substrate=substrate,
        execution_sessions=execution_sessions,
        launch_sessions=launch_sessions,
    )
    artifact_path = workspace_export.root_path if workspace_export else "workspace export unavailable"

    console.print("")
    console.print("Derivative - Self-Executing Reasoning Engine")
    console.print(
        f"Confidence: {report.confidence_adjusted:.2f}. Cycles used: {cycles_used}/{MAX_EXECUTION_CYCLES}. "
        f"Status: {status_text}."
    )
    console.print("")
    console.print(narrative)
    console.print("")
    console.print(f"Artifacts saved to: {artifact_path}")
    console.print(f"Execution time: {time.time() - start_time:.2f}s")

    try:
        memory.record(result, full_problem)
    except OSError as exc:
        console.print(f"[yellow]Warning: unable to persist memory: {exc}[/yellow]")

    try:
        gene_pool.record_execution(result, full_problem)
    except OSError as exc:
        console.print(f"[yellow]Warning: unable to persist verified gene pool: {exc}[/yellow]")

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
                execution_code=result.execution_result.final_code if result.execution_result else "",
                execution_prediction=result.execution_result.final_prediction if result.execution_result else "",
                execution_output=result.execution_result.final_output if result.execution_result else "",
                execution_delta=(
                    result.execution_result.history[-1].delta
                    if result.execution_result and result.execution_result.history
                    else 0.0
                ),
                execution_residual=(
                    result.execution_result.final_residual
                    if result.execution_result
                    else 0.0
                ),
            )
        )
    except OSError as exc:
        console.print(f"[yellow]Warning: unable to persist audit trail: {exc}[/yellow]")


if __name__ == "__main__":
    app()
