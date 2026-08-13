import typer

from core.forge.heldout_benchmark import (
    HeldoutThresholds,
    bundled_heldout_dataset_path,
    evaluate_heldout_thresholds,
    load_heldout_cases,
    persist_heldout_summary,
    render_heldout_summary,
    run_heldout_cases,
)
from forge import run_forge


app = typer.Typer(help="Run Forge against held-out requirements and independent acceptance oracles.")


@app.command()
def main(
    dataset: str = typer.Option(
        "",
        "--dataset",
        help="Held-out dataset path. Defaults to the bundled frozen dataset.",
    ),
    mode: str = typer.Option("local-only", "--mode", help="Forge execution mode."),
    output_root: str = typer.Option(
        "generated_artifacts/forge_heldout_runs",
        "--output-root",
    ),
    packaging_root: str = typer.Option(
        "generated_artifacts/forge_heldout_packages",
        "--packaging-root",
    ),
    benchmark_output_root: str = typer.Option(
        "generated_artifacts/forge_heldout_benchmarks",
        "--benchmark-output-root",
    ),
    max_planner_attempts: int = typer.Option(1, "--max-planner-attempts", min=1),
    max_coder_attempts: int = typer.Option(3, "--max-coder-attempts", min=1),
    min_status_accuracy: float = typer.Option(0.0, min=0.0, max=1.0),
    min_external_verified_at_1: float = typer.Option(0.0, min=0.0, max=1.0),
    max_external_false_verified_rate: float = typer.Option(0.0, min=0.0, max=1.0),
    min_infeasible_detection_rate: float = typer.Option(0.0, min=0.0, max=1.0),
    enforce_thresholds: bool = typer.Option(False, "--enforce-thresholds/--no-enforce-thresholds"),
) -> None:
    cases = load_heldout_cases(dataset or bundled_heldout_dataset_path())
    summary = run_heldout_cases(
        cases,
        run_case=lambda requirement: run_forge(
            requirement=requirement,
            execution_mode=mode,
            output_root=output_root,
            packaging_output_root=packaging_root,
            max_planner_attempts=max_planner_attempts,
            max_coder_attempts=max_coder_attempts,
        ),
    )
    report_path = persist_heldout_summary(summary, benchmark_output_root)
    typer.echo(render_heldout_summary(summary, report_path))
    failures = evaluate_heldout_thresholds(
        summary,
        HeldoutThresholds(
            min_status_accuracy=min_status_accuracy,
            min_external_verified_at_1=min_external_verified_at_1,
            max_external_false_verified_rate=max_external_false_verified_rate,
            min_infeasible_detection_rate=min_infeasible_detection_rate,
        ),
    )
    if failures:
        typer.echo("Held-out threshold gate: failed")
        for failure in failures:
            typer.echo(f"- {failure}")
        if enforce_thresholds:
            raise typer.Exit(code=1)
    else:
        typer.echo("Held-out threshold gate: passed")


if __name__ == "__main__":
    app()
