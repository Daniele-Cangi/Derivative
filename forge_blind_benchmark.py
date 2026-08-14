import typer

from core.forge.blind_benchmark import (
    bundled_blind_manifest_path,
    evaluate_blind_thresholds,
    load_blind_bundle,
    persist_blind_report,
    render_blind_report,
    run_blind_bundle,
)
from core.forge.heldout_benchmark import HeldoutThresholds
from forge import run_forge


app = typer.Typer(help="Run Forge against a sealed blind benchmark bundle.")


@app.command()
def main(
    manifest: str = typer.Option(
        "",
        "--manifest",
        help="Sealed bundle manifest. Defaults to the bundled blind-v2 calibration set.",
    ),
    mode: str = typer.Option("local-only", "--mode", help="Forge execution mode."),
    output_root: str = typer.Option(
        "generated_artifacts/forge_blind_runs",
        "--output-root",
    ),
    packaging_root: str = typer.Option(
        "generated_artifacts/forge_blind_packages",
        "--packaging-root",
    ),
    benchmark_output_root: str = typer.Option(
        "generated_artifacts/forge_blind_benchmarks",
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
    bundle = load_blind_bundle(manifest or bundled_blind_manifest_path())
    report = run_blind_bundle(
        bundle,
        run_case=lambda requirement: run_forge(
            requirement=requirement,
            execution_mode=mode,
            output_root=output_root,
            packaging_output_root=packaging_root,
            max_planner_attempts=max_planner_attempts,
            max_coder_attempts=max_coder_attempts,
        ),
    )
    report_path = persist_blind_report(report, benchmark_output_root)
    typer.echo(render_blind_report(report, report_path))
    failures = evaluate_blind_thresholds(
        report,
        HeldoutThresholds(
            min_status_accuracy=min_status_accuracy,
            min_external_verified_at_1=min_external_verified_at_1,
            max_external_false_verified_rate=max_external_false_verified_rate,
            min_infeasible_detection_rate=min_infeasible_detection_rate,
        ),
    )
    if not enforce_thresholds:
        typer.echo("Blind benchmark threshold enforcement: disabled")
    elif failures:
        typer.echo("Blind benchmark threshold gate: failed")
        for failure in failures:
            typer.echo(f"- {failure}")
        raise typer.Exit(code=1)
    else:
        typer.echo("Blind benchmark threshold gate: passed")


if __name__ == "__main__":
    app()
