import typer

from core.forge.benchmark import (
    BenchmarkThresholds,
    default_forge_benchmark_cases,
    evaluate_benchmark_thresholds,
    extended_forge_benchmark_cases,
    load_benchmark_cases,
    persist_benchmark_summary,
    render_benchmark_summary,
    run_benchmark_cases,
)
from forge import run_forge


app = typer.Typer(help="Run Forge benchmark cases and emit baseline metrics.")


@app.command()
def main(
    preset: str = typer.Option(
        "default",
        "--preset",
        help="Benchmark preset: default or extended.",
    ),
    dataset: str = typer.Option(
        "",
        "--dataset",
        help="Optional JSON dataset path. If omitted, built-in benchmark cases are used.",
    ),
    mode: str = typer.Option("local-only", "--mode", help="Execution mode for Forge planner substrate."),
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
    benchmark_output_root: str = typer.Option(
        "generated_artifacts/forge_benchmarks",
        "--benchmark-output-root",
        help="Directory for benchmark summary reports.",
    ),
    max_planner_attempts: int = typer.Option(
        1,
        "--max-planner-attempts",
        min=1,
        help="Maximum planner attempts for each case.",
    ),
    max_coder_attempts: int = typer.Option(
        1,
        "--max-coder-attempts",
        min=1,
        help="Maximum coder attempts for each case.",
    ),
    min_status_accuracy: float = typer.Option(
        0.95,
        "--min-status-accuracy",
        min=0.0,
        max=1.0,
        help="Minimum status accuracy threshold for quality gating.",
    ),
    min_verified_at_1: float = typer.Option(
        0.90,
        "--min-verified-at-1",
        min=0.0,
        max=1.0,
        help="Minimum Verified@1 threshold for quality gating.",
    ),
    max_false_verified_rate: float = typer.Option(
        0.00,
        "--max-false-verified-rate",
        min=0.0,
        max=1.0,
        help="Maximum false-verified rate threshold for quality gating.",
    ),
    min_infeasible_detection_rate: float = typer.Option(
        1.00,
        "--min-infeasible-detection-rate",
        min=0.0,
        max=1.0,
        help="Minimum infeasible detection rate threshold for quality gating.",
    ),
    enforce_thresholds: bool = typer.Option(
        False,
        "--enforce-thresholds/--no-enforce-thresholds",
        help="Exit with non-zero code when benchmark thresholds are violated.",
    ),
) -> None:
    normalized_preset = preset.strip().lower()
    if dataset:
        cases = load_benchmark_cases(dataset)
    elif normalized_preset == "extended":
        cases = extended_forge_benchmark_cases()
    elif normalized_preset == "default":
        cases = default_forge_benchmark_cases()
    else:
        raise typer.BadParameter("Unsupported --preset value. Use 'default' or 'extended'.")
    summary = run_benchmark_cases(
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
    report_path = persist_benchmark_summary(summary, benchmark_output_root)
    typer.echo(render_benchmark_summary(summary, report_path))
    threshold_failures = evaluate_benchmark_thresholds(
        summary,
        BenchmarkThresholds(
            min_status_accuracy=min_status_accuracy,
            min_verified_at_1=min_verified_at_1,
            max_false_verified_rate=max_false_verified_rate,
            min_infeasible_detection_rate=min_infeasible_detection_rate,
        ),
    )
    if threshold_failures:
        typer.echo("Threshold gate: failed")
        for failure in threshold_failures:
            typer.echo(f"- {failure}")
        if enforce_thresholds:
            raise typer.Exit(code=1)
    else:
        typer.echo("Threshold gate: passed")


if __name__ == "__main__":
    app()
