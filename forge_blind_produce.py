import hashlib
import os
from pathlib import Path

import typer
from dotenv import load_dotenv

from core.forge.blind_producer import BlindProducerConfig, produce_and_freeze_blind_bundle
from core.forge.telemetry import track_model_usage


load_dotenv(Path(__file__).resolve().with_name(".env"))

app = typer.Typer(
    help="Produce and freeze an isolated OpenAI-authored blind bundle before Forge execution.",
    pretty_exceptions_show_locals=False,
)


@app.command()
def main(
    output_root: str = typer.Argument(..., help="New directory for the sealed bundle."),
    bundle_id: str = typer.Option(..., "--bundle-id"),
    benchmark_version: str = typer.Option("v3", "--benchmark-version"),
    model: str | None = typer.Option(None, "--model"),
    verified_cases: int = typer.Option(6, "--verified-cases", min=1),
    validation_failed_cases: int = typer.Option(3, "--validation-failed-cases", min=1),
    infeasible_cases: int = typer.Option(3, "--infeasible-cases", min=1),
    repository_root: str = typer.Option(".", "--repository-root"),
    input_cost_per_1m: float | None = typer.Option(
        None,
        "--input-cost-per-1m",
        min=0.0,
    ),
    output_cost_per_1m: float | None = typer.Option(
        None,
        "--output-cost-per-1m",
        min=0.0,
    ),
) -> None:
    if (input_cost_per_1m is None) != (output_cost_per_1m is None):
        raise typer.BadParameter("Both token cost rates must be provided together.")
    if input_cost_per_1m is not None and output_cost_per_1m is not None:
        os.environ["OPENAI_INPUT_COST_PER_1M_TOKENS"] = str(input_cost_per_1m)
        os.environ["OPENAI_OUTPUT_COST_PER_1M_TOKENS"] = str(output_cost_per_1m)

    try:
        with track_model_usage() as usage:
            bundle = produce_and_freeze_blind_bundle(
                output_root=output_root,
                repository_root=repository_root,
                config=BlindProducerConfig(
                    bundle_id=bundle_id,
                    benchmark_version=benchmark_version,
                    verified_cases=verified_cases,
                    validation_failed_cases=validation_failed_cases,
                    infeasible_cases=infeasible_cases,
                ),
                model=model,
            )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        category = _safe_failure_category(exc)
        failure_id = hashlib.sha256(str(exc).encode("utf-8")).hexdigest()[:12]
        typer.echo(
            f"Blind production failed: {category}. No bundle was published. "
            f"Failure id: {failure_id}",
            err=True,
        )
        raise typer.Exit(code=1) from None
    estimated_cost, pricing_source = usage.estimated_cost()
    typer.echo("Forge Blind Producer")
    typer.echo(f"Bundle: {bundle.bundle_id}")
    typer.echo(f"Manifest: {bundle.manifest_path}")
    typer.echo(f"Manifest SHA-256: {bundle.manifest_sha256}")
    typer.echo(f"Dataset SHA-256: {bundle.dataset_sha256}")
    typer.echo(f"Forge baseline SHA-256: {bundle.baseline_sha256}")
    typer.echo(f"Cases: {len(bundle.cases)}")
    typer.echo(f"Model requests: {usage.request_count}")
    typer.echo(f"Model tokens: {usage.total_tokens}")
    typer.echo(
        "Estimated producer cost: "
        + ("unavailable" if estimated_cost is None else f"${estimated_cost:.8f}")
    )
    typer.echo(f"Pricing source: {pricing_source}")
    typer.echo("Status: frozen_before_execution")


def _safe_failure_category(exc: Exception) -> str:
    message = str(exc)
    if "Requirement producer failed validation" in message:
        return "requirement_preflight_failed"
    if "Oracle producer failed validation" in message:
        return "oracle_preflight_failed"
    if isinstance(exc, FileExistsError):
        return "destination_exists"
    if isinstance(exc, RuntimeError):
        return "provider_unavailable"
    return "production_failed"


if __name__ == "__main__":
    app()
