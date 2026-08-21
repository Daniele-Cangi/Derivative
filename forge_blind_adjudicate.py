import hashlib
from pathlib import Path

import typer
from dotenv import load_dotenv

from core.forge.blind_adjudication import (
    AdjudicationReviewer,
    adjudicate_blind_manifest,
)


load_dotenv(Path(__file__).resolve().with_name(".env"))

app = typer.Typer(
    help="Independently adjudicate frozen blind requirement labels without Forge results.",
    pretty_exceptions_show_locals=False,
)


@app.command()
def main(
    manifest: str = typer.Option(..., "--manifest"),
    output: str = typer.Option(..., "--output"),
    reviewer_a_model: str = typer.Option("gpt-5.6-terra", "--reviewer-a-model"),
    reviewer_a_input_cost: float = typer.Option(..., "--reviewer-a-input-cost", min=0.0),
    reviewer_a_output_cost: float = typer.Option(..., "--reviewer-a-output-cost", min=0.0),
    reviewer_b_model: str = typer.Option("gpt-5.1", "--reviewer-b-model"),
    reviewer_b_input_cost: float = typer.Option(..., "--reviewer-b-input-cost", min=0.0),
    reviewer_b_output_cost: float = typer.Option(..., "--reviewer-b-output-cost", min=0.0),
    repository_root: str = typer.Option(".", "--repository-root"),
) -> None:
    try:
        receipt = adjudicate_blind_manifest(
            manifest_path=manifest,
            output_path=output,
            repository_root=repository_root,
            reviewers=(
                AdjudicationReviewer(
                    reviewer_a_model,
                    reviewer_a_input_cost,
                    reviewer_a_output_cost,
                ),
                AdjudicationReviewer(
                    reviewer_b_model,
                    reviewer_b_input_cost,
                    reviewer_b_output_cost,
                ),
            ),
        )
    except (FileExistsError, RuntimeError, ValueError) as exc:
        failure_id = hashlib.sha256(str(exc).encode("utf-8")).hexdigest()[:12]
        typer.echo(
            f"Blind adjudication failed closed. No receipt was written. Failure id: {failure_id}",
            err=True,
        )
        raise typer.Exit(code=1) from None

    summary = receipt["summary"]
    typer.echo("Forge Blind Requirement Adjudication")
    typer.echo(f"Bundle: {receipt['bundle']['bundle_id']}")
    typer.echo(f"Cases: {summary['total_cases']}")
    typer.echo(f"Label valid: {summary['label_valid']}")
    typer.echo(f"Label invalid: {summary['label_invalid']}")
    typer.echo(f"Unresolved: {summary['unresolved']}")
    typer.echo(f"Model requests: {summary['total_model_requests']}")
    typer.echo(f"Model tokens: {summary['total_model_tokens']}")
    typer.echo(f"Estimated cost: ${summary['total_estimated_cost_usd']:.8f}")
    typer.echo(f"Receipt: {Path(output).resolve()}")
    typer.echo("Status: adjudicated_without_forge_results")


if __name__ == "__main__":
    app()
