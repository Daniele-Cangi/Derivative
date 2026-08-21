import hashlib
from pathlib import Path

import typer

from core.forge.blind_metrics import derive_adjudicated_metrics_from_files


app = typer.Typer(
    help="Derive definitive blind metrics from sealed raw and adjudication receipts.",
    pretty_exceptions_show_locals=False,
)


@app.command()
def main(
    manifest: str = typer.Option(..., "--manifest"),
    baseline_report: str = typer.Option(..., "--baseline-report"),
    adjudication: str = typer.Option(..., "--adjudication"),
    output: str = typer.Option(..., "--output"),
    repository_root: str = typer.Option(".", "--repository-root"),
) -> None:
    try:
        receipt = derive_adjudicated_metrics_from_files(
            manifest_path=manifest,
            baseline_report_path=baseline_report,
            adjudication_path=adjudication,
            output_path=output,
            repository_root=repository_root,
        )
    except (FileExistsError, ValueError) as exc:
        failure_id = hashlib.sha256(str(exc).encode("utf-8")).hexdigest()[:12]
        typer.echo(
            "Adjudicated metrics derivation failed closed. "
            f"No receipt was written. Failure id: {failure_id}",
            err=True,
        )
        raise typer.Exit(code=1) from None

    summary = receipt["summary"]
    typer.echo("Forge Blind Adjudicated Metrics")
    typer.echo(f"Bundle: {receipt['sources']['bundle_id']}")
    typer.echo(f"Definitive status cases: {summary['definitive_status_cases']}")
    typer.echo(f"Excluded cases: {summary['excluded_cases']}")
    typer.echo(f"Unresolved: {summary['unresolved_cases']}")
    typer.echo(f"Invalid adjudications: {summary['invalid_adjudication_cases']}")
    typer.echo(f"Receipt: {Path(output).resolve()}")
    typer.echo("Status: adjudicated_metrics_derived")


if __name__ == "__main__":
    app()
