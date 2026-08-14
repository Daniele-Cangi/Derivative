import typer

from core.forge.blind_freeze import BlindFreezeProvenance, freeze_blind_bundle


app = typer.Typer(help="Freeze an externally authored blind benchmark before execution.")


@app.command()
def main(
    bundle_root: str = typer.Argument(..., help="Directory containing cases and oracles."),
    bundle_id: str = typer.Option(..., "--bundle-id"),
    producer: str = typer.Option(..., "--producer"),
    requirements_origin: str = typer.Option(..., "--requirements-origin"),
    oracle_origin: str = typer.Option(..., "--oracle-origin"),
    declaration: str = typer.Option(..., "--declaration"),
    source_url: list[str] = typer.Option([], "--source-url"),
    dataset: str = typer.Option("cases.json", "--dataset"),
    repository_root: str = typer.Option(".", "--repository-root"),
) -> None:
    bundle = freeze_blind_bundle(
        bundle_root=bundle_root,
        bundle_id=bundle_id,
        provenance=BlindFreezeProvenance(
            producer=producer,
            requirements_origin=requirements_origin,
            oracle_origin=oracle_origin,
            declaration=declaration,
        ),
        source_urls=source_url,
        repository_root=repository_root,
        dataset_path=dataset,
    )
    typer.echo("Forge Blind Freeze")
    typer.echo(f"Bundle: {bundle.bundle_id}")
    typer.echo(f"Manifest: {bundle.manifest_path}")
    typer.echo(f"Manifest SHA-256: {bundle.manifest_sha256}")
    typer.echo(f"Dataset SHA-256: {bundle.dataset_sha256}")
    typer.echo(f"Forge baseline SHA-256: {bundle.baseline_sha256}")
    typer.echo(f"Cases: {len(bundle.cases)}")
    typer.echo("Status: frozen_before_execution")


if __name__ == "__main__":
    app()
