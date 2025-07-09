"""Command line interface for the OceanData SDK."""

from __future__ import annotations

import pandas as pd
import typer
from dotenv import load_dotenv

from ocean_sdk import (
    publish_dataset,
    run_analysis,
    sync_marketplace,
    train_model,
)
from src.modules.calculator import add

load_dotenv()

app = typer.Typer(add_completion=False)


@app.callback()
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
) -> None:
    """OceanData CLI entry point."""
    ctx.obj = {"DEBUG": debug}
    if debug:
        typer.echo("Debug mode activated")


@app.command()
def hello(name: str = "World") -> None:
    """Simple greeting."""
    typer.echo(f"Hello, {name}!")


@app.command(name="add")
def add_cmd(x: float, y: float) -> None:
    """Add two numbers and output the result."""
    result = add(x, y)
    if result.is_integer():
        result = int(result)
    a = int(x) if float(x).is_integer() else x
    b = int(y) if float(y).is_integer() else y
    typer.echo(f"{a} + {b} = {result}")


@app.command()
def analyze(source_type: str) -> None:
    """Run analysis for the given data source using dummy data."""
    data = pd.DataFrame()
    result = run_analysis(data, source_type)
    typer.echo(result)


@app.command()
def publish(name: str, price: float) -> None:
    """Publish a dataset via the blockchain helper."""
    result = publish_dataset(name, {"title": name}, price)
    typer.echo(result)


@app.command()
def train(data_id: str) -> None:
    """Trigger Compute-to-Data training for a dataset."""
    result = train_model(data_id)
    typer.echo(result)


@app.command()
def sync() -> None:
    """Synchronize with the marketplace backend."""
    result = sync_marketplace()
    typer.echo(result)


def cli_main() -> None:
    """Entry point for poetry script."""
    app()


if __name__ == "__main__":
    cli_main()
