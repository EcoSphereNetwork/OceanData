"""Command line interface for the OceanData SDK."""
from __future__ import annotations

import pandas as pd
import typer
from dotenv import load_dotenv
from rich.progress import Progress

from ocean_sdk import (
    publish_dataset,
    run_analysis,
    sync_marketplace,
    train_model,
    get_user_identity,
    register_model,
    evaluate_model,
    retrieve_model_outputs,
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


@app.command()
def whoami() -> None:
    """Display current user identity."""
    identity = get_user_identity()
    typer.echo(identity.model_dump())


@app.command()
def register(name: str, schema: str, version: str) -> None:
    """Register a model."""
    info = register_model(name, schema, version)
    typer.echo(info.model_dump())


@app.command()
def evaluate(model_id: str, dataset_id: str) -> None:
    """Evaluate a model."""
    with Progress() as progress:
        task = progress.add_task("evaluating", total=1)
        result = evaluate_model(model_id, dataset_id)
        progress.update(task, advance=1)
    typer.echo(result.model_dump())


@app.command()
def results(model_id: str) -> None:
    """Retrieve evaluation outputs."""
    try:
        result = retrieve_model_outputs(model_id)
        typer.echo(result.model_dump())
    except KeyError:
        typer.echo(f"No results for model {model_id}")


def cli_main() -> None:
    """Entry point for poetry script."""
    app()


if __name__ == "__main__":
    cli_main()
