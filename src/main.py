"""Command line interface for the OceanData SDK."""

from __future__ import annotations

import json
import pandas as pd
import requests
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
from src.auth import AuthManager

load_dotenv()

app = typer.Typer(add_completion=True)


def _echo(ctx: typer.Context, value: object) -> None:
    """Output value respecting --json flag."""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if ctx.obj.get("JSON"):
        typer.echo(json.dumps(value))
    else:
        typer.echo(value)


@app.callback()
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
    token: str | None = typer.Option(None, "--token", help="Access token"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """OceanData CLI entry point."""
    auth = AuthManager(token)
    ctx.obj = {"DEBUG": debug, "JSON": json_output, "AUTH": auth}
    if debug:
        typer.echo("Debug mode activated")


@app.command()
def hello(ctx: typer.Context, name: str = "World") -> None:
    """Simple greeting."""
    _echo(ctx, f"Hello, {name}!")


@app.command(name="add")
def add_cmd(ctx: typer.Context, x: float, y: float) -> None:
    """Add two numbers and output the result."""
    result = add(x, y)
    if result.is_integer():
        result = int(result)
    a = int(x) if float(x).is_integer() else x
    b = int(y) if float(y).is_integer() else y
    _echo(ctx, f"{a} + {b} = {result}")


@app.command()
def analyze(ctx: typer.Context, source_type: str) -> None:
    """Run analysis for the given data source using dummy data."""
    data = pd.DataFrame()
    result = run_analysis(data, source_type)
    _echo(ctx, result)


@app.command()
def publish(ctx: typer.Context, name: str, price: float) -> None:
    """Publish a dataset via the blockchain helper."""
    result = publish_dataset(name, {"title": name}, price)
    _echo(ctx, result)


@app.command()
def train(ctx: typer.Context, data_id: str) -> None:
    """Trigger Compute-to-Data training for a dataset."""
    result = train_model(data_id)
    _echo(ctx, result)


@app.command()
def sync(ctx: typer.Context) -> None:
    """Synchronize with the marketplace backend."""
    result = sync_marketplace()
    _echo(ctx, result)


@app.command()
def whoami(ctx: typer.Context) -> None:
    """Display current user identity."""
    identity = get_user_identity()
    _echo(ctx, identity)


@app.command()
def register(ctx: typer.Context, name: str, schema: str, version: str) -> None:
    """Register a model."""
    info = register_model(name, schema, version)
    _echo(ctx, info)


@app.command()
def evaluate(
    ctx: typer.Context,
    model_id: str,
    dataset_id: str,
    webhook: str | None = typer.Option(None, "--webhook", help="Send result to URL"),
) -> None:
    """Evaluate a model."""
    with Progress() as progress:
        task = progress.add_task("evaluating", total=1)
        result = evaluate_model(model_id, dataset_id)
        progress.update(task, advance=1)
    if webhook:
        headers = ctx.obj["AUTH"].inject_auth_header({"Content-Type": "application/json"})
        requests.post(webhook, headers=headers, data=json.dumps(result.model_dump()))
    _echo(ctx, result)


@app.command()
def results(ctx: typer.Context, model_id: str) -> None:
    """Retrieve evaluation outputs."""
    try:
        result = retrieve_model_outputs(model_id)
        _echo(ctx, result)
    except KeyError:
        _echo(ctx, f"No results for model {model_id}")


@app.command()
def api(ctx: typer.Context, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the REST API server."""
    from src.api import app as api_app, save_openapi
    import uvicorn

    save_openapi()
    uvicorn.run(api_app, host=host, port=port, log_level="info")  # pragma: no cover


def cli_main() -> None:
    """Entry point for poetry script."""
    app()  # pragma: no cover


if __name__ == "__main__":
    cli_main()  # pragma: no cover
