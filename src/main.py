"""Command line interface for OceanData template."""

from __future__ import annotations

import click
from dotenv import load_dotenv

from src.modules.calculator import add

load_dotenv()

@click.group()
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """OceanData CLI entry point."""
    ctx.ensure_object(dict)
    ctx.obj["DEBUG"] = debug
    if debug:
        click.echo("Debug mode activated")

@cli.command()
@click.option("--name", default="World", help="Name to greet")
def hello(name: str) -> None:
    """Simple greeting."""
    click.echo(f"Hello, {name}!")

@cli.command()
@click.argument("x", type=float)
@click.argument("y", type=float)
def add_cmd(x: float, y: float) -> None:
    """Add two numbers and output the result."""
    result = add(x, y)
    click.echo(f"{x} + {y} = {result}")


def main() -> None:
    """Run the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
