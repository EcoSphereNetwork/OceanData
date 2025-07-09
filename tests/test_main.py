"""Unit tests for the CLI."""
from typer.testing import CliRunner

from src.main import app


def test_hello_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["hello"])
    assert result.exit_code == 0
    assert "Hello, World!" in result.output


def test_add_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["add", "2", "3"])
    assert result.exit_code == 0
    assert "2 + 3 = 5" in result.output


def test_sync_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["sync"])
    assert result.exit_code == 0
    assert "success" in result.output
