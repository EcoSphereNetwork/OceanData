from __future__ import annotations

"""Unit tests for the CLI."""

from typer.testing import CliRunner
from pathlib import Path
import importlib.util
import sys
import ast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

main_path = ROOT / "src" / "main.py"
spec = importlib.util.spec_from_file_location("cli_main", main_path)
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)
app = main.app


def test_hello_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["hello"])
    assert result.exit_code == 0
    assert "Hello" in result.output


def test_add_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["add", "2", "3"])
    assert result.exit_code == 0
    assert "= 5" in result.output


def test_sync_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["sync"])
    assert result.exit_code == 0
    assert "success" in result.output


def test_register_and_results() -> None:
    runner = CliRunner()
    reg = runner.invoke(app, ["register", "test", "schema", "1.0"])
    assert reg.exit_code == 0
    data = ast.literal_eval(reg.output)
    model_id = data["model_id"]
    eval_res = runner.invoke(app, ["evaluate", model_id, "data"])
    assert eval_res.exit_code == 0
    res = runner.invoke(app, ["results", model_id])
    assert res.exit_code == 0
    assert "accuracy" in res.output


def test_results_missing() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["results", "missing"])
    assert result.exit_code == 0
    assert "No results" in result.output
