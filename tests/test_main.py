from __future__ import annotations

"""Unit tests for the CLI."""

from typer.testing import CliRunner
from pathlib import Path
import sys
import ast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.main import app


def _run_cli(args):
    runner = CliRunner()
    return runner.invoke(app, args)


def test_hello_command() -> None:
    result = _run_cli(["hello"])
    assert result.exit_code == 0
    assert "Hello, World!" in result.output


def test_add_command() -> None:
    result = _run_cli(["add", "2", "3"])
    assert result.exit_code == 0
    assert "2 + 3 = 5" in result.output


def test_sync_command() -> None:
    result = _run_cli(["sync"])
    assert result.exit_code == 0
    assert "success" in result.output


def test_register_and_results() -> None:
    reg = _run_cli(["register", "test", "schema", "1.0"])
    assert reg.exit_code == 0
    data = ast.literal_eval(reg.output)
    model_id = data["model_id"]
    eval_res = _run_cli(["evaluate", model_id, "data"])
    assert eval_res.exit_code == 0
    res = _run_cli(["results", model_id])
    assert res.exit_code == 0
    assert "accuracy" in res.output


def test_results_missing() -> None:
    result = _run_cli(["results", "missing"])
    assert result.exit_code == 0
    assert "No results" in result.output


def test_json_flag() -> None:
    result = _run_cli(["--json", "hello"])
    assert result.exit_code == 0
    assert result.output.strip() == '"Hello, World!"'


def test_auxiliary_commands() -> None:
    assert _run_cli(["analyze", "dummy"]).exit_code == 0
    assert _run_cli(["publish", "x", "1"]).exit_code == 0
    assert _run_cli(["train", "data_id"]).exit_code == 0
    assert _run_cli(["whoami"]).exit_code == 0


def test_debug_flag() -> None:
    result = _run_cli(["--debug", "hello"])
    assert result.exit_code == 0
    assert "Debug mode activated" in result.output


def test_webhook_evaluate() -> None:
    import http.server
    import threading
    import json as js

    received: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # type: ignore[override]
            length = int(self.headers.get("Content-Length", "0"))
            data = self.rfile.read(length).decode()
            received.append(js.loads(data))
            self.send_response(200)
            self.end_headers()

    server = http.server.HTTPServer(("localhost", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    port = server.server_address[1]

    reg = _run_cli(["register", "hook", "schema", "1.0"])
    model_id = ast.literal_eval(reg.output)["model_id"]
    res = _run_cli(
        ["--token", "abc", "evaluate", model_id, "data", "--webhook", f"http://localhost:{port}"]
    )
    server.shutdown()
    thread.join()
    assert res.exit_code == 0
    assert received and received[0]["model_id"] == model_id
