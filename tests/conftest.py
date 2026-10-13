"""Shared fixtures: the conformance suite's mock API server, the MCP server in either
mode, and helpers to read what reached the mock.

The tests never call a real endpoint. They need a checkout of the DecisionNode
conformance suite: set DECISIONNODE_CONFORMANCE_DIR, or check it out beside this
repository as `decision-node-conformance`.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pytest
import uvicorn

from decisionnode_mcp import api
from decisionnode_mcp.server import create_server

ROOT = Path(__file__).resolve().parent.parent
TEST_KEY = "dn_test_mcp_key"


def conformance_dir() -> Optional[Path]:
    configured = os.environ.get("DECISIONNODE_CONFORMANCE_DIR")
    candidates = [Path(configured)] if configured else [ROOT.parent / "decision-node-conformance"]
    for path in candidates:
        if (path / "mock" / "server.py").is_file():
            return path
    return None


@pytest.fixture(scope="session")
def suite_dir() -> Path:
    path = conformance_dir()
    if path is None:
        pytest.fail(
            "The conformance suite was not found. Set DECISIONNODE_CONFORMANCE_DIR to a checkout of it.",
            pytrace=False,
        )
    return path


@pytest.fixture(scope="session")
def mock_url(suite_dir: Path) -> Iterator[str]:
    proc = subprocess.Popen(
        [sys.executable, str(suite_dir / "mock" / "server.py"), "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    assert line.startswith("listening on "), line
    url = line[len("listening on ") :]
    try:
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


class Mock:
    """The mock API, seen from one test: a fresh session id and the requests it recorded."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.session = "mcp-" + uuid.uuid4().hex

    def headers(self, scenario: Optional[str] = None) -> Dict[str, str]:
        h = {"x-mock-session": self.session}
        if scenario:
            h["x-mock-scenario"] = scenario
        return h

    def requests(self) -> List[Dict[str, Any]]:
        with urllib.request.urlopen("%s/_mock/requests?session=%s" % (self.url, self.session)) as resp:
            return json.loads(resp.read())["requests"]

    def max_in_flight(self) -> int:
        with urllib.request.urlopen("%s/_mock/requests?session=%s" % (self.url, self.session)) as resp:
            return int(json.loads(resp.read())["max_in_flight"])

    def add_scenario(self, scenario: Dict[str, Any]) -> str:
        req = urllib.request.Request(
            self.url + "/_mock/scenarios",
            data=json.dumps(scenario).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            resp.read()
        return scenario["name"]


@pytest.fixture
def mock(mock_url: str, monkeypatch: pytest.MonkeyPatch) -> Mock:
    for name in ("DECISIONNODE_API_KEY", "DECISIONNODE_BASE_URL", "DECISIONNODE_MODEL", "DECISIONNODE_MCP_RATE_LIMIT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DECISIONNODE_BASE_URL", mock_url)
    m = Mock(mock_url)
    monkeypatch.setattr(api.settings, "extra_headers", m.headers())
    monkeypatch.setattr(api.settings, "rate_limit_per_minute", 0)
    monkeypatch.setattr(api, "rejected_keys", api._RejectedKeys())
    return m


def use_scenario(mock: Mock, monkeypatch: pytest.MonkeyPatch, scenario: str) -> None:
    monkeypatch.setattr(api.settings, "extra_headers", mock.headers(scenario))


@pytest.fixture
def local(mock: Mock, monkeypatch: pytest.MonkeyPatch) -> Mock:
    """The server as the stdio entry point configures it: key from the environment."""
    monkeypatch.setenv("DECISIONNODE_API_KEY", TEST_KEY)
    monkeypatch.setattr(api.settings, "local", True)
    return mock


@pytest.fixture
def hosted(mock: Mock, monkeypatch: pytest.MonkeyPatch) -> Mock:
    """The server as the HTTP entry point configures it: key from each request."""
    monkeypatch.setattr(api.settings, "local", False)
    return mock


# --------------------------------------------------------------------------- HTTP


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def serve_http() -> Iterator[str]:
    """One instance of the hosted server, as `main --transport http` builds it, on a real port."""
    app = create_server().http_app(path="/", stateless_http=True)
    port = free_port()
    # log_config and log_level None: uvicorn leaves the logging configuration alone, so a
    # test that sets loggers to DEBUG (test_key_logging.py) keeps them at DEBUG.
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, log_level=None, lifespan="on")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("the HTTP server did not start")
        time.sleep(0.02)
    try:
        yield "http://127.0.0.1:%d" % port
    finally:
        server.should_exit = True
        thread.join(timeout=10)
