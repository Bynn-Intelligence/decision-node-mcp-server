"""The caller's key never reaches a log: stderr, and the decisionnode and fastmcp loggers at DEBUG."""

from __future__ import annotations

import io
import logging
from typing import Iterator, List

import pytest
from conftest import Mock, serve_http
from fastmcp import Client
from test_http import result, rpc

from decisionnode_mcp import api
from decisionnode_mcp.server import create_server

SECRET_KEY = "dn_live_key_that_must_never_be_logged_7f3a9c"
# The mock API answers 401 for a key that starts with "invalid".
REJECTED_KEY = "invalid_dn_key_rejected_4e81c2d9"
LOGGERS = ("decisionnode", "fastmcp", "mcp", "httpx", "httpcore", "uvicorn", "uvicorn.access", "uvicorn.error")

QUESTIONS = {"q": {"type": "truth", "instructions": "?"}}
CALLS = [
    ("truth", {"instructions": "?", "state": "s"}),
    ("choice", {"instructions": "?", "options": ["a", "b"]}),
    ("score", {"instructions": "?", "levels": ["low", "high"]}),
    ("number", {"instructions": "?", "min": 0, "max": 9}),
    ("point", {"instructions": "?", "images": [{"id": "a", "url": "https://example.com/a.jpg"}]}),
    ("box", {"instructions": "?", "images": [{"id": "a", "url": "https://example.com/a.jpg"}]}),
    ("decide", {"questions": QUESTIONS}),
    ("decide_many", {"items": [{"id": "1"}, {"id": "2"}], "questions": QUESTIONS}),
    ("decide_video", {"questions": QUESTIONS, "videos": [{"id": "v", "frames": [{"t": 0, "media_type": "image/jpeg", "data": "/9j/"}]}]}),
    ("models", {}),
    ("truth", {"instructions": "?", "images": [{"id": "a", "url": "http://example.com/a.jpg"}]}),
]


@pytest.fixture
def captured_logs() -> Iterator[io.StringIO]:
    """Every record of the named loggers, and of the root logger, at DEBUG, into one buffer."""
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s %(args)s"))
    loggers: List[logging.Logger] = [logging.getLogger(name) for name in LOGGERS] + [logging.getLogger()]
    saved = [(lg, lg.level, lg.propagate) for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.DEBUG)
        lg.addHandler(handler)
        lg.propagate = False
    try:
        yield buffer
    finally:
        for lg, level, propagate in saved:
            lg.removeHandler(handler)
            lg.setLevel(level)
            lg.propagate = propagate


def assert_no_key(text: str, *keys: str) -> None:
    for key in keys:
        assert key not in text


async def test_local_tools_never_log_the_key(
    local: Mock, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str], captured_logs: io.StringIO
) -> None:
    monkeypatch.setenv("DECISIONNODE_API_KEY", SECRET_KEY)
    monkeypatch.setattr(api.settings, "rate_limit_per_minute", 3)
    async with Client(create_server()) as client:
        for name, args in CALLS + CALLS:  # the second round also hits the rate limit
            await client.call_tool(name, args, raise_on_error=False)
    monkeypatch.setenv("DECISIONNODE_API_KEY", REJECTED_KEY)
    async with Client(create_server()) as client:
        for _ in range(2):  # a 401 from the API, then the remembered refusal
            await client.call_tool("truth", {"instructions": "?"}, raise_on_error=False)
    out, err = capfd.readouterr()
    logs = captured_logs.getvalue()
    assert logs  # the loggers were live at DEBUG
    assert_no_key(out + err + logs, SECRET_KEY, REJECTED_KEY)


def test_hosted_server_never_logs_the_key(
    hosted: Mock, capfd: pytest.CaptureFixture[str], captured_logs: io.StringIO
) -> None:
    with serve_http() as url:
        for name, args in CALLS:
            resp = rpc(url, "tools/call", {"name": name, "arguments": args}, key=SECRET_KEY)
            assert "result" in result(resp)
        for _ in range(2):
            rpc(url, "tools/call", {"name": "truth", "arguments": {"instructions": "?"}}, key=REJECTED_KEY)
    sent = [r["headers"]["authorization"] for r in hosted.requests()]
    assert "Bearer " + SECRET_KEY in sent  # the keys did travel, to the API only
    assert "Bearer " + REJECTED_KEY in sent
    out, err = capfd.readouterr()
    logs = captured_logs.getvalue()
    # uvicorn's own loggers stayed at DEBUG while the server ran: its access log is there.
    assert "uvicorn.access" in logs
    assert_no_key(out + err + logs, SECRET_KEY, REJECTED_KEY)
