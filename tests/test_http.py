"""The hosted server: Streamable HTTP, stateless, at the root path, with the caller's key."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import httpx
import pytest
from conftest import Mock, serve_http, use_scenario
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

TOOLS = ["guide", "truth", "choice", "score", "number", "point", "box", "decide", "decide_many", "decide_video", "models"]
CALLER_KEY = "dn_caller_key_1"


def rpc(url: str, method: str, params: Optional[Dict[str, Any]] = None, key: Optional[str] = CALLER_KEY, **extra: str) -> httpx.Response:
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", **extra}
    if key is not None:
        headers["Authorization"] = "Bearer " + key
    body: Dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if not method.startswith("notifications/"):
        body["id"] = 1
    return httpx.post(url + "/", headers=headers, json=body, timeout=30)


def result(resp: httpx.Response) -> Dict[str, Any]:
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    return resp.json()


INIT = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}


def test_two_instances_answer_one_client_without_a_session(hosted: Mock) -> None:
    # Behind a load balancer each request may land on another replica. In stateless mode
    # initialize on one instance and every later request on the other must work.
    with serve_http() as a, serve_http() as b:
        init = rpc(a, "initialize", INIT)
        assert init.status_code == 200
        assert result(init)["result"]["serverInfo"]["name"] == "DecisionNode"
        assert "mcp-session-id" not in init.headers
        assert rpc(b, "notifications/initialized").status_code == 202

        for url in (b, a):
            listed = rpc(url, "tools/list")
            assert listed.status_code == 200
            assert [t["name"] for t in result(listed)["result"]["tools"]] == TOOLS

        called = rpc(b, "tools/call", {"name": "truth", "arguments": {"instructions": "Refund?", "state": "Refund me."}})
        data = result(called)["result"]
        assert data["isError"] is False
        assert set(data["structuredContent"]) == {"truth", "cost"}
        assert json.loads(data["content"][0]["text"]) == data["structuredContent"]


def test_the_callers_key_is_forwarded_and_never_the_servers(hosted: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECISIONNODE_API_KEY", "dn_server_environment_key")
    with serve_http() as url:
        for key in ("dn_caller_one", "dn_caller_two"):
            resp = rpc(url, "tools/call", {"name": "truth", "arguments": {"instructions": "?"}}, key=key)
            assert result(resp)["result"]["isError"] is False
    sent = [r["headers"]["authorization"] for r in hosted.requests()]
    assert sent == ["Bearer dn_caller_one", "Bearer dn_caller_two"]


def test_no_key_is_refused_at_the_transport(hosted: Mock) -> None:
    with serve_http() as url:
        for headers in ({}, {"Authorization": "Bearer"}, {"Authorization": "Basic abc"}):
            resp = httpx.post(
                url + "/",
                headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream", **headers},
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            )
            assert resp.status_code == 401, headers
    assert hosted.requests() == []


def test_a_key_the_api_does_not_know_is_its_401(hosted: Mock) -> None:
    with serve_http() as url:
        resp = rpc(url, "tools/call", {"name": "truth", "arguments": {"instructions": "?"}}, key="invalid_key")
    data = result(resp)["result"]
    assert data["isError"] is True
    assert data["structuredContent"]["status"] == 401
    assert data["structuredContent"]["error_type"] == "authentication_error"


def test_the_hosted_server_never_reads_a_path(hosted: Mock, suite_dir: Any) -> None:
    path = str(suite_dir / "fixtures" / "images" / "pixel.jpg")
    with serve_http() as url:
        resp = rpc(
            url,
            "tools/call",
            {"name": "truth", "arguments": {"instructions": "?", "images": [{"id": "a", "path": path}]}},
        )
    data = result(resp)["result"]
    assert data["isError"] is True
    assert "local server" in data["structuredContent"]["message"]
    assert hosted.requests() == []


def test_health(hosted: Mock) -> None:
    with serve_http() as url:
        resp = httpx.get(url + "/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


async def test_mcp_client_over_http(hosted: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    with serve_http() as url:
        transport = StreamableHttpTransport(url + "/", headers={"Authorization": "Bearer " + CALLER_KEY})
        async with Client(transport) as client:
            assert client.initialize_result is not None
            assert "guide" in (client.initialize_result.instructions or "")
            tools = await client.list_tools()
            assert [t.name for t in tools] == TOOLS
            guide = await client.call_tool("guide", {})
            assert "Pick the tool" in guide.structured_content["guide"]
            ok = await client.call_tool(
                "choice", {"instructions": "Topic?", "options": {"billing": "money", "bug": "broken"}, "state": "x"}
            )
            assert ok.structured_content["choice"] in ("billing", "bug")
            many = await client.call_tool(
                "decide_many",
                {"items": [{"id": "a", "state": "x"}, {"id": "b", "state": "y"}], "questions": {"q": {"type": "truth", "instructions": "?"}}},
            )
            assert list(many.structured_content["results"]) == ["a", "b"]
            models = await client.call_tool("models", {})
            assert len(models.structured_content["models"]) == 4
            # A fixed rule the SDK checks before sending: the API's own refusal, with no status
            # and no request id, since nothing reached the API.
            sent = len(hosted.requests())
            bad = await client.call_tool("score", {"instructions": "?", "levels": [str(i) for i in range(11)]}, raise_on_error=False)
            assert bad.is_error and bad.structured_content["status"] is None and bad.structured_content["request_id"] is None
            assert len(hosted.requests()) == sent
            # An error the API answers arrives with its status and request id.
            use_scenario(hosted, monkeypatch, "error/400-max-tokens-exceeded")
            refused = await client.call_tool("truth", {"instructions": "?"}, raise_on_error=False)
            assert refused.is_error and refused.structured_content["status"] == 400
            assert refused.structured_content["error_type"] == "max_tokens_exceeded" and refused.structured_content["request_id"]


def test_a_rejected_key_is_not_sent_again(hosted: Mock) -> None:
    with serve_http() as url:
        for _ in range(3):
            resp = rpc(url, "tools/call", {"name": "truth", "arguments": {"instructions": "?"}}, key="invalid_key")
            data = result(resp)["result"]
            assert data["isError"] is True and data["structuredContent"]["status"] == 401
    assert len(hosted.requests()) == 1


# --------------------------------------------------------------------------- body size

MIB = 1024 * 1024


def test_the_installed_mcp_library_limits_the_body_to_4_mib(hosted: Mock) -> None:
    # The image limit over HTTP (3 MiB of base64) assumes
    # the transport refuses anything over 4 MiB. Installing from uv.lock keeps that
    # version; this test catches an upgrade that drops or raises the limit.
    from mcp.server import transport_security

    assert transport_security.DEFAULT_MAX_REQUEST_BODY_SIZE == 4 * MIB
    images = [{"id": "a", "data": "A" * (4 * MIB + 16), "media_type": "image/png"}]
    with serve_http() as url:
        resp = rpc(url, "tools/call", {"name": "truth", "arguments": {"instructions": "?", "images": images}})
    assert resp.status_code == 413
    assert hosted.requests() == []


def test_image_data_between_3_and_4_mib_gets_a_clear_tool_error(hosted: Mock) -> None:
    images = [{"id": "a", "data": "A" * (3 * MIB + 16), "media_type": "image/png"}]
    with serve_http() as url:
        resp = rpc(url, "tools/call", {"name": "truth", "arguments": {"instructions": "?", "images": images}})
    assert resp.status_code == 200
    data = result(resp)["result"]
    assert data["isError"] is True
    assert "https 'url'" in data["structuredContent"]["message"]
    assert hosted.requests() == []
