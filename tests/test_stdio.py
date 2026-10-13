"""The local server: the installed entry point over stdio, as an MCP client launches it."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from conftest import TEST_KEY, Mock
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from decisionnode_mcp import texts

TOOLS = ["guide", "truth", "choice", "score", "number", "point", "box", "decide", "decide_many", "decide_video", "models"]


def transport(env: dict) -> StdioTransport:
    entry = shutil.which("bynn-decisionnode-mcp", path=str(Path(sys.executable).parent))
    assert entry, "the bynn-decisionnode-mcp entry point is not installed"
    base = {k: v for k, v in os.environ.items() if not k.startswith("DECISIONNODE_")}
    return StdioTransport(command=entry, args=[], env={**base, **env})


async def test_stdio_by_default(mock: Mock, suite_dir: Path) -> None:
    env = {"DECISIONNODE_API_KEY": TEST_KEY, "DECISIONNODE_BASE_URL": mock.url}
    async with Client(transport(env)) as client:
        assert client.initialize_result is not None
        assert client.initialize_result.instructions == texts.INSTRUCTIONS
        tools = await client.list_tools()
        assert [t.name for t in tools] == TOOLS
        truth = await client.call_tool("truth", {"instructions": "Refund?", "state": "Refund me."})
        assert set(truth.structured_content) == {"truth", "cost"}
        # The local server reads an image from the user's disk.
        image = {"id": "r", "path": str(suite_dir / "fixtures" / "images" / "pixel.png")}
        withimg = await client.call_tool("truth", {"instructions": "Readable?", "images": [image]})
        assert not withimg.is_error
        models = await client.call_tool("models", {})
        assert len(models.structured_content["models"]) == 4


async def test_stdio_without_a_key(mock: Mock) -> None:
    async with Client(transport({"DECISIONNODE_BASE_URL": mock.url})) as client:
        result = await client.call_tool("truth", {"instructions": "?"}, raise_on_error=False)
        assert result.is_error
        assert result.structured_content["error_type"] == "authentication_error"
