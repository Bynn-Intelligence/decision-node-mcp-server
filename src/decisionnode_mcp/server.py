"""Builds the MCP server: the instructions, the auth, the mounted tool groups and `/health`."""

from __future__ import annotations

from fastmcp import FastMCP

from . import auth, decisions, guide, models, questions, texts, videos
from .health import add_health_endpoint

NAME = "DecisionNode"

# Each tool group registers its tools on its own FastMCP instance. Importing a module
# does not expose its tools: every group must be mounted onto the main server, or
# tools/list returns no tools while the server otherwise looks healthy.
TOOL_GROUPS = (guide.mcp, questions.mcp, decisions.mcp, videos.mcp, models.mcp)


def create_server() -> FastMCP:
    """A new main server. The tool groups are shared; the server keeps no state between requests."""
    server = FastMCP(NAME, instructions=texts.INSTRUCTIONS, auth=auth.ApiKeyPassThrough())
    for group in TOOL_GROUPS:
        server.mount(group)
    add_health_endpoint(server)
    return server
