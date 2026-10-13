"""The `guide` tool: how to use the other tools well. No API call."""

from __future__ import annotations

from mcp.types import TextContent
from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult

from . import schemas, texts

mcp = FastMCP("DecisionNode guide")


@mcp.tool(
    name="guide",
    description=texts.GUIDE_DESCRIPTION,
    annotations=schemas.annotations("Guide"),
    output_schema=schemas.GUIDE_OUTPUT,
)
def guide() -> ToolResult:
    return ToolResult(
        content=[TextContent(type="text", text=texts.GUIDE)],
        structured_content={"guide": texts.GUIDE},
    )
