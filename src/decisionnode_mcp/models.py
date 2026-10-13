"""The `models` tool: the models a key may use, with their limits."""

from __future__ import annotations

from typing import Any, Dict

from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult

from . import api, schemas, texts

mcp = FastMCP("DecisionNode models")


@mcp.tool(
    name="models",
    description=texts.MODELS_DESCRIPTION,
    annotations=schemas.annotations("Models"),
    output_schema=schemas.MODELS_OUTPUT,
)
async def models() -> ToolResult:
    async def call() -> Dict[str, Any]:
        async with api.client() as dn:
            entries = await dn.models()
        return {"models": [dict(m.raw) for m in entries]}

    return await api.run(call)
