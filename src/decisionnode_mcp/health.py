"""`GET /health`, the liveness and readiness probe of the hosted server. No key needed."""

from __future__ import annotations

from datetime import datetime, timezone

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import __version__


def add_health_endpoint(mcp: FastMCP) -> None:
    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "healthy",
                "version": __version__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
