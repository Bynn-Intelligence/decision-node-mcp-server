.PHONY: deps-dev test run run-http

deps-dev:
	uv sync

test:
	uv run pytest

# Local server over stdio (what an MCP client launches)
run:
	uv run bynn-decisionnode-mcp

# Streamable HTTP on http://127.0.0.1:8000/
run-http:
	uv run bynn-decisionnode-mcp --transport http
