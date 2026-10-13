"""Entry point: `bynn-decisionnode-mcp` runs over stdio by default, `--transport http` hosted.

stdio      launched by an MCP client on the user's machine; the key comes from
           DECISIONNODE_API_KEY, and images and video files may be given by file path.
http       Streamable HTTP, stateless, at the root path (`MCP_PATH`, default `/`), with
           `GET /health`. Each request carries the caller's key in `Authorization`.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import __version__, api, auth
from .server import create_server

HTTP_TRANSPORTS = ("http", "streamable-http")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bynn-decisionnode-mcp", description="The DecisionNode MCP server.")
    p.add_argument(
        "--transport",
        choices=("stdio",) + HTTP_TRANSPORTS,
        default=os.environ.get("MCP_TRANSPORT", "stdio"),
        help="stdio (default) for a local client, http for the hosted server",
    )
    p.add_argument("--host", default=os.environ.get("MCP_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("MCP_PORT", "8000")))
    p.add_argument("--path", default=os.environ.get("MCP_PATH", "/"))
    p.add_argument("--version", action="version", version="%(prog)s " + __version__)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _parser().parse_args(argv)
    local = args.transport == "stdio"
    try:
        api.configure(local=local)
    except api.ConfigurationError as exc:
        print("bynn-decisionnode-mcp: %s" % exc, file=sys.stderr)
        sys.exit(2)
    server = create_server()
    base_url = os.environ.get("DECISIONNODE_BASE_URL") or "https://api.decisionnode.com (default)"

    # stdout carries the protocol on stdio, so everything for people goes to stderr.
    log = sys.stderr
    if local:
        print("DecisionNode MCP server %s on stdio; API %s" % (__version__, base_url), file=log)
        server.run(transport="stdio", show_banner=False)
        return

    print(
        "DecisionNode MCP server %s on http://%s:%d at path %r; API %s"
        % (__version__, args.host, args.port, args.path, base_url),
        file=log,
    )
    print(auth.describe(), file=log)
    if api.settings.rate_limit_per_minute:
        print("Rate limit: %d requests a minute per key on this instance" % api.settings.rate_limit_per_minute, file=log)
    # stateless_http is required behind a load balancer with several replicas: without it
    # the session made by `initialize` lives in one process's memory, and a follow-up
    # request routed to another replica is refused ("No valid session ID provided"). It is
    # set here, at run(), where FastMCP 2.x reads it.
    server.run(
        transport="streamable-http",
        host=args.host,
        port=args.port,
        path=args.path,
        stateless_http=True,
        show_banner=False,
    )


if __name__ == "__main__":
    main()
