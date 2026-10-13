"""The connection credential of the hosted server: a DecisionNode API key, passed through.

Every HTTP request to the MCP endpoint carries the caller's key as
`Authorization: Bearer <key>`. The server does not judge the key: it forwards it with
each API call, and the API, the one judge of a key, answers 401 for a key it does not
know. The transport only refuses a request that carries no bearer value at all, or one
that cannot be a key (empty, or with characters an HTTP header cannot carry), so a
client without a key learns it at connect time instead of on its first tool call.

Nothing about the key is stored or logged.

A sign-in for clients that cannot send a header (OAuth 2.1) would verify its access
tokens here, before any tool runs, the way this module checks the key's form.
"""

from __future__ import annotations

from typing import Optional

from fastmcp.server.auth import AccessToken, TokenVerifier

CLIENT_ID = "decisionnode-api-key"


def plausible_key(token: str) -> bool:
    """A key is plain visible ASCII without whitespace."""
    return bool(token) and len(token) <= 512 and all(33 <= ord(c) <= 126 for c in token)


class ApiKeyPassThrough(TokenVerifier):
    """Accept any well-formed bearer value; the DecisionNode API authenticates it on every call."""

    def __init__(self) -> None:
        super().__init__(base_url=None, required_scopes=None)

    async def verify_token(self, token: str) -> Optional[AccessToken]:
        if not plausible_key(token):
            return None
        return AccessToken(token=token, client_id=CLIENT_ID, scopes=[])


def describe() -> str:
    """The one startup line naming the auth mode."""
    return "Auth: API key pass-through; the caller's bearer key is forwarded to the DecisionNode API, which checks it"
