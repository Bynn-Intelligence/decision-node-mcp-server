"""Calls to the DecisionNode API, made through the Python SDK, and the shape of tool results.

Where the API key comes from depends on how the server runs:

- Hosted (HTTP transport): the caller's own key, from the `Authorization: Bearer <key>`
  header of the MCP request, read with FastMCP's `get_http_headers()`. The server never
  falls back to a key of its own, so one caller can never spend another's key.
- Local (stdio transport): `DECISIONNODE_API_KEY` from the environment the client
  launched the server with.

Nothing is stored: a client is made for the call and closed after it.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
import time
import urllib.parse
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Deque, Dict, List, Mapping, Optional, Tuple

import decisionnode
import mcp.types

# The SDK's own decide body, transport and parser, for what its decide() does not take yet (decide below).
from decisionnode import _base as sdk_base
from decisionnode._types import parse_decision
from decisionnode import (
    AsyncDecisionNode,
    AuthenticationError,
    BadRequestError,
    ConnectionError,
    Decision,
    DecisionNodeError,
    Image,
    InsufficientCreditError,
    NotFoundError,
    OverloadedError,
    RateLimitError,
    RefusalError,
    ServerError,
    TimeoutError,
    ValidationError,
)
from fastmcp.server.dependencies import get_http_headers
from fastmcp.tools.tool import ToolResult

from .schemas import PLACE_TYPES

# --------------------------------------------------------------------------- settings

# Requests per minute per API key on each instance of the hosted server. The limit is
# kept per process, so the effective limit across a deployment is this value times the
# number of replicas.
DEFAULT_RATE_LIMIT_PER_MINUTE = 120


@dataclass
class Settings:
    """How this process runs. `main` sets `local` for the stdio transport.

    The defaults are the hosted ones, so a server started any other way neither reads
    files from its own disk nor uses a key from its own environment.
    """

    local: bool = False
    # Extra headers sent with every API request (a deployment's own routing headers, or
    # a test harness's). Never a key.
    extra_headers: Dict[str, str] = field(default_factory=dict)
    # Requests per minute per API key on this instance; 0 turns the limit off. On by
    # default for the hosted server; the local server, which serves one user's own key,
    # starts with it off.
    rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE


settings = Settings()


class ConfigurationError(Exception):
    """A setting the server refuses to start with. `main` prints it and exits."""


_LOOPBACK_HOSTS = ("localhost",)


def _is_loopback(host: str) -> bool:
    host = host.strip("[]").lower()
    if host in _LOOPBACK_HOSTS or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_base_url(value: str) -> None:
    """The API address must be https; plain http is allowed only to this machine.

    Every call carries an API key, and plain http would send it in clear over the network.
    """
    parts = urllib.parse.urlsplit(value.strip())
    scheme = parts.scheme.lower()
    if scheme == "https" and parts.hostname:
        return
    if scheme == "http" and parts.hostname and _is_loopback(parts.hostname):
        return
    if scheme == "http":
        raise ConfigurationError(
            "DECISIONNODE_BASE_URL must use https: plain http would send API keys in clear. "
            "http is accepted only for a loopback address (localhost, 127.0.0.1, ::1)."
        )
    raise ConfigurationError("DECISIONNODE_BASE_URL must be an https URL, got %r." % value)


def configure(*, local: bool, env: Mapping[str, str] = os.environ) -> None:
    """Read the environment. Raises ConfigurationError for a setting the server cannot run with."""
    base_url = (env.get("DECISIONNODE_BASE_URL") or "").strip()
    if base_url:
        check_base_url(base_url)
    default = 0 if local else DEFAULT_RATE_LIMIT_PER_MINUTE
    raw = (env.get("DECISIONNODE_MCP_RATE_LIMIT") or "").strip()
    if raw:
        try:
            limit = int(raw)
        except ValueError:
            limit = -1
        if limit < 0:
            raise ConfigurationError(
                "DECISIONNODE_MCP_RATE_LIMIT must be a whole number of requests a minute, 0 or more "
                "(0 turns the limit off), got %r." % raw
            )
    else:
        limit = default
    settings.local = local
    settings.rate_limit_per_minute = limit


# --------------------------------------------------------------------------- errors


class ToolFailure(Exception):
    """A failure the tool reports as an `isError` result rather than raising."""

    def __init__(self, error_type: str, message: str, *, status: Optional[int] = None, **extra: Any) -> None:
        super().__init__(message)
        self.payload: Dict[str, Any] = {
            "error_type": error_type,
            "message": message,
            "reason": None,
            "status": status,
            "request_id": None,
        }
        self.payload.update(extra)


# The error_type to report when the API's body carries none (a plain-text 400, a 5xx
# without a body, no response at all). A 403 that is not a refusal is
# PermissionDeniedError in SDK versions that have it.
_PERMISSION_DENIED = getattr(decisionnode, "PermissionDeniedError", None)

_FALLBACK_TYPES = ([(_PERMISSION_DENIED, "permission_error")] if _PERMISSION_DENIED is not None else []) + [
    (RefusalError, "refusal_error"),
    (ValidationError, "validation_error"),
    (BadRequestError, "invalid_request_error"),
    (AuthenticationError, "authentication_error"),
    (InsufficientCreditError, "insufficient_credit_error"),
    (NotFoundError, "not_found_error"),
    (RateLimitError, "rate_limit_error"),
    (OverloadedError, "overloaded_error"),
    (ServerError, "api_error"),
    (TimeoutError, "timeout_error"),
    (ConnectionError, "connection_error"),
]


def _describe(exc: DecisionNodeError) -> str:
    """A message for an error whose body carries none, without the request id (a field of its own)."""
    if isinstance(exc, ValidationError) and exc.entries:
        parts = []
        for entry in exc.entries[:5]:
            if isinstance(entry, Mapping):
                loc = ".".join(str(p) for p in entry.get("loc") or [])
                parts.append("%s: %s" % (loc, entry.get("msg")) if loc else str(entry.get("msg")))
        more = " (and %d more)" % (len(exc.entries) - 5) if len(exc.entries) > 5 else ""
        if parts:
            return "; ".join(parts) + more
    if exc.status is not None:
        doc = (type(exc).__doc__ or "").strip().splitlines()
        return doc[0] if doc else "HTTP %d" % exc.status
    return str(exc)


def error_payload(exc: DecisionNodeError) -> Dict[str, Any]:
    """The error fields an agent reads to correct its call."""
    error_type = exc.error_type
    if not error_type:
        error_type = next((name for cls, name in _FALLBACK_TYPES if isinstance(exc, cls)), "error")
    message = exc.message or exc.reason or _describe(exc)
    payload: Dict[str, Any] = {
        "error_type": error_type,
        "message": message,
        "reason": exc.reason,
        "status": exc.status,
        "request_id": exc.request_id,
    }
    if isinstance(exc, RefusalError):
        payload["category"] = exc.category
        if exc.safety is not None:
            payload["safety"] = exc.safety.to_dict()
    if isinstance(exc, ValidationError) and exc.entries is not None:
        payload["entries"] = exc.entries
    if isinstance(exc, (RateLimitError, OverloadedError)) and exc.retry_after is not None:
        payload["retry_after"] = exc.retry_after
    boosters = getattr(exc, "boosters", None)
    if boosters:
        # The x-decisionnode-boosters header: what each booster of the request did.
        payload["boosters"] = {
            name: {"status": entry.status, "latency_ms": entry.latency_ms} for name, entry in boosters.items()
        }
    return payload


class _ErrorResult(ToolResult):
    """A tool result with `isError` set, carrying the error as structured content and text."""

    def to_mcp_result(self) -> mcp.types.CallToolResult:  # type: ignore[override]
        return mcp.types.CallToolResult(content=self.content, structuredContent=self.structured_content, isError=True)


def _text(data: Dict[str, Any]) -> List[mcp.types.TextContent]:
    return [mcp.types.TextContent(type="text", text=json.dumps(data, ensure_ascii=False))]


def ok(data: Dict[str, Any]) -> ToolResult:
    """A result as structured content and the same JSON as text, for clients that read only text."""
    return ToolResult(content=_text(data), structured_content=data)


def fail(payload: Dict[str, Any]) -> ToolResult:
    return _ErrorResult(content=_text(payload), structured_content=payload)


# --------------------------------------------------------------------------- the client


def _bearer_from_headers() -> Optional[str]:
    value = get_http_headers().get("authorization", "")
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def _resolve_key() -> Optional[str]:
    if settings.local:
        return None  # the SDK reads DECISIONNODE_API_KEY
    key = _bearer_from_headers()
    if key is None:
        raise ToolFailure(
            "authentication_error",
            "No API key. Send 'Authorization: Bearer <DecisionNode API key>' on the MCP connection.",
            status=401,
        )
    return key


class _RateLimiter:
    """A sliding one-minute window per key, on this instance. Off when the limit is 0.

    Keys are held only as a SHA-256 digest, never in clear. At most `max_size` keys are
    tracked, so memory stays bounded however many distinct keys arrive. The keys are kept
    in order of their last counted call, so the first key is the one whose window expires
    soonest. When the table is full, a new key takes the first key's place only if that
    key's window has expired; otherwise every tracked key still has calls inside the
    minute and the new key's call is refused with the rate-limit error. A key at its limit
    is never forgotten while its window is live, so flooding the table with other keys
    cannot reset it. A key the API rejects is dropped at once, so a flood of invalid keys
    holds a slot only while its one request is in flight and cannot lock new keys out.
    Each call does constant work, apart from dropping its own expired timestamps.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self.max_size = max_size
        self._calls: "OrderedDict[str, Deque[float]]" = OrderedDict()

    def _refuse(self, limit: int, retry_after: float) -> None:
        raise ToolFailure(
            "rate_limit_error",
            "Too many DecisionNode requests from this key through this server: the limit is %d a minute." % limit,
            status=429,
            retry_after=round(max(0.0, retry_after), 3),
        )

    def take(self, key: str, cost: int = 1) -> None:
        limit = settings.rate_limit_per_minute
        if limit <= 0:
            return
        now = time.monotonic()
        key = key_digest(key)
        window = self._calls.get(key)
        if window is not None:
            while window and now - window[0] >= 60.0:
                window.popleft()
        used = len(window) if window is not None else 0
        if used + cost > limit:
            self._refuse(limit, 60.0 - (now - window[0]) if window else 60.0)
        if window is None:
            if len(self._calls) >= self.max_size:
                # The first key has the oldest last call: if its window is still live,
                # so is every other key's, and none may be forgotten.
                first = next(iter(self._calls.values()))
                if first and now - first[-1] < 60.0:
                    self._refuse(limit, 60.0 - (now - first[-1]))
                self._calls.popitem(last=False)
            window = self._calls[key] = deque()
        else:
            self._calls.move_to_end(key)
        window.extend([now] * cost)

    def forget(self, key: str) -> None:
        """Drop a key the API rejected, so keys that were never valid cannot fill the table."""
        self._calls.pop(key_digest(key), None)

    def __len__(self) -> int:
        return len(self._calls)


rate_limiter = _RateLimiter()


def key_digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class _RejectedKeys:
    """Keys the API has just answered 401 for, refused for a short while without a request.

    A client looping on a bad key would otherwise cost one API request per call. Each
    entry is the SHA-256 digest of the key (never the key) and the error result the API
    gave, kept for `ttl` seconds; at most `max_size` keys are held, the oldest dropped first.
    """

    def __init__(self, ttl: float = 60.0, max_size: int = 10_000) -> None:
        self.ttl = ttl
        self.max_size = max_size
        self._entries: "OrderedDict[str, Tuple[float, Dict[str, Any]]]" = OrderedDict()

    def remember(self, key: str, payload: Dict[str, Any]) -> None:
        digest = key_digest(key)
        rate_limiter.forget(key)
        self._entries.pop(digest, None)
        # No request is made for a refusal from here, so it carries no request id.
        self._entries[digest] = (time.monotonic() + self.ttl, dict(payload, request_id=None))
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)

    def check(self, key: str) -> None:
        """Raise the remembered error result for a key the API rejected within the last `ttl` seconds."""
        digest = key_digest(key)
        entry = self._entries.get(digest)
        if entry is None:
            return
        expires, payload = entry
        if time.monotonic() >= expires:
            del self._entries[digest]
            return
        raise ToolFailure(payload["error_type"], payload["message"], **{
            k: v for k, v in payload.items() if k not in ("error_type", "message")
        })

    def __len__(self) -> int:
        return len(self._entries)


rejected_keys = _RejectedKeys()


@asynccontextmanager
async def client(cost: int = 1) -> AsyncIterator["_KeyedClient"]:
    """An SDK client for one tool call, with the caller's key and the SDK's retry policy.

    The base URL and the default model come from `DECISIONNODE_BASE_URL` and
    `DECISIONNODE_MODEL`, as in every DecisionNode product.
    """
    key = _resolve_key()
    if settings.local and not os.environ.get("DECISIONNODE_API_KEY"):
        raise ToolFailure(
            "authentication_error",
            "No API key. Set DECISIONNODE_API_KEY in this server's environment (the 'env' of its "
            "entry in the MCP client's configuration).",
            status=401,
        )
    effective_key = key or os.environ.get("DECISIONNODE_API_KEY", "")
    rejected_keys.check(effective_key)
    rate_limiter.take(effective_key, cost)
    dn = AsyncDecisionNode(api_key=key, headers=dict(settings.extra_headers) or None)
    try:
        yield _KeyedClient(dn, effective_key)
    except AuthenticationError as exc:
        rejected_keys.remember(effective_key, error_payload(exc))
        raise
    finally:
        await dn.close()


class _KeyedClient:
    """The SDK client, plus a way for a tool to report a 401 it caught itself (`decide_many`)."""

    def __init__(self, dn: AsyncDecisionNode, key: str) -> None:
        self._dn = dn
        self._key = key

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dn, name)

    def rejected(self, exc: AuthenticationError) -> None:
        rejected_keys.remember(self._key, error_payload(exc))


async def run(call: Any) -> ToolResult:
    """Run a tool body, turning SDK errors and argument problems into `isError` results."""
    try:
        return ok(await call())
    except ToolFailure as exc:
        return fail(exc.payload)
    except DecisionNodeError as exc:
        return fail(error_payload(exc))


# --------------------------------------------------------------------------- arguments


# The hosted server's transport refuses a request body over 4 MiB (the mcp library's
# limit for Streamable HTTP). Base64 image data in one call is held to 3 MiB in total
# (3,145,728 characters, about 2.25 MiB of image bytes), which leaves room for the rest
# of the call. Larger images go by https url, which the API fetches itself.
HOSTED_IMAGE_DATA_LIMIT = 3 * 1024 * 1024

# The rate and the most frames taken from a video file when the call names neither.
DEFAULT_VIDEO_FPS = 2.0
DEFAULT_VIDEO_MAX_FRAMES = 32


def images_for_api(images: Optional[List[Dict[str, Any]]]) -> Optional[List[Any]]:
    """Turn the tool's image objects into what the SDK sends.

    `{id, url}` and `{id, data, media_type}` go to the API as they are. `{id, path}` is
    read from the user's disk, and only by the local server: the hosted server never
    opens a path a caller names.
    """
    if not images:
        return None
    out: List[Any] = []
    data_total = 0
    for img in images:
        img = {k: v for k, v in img.items() if v is not None}
        image_id = img.get("id")
        sources = [k for k in ("url", "data", "path") if k in img]
        if not isinstance(image_id, str) or len(sources) != 1:
            raise ToolFailure(
                "invalid_request_error",
                "Each image needs an 'id' and exactly one of 'url', 'data' (with 'media_type') or 'path'.",
            )
        if sources[0] == "path":
            if not settings.local:
                raise ToolFailure(
                    "invalid_request_error",
                    "image %r: file paths are read only by the local server. Send the image as an https "
                    "'url' or as base64 'data' with its 'media_type'." % image_id,
                )
            try:
                out.append(Image(image_id, path=os.path.expanduser(str(img["path"]))))
            except (OSError, ValueError) as exc:
                # The SDK reports an unreadable or unrecognised file as ValueError.
                raise ToolFailure("invalid_request_error", str(exc)) from None
        elif sources[0] == "data":
            if not img.get("media_type"):
                raise ToolFailure(
                    "invalid_request_error",
                    "image %r: 'data' needs its 'media_type' (image/jpeg, image/png or image/webp)." % image_id,
                )
            data_total += len(str(img["data"]))
            if not settings.local and data_total > HOSTED_IMAGE_DATA_LIMIT:
                raise ToolFailure(
                    "invalid_request_error",
                    "The images' base64 'data' totals more than 3 MiB, the most one call to the hosted "
                    "server can carry. Send larger images by https 'url' instead.",
                )
            out.append({"id": image_id, "media_type": img["media_type"], "data": img["data"]})
        else:
            url = str(img["url"])
            if not url.lower().startswith("https://"):
                raise ToolFailure(
                    "invalid_request_error",
                    "image %r: the url must start with https://. Send the image over https, or as "
                    "base64 'data' with its 'media_type'." % image_id,
                )
            out.append({"id": image_id, "url": url})
    return out


@dataclass
class VideoFile:
    """A video file on the user's disk, sampled into frames by the local server before the request."""

    id: str
    path: str
    fps: float
    start: float
    end: float
    max_frames: int


def videos_for_api(videos: List[Dict[str, Any]], images: Optional[List[Dict[str, Any]]] = None) -> List[Any]:
    """Check the tool's video objects and turn each into what is sent.

    `{id, frames, fps, start, end}` goes to the API as it is. `{id, path}` becomes a
    `VideoFile` for the local server to sample; the hosted server never opens a path a
    caller names. On the hosted server the frames' base64 data counts toward the same
    3 MiB as the images' data.
    """
    data_total = sum(len(str(img["data"])) for img in images or [] if img.get("data") is not None)
    out: List[Any] = []
    for video in videos:
        video = {k: v for k, v in video.items() if v is not None}
        video_id = video.get("id")
        sources = [k for k in ("frames", "path") if k in video]
        if len(sources) != 1:
            raise ToolFailure(
                "invalid_request_error",
                "video %r: give exactly one of 'frames' (each {t, media_type, data}) or 'path'." % video_id,
            )
        if sources[0] == "path":
            if not settings.local:
                raise ToolFailure(
                    "invalid_request_error",
                    "video %r: file paths are read only by the local server. Send the video as 'frames', "
                    "each {t, media_type, data} with its time in seconds." % video_id,
                )
            out.append(
                VideoFile(
                    id=str(video_id),
                    path=os.path.expanduser(str(video["path"])),
                    fps=float(video.get("fps", DEFAULT_VIDEO_FPS)),
                    start=float(video.get("start", 0.0)),
                    end=float(video.get("end", float("inf"))),
                    max_frames=int(video.get("max_frames", DEFAULT_VIDEO_MAX_FRAMES)),
                )
            )
            continue
        if "max_frames" in video:
            raise ToolFailure(
                "invalid_request_error",
                "video %r: 'max_frames' applies to a 'path' only. Send fewer frames instead." % video_id,
            )
        data_total += sum(len(str(frame.get("data", ""))) for frame in video["frames"])
        if not settings.local and data_total > HOSTED_IMAGE_DATA_LIMIT:
            raise ToolFailure(
                "invalid_request_error",
                "The base64 'data' of the images and video frames totals more than 3 MiB, the most one "
                "call to the hosted server can carry. Send fewer or smaller frames, or the images by "
                "https 'url'.",
            )
        out.append({k: video[k] for k in ("id", "fps", "start", "end", "frames") if k in video})
    return out


def has_places(questions: Mapping[str, Any]) -> bool:
    """Whether any question is a point or a box."""
    return any(isinstance(q, Mapping) and q.get("type") in PLACE_TYPES for q in questions.values())


async def decide(
    dn: Any,
    state: Any,
    questions: Dict[str, Any],
    images: Optional[List[Any]],
    model: Optional[str],
    videos: Optional[List[Dict[str, Any]]] = None,
    boosters: Optional[List[Any]] = None,
) -> Decision:
    """One decide request.

    It goes through the SDK's decide() when that takes everything the call holds. The
    SDK's decide() takes no videos yet, and an SDK may not know the point and box types
    yet, so a call with either sends the SDK's decide body with them, on the SDK's
    transport (its retries, its errors, the response headers), read by its own parser:
    the result and every error are what decide() would give.
    """
    if not videos and not has_places(questions):
        if boosters:
            return await dn.decide(state, questions, images, model, boosters=boosters)
        return await dn.decide(state, questions, images, model)
    body: Dict[str, Any] = {
        "model": model or dn.model,
        "state": "" if state is None else state,
        "questions": questions,
    }
    if images:
        body["images"] = [img.to_dict() if isinstance(img, Image) else dict(img) for img in images]
    if videos:
        body["videos"] = videos
    if boosters:
        body["boosters"] = boosters
    status, headers, text = await dn._send("POST", sdk_base.DECIDE_PATH, sdk_base.encode(body))
    return parse_decision(sdk_base.parse_json(status, headers, text), headers)


async def decide_many(dn: Any, requests: List[Dict[str, Any]], concurrency: int) -> List[Any]:
    """Many decide requests, at most `concurrency` at a time, in input order: each result is
    the Decision or the error that request ended with. Through the SDK's decide_many()
    unless a request holds what only `decide` above sends."""
    if not any(has_places(req["questions"]) for req in requests):
        return list(await dn.decide_many(requests, concurrency=concurrency))
    limit = asyncio.Semaphore(concurrency)

    async def one(req: Dict[str, Any]) -> Any:
        async with limit:
            try:
                return await decide(dn, req.get("state"), req["questions"], req.get("images"), req.get("model"))
            except DecisionNodeError as exc:
                return exc

    return list(await asyncio.gather(*(one(req) for req in requests)))


# --------------------------------------------------------------------------- results


def with_safety(result: Dict[str, Any], decision: Decision) -> Dict[str, Any]:
    """Add `safety` only when the hosted check flagged a category, so ordinary results stay short."""
    if decision.safety is not None and decision.safety.flagged:
        result["safety"] = decision.safety.to_dict()
    return result


def with_evidence(result: Dict[str, Any], decision: Decision) -> Dict[str, Any]:
    """Add the boosters' `evidence` exactly as the API sent it, when the call named boosters."""
    evidence = decision.raw.get("evidence") if isinstance(decision.raw, Mapping) else None
    if isinstance(evidence, Mapping):
        result["evidence"] = dict(evidence)
    return result


def with_cost(result: Dict[str, Any], decision: Decision) -> Dict[str, Any]:
    """Add the answer's `usage.cost` exactly as the API sent it, when it sent one, so an
    agent sees what each call cost."""
    usage = decision.raw.get("usage") if isinstance(decision.raw, Mapping) else None
    cost = usage.get("cost") if isinstance(usage, Mapping) else None
    if isinstance(cost, Mapping):
        result["cost"] = dict(cost)
    return result


def answer_fields(decision: Decision, question_id: str) -> Dict[str, Any]:
    """One answer's fields exactly as the API sent them, without its `type`."""
    raw = (decision.raw.get("answers") or {}).get(question_id)
    if not isinstance(raw, Mapping):
        raise ToolFailure("api_error", "The API's response has no answer for the question.")
    return {k: v for k, v in raw.items() if k != "type"}
