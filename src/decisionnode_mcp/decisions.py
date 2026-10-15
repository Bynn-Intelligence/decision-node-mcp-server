"""The full-request tools: `decide` and `decide_many`."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from decisionnode import AuthenticationError, Decision, DecisionNodeError
from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from pydantic import Field

from . import api, schemas, texts
from .schemas import (
    BOOSTERS_DESCRIPTION,
    IMAGES_DESCRIPTION,
    MODEL_DESCRIPTION,
    STATE_DESCRIPTION,
    BoosterEntry,
    ImageInput,
    ItemInput,
    QuestionInput,
)

mcp = FastMCP("DecisionNode decisions")

MAX_ITEMS = 50
CONCURRENCY = 8

Questions = Annotated[
    Dict[str, QuestionInput],
    Field(
        min_length=1,
        description=(
            "Question id to question, exactly as the API takes it: {type, instructions, criteria}; for a "
            "number {type, instructions, min, max, step}; for a point or a box {type, instructions, image, "
            "units}, which needs at least one image. Ids are yours and come back as the answer keys."
        ),
    ),
]


def _answers(decision: Decision) -> Dict[str, Any]:
    answers = decision.raw.get("answers")
    return dict(answers) if isinstance(answers, dict) else {}


@mcp.tool(
    name="decide",
    description=texts.DECIDE_DESCRIPTION,
    annotations=schemas.annotations("Decide"),
    output_schema=schemas.DECIDE_OUTPUT,
)
async def decide(
    questions: Questions,
    state: Annotated[Any, Field(description=STATE_DESCRIPTION)] = None,
    images: Annotated[Optional[List[ImageInput]], Field(description=IMAGES_DESCRIPTION)] = None,
    boosters: Annotated[Optional[List[BoosterEntry]], Field(max_length=4, description=BOOSTERS_DESCRIPTION)] = None,
    model: Annotated[Optional[str], Field(description=MODEL_DESCRIPTION)] = None,
) -> ToolResult:
    async def call() -> Dict[str, Any]:
        sdk_images = api.images_for_api([i.model_dump() for i in images] if images else None)
        wire = {qid: q.wire() for qid, q in questions.items()}
        booster_wire = [b if isinstance(b, str) else b.wire() for b in boosters] if boosters else None
        async with api.client() as dn:
            decision = await api.decide(dn, state, wire, sdk_images, model, boosters=booster_wire)
        result = {
            "answers": _answers(decision),
            "model": decision.model,
            "input_tokens": decision.usage.input_tokens,
        }
        return api.with_safety(api.with_cost(api.with_evidence(result, decision), decision), decision)

    return await api.run(call)


@mcp.tool(
    name="decide_many",
    description=texts.DECIDE_MANY_DESCRIPTION,
    annotations=schemas.annotations("Decide many"),
    output_schema=schemas.DECIDE_MANY_OUTPUT,
)
async def decide_many(
    items: Annotated[
        List[ItemInput],
        Field(
            min_length=1,
            max_length=MAX_ITEMS,
            description="The states to judge, 1 to 50, each {id, state} and, when its questions look at images, its `images`.",
        ),
    ],
    questions: Questions,
    model: Annotated[Optional[str], Field(description=MODEL_DESCRIPTION)] = None,
) -> ToolResult:
    async def call() -> Dict[str, Any]:
        ids = [item.id for item in items]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise api.ToolFailure(
                "invalid_request_error",
                "Item ids must be unique; repeated: %s." % ", ".join(repr(d) for d in duplicates),
            )
        item_images = [[i.model_dump() for i in item.images] if item.images else None for item in items]
        data = sum(len(img["data"]) for images in item_images for img in images or [] if img.get("data"))
        if not api.settings.local and data > api.HOSTED_IMAGE_DATA_LIMIT:
            raise api.ToolFailure(
                "invalid_request_error",
                "The items' images' base64 'data' totals more than 3 MiB, the most one call to the hosted "
                "server can carry. Send larger images by https 'url' instead, or fewer items a call.",
            )
        wire = {qid: q.wire() for qid, q in questions.items()}
        requests = []
        for item, images in zip(items, item_images):
            request: Dict[str, Any] = {"state": item.state, "questions": wire, "model": model}
            if images:
                request["images"] = api.images_for_api(images)
            requests.append(request)
        async with api.client(cost=len(items)) as dn:
            outcomes = await api.decide_many(dn, requests, CONCURRENCY)
        results: Dict[str, Any] = {}
        input_tokens = 0
        for item_id, outcome in zip(ids, outcomes):
            if isinstance(outcome, AuthenticationError):
                dn.rejected(outcome)
            if isinstance(outcome, DecisionNodeError):
                results[item_id] = {"error": api.error_payload(outcome)}
                continue
            input_tokens += outcome.usage.input_tokens or 0
            results[item_id] = api.with_safety(api.with_cost({"answers": _answers(outcome), "model": outcome.model}, outcome), outcome)
        return {"results": results, "input_tokens": input_tokens}

    return await api.run(call)
