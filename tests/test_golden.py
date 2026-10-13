"""Golden files: each tool's schemas, and its result for recorded API responses.

A change to a schema, a description or a result shape changes what every agent pays
for and reads, so it must show up in review. Regenerate after a deliberate change:

    UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
from conftest import Mock, use_scenario
from fastmcp import Client

from decisionnode_mcp.server import create_server

GOLDEN = Path(__file__).parent / "golden"


def check_golden(name: str, data: Any) -> None:
    path = GOLDEN / (name + ".json")
    text = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    if os.environ.get("UPDATE_GOLDEN") or not path.exists():
        path.write_text(text)
        if not os.environ.get("UPDATE_GOLDEN"):
            pytest.fail("wrote the missing golden file %s; check it in" % path.name)
        return
    assert json.loads(path.read_text()) == data, "golden file %s differs; regenerate with UPDATE_GOLDEN=1" % path.name


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="before 3.11, typing wraps a parameter with a None default in Optional, so the "
    "generated input schemas differ in form (not in meaning); the goldens are written on 3.11+",
)
async def test_tool_schemas() -> None:
    async with Client(create_server()) as client:
        tools = await client.list_tools()
    for tool in tools:
        check_golden(
            "tool-" + tool.name,
            {
                "name": tool.name,
                "description": tool.description,
                "annotations": tool.annotations.model_dump(exclude_none=True) if tool.annotations else None,
                "inputSchema": tool.inputSchema,
                "outputSchema": tool.outputSchema,
            },
        )


Q = {"q": {"type": "truth", "instructions": "?"}}

CASES = [
    ("truth", "response/check-returns-truth-answer", {"instructions": "Refund request?", "state": "Customer: refund me."}),
    ("choice", "response/classify-returns-choice-answer", {"instructions": "Where?", "options": ["billing", "bug", "account"]}),
    ("score", "response/score-returns-score-answer", {"instructions": "How urgent?", "levels": ["routine", "today", "urgent", "critical"]}),
    ("decide", "response/decide-three-types", {"state": "s", "questions": Q}),
    ("decide", "response/score-legend-structured", {"state": "s", "questions": Q}),
    ("decide", "response/choice-empty-name-chosen", {"state": "s", "questions": Q}),
    ("decide", "response/unknown-answer-type", {"state": "s", "questions": Q}),
    ("decide", "response/safety-flagged", {"state": "s", "questions": Q}),
    ("decide_many", "behaviour/decide-many-one-failure", {"items": [{"id": "a", "state": "a"}, {"id": "b", "state": "b"}, {"id": "c", "state": "c"}], "questions": Q}),
    ("models", "response/models-list", {}),
    ("decide", "error/403-refusal-harm", {"state": "s", "questions": Q}),
    ("decide", "error/403-refusal-self-harm", {"state": "s", "questions": Q}),
    ("decide", "error/400-max-tokens-exceeded", {"state": "s", "questions": Q}),
    ("decide", "error/400-too-many-options", {"state": "s", "questions": Q}),
    ("decide", "error/401-authentication", {"state": "s", "questions": Q}),
    ("decide", "error/402-insufficient-credit", {"state": "s", "questions": Q}),
    ("decide", "error/404-not-found", {"state": "s", "questions": Q}),
    ("decide", "error/422-missing-criteria", {"state": "s", "questions": Q}),
    ("decide", "error/reason-escaped-ascii", {"state": "s", "questions": Q}),
]


@pytest.mark.parametrize("tool, scenario, args", CASES, ids=["%s-%s" % (c[0], c[1].split("/")[1]) for c in CASES])
async def test_results(local: Mock, monkeypatch: pytest.MonkeyPatch, tool: str, scenario: str, args: Dict[str, Any]) -> None:
    use_scenario(local, monkeypatch, scenario)
    async with Client(create_server()) as client:
        result = await client.call_tool(tool, args, raise_on_error=False)
    content = result.structured_content
    if tool == "decide_many":
        # The mock gives a response without its own request id a fresh one each time.
        for item in content["results"].values():
            if "error" in item:
                item["error"]["request_id"] = "<request id>"
    check_golden(
        "result-%s-%s" % (tool, scenario.split("/")[1]),
        {"isError": result.is_error, "structuredContent": content},
    )


# Responses of the additions to the API's first contract (number, point and box questions, videos),
# recorded here rather than taken from the conformance suite.
RECORDED = [
    (
        "number",
        "number-returns-number-answer",
        {"instructions": "How many cars are visible?", "min": 0, "max": 50},
        {
            "model": "decisionnode-moe-0.9.0-rc1",
            "answers": {"number": {"type": "number", "number": 7, "expected": 7.3, "confidence": 0.62, "probabilities": {"6": 0.21, "7": 0.62, "8": 0.14}}},
            "usage": {"input_tokens": 530, "output_tokens": 0},
        },
    ),
    (
        "point",
        "point-returns-point-answer",
        {"instructions": "Point to the yellow package closest to the robot arm.", "images": [{"id": "belt", "url": "https://files.example.com/belt.jpg"}]},
        {
            "model": "decisionnode-moe-0.9.0-rc1",
            "answers": {"point": {"type": "point", "x": 812, "y": 455, "present": 0.97, "confidence": 0.91, "image": "belt", "width": 1920, "height": 1080}},
            "usage": {"input_tokens": 1302, "output_tokens": 0},
        },
    ),
    (
        "box",
        "box-returns-box-answer",
        {"instructions": "Draw a box around the damaged package.", "images": [{"id": "belt", "url": "https://files.example.com/belt.jpg"}], "units": "normalized"},
        {
            "model": "decisionnode-moe-0.9.0-rc1",
            "answers": {"box": {"type": "box", "x1": 0.1563, "y1": 0.1111, "x2": 0.2708, "y2": 0.3796, "present": 0.95, "confidence": 0.88, "image": "belt", "width": 1920, "height": 1080}},
            "usage": {"input_tokens": 1302, "output_tokens": 0},
        },
    ),
    (
        "decide_video",
        "video-frames-with-warnings",
        {
            "questions": {"fall": {"type": "truth", "instructions": "Does a person fall?"}, "cars": {"type": "number", "instructions": "How many cars pass?", "min": 0, "max": 50}},
            "videos": [{"id": "gate", "fps": 2, "frames": [{"t": 0.0, "media_type": "image/jpeg", "data": "/9j/"}, {"t": 0.5, "media_type": "image/jpeg", "data": "/9j/"}]}],
        },
        {
            "model": "decisionnode-moe-0.9.0-rc1",
            "answers": {
                "fall": {"type": "truth", "truth": 0.04},
                "cars": {"type": "number", "number": 12, "expected": 12.4, "confidence": 0.41, "probabilities": {"11": 0.22, "12": 0.41, "13": 0.27}},
            },
            "usage": {"input_tokens": 1342, "output_tokens": 0, "video_frames": 2, "video_seconds": 1.0},
            "warnings": ["Video 'gate' was read at 2 frames a second."],
        },
    ),
]


@pytest.mark.parametrize("tool, name, args, body", RECORDED, ids=[r[1] for r in RECORDED])
async def test_recorded_results(
    local: Mock, monkeypatch: pytest.MonkeyPatch, tool: str, name: str, args: Dict[str, Any], body: Dict[str, Any]
) -> None:
    scenario = local.add_scenario({"name": "golden-" + name, "steps": [{"status": 200, "body": body}], "after": "repeat_last"})
    use_scenario(local, monkeypatch, scenario)
    async with Client(create_server()) as client:
        result = await client.call_tool(tool, args, raise_on_error=False)
    check_golden(
        "result-%s-%s" % (tool, name),
        {"isError": result.is_error, "structuredContent": result.structured_content},
    )
