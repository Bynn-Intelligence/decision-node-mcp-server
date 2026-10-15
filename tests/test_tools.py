"""Every tool against the mock API, in process, as the local server runs them."""

from __future__ import annotations

import base64
import json
import struct
import zlib
from typing import Any, Dict

import pytest
from conftest import TEST_KEY, Mock, use_scenario
from fastmcp import Client

from decisionnode_mcp import api, texts
from decisionnode_mcp.server import create_server

TOOLS = ["guide", "truth", "choice", "score", "number", "point", "box", "decide", "decide_many", "decide_video", "models"]


async def call(name: str, args: Dict[str, Any]) -> Any:
    async with Client(create_server()) as client:
        return await client.call_tool(name, args, raise_on_error=False)


def payload(result: Any) -> Dict[str, Any]:
    """The structured content, after checking the text block carries the same JSON."""
    assert result.structured_content is not None
    assert len(result.content) == 1 and result.content[0].type == "text"
    assert json.loads(result.content[0].text) == result.structured_content
    return result.structured_content


def sent_body(mock: Mock) -> Dict[str, Any]:
    reqs = mock.requests()
    assert len(reqs) == 1, reqs
    return reqs[0]["body"]


# --------------------------------------------------------------------------- the list


async def test_lists_seven_read_only_tools_with_schemas(local: Mock) -> None:
    async with Client(create_server()) as client:
        tools = await client.list_tools()
        init = client.initialize_result
    assert [t.name for t in tools] == TOOLS
    for tool in tools:
        a = tool.annotations
        assert a is not None and a.readOnlyHint is True and a.idempotentHint is True and a.destructiveHint is False
        assert tool.outputSchema and tool.outputSchema["type"] == "object"
        assert tool.inputSchema["type"] == "object"
    assert init is not None and init.instructions == texts.INSTRUCTIONS
    assert local.requests() == []


async def test_guide_makes_no_request(local: Mock) -> None:
    result = await call("guide", {})
    assert not result.is_error
    assert result.structured_content == {"guide": texts.GUIDE}
    assert result.content[0].text == texts.GUIDE
    assert local.requests() == []


# --------------------------------------------------------------------------- truth, choice, score


async def test_truth_sends_one_truth_question(local: Mock) -> None:
    result = await call("truth", {"instructions": "Is this a refund request?", "state": "Refund me now."})
    data = payload(result)
    assert set(data) == {"truth", "cost"} and 0.0 <= data["truth"] <= 1.0
    body = sent_body(local)
    assert body == {
        "model": "decisionnode-moe",
        "state": "Refund me now.",
        "questions": {"check": {"type": "truth", "instructions": "Is this a refund request?"}},
    }
    assert local.requests()[0]["headers"]["authorization"] == "Bearer " + TEST_KEY


async def test_truth_criteria_and_no_state(local: Mock) -> None:
    await call("truth", {"instructions": "Is a hotdog a sandwich?", "if_true": "it is", "if_false": "it is not"})
    body = sent_body(local)
    assert body["state"] == ""
    assert body["questions"]["check"]["criteria"] == {"true": "it is", "false": "it is not"}


async def test_truth_returns_the_values_unchanged(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, "response/check-returns-truth-answer")
    assert payload(await call("truth", {"instructions": "Refund request?", "state": "Customer: refund me."})) == {
        "truth": 0.97
    }


async def test_choice_from_a_list_and_from_an_object(local: Mock) -> None:
    data = payload(await call("choice", {"instructions": "Topic?", "options": ["shipping", "billing"], "state": "x"}))
    assert data["choice"] in ("shipping", "billing")
    assert set(data) == {"choice", "confidence", "probabilities", "cost"}
    assert sent_body(local)["questions"]["classify"] == {
        "type": "choice",
        "instructions": "Topic?",
        "criteria": {"shipping": None, "billing": None},
    }


async def test_choice_keeps_long_floats(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, "response/classify-returns-choice-answer")
    data = payload(
        await call(
            "choice",
            {
                "instructions": "Where should this go?",
                "options": {"billing": "money", "bug": "broken", "account": "login"},
                "state": {"ticket": "I was charged twice and nobody has replied for 3 days."},
            },
        )
    )
    assert data == {
        "choice": "billing",
        "confidence": 0.73,
        "probabilities": {"account": 0.12, "billing": 0.8200000000000001, "bug": 0.060000000000000005},
    }
    assert "0.8200000000000001" in json.dumps(data)


async def test_score(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, "response/score-returns-score-answer")
    data = payload(await call("score", {"instructions": "How urgent?", "levels": ["routine", "today", "urgent", "critical"]}))
    assert data == {
        "score": 2.31,
        "confidence": 0.31,
        "legend": {"0": "routine", "1": "today", "2": "urgent", "3": "critical"},
        "probabilities": {"0": 0.01, "1": 0.09, "2": 0.48, "3": 0.42},
    }


async def test_score_against_the_normal_mock(local: Mock) -> None:
    data = payload(await call("score", {"instructions": "Severity?", "levels": ["low", "high"], "state": "Site down"}))
    assert set(data["legend"]) == {"0", "1"}
    assert sent_body(local)["questions"]["score"]["criteria"] == ["low", "high"]


def number_scenario(mock: Mock, question_id: str, answer: Dict[str, Any]) -> str:
    """A recorded answer for a number question, whatever the mock's own handler knows."""
    return mock.add_scenario(
        {
            "name": "mcp-number-" + question_id,
            "steps": [
                {
                    "status": 200,
                    "body": {
                        "model": "decisionnode-moe-0.9.0-rc1",
                        "answers": {question_id: dict({"type": "number"}, **answer)},
                        "usage": {"input_tokens": 530, "output_tokens": 0},
                    },
                }
            ],
            "after": "repeat_last",
        }
    )


NUMBER_ANSWER = {"number": 7, "expected": 7.3, "confidence": 0.62, "probabilities": {"6": 0.21, "7": 0.62, "8": 0.14}}


async def test_number_sends_one_number_question(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, number_scenario(local, "number", NUMBER_ANSWER))
    data = payload(
        await call("number", {"instructions": "How many cars are visible?", "min": 0, "max": 50, "state": "Car park, level 2."})
    )
    assert data == NUMBER_ANSWER
    body = sent_body(local)
    assert body["questions"] == {
        "number": {"type": "number", "instructions": "How many cars are visible?", "min": 0, "max": 50}
    }
    # Whole numbers stay whole on the wire.
    assert '"min":0,"max":50' in local.requests()[0]["body_text"]


async def test_number_bounds_are_optional_and_steps_can_be_fractional(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, number_scenario(local, "number", NUMBER_ANSWER))
    await call("number", {"instructions": "How many line items?"})
    assert sent_body(local)["questions"]["number"] == {"type": "number", "instructions": "How many line items?"}
    local.session += "-half"
    monkeypatch.setattr(api.settings, "extra_headers", local.headers("mcp-number-number"))
    await call("number", {"instructions": "Rating?", "min": 0, "max": 5, "step": 0.5})
    assert sent_body(local)["questions"]["number"] == {
        "type": "number",
        "instructions": "Rating?",
        "min": 0,
        "max": 5,
        "step": 0.5,
    }


@pytest.mark.parametrize("step", [0, -1])
async def test_number_step_must_be_above_0(local: Mock, step: int) -> None:
    result = await call("number", {"instructions": "?", "step": step})
    assert result.is_error
    assert local.requests() == []


async def test_decide_takes_number_questions(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, number_scenario(local, "cars", NUMBER_ANSWER))
    questions = {
        "cars": {"type": "number", "instructions": "How many cars?", "min": 0, "max": 50, "step": None},
    }
    data = payload(await call("decide", {"state": "s", "questions": questions}))
    assert data["answers"]["cars"] == dict({"type": "number"}, **NUMBER_ANSWER)
    # A bound sent as null is left out, as if it was never given.
    assert sent_body(local)["questions"] == {"cars": {"type": "number", "instructions": "How many cars?", "min": 0, "max": 50}}


async def test_the_older_name_of_truth_is_accepted_and_asked_as_truth(local: Mock) -> None:
    data = payload(await call("decide", {"state": "s", "questions": {"q": {"type": "noul", "instructions": "Refund?"}}}))
    assert data["answers"]["q"]["type"] == "truth"
    assert sent_body(local)["questions"] == {"q": {"type": "truth", "instructions": "Refund?"}}


async def test_no_schema_or_text_shows_the_older_name_of_truth(local: Mock) -> None:
    async with Client(create_server()) as client:
        tools = await client.list_tools()
        guide = await client.call_tool("guide", {})
    shown = json.dumps([t.model_dump() for t in tools]) + guide.content[0].text + texts.INSTRUCTIONS
    assert "noul" not in shown.lower()
    decide = next(t for t in tools if t.name == "decide")
    question = decide.inputSchema["properties"]["questions"]["additionalProperties"]
    assert question["properties"]["type"]["enum"] == ["truth", "choice", "score", "number", "point", "box"]


async def test_sixteen_images_reach_the_sdk(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    # The API takes up to 16 images a request. The server neither caps nor drops any: all
    # of them are handed to the SDK as they came.
    seen: Dict[str, Any] = {}

    async def decide(self: Any, state: Any = None, questions: Any = None, images: Any = None, model: Any = None) -> Any:
        seen["images"] = images
        body = {"model": "decisionnode-moe-0.9.0-rc1", "answers": {"check": {"type": "truth", "truth": 0.5}}, "usage": {"input_tokens": 1}}
        return api.parse_decision(body, {})

    monkeypatch.setattr(api.AsyncDecisionNode, "decide", decide)
    images = [{"id": "p%02d" % i, "url": "https://example.com/p%02d.jpg" % i} for i in range(16)]
    data = payload(await call("truth", {"instructions": "Same product on every photo?", "images": images}))
    assert data == {"truth": 0.5}
    assert seen["images"] == images


# --------------------------------------------------------------------------- point and box

BELT = {"id": "belt", "url": "https://files.example.com/line-3/cam-2/0917.jpg"}
REFERENCE = {"id": "reference", "url": "https://files.example.com/line-3/reference.jpg"}
POINT = {"x": 812, "y": 455, "present": 0.97, "confidence": 0.91, "image": "belt", "width": 1920, "height": 1080}
BOX = {"x1": 300, "y1": 120, "x2": 520, "y2": 410, "present": 0.95, "confidence": 0.88, "image": "belt", "width": 1920, "height": 1080}


def answers_scenario(mock: Mock, name: str, answers: Dict[str, Any]) -> str:
    """Recorded answers, whatever question types the mock's own handler knows."""
    body = {"model": "decisionnode-moe-0.9.0-rc1", "answers": answers, "usage": {"input_tokens": 1302, "output_tokens": 0}}
    return mock.add_scenario({"name": "mcp-" + name, "steps": [{"status": 200, "body": body}], "after": "repeat_last"})


async def test_point_sends_one_point_question(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, answers_scenario(local, "point", {"point": dict(POINT, type="point")}))
    data = payload(await call("point", {"instructions": "Point to the yellow package closest to the robot arm.", "images": [BELT]}))
    assert data == POINT
    body = sent_body(local)
    assert body == {
        "model": "decisionnode-moe",
        "state": "",
        "questions": {"point": {"type": "point", "instructions": "Point to the yellow package closest to the robot arm."}},
        "images": [BELT],
    }


async def test_box_names_its_image_and_units(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    normalized = dict(BOX, x1=0.1563, y1=0.1111, x2=0.2708, y2=0.3796, type="box")
    use_scenario(local, monkeypatch, answers_scenario(local, "box", {"box": normalized}))
    args = {
        "instructions": "Draw a box around the part that differs from the reference image.",
        "images": [BELT, REFERENCE],
        "image": "belt",
        "units": "normalized",
        "state": {"order": "A-1043"},
    }
    data = payload(await call("box", args))
    assert data["x1"] == 0.1563 and data["image"] == "belt" and "type" not in data
    body = sent_body(local)
    assert body["questions"] == {
        "box": {
            "type": "box",
            "instructions": "Draw a box around the part that differs from the reference image.",
            "image": "belt",
            "units": "normalized",
        }
    }
    assert body["images"] == [BELT, REFERENCE] and body["state"] == {"order": "A-1043"}


@pytest.mark.parametrize("tool", ["point", "box"])
async def test_a_point_or_a_box_needs_an_image(local: Mock, tool: str) -> None:
    for args in ({"instructions": "the door"}, {"instructions": "the door", "images": []}):
        assert (await call(tool, args)).is_error
    data = error_of(await call(tool, {"instructions": "the door", "images": [BELT, REFERENCE]}))
    assert data["error_type"] == "invalid_request_error" and "'image' names the one to look in" in data["message"]
    data = error_of(await call(tool, {"instructions": "the door", "images": [BELT], "image": "gate"}))
    assert "'gate'" in data["message"]
    assert (await call(tool, {"instructions": "the door", "images": [BELT], "units": "inches"})).is_error
    assert local.requests() == []


async def test_decide_mixes_points_and_boxes_with_the_other_types(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    answers = {
        "pick": dict(POINT, type="point"),
        "damage": dict(BOX, type="box"),
        "any_damage": {"type": "truth", "truth": 0.96},
    }
    use_scenario(local, monkeypatch, answers_scenario(local, "decide-places", answers))
    questions = {
        "pick": {"type": "point", "instructions": "Point to the yellow package closest to the robot arm."},
        "damage": {"type": "box", "instructions": "Draw a box around the damaged package.", "image": "belt", "units": None},
        "any_damage": {"type": "truth", "instructions": "Is any package on the belt damaged?"},
    }
    data = payload(await call("decide", {"questions": questions, "images": [BELT]}))
    assert data["answers"] == answers and data["input_tokens"] == 1302
    sent = sent_body(local)
    assert sent["images"] == [BELT]
    assert sent["questions"]["damage"] == {"type": "box", "instructions": "Draw a box around the damaged package.", "image": "belt"}
    assert sent["questions"]["pick"] == questions["pick"]


async def test_decide_many_items_carry_their_own_images(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, answers_scenario(local, "many-boxes", {"damage": dict(BOX, type="box")}))
    items = [
        {"id": "cam-%d" % i, "state": "Line 3, camera %d." % i, "images": [{"id": "belt", "url": "https://files.example.com/cam-%d.jpg" % i}]}
        for i in range(12)
    ]
    questions = {"damage": {"type": "box", "instructions": "Draw a box around the damaged package."}}
    data = payload(await call("decide_many", {"items": items, "questions": questions}))
    assert list(data["results"]) == [i["id"] for i in items]
    assert all(r["answers"]["damage"] == dict(BOX, type="box") for r in data["results"].values())
    assert data["input_tokens"] == 12 * 1302
    sent = sorted((r["body"]["state"], r["body"]["images"][0]["url"]) for r in local.requests())
    assert sent == sorted(("Line 3, camera %d." % i, "https://files.example.com/cam-%d.jpg" % i) for i in range(12))
    assert local.max_in_flight() <= 8


async def test_decide_many_reports_each_item_s_error_for_points_and_boxes(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, "error/422-missing-criteria")
    items = [{"id": "a", "images": [BELT]}, {"id": "b", "images": [BELT]}]
    result = await call("decide_many", {"items": items, "questions": {"p": {"type": "point", "instructions": "the door"}}})
    assert not result.is_error
    data = payload(result)
    assert [data["results"][i]["error"]["status"] for i in ("a", "b")] == [422, 422]
    assert data["input_tokens"] == 0
    assert len(local.requests()) == 2  # a 422 is not retried


async def test_hosted_decide_many_images_share_the_3_mib(hosted: Mock) -> None:
    half = "A" * (api.HOSTED_IMAGE_DATA_LIMIT // 2 + 4)
    items = [{"id": str(i), "images": [{"id": "a", "data": half, "media_type": "image/png"}]} for i in range(2)]
    data = error_of(await call("decide_many", {"items": items, "questions": {"q": {"type": "truth", "instructions": "?"}}}))
    assert data["error_type"] == "invalid_request_error" and "3 MiB" in data["message"]
    assert hosted.requests() == []


async def test_model_argument_and_environment(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    await call("truth", {"instructions": "?", "model": "decisionnode-flash"})
    assert sent_body(local)["model"] == "decisionnode-flash"
    local.session += "-env"
    monkeypatch.setattr(api.settings, "extra_headers", local.headers())
    monkeypatch.setenv("DECISIONNODE_MODEL", "decisionnode-flash")
    await call("truth", {"instructions": "?"})
    assert sent_body(local)["model"] == "decisionnode-flash"


# --------------------------------------------------------------------------- decide, decide_many


QUESTIONS = {
    "route": {"type": "choice", "instructions": "Where should this go?", "criteria": {"billing": "money", "bug": "broken"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["routine", "urgent"]},
    "escalate": {"type": "truth", "instructions": "Escalate to a human now?"},
}


async def test_decide_sends_the_questions_as_given(local: Mock) -> None:
    data = payload(await call("decide", {"state": {"ticket": "Charged twice."}, "questions": QUESTIONS}))
    assert list(data["answers"]) == ["route", "urgency", "escalate"]
    assert data["answers"]["escalate"]["type"] == "truth"
    assert data["model"].startswith("decisionnode-moe-")
    assert isinstance(data["input_tokens"], int)
    assert "safety" not in data
    assert sent_body(local) == {"model": "decisionnode-moe", "state": {"ticket": "Charged twice."}, "questions": QUESTIONS}


async def test_decide_several_places(local: Mock) -> None:
    # Max on a box question is sent as given, and the list the API
    # answers (count, complete, boxes) comes back as sent.
    question = {"type": "box", "instructions": "Every package on the belt.", "max": 8}
    data = payload(await call("decide", {"questions": {"packages": question}, "images": [BELT]}))
    assert sent_body(local)["questions"]["packages"] == question
    answer = data["answers"]["packages"]
    assert answer["type"] == "box" and "x1" not in answer
    assert set(answer) >= {"count", "present", "complete", "boxes", "image", "width", "height"}
    assert len(answer["boxes"]) == answer["count"] <= 8


DEPTH_READING = {
    "image": "belt", "ahead": 3.21, "directions": [5.12, 4.03, None, 2.81, 6.02, 7.15, 9.33], "ground": True,
    "grid_rows": 2, "grid_cols": 2, "grid": [[9.1, 8.7], [3.4, 2.9]], "depth_min": 1.01, "depth_max": 14.98,
    "depth_median": 4.37, "width": 1920, "height": 1080,
}
SEGMENT_READING = {
    "image": "belt", "width": 1920, "height": 1080, "prompts": ["purple gate"],
    "instances": [{
        "label": "purple gate", "score": 0.91, "box": [0.31, 0.22, 0.58, 0.81],
        "polygon": [[0.31, 0.22], [0.58, 0.22], [0.58, 0.81]], "area": 0.12, "centroid": [0.45, 0.47],
        "opening": None, "depth": {"median": 4.2, "opening_median": None, "opening_rim_median": None},
        "size_m": {"width": 1.6, "height": 2.1, "opening": None},
    }],
    "depth_source": "bynn_depth", "assumed_hfov_deg": 65,
}
EVIDENCE = {
    "bynn_depth": {"version": "bynn_depth-v1", "status": "ok", "latency_ms": 41, "billed_calls": 1, "results": [DEPTH_READING]},
    "bynn_segment": {"version": "bynn_segment-v1", "status": "ok", "latency_ms": 88, "billed_calls": 1, "results": [SEGMENT_READING]},
}


def evidence_scenario(mock: Mock, name: str, answers: Dict[str, Any]) -> str:
    body = {"model": "decisionnode-moe-0.9.0-rc1", "answers": answers, "evidence": EVIDENCE,
            "usage": {"input_tokens": 512, "output_tokens": 0, "booster_tokens": 180, "boosters": 2}}
    return mock.add_scenario({"name": name, "steps": [{"status": 200, "body": body}], "after": "repeat_last"})


async def test_decide_sends_boosters_as_given_and_returns_their_evidence(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, evidence_scenario(local, "mcp-boosters", {"clear": {"type": "truth", "truth": 0.12}}))
    boosters = [
        "age",
        {"name": "bynn_depth", "options": {"grid_size": 2}},
        {"name": "bynn_segment", "images": ["belt"], "require": True, "options": {"prompt": ["purple gate"], "depth": True}},
    ]
    questions = {"clear": {"type": "truth", "instructions": "Is the path straight ahead clear for 2 meters?"}}
    data = payload(await call("decide", {"questions": questions, "images": [BELT], "boosters": boosters}))
    assert sent_body(local)["boosters"] == boosters
    assert data["evidence"] == EVIDENCE
    assert data["answers"]["clear"]["truth"] == 0.12


async def test_decide_with_a_place_question_sends_its_boosters(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, evidence_scenario(local, "mcp-boosters-place", {"gate": dict(POINT, type="point")}))
    boosters = [{"name": "bynn_segment", "options": {"prompt": "purple gate"}}]
    data = payload(await call("decide", {"questions": {"gate": {"type": "point", "instructions": "the gate"}}, "images": [BELT], "boosters": boosters}))
    assert sent_body(local)["boosters"] == boosters
    assert data["evidence"]["bynn_segment"]["results"] == [SEGMENT_READING]


async def test_decide_with_a_booster_against_the_mock(local: Mock) -> None:
    questions = {"adult": {"type": "truth", "instructions": "Is the person an adult?"}}
    data = payload(await call("decide", {"questions": questions, "images": [BELT], "boosters": ["age"]}))
    assert sent_body(local)["boosters"] == ["age"]
    assert data["evidence"]["age"]["status"] == "ok"
    assert [r["image"] for r in data["evidence"]["age"]["results"]] == ["belt"]


async def test_decide_without_boosters_has_no_evidence(local: Mock) -> None:
    data = payload(await call("decide", {"questions": {"q": {"type": "truth", "instructions": "?"}}}))
    assert "boosters" not in sent_body(local)
    assert "evidence" not in data


async def test_decide_takes_at_most_four_boosters(local: Mock) -> None:
    result = await call("decide", {"questions": {"q": {"type": "truth", "instructions": "?"}}, "boosters": ["a", "b", "c", "d", "e"]})
    assert result.is_error
    assert local.requests() == []


async def test_decide_passes_a_segment_vision_booster_through_unchanged(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, evidence_scenario(local, "mcp-segment", {"pad": {"type": "truth", "truth": 0.88}}))
    segment = {
        "name": "bynn_segment",
        "images": ["belt"],
        "require": True,
        "timeout_ms": 2000,
        "options": {"prompt": ["orange H landing pad", "purple gate"], "max_instances": 4, "min_score": 0.3,
                    "polygon_points": 12, "depth": False},
    }
    questions = {"pad": {"type": "truth", "instructions": "Is a landing pad in view?"}}
    data = payload(await call("decide", {"questions": questions, "images": [BELT], "boosters": [segment, "face_detection"]}))
    assert sent_body(local)["boosters"] == [segment, "face_detection"]
    assert data["evidence"]["bynn_segment"]["billed_calls"] == 1
    assert data["evidence"]["bynn_segment"]["results"] == [SEGMENT_READING]


async def test_a_booster_refusal_reaches_the_caller_with_its_reason(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    reason = "\"options\": 'prompt' is required"
    step = {"status": 400, "headers": {"x-decisionnode-error": reason},
            "body": {"detail": {"error_type": "api_usage_error", "message": "Invalid request."}}}
    use_scenario(local, monkeypatch, local.add_scenario({"name": "mcp-segment-no-prompt", "steps": [step], "after": "repeat_last"}))
    result = await call("decide", {"questions": {"q": {"type": "truth", "instructions": "?"}}, "images": [BELT],
                                   "boosters": [{"name": "bynn_segment"}]})
    data = error_of(result)
    assert data["status"] == 400 and data["error_type"] == "api_usage_error"
    assert data["message"] == "Invalid request." and data["reason"] == reason
    assert len(local.requests()) == 1  # a 400 is never retried


async def test_a_required_booster_that_failed_names_its_status(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    step = {"status": 529, "headers": {"retry-after-ms": "1", "x-decisionnode-boosters": "bynn_depth=ok:41,bynn_segment=error:61"},
            "body": {"detail": {"error_type": "booster_unavailable", "message": "The booster 'bynn_segment' failed."}}}
    use_scenario(local, monkeypatch, local.add_scenario({"name": "mcp-segment-failed", "steps": [step], "after": "repeat_last"}))
    boosters = ["bynn_depth", {"name": "bynn_segment", "require": True, "options": {"prompt": "purple gate"}}]
    data = error_of(await call("decide", {"questions": {"q": {"type": "truth", "instructions": "?"}}, "images": [BELT], "boosters": boosters}))
    assert data["status"] == 529 and data["error_type"] == "booster_unavailable"
    assert data["boosters"] == {"bynn_depth": {"status": "ok", "latency_ms": 41}, "bynn_segment": {"status": "error", "latency_ms": 61}}


async def test_decide_carries_the_cost_as_sent(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    # usage.cost is passed through whole, never recomputed: here usd is
    # not the doubles' sum of its parts, which a client adding them would get wrong.
    cost = {"usd": 0.036014322, "model_usd": 0.000014322, "boosters_usd": 0.036, "free_tier": False}
    body = {"model": "decisionnode-moe-0.9.0-rc1", "answers": {"q": {"type": "truth", "truth": 0.5}}, "usage": {"input_tokens": 341, "output_tokens": 0, "cost": cost}}
    name = local.add_scenario({"name": "mcp-cost", "steps": [{"status": 200, "body": body}], "after": "repeat_last"})
    use_scenario(local, monkeypatch, name)
    data = payload(await call("decide", {"questions": {"q": {"type": "truth", "instructions": "?"}}}))
    assert data["cost"] == cost
    many = payload(await call("decide_many", {"items": [{"id": "a"}], "questions": {"q": {"type": "truth", "instructions": "?"}}}))
    assert many["results"]["a"]["cost"] == cost


async def test_decide_without_a_cost(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"model": "decisionnode-moe-0.9.0-rc1", "answers": {"q": {"type": "truth", "truth": 0.5}}, "usage": {"input_tokens": 1, "output_tokens": 0}}
    name = local.add_scenario({"name": "mcp-no-cost", "steps": [{"status": 200, "body": body}], "after": "repeat_last"})
    use_scenario(local, monkeypatch, name)
    data = payload(await call("decide", {"questions": {"q": {"type": "truth", "instructions": "?"}}}))
    assert "cost" not in data


async def test_one_question_tools_carry_the_cost_as_sent(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    # Every tool that asks a question carries the answer's usage.cost whole,
    # beside the answer's own fields.
    cost = {"usd": 0.000014322, "model_usd": 0.000014322, "boosters_usd": 0.0, "free_tier": False}
    body = {"model": "decisionnode-moe-0.9.0-rc1", "answers": {"check": {"type": "truth", "truth": 0.5}}, "usage": {"input_tokens": 341, "output_tokens": 0, "cost": cost}}
    name = local.add_scenario({"name": "mcp-truth-cost", "steps": [{"status": 200, "body": body}], "after": "repeat_last"})
    use_scenario(local, monkeypatch, name)
    data = payload(await call("truth", {"instructions": "?"}))
    assert data == {"truth": 0.5, "cost": cost}


async def test_decide_replays_recorded_answers(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, "response/decide-three-types")
    data = payload(await call("decide", {"state": "s", "questions": QUESTIONS}))
    assert data["answers"]["route"]["probabilities"] == {"account": 0.05, "billing": 0.87, "bug": 0.08}
    assert data["answers"]["urgency"]["legend"]["2"] == "urgent"
    assert data["input_tokens"] == 62


async def test_safety_appears_only_when_flagged(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, "response/safety-flagged")
    data = payload(await call("decide", {"state": "s", "questions": {"q": {"type": "truth", "instructions": "?"}}}))
    assert data["safety"] == {"checked": True, "harm": 0.912, "self_harm": 0.004, "flagged": ["harm"]}
    use_scenario(local, monkeypatch, "response/safety-unchecked")
    data = payload(await call("decide", {"state": "s", "questions": {"q": {"type": "truth", "instructions": "?"}}}))
    assert "safety" not in data


async def test_truth_with_a_flag(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    name = local.add_scenario(
        {
            "name": "mcp-truth-flagged",
            "steps": [
                {
                    "status": 200,
                    "headers": {"x-decisionnode-safety": "harm=0.031, self_harm=0.81;flagged"},
                    "body": {
                        "model": "decisionnode-moe-0.9.0-rc1",
                        "answers": {"check": {"type": "truth", "truth": 0.5}},
                        "usage": {"input_tokens": 3, "output_tokens": 0},
                    },
                }
            ],
            "after": "repeat_last",
        }
    )
    use_scenario(local, monkeypatch, name)
    data = payload(await call("truth", {"instructions": "?"}))
    assert data == {
        "truth": 0.5,
        "safety": {"checked": True, "harm": 0.031, "self_harm": 0.81, "flagged": ["self_harm"]},
    }


async def test_decide_many_one_failure_does_not_fail_the_call(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, "behaviour/decide-many-one-failure")
    result = await call(
        "decide_many",
        {
            "items": [{"id": "first", "state": "a"}, {"id": "second", "state": "b"}, {"id": "third", "state": "c"}],
            "questions": {"q": {"type": "truth", "instructions": "Item?"}},
        },
    )
    assert not result.is_error
    data = payload(result)
    assert list(data["results"]) == ["first", "second", "third"]
    assert data["results"]["first"]["answers"]["q"]["truth"] == 0.1
    assert data["results"]["second"]["error"]["status"] == 422
    assert "answers" in data["results"]["third"]
    assert data["results"]["third"]["answers"]["q"]["truth"] == 0.3
    assert data["input_tokens"] == 10  # the two answered items, 5 each
    assert len(local.requests()) == 3  # the 422 is not retried


async def test_decide_many_sends_eight_at_a_time(local: Mock) -> None:
    items = [{"id": "m%02d" % i, "state": "message %d" % i} for i in range(20)]
    data = payload(await call("decide_many", {"items": items, "questions": {"refund": {"type": "truth", "instructions": "Refund?"}}}))
    assert list(data["results"]) == [i["id"] for i in items]
    assert all("answers" in r for r in data["results"].values())
    assert len(local.requests()) == 20
    assert local.max_in_flight() <= 8


async def test_decide_many_input_limits(local: Mock) -> None:
    q = {"q": {"type": "truth", "instructions": "?"}}
    too_many = await call("decide_many", {"items": [{"id": str(i)} for i in range(51)], "questions": q})
    assert too_many.is_error
    dupes = await call("decide_many", {"items": [{"id": "a"}, {"id": "a"}], "questions": q})
    assert dupes.is_error and dupes.structured_content["error_type"] == "invalid_request_error"
    assert local.requests() == []


async def test_models(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, "response/models-list")
    data = payload(await call("models", {}))
    names = [m["name"] for m in data["models"]]
    assert any(n.startswith("decisionnode-moe-") for n in names) and any(n.startswith("decisionnode-flash-") for n in names)
    assert data["models"][0]["max_input_tokens"] == 32768
    req = local.requests()[0]
    assert (req["method"], req["path"]) == ("GET", "/v1/models")


# --------------------------------------------------------------------------- errors

ERROR_KEYS = {"error_type", "message", "reason", "status", "request_id"}


def error_of(result: Any) -> Dict[str, Any]:
    assert result.is_error
    data = payload(result)
    assert ERROR_KEYS <= set(data)
    return data


@pytest.mark.parametrize(
    "case, status, error_type",
    [
        ("400-too-many-levels", 400, "invalid_request_error"),
        ("400-unknown-model", 400, "api_usage_error"),
        ("400-max-tokens-exceeded", 400, "max_tokens_exceeded"),
        ("401-authentication", 401, "authentication_error"),
        ("402-insufficient-credit", 402, "insufficient_credit_error"),
        ("404-not-found", 404, "not_found_error"),
        ("422-missing-criteria", 422, "validation_error"),
    ],
)
async def test_api_errors_are_error_results(
    local: Mock, monkeypatch: pytest.MonkeyPatch, case: str, status: int, error_type: str
) -> None:
    use_scenario(local, monkeypatch, "error/" + case)
    data = error_of(await call("decide", {"state": "s", "questions": {"q": {"type": "truth", "instructions": "?"}}}))
    assert data["status"] == status
    assert data["error_type"] == error_type
    assert data["request_id"]
    assert len(local.requests()) == 1  # never retried


@pytest.mark.parametrize(
    "case, status, error_type",
    [("403-permission-denied", 403, "permission_error"), ("418-unmapped-status", 418, None)],
)
async def test_permission_denied_and_unmapped_statuses(
    local: Mock, monkeypatch: pytest.MonkeyPatch, case: str, status: int, error_type: Any
) -> None:
    use_scenario(local, monkeypatch, "error/" + case)
    data = error_of(await call("decide", {"state": "s", "questions": {"q": {"type": "truth", "instructions": "?"}}}))
    assert data["status"] == status
    assert "category" not in data  # not a refusal
    if error_type:
        assert data["error_type"] == error_type
    assert data["message"]


async def test_reason_is_the_servers_words(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, "error/400-too-many-levels")
    data = error_of(await call("score", {"instructions": "?", "levels": [str(i) for i in range(11)]}))
    assert data["reason"] and "10 levels" in data["reason"]


async def test_validation_entries(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, "error/422-missing-criteria")
    data = error_of(await call("decide", {"questions": {"q": {"type": "score", "instructions": "?"}}}))
    assert isinstance(data["entries"], list) and data["entries"]


async def test_refused_request(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, "error/403-refusal-harm")
    data = error_of(await call("truth", {"instructions": "?", "state": "s"}))
    assert data["error_type"] == "refusal_error"
    assert data["category"] == "harm"
    assert data["status"] == 403
    assert data["message"].startswith("This request was refused under the usage policy")
    assert len(local.requests()) == 1  # a refusal is not retried


async def test_temporary_failures_are_retried_first(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, "behaviour/retry-529-honours-retry-after-ms")
    result = await call("decide", {"state": "s", "questions": {"q": {"type": "truth", "instructions": "?"}}})
    assert not result.is_error
    assert len(local.requests()) > 1


async def test_retries_exhausted(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    use_scenario(local, monkeypatch, "error/500-server-error")
    data = error_of(await call("truth", {"instructions": "?"}))
    assert data["status"] == 500 and data["error_type"]
    assert len(local.requests()) == 3  # the first attempt and two retries


async def test_rate_limit_error_carries_retry_after(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    name = local.add_scenario(
        {
            "name": "mcp-429-fast",
            "steps": [
                {
                    "status": 429,
                    "headers": {"retry-after-ms": "10"},
                    "body": {"detail": {"error_type": "rate_limit_error", "message": "Rate limited."}},
                }
            ],
            "after": "repeat_last",
        }
    )
    use_scenario(local, monkeypatch, name)
    data = error_of(await call("truth", {"instructions": "?"}))
    assert data["error_type"] == "rate_limit_error" and data["retry_after"] == 0.01


async def test_bad_key_from_the_api(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECISIONNODE_API_KEY", "invalid_key")
    data = error_of(await call("truth", {"instructions": "?"}))
    assert data["status"] == 401 and data["error_type"] == "authentication_error"


async def test_missing_key_fails_before_sending(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DECISIONNODE_API_KEY")
    data = error_of(await call("truth", {"instructions": "?"}))
    assert data["error_type"] == "authentication_error"
    assert "DECISIONNODE_API_KEY" in data["message"]
    assert local.requests() == []


# --------------------------------------------------------------------------- images


async def test_images_by_url_data_and_path(local: Mock, suite_dir: Any) -> None:
    jpeg = (suite_dir / "fixtures" / "images" / "pixel.jpg").read_bytes()
    png = base64.b64encode((suite_dir / "fixtures" / "images" / "pixel.png").read_bytes()).decode()
    images = [
        {"id": "a", "url": "https://example.com/a.jpg"},
        {"id": "b", "data": png, "media_type": "image/png"},
        {"id": "c", "path": str(suite_dir / "fixtures" / "images" / "pixel.jpg")},
    ]
    result = await call("truth", {"instructions": "Same receipt?", "images": images})
    assert not result.is_error, result.content
    sent = sent_body(local)["images"]
    assert sent[0] == {"id": "a", "url": "https://example.com/a.jpg"}
    assert sent[1] == {"id": "b", "media_type": "image/png", "data": png}
    assert sent[2] == {"id": "c", "media_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}


async def test_image_needs_one_source(local: Mock) -> None:
    data = error_of(await call("truth", {"instructions": "?", "images": [{"id": "a"}]}))
    assert data["error_type"] == "invalid_request_error"
    data = error_of(await call("truth", {"instructions": "?", "images": [{"id": "a", "data": "AAAA"}]}))
    assert "media_type" in data["message"]
    assert local.requests() == []


async def test_hosted_image_data_over_3_mib_is_refused_before_a_request(hosted: Mock) -> None:
    half = "A" * (api.HOSTED_IMAGE_DATA_LIMIT // 2 + 4)
    images = [{"id": "a", "data": half, "media_type": "image/png"}, {"id": "b", "data": half, "media_type": "image/png"}]
    data = error_of(await call("truth", {"instructions": "?", "images": images}))
    assert data["error_type"] == "invalid_request_error"
    assert "3 MiB" in data["message"] and "https 'url'" in data["message"]
    assert hosted.requests() == []


def png(size: int) -> bytes:
    """A PNG of `size` bytes with a real 1x1 header, which the SDK reads before sending."""
    ihdr = b"IHDR" + struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    head = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + ihdr + struct.pack(">I", zlib.crc32(ihdr))
    return head + bytes(size - len(head))


async def test_local_image_data_has_no_3_mib_limit(local: Mock) -> None:
    big = base64.b64encode(png(api.HOSTED_IMAGE_DATA_LIMIT)).decode()
    assert len(big) > api.HOSTED_IMAGE_DATA_LIMIT
    result = await call("truth", {"instructions": "?", "images": [{"id": "a", "data": big, "media_type": "image/png"}]})
    assert not result.is_error, payload(result)
    assert "3 MiB" not in json.dumps(payload(result))
    assert sent_body(local)["images"] == [{"id": "a", "media_type": "image/png", "data": big}]


async def test_unreadable_path(local: Mock) -> None:
    data = error_of(await call("truth", {"instructions": "?", "images": [{"id": "a", "path": "/no/such/file.jpg"}]}))
    assert "cannot read" in data["message"]
    assert local.requests() == []


# --------------------------------------------------------------------------- rate limit


async def test_rate_limit_per_key(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.settings, "rate_limit_per_minute", 2)
    monkeypatch.setattr(api, "rate_limiter", api._RateLimiter())
    assert not (await call("truth", {"instructions": "?"})).is_error
    assert not (await call("truth", {"instructions": "?"})).is_error
    data = error_of(await call("truth", {"instructions": "?"}))
    assert data["error_type"] == "rate_limit_error" and data["status"] == 429 and data["retry_after"] > 0
    assert len(local.requests()) == 2
    # decide_many counts one request per item
    monkeypatch.setattr(api, "rate_limiter", api._RateLimiter())
    many = await call("decide_many", {"items": [{"id": "1"}, {"id": "2"}, {"id": "3"}], "questions": {"q": {"type": "truth", "instructions": "?"}}})
    assert error_of(many)["error_type"] == "rate_limit_error"


def test_rate_limit_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    saved = (api.settings.local, api.settings.rate_limit_per_minute)
    try:
        api.configure(local=False, env={})
        assert api.settings.rate_limit_per_minute == 120 == api.DEFAULT_RATE_LIMIT_PER_MINUTE
        api.configure(local=False, env={"DECISIONNODE_MCP_RATE_LIMIT": "30"})
        assert api.settings.rate_limit_per_minute == 30
        api.configure(local=False, env={"DECISIONNODE_MCP_RATE_LIMIT": "0"})
        assert api.settings.rate_limit_per_minute == 0
        api.configure(local=True, env={})
        assert api.settings.rate_limit_per_minute == 0
    finally:
        api.settings.local, api.settings.rate_limit_per_minute = saved


@pytest.mark.parametrize("value", ["-1", "-120", "not a number", "1.5"])
def test_a_bad_rate_limit_is_a_startup_error(value: str) -> None:
    saved = (api.settings.local, api.settings.rate_limit_per_minute)
    try:
        with pytest.raises(api.ConfigurationError, match="DECISIONNODE_MCP_RATE_LIMIT"):
            api.configure(local=False, env={"DECISIONNODE_MCP_RATE_LIMIT": value})
        assert (api.settings.local, api.settings.rate_limit_per_minute) == saved
    finally:
        api.settings.local, api.settings.rate_limit_per_minute = saved


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_the_rate_limiter_tracks_a_bounded_number_of_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.settings, "rate_limit_per_minute", 5)
    limiter = api._RateLimiter(max_size=100)
    refused = 0
    for i in range(1000):  # all of them within the minute, so none may be forgotten
        try:
            limiter.take("key-%d" % i)
        except api.ToolFailure as exc:
            assert exc.payload["error_type"] == "rate_limit_error" and exc.payload["status"] == 429
            refused += 1
    assert len(limiter) == 100 and refused == 900
    # the first keys are the ones kept, with their counts intact
    for _ in range(4):
        limiter.take("key-0")
    with pytest.raises(api.ToolFailure):
        limiter.take("key-0")


def test_a_key_at_its_limit_stays_limited_after_10000_other_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.settings, "rate_limit_per_minute", 5)
    clock = _Clock()
    monkeypatch.setattr(api.time, "monotonic", clock)
    limiter = api._RateLimiter()
    for _ in range(5):
        limiter.take("victim")
    with pytest.raises(api.ToolFailure):
        limiter.take("victim")
    for i in range(10_000):
        clock.now += 0.001
        try:
            limiter.take("junk-%d" % i)
        except api.ToolFailure as exc:
            assert exc.payload["error_type"] == "rate_limit_error"
    assert len(limiter) == 10_000
    assert api.key_digest("victim") in limiter._calls
    with pytest.raises(api.ToolFailure) as caught:
        limiter.take("victim")
    assert caught.value.payload["error_type"] == "rate_limit_error" and caught.value.payload["status"] == 429
    # once its minute is over the key is served again
    clock.now = 1000.0 + 60.0
    limiter.take("victim")


def test_rejected_keys_do_not_lock_new_keys_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.settings, "rate_limit_per_minute", 5)
    clock = _Clock()
    monkeypatch.setattr(api.time, "monotonic", clock)
    limiter = api._RateLimiter(max_size=100)
    monkeypatch.setattr(api, "rate_limiter", limiter)
    rejected = api._RejectedKeys()
    payload = {"error_type": "authentication_error", "message": "Invalid API key.", "status": 401}
    # a flood of invalid keys, each answered 401 by the API, as the client() wrapper does
    for i in range(1_000):
        clock.now += 0.001
        limiter.take("junk-%d" % i)
        rejected.remember("junk-%d" % i, payload)
    assert len(limiter) == 0
    # a new, valid key is still served
    limiter.take("valid")
    assert api.key_digest("valid") in limiter._calls


def test_the_rate_limiter_replaces_only_an_expired_key_at_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.settings, "rate_limit_per_minute", 5)
    clock = _Clock()
    monkeypatch.setattr(api.time, "monotonic", clock)
    limiter = api._RateLimiter(max_size=3)
    for name in ("a", "b", "c"):
        limiter.take(name)
        clock.now += 10.0
    # every window is live: the new key is refused and nothing is forgotten
    with pytest.raises(api.ToolFailure) as caught:
        limiter.take("d")
    assert caught.value.payload["error_type"] == "rate_limit_error"
    assert caught.value.payload["retry_after"] == pytest.approx(30.0)
    assert {api.key_digest(k) for k in "abc"} == set(limiter._calls)
    limiter.take("a")  # "a" now has the newest call, "b" the oldest
    clock.now = 1010.0 + 60.0  # "b"'s window has expired, "c"'s has not
    limiter.take("d")
    assert {api.key_digest(k) for k in "acd"} == set(limiter._calls)
    with pytest.raises(api.ToolFailure):
        limiter.take("e")  # "c" is live until 1080
    clock.now = 1080.0
    limiter.take("e")
    assert {api.key_digest(k) for k in "ade"} == set(limiter._calls)
    # a refused call does not count as use and does not reorder the keys
    for _ in range(4):
        limiter.take("a")
    with pytest.raises(api.ToolFailure):
        limiter.take("a")


@pytest.mark.parametrize(
    "url",
    ["https://api.example.com", "http://127.0.0.1:8080", "http://localhost:9000/v1", "http://[::1]:8000", "http://127.1.2.3"],
)
def test_base_url_accepted(url: str) -> None:
    api.check_base_url(url)


@pytest.mark.parametrize(
    "url", ["http://api.example.com", "http://10.0.0.5:8080", "http://localhost.example.com", "ftp://api.example.com", "api.example.com"]
)
def test_base_url_refused(url: str) -> None:
    with pytest.raises(api.ConfigurationError, match="DECISIONNODE_BASE_URL"):
        api.check_base_url(url)


def test_plain_http_base_url_is_a_startup_error() -> None:
    with pytest.raises(api.ConfigurationError, match="https"):
        api.configure(local=True, env={"DECISIONNODE_BASE_URL": "http://api.example.com"})


# --------------------------------------------------------------------------- rejected keys


async def test_a_rejected_key_is_refused_without_a_request(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECISIONNODE_API_KEY", "invalid_key")
    first = error_of(await call("truth", {"instructions": "?"}))
    assert first["status"] == 401 and first["request_id"]
    assert len(local.requests()) == 1
    second = error_of(await call("truth", {"instructions": "?"}))
    assert len(local.requests()) == 1  # refused here, no second request
    assert second == dict(first, request_id=None)
    # A different key is not affected.
    monkeypatch.setenv("DECISIONNODE_API_KEY", TEST_KEY)
    assert not (await call("truth", {"instructions": "?"})).is_error
    assert len(local.requests()) == 2


async def test_a_key_rejected_inside_decide_many_is_remembered(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECISIONNODE_API_KEY", "invalid_key")
    many = await call("decide_many", {"items": [{"id": "1"}], "questions": {"q": {"type": "truth", "instructions": "?"}}})
    assert payload(many)["results"]["1"]["error"]["status"] == 401
    data = error_of(await call("truth", {"instructions": "?"}))
    assert data["status"] == 401 and data["error_type"] == "authentication_error"
    assert len(local.requests()) == 1


def test_rejected_keys_expire_and_stay_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(api.time, "monotonic", lambda: clock[0])
    cache = api._RejectedKeys(ttl=60.0, max_size=3)
    error = {"error_type": "authentication_error", "message": "bad key", "reason": None, "status": 401, "request_id": "r"}
    cache.remember("k0", error)
    with pytest.raises(api.ToolFailure) as caught:
        cache.check("k0")
    assert caught.value.payload == dict(error, request_id=None)
    clock[0] += 60.0
    cache.check("k0")  # expired: let through
    for i in range(5):
        cache.remember("k%d" % i, error)
    assert len(cache) == 3
    cache.check("k0")  # the oldest were dropped
    with pytest.raises(api.ToolFailure):
        cache.check("k4")
    assert all("k" not in digest[:2] and len(digest) == 64 for digest in cache._entries)  # digests only


# --------------------------------------------------------------------------- image urls


@pytest.mark.parametrize(
    "url", ["http://example.com/a.jpg", "file:///etc/passwd", "ftp://example.com/a.jpg", "//example.com/a.jpg", "a.jpg"]
)
async def test_image_url_must_be_https(local: Mock, url: str) -> None:
    data = error_of(await call("truth", {"instructions": "?", "images": [{"id": "a", "url": url}]}))
    assert data["error_type"] == "invalid_request_error"
    assert "https://" in data["message"]
    assert local.requests() == []
