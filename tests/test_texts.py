"""The texts the agent reads follow their rules."""

from __future__ import annotations

import re
from pathlib import Path

from decisionnode_mcp import texts

DESCRIPTIONS = {
    "guide": texts.GUIDE_DESCRIPTION,
    "truth": texts.TRUTH_DESCRIPTION,
    "choice": texts.CHOICE_DESCRIPTION,
    "score": texts.SCORE_DESCRIPTION,
    "number": texts.NUMBER_DESCRIPTION,
    "point": texts.POINT_DESCRIPTION,
    "box": texts.BOX_DESCRIPTION,
    "decide": texts.DECIDE_DESCRIPTION,
    "decide_many": texts.DECIDE_MANY_DESCRIPTION,
    "decide_video": texts.DECIDE_VIDEO_DESCRIPTION,
    "models": texts.MODELS_DESCRIPTION,
}

README = Path(__file__).resolve().parent.parent / "README.md"


def sentences(text: str) -> list:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z`])", text) if s]


def test_each_description_says_when_then_what_then_cost() -> None:
    for name, text in DESCRIPTIONS.items():
        parts = sentences(text)
        assert parts[0].startswith("Use this"), name
        assert parts[1].startswith("Returns"), name
        assert re.search(r"Costs (nothing|one request)", text), name


def test_api_tools_name_the_flash_alias() -> None:
    for name in ("truth", "choice", "score", "number", "point", "box", "decide", "decide_many", "decide_video"):
        assert "decisionnode-flash" in DESCRIPTIONS[name], name


def test_instructions() -> None:
    text = texts.INSTRUCTIONS
    assert "`guide`" in text and "once" in text
    assert "knows nothing else" in text
    assert "does not write text" in text
    assert "(number)" in text and "(point, box)" in text
    assert len(sentences(text)) <= 6


def test_guide_covers_the_essentials() -> None:
    g = texts.GUIDE
    for needle in ("`truth`", "`choice`", "`score`", "`number`", "`decide`", "`decide_many`",
                   "`point`", "`box`", "## Points and boxes", "hand the image to a person",
                   "`decide_video`", "## Video", '"path"', "ffmpeg", "no video file, URL or sound",
                   "0.5 means the model cannot tell", "`expected`", "Up to 16 images",
                   "Ask one narrow thing", "Put everything the model needs in `state`", "Limits"):
        assert needle in g, needle
    assert len(g.splitlines()) < 120  # one page


def test_point_and_box_descriptions_explain_present_and_confidence() -> None:
    for text in (texts.POINT_DESCRIPTION, texts.BOX_DESCRIPTION):
        assert "`present`" in text and "`confidence`" in text
        assert "hand the image to a person" in text
        assert "at least one image" in text and "`image` names the one to look in" in text


def test_video_is_frames_only_in_every_text() -> None:
    assert "Only frames are sent" in texts.DECIDE_VIDEO_DESCRIPTION
    assert "no video file, URL or sound" in texts.DECIDE_VIDEO_DESCRIPTION


def test_no_dashes_and_no_older_type_name_in_what_people_and_agents_read() -> None:
    shown = [texts.INSTRUCTIONS, texts.GUIDE, *DESCRIPTIONS.values(), README.read_text()]
    for text in shown:
        assert "\u2014" not in text and "\u2013" not in text  # em and en dashes
        assert "noul" not in text.lower()


def test_everything_is_written_as_available() -> None:
    shown = [texts.INSTRUCTIONS, texts.GUIDE, *DESCRIPTIONS.values(), README.read_text()]
    for text in shown:
        lowered = text.lower()
        for phrase in ("coming soon", "model release", "not open yet", "when the api enables", "until then", "is planned"):
            assert phrase not in lowered, phrase
