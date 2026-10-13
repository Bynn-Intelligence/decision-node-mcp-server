"""`decide_video`: frames sent as given, video files sampled into frames on the user's
machine by the local server, and nothing but frames sent."""

from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest
from conftest import Mock, use_scenario
from fastmcp import Client
from test_tools import call, error_of, payload, sent_body

from decisionnode_mcp import api, frames, texts
from decisionnode_mcp.server import create_server

FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg and ffprobe are not installed")

QUESTIONS = {
    "fall": {"type": "truth", "instructions": "Does a person fall in the video?"},
    "forklifts": {"type": "number", "instructions": "How many forklifts pass the door?", "min": 0, "max": 50},
}
ANSWERS = {
    "fall": {"type": "truth", "truth": 0.08},
    "forklifts": {"type": "number", "number": 3, "expected": 3.2, "confidence": 0.71, "probabilities": {"2": 0.12, "3": 0.71, "4": 0.15}},
}
PIXEL = base64.b64encode(bytes.fromhex("ffd8ffe000104a46494600")).decode()


def frames_of(times: List[float]) -> List[Dict[str, Any]]:
    return [{"t": t, "media_type": "image/jpeg", "data": PIXEL} for t in times]


def answered(mock: Mock, monkeypatch: pytest.MonkeyPatch, **extra: Any) -> None:
    """The mock answers like an API that reads videos."""
    headers = extra.pop("headers", {})
    body = {
        "model": "decisionnode-moe-0.9.0-rc1",
        "answers": ANSWERS,
        "usage": {"input_tokens": 1342, "output_tokens": 0, "video_frames": 20, "video_seconds": 10.0},
        **extra,
    }
    name = mock.add_scenario({"name": "mcp-video-" + mock.session, "steps": [{"status": 200, "headers": headers, "body": body}], "after": "repeat_last"})
    use_scenario(mock, monkeypatch, name)


# --------------------------------------------------------------------------- listing


async def test_decide_video_is_listed_with_the_video_guide(local: Mock) -> None:
    async with Client(create_server()) as client:
        tools = await client.list_tools()
        guide = (await client.call_tool("guide", {})).structured_content["guide"]
    assert [t.name for t in tools][-2:] == ["decide_video", "models"]
    assert guide == texts.GUIDE and "## Video" in guide
    tool = next(t for t in tools if t.name == "decide_video")
    assert tool.annotations is not None and tool.annotations.readOnlyHint is True
    assert tool.inputSchema["required"] == ["questions", "videos"]
    assert local.requests() == []



# --------------------------------------------------------------------------- frames


async def test_frames_are_sent_as_given(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    answered(local, monkeypatch, warnings=["Video 'dock' was read at 1 frame a second."])
    video = {"id": "dock", "fps": 2, "start": 30, "end": 40, "frames": frames_of([30.0, 30.5, 31.0])}
    image = {"id": "plan", "url": "https://example.com/plan.png"}
    data = payload(
        await call("decide_video", {"state": "Warehouse camera 4.", "questions": QUESTIONS, "videos": [video], "images": [image]})
    )
    assert data == {
        "answers": ANSWERS,
        "model": "decisionnode-moe-0.9.0-rc1",
        "input_tokens": 1342,
        "video_frames": 20,
        "video_seconds": 10.0,
        "warnings": ["Video 'dock' was read at 1 frame a second."],
    }
    req = local.requests()[0]
    assert (req["method"], req["path"]) == ("POST", "/v1/decide")
    assert req["body"] == {
        "model": "decisionnode-moe",
        "state": "Warehouse camera 4.",
        "questions": QUESTIONS,
        "images": [image],
        "videos": [video],
    }
    assert req["headers"]["authorization"].startswith("Bearer ")


async def test_a_request_with_videos_keeps_the_safety_flag(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    answered(local, monkeypatch, headers={"x-decisionnode-safety": "harm=0.912;flagged, self_harm=0.004"})
    data = payload(await call("decide_video", {"questions": QUESTIONS, "videos": [{"id": "v", "frames": frames_of([0.0])}]}))
    assert data["safety"] == {"checked": True, "harm": 0.912, "self_harm": 0.004, "flagged": ["harm"]}
    assert "warnings" not in data


async def test_api_errors_and_refusals_are_error_results(local: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    args = {"questions": QUESTIONS, "videos": [{"id": "v", "frames": frames_of([0.0])}]}
    use_scenario(local, monkeypatch, "error/400-max-tokens-exceeded")
    data = error_of(await call("decide_video", args))
    assert data["status"] == 400 and data["error_type"] == "max_tokens_exceeded" and data["request_id"]
    use_scenario(local, monkeypatch, "error/403-refusal-harm")
    data = error_of(await call("decide_video", args))
    assert data["error_type"] == "refusal_error" and data["category"] == "harm"
    assert len(local.requests()) == 2  # neither is retried


async def test_a_video_needs_frames_or_a_path(local: Mock) -> None:
    for video in ({"id": "v"}, {"id": "v", "frames": frames_of([0.0]), "path": "/tmp/v.mp4"}):
        data = error_of(await call("decide_video", {"questions": QUESTIONS, "videos": [video]}))
        assert data["error_type"] == "invalid_request_error" and "'frames'" in data["message"]
    data = error_of(await call("decide_video", {"questions": QUESTIONS, "videos": [{"id": "v", "frames": frames_of([0.0]), "max_frames": 4}]}))
    assert "'path' only" in data["message"]
    assert local.requests() == []


@pytest.mark.parametrize("video", [{"id": "v", "url": "https://example.com/v.mp4"}, {"id": "v", "data": "AAAA", "media_type": "video/mp4"}])
async def test_a_video_file_or_url_is_never_sent(local: Mock, video: Dict[str, Any]) -> None:
    result = await call("decide_video", {"questions": QUESTIONS, "videos": [video]})
    assert result.is_error
    assert local.requests() == []


async def test_the_hosted_server_never_reads_a_video_path(hosted: Mock, tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not read")
    data = error_of(await call("decide_video", {"questions": QUESTIONS, "videos": [{"id": "v", "path": str(path)}]}))
    assert "local server" in data["message"] and "'frames'" in data["message"]
    assert hosted.requests() == []


async def test_hosted_frames_count_toward_the_3_mib_of_data(hosted: Mock) -> None:
    half = "A" * (api.HOSTED_IMAGE_DATA_LIMIT // 2 + 4)
    images = [{"id": "a", "data": half, "media_type": "image/jpeg"}]
    video = {"id": "v", "frames": [{"t": 0, "media_type": "image/jpeg", "data": half}]}
    data = error_of(await call("decide_video", {"questions": QUESTIONS, "videos": [video], "images": images}))
    assert data["error_type"] == "invalid_request_error"
    assert "3 MiB" in data["message"] and "video frames" in data["message"]
    assert hosted.requests() == []


# --------------------------------------------------------------------------- video files


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Five seconds at 10 frames a second, 320 x 240, with a sound track."""
    if not FFMPEG:
        pytest.skip("ffmpeg and ffprobe are not installed")
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=5", "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )
    return path


@needs_ffmpeg
async def test_a_video_file_is_sent_as_frames_only(local: Mock, monkeypatch: pytest.MonkeyPatch, clip: Path) -> None:
    answered(local, monkeypatch)
    video = {"id": "dock", "path": str(clip), "fps": 2, "start": 1, "end": 3}
    data = payload(await call("decide_video", {"questions": QUESTIONS, "videos": [video]}))
    assert "warnings" not in data
    sent = sent_body(local)["videos"]
    assert [sorted(v) for v in sent] == [["fps", "frames", "id"]]
    assert sent[0]["id"] == "dock" and sent[0]["fps"] == 2.0
    assert [f["t"] for f in sent[0]["frames"]] == [1.0, 1.5, 2.0, 2.5]
    for frame in sent[0]["frames"]:
        assert frame["media_type"] == "image/jpeg"
        assert base64.b64decode(frame["data"]).startswith(b"\xff\xd8")  # a JPEG, never the file
    body_text = local.requests()[0]["body_text"]
    assert base64.b64encode(clip.read_bytes()[:48]).decode()[:40] not in body_text


@needs_ffmpeg
async def test_a_long_window_is_cut_and_the_result_says_so(local: Mock, monkeypatch: pytest.MonkeyPatch, clip: Path) -> None:
    answered(local, monkeypatch, warnings=["from the API"])
    data = payload(await call("decide_video", {"questions": QUESTIONS, "videos": [{"id": "dock", "path": str(clip), "max_frames": 3}]}))
    assert [f["t"] for f in sent_body(local)["videos"][0]["frames"]] == [0.0, 0.5, 1.0]
    assert len(data["warnings"]) == 2
    assert data["warnings"][0].startswith("video 'dock': the window holds more than 3 frames at 2 a second")
    assert data["warnings"][1] == "from the API"


@needs_ffmpeg
def test_the_sampler_takes_frames_by_the_api_rule(clip: Path) -> None:
    sampled = frames.sample(str(clip), fps=4, start=0, end=2, max_frames=32)
    # 10 frames a second sampled at 4 a second: a frame at least 0.25 s after the one before.
    assert [t for t, _ in sampled.frames] == [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
    assert not sampled.cut


@needs_ffmpeg
async def test_an_unreadable_video_file(local: Mock, tmp_path: Path) -> None:
    path = tmp_path / "notes.mp4"
    path.write_text("not a video")
    data = error_of(await call("decide_video", {"questions": QUESTIONS, "videos": [{"id": "v", "path": str(path)}]}))
    assert data["error_type"] == "invalid_request_error" and "ffprobe cannot read the file" in data["message"]
    assert local.requests() == []


@pytest.mark.parametrize("path", ["/no/such/clip.mp4", "https://example.com/clip.mp4", "-i"])
async def test_a_path_is_only_ever_a_local_file(local: Mock, path: str) -> None:
    data = error_of(await call("decide_video", {"questions": QUESTIONS, "videos": [{"id": "v", "path": path}]}))
    assert "no video file at" in data["message"]
    assert local.requests() == []


async def test_without_ffmpeg_a_video_file_is_refused_clearly(
    local: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00")
    monkeypatch.setattr(frames.shutil, "which", lambda name: None)
    data = error_of(await call("decide_video", {"questions": QUESTIONS, "videos": [{"id": "v", "path": str(path)}]}))
    assert data["error_type"] == "invalid_request_error"
    assert "needs ffmpeg and ffprobe" in data["message"] and "send the video as frames" in data["message"]
    assert local.requests() == []
