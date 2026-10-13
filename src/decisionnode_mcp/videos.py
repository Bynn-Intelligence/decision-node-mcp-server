"""The `decide_video` tool: questions about videos sent as frames."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, Dict, List, Optional, Tuple

from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from pydantic import Field

from . import api, frames, schemas, texts
from .decisions import Questions
from .schemas import IMAGES_DESCRIPTION, MODEL_DESCRIPTION, STATE_DESCRIPTION, VIDEOS_DESCRIPTION, ImageInput, VideoInput

mcp = FastMCP("DecisionNode videos")


async def _sample(video: api.VideoFile) -> Tuple[Dict[str, Any], Optional[str]]:
    """A video file as the frames form, and a warning when its window held more frames than were taken."""
    try:
        sampled = await asyncio.to_thread(
            frames.sample, video.path, fps=video.fps, start=video.start, end=video.end, max_frames=video.max_frames
        )
    except frames.FrameError as exc:
        raise api.ToolFailure("invalid_request_error", "video %r: %s." % (video.id, str(exc).rstrip("."))) from None
    warning = None
    if sampled.cut:
        warning = (
            "video %r: the window holds more than %d frames at %g a second; the first %d were sent, "
            "from %g to %g s. Send a shorter window, a lower fps or a higher max_frames."
            % (video.id, video.max_frames, video.fps, video.max_frames, sampled.frames[0][0], sampled.frames[-1][0])
        )
    return frames.video_object(video.id, sampled, video.fps), warning


@mcp.tool(
    name="decide_video",
    description=texts.DECIDE_VIDEO_DESCRIPTION,
    annotations=schemas.annotations("Decide about videos"),
    output_schema=schemas.DECIDE_VIDEO_OUTPUT,
)
async def decide_video(
    questions: Questions,
    videos: Annotated[List[VideoInput], Field(min_length=1, description=VIDEOS_DESCRIPTION)],
    state: Annotated[Any, Field(description=STATE_DESCRIPTION)] = None,
    images: Annotated[Optional[List[ImageInput]], Field(description=IMAGES_DESCRIPTION)] = None,
    model: Annotated[Optional[str], Field(description=MODEL_DESCRIPTION)] = None,
) -> ToolResult:
    async def call() -> Dict[str, Any]:
        image_args = [i.model_dump() for i in images] if images else None
        sdk_images = api.images_for_api(image_args)
        planned = api.videos_for_api([v.model_dump() for v in videos], image_args)
        wire = {qid: q.wire() for qid, q in questions.items()}
        warnings: List[str] = []
        async with api.client() as dn:
            # Files are sampled once the key is known to be there, and one after the other:
            # ffmpeg already uses every core for one.
            sent = []
            for video in planned:
                if isinstance(video, api.VideoFile):
                    video, warning = await _sample(video)
                    if warning:
                        warnings.append(warning)
                sent.append(video)
            decision = await api.decide(dn, state, wire, sdk_images, model, videos=sent)
        answers = decision.raw.get("answers")
        usage = decision.raw.get("usage") if isinstance(decision.raw.get("usage"), dict) else {}
        result: Dict[str, Any] = {
            "answers": dict(answers) if isinstance(answers, dict) else {},
            "model": decision.model,
            "input_tokens": decision.usage.input_tokens,
        }
        for key in ("video_frames", "video_seconds"):
            if key in usage:
                result[key] = usage[key]
        api.with_cost(result, decision)
        sent_warnings = decision.raw.get("warnings")
        if isinstance(sent_warnings, list):
            warnings += [str(w) for w in sent_warnings]
        if warnings:
            result["warnings"] = warnings
        return api.with_safety(result, decision)

    return await api.run(call)
