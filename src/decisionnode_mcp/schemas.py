"""Input types and output schemas shared by the tools, and their annotations."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, field_validator


def annotations(title: str) -> ToolAnnotations:
    """Every tool only reads: nothing it does changes anything, and repeating it is safe."""
    return ToolAnnotations(
        title=title,
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )


# --------------------------------------------------------------------------- inputs

STATE_DESCRIPTION = (
    "What to judge: a string, or any JSON value (an object with named fields works well). "
    "The model sees only this, the question and any images. Leave it out for a question that "
    "stands on its own."
)
MODEL_DESCRIPTION = (
    "The model id. Leave it out for the default, `decisionnode-moe`; `decisionnode-flash` is the "
    "faster, lower-cost tier. The `models` tool lists the ids your key may use."
)
IMAGES_DESCRIPTION = (
    "Up to 16 images every question can see, within the model's token limit, each with a short, "
    "meaningful id (`receipt`, not `2`). The instructions refer to an image by its id in backticks "
    '(recommended) or by its number from 1 in this order ("image 2", "Picture 2"). Each is '
    "{id, url} with an https URL, {id, data, media_type} with base64 data, or, on the local "
    "server only, {id, path} for a file on the user's machine. On the hosted server the base64 data "
    "of one call is limited to about 3 MiB in total; send larger images by https url."
)
VIDEOS_DESCRIPTION = (
    "The videos every question can see, each referred to by its id (recommended) or by its number "
    'from 1 in this order ("video 1"), sent as frames with their times: '
    "{id, frames: [{t, media_type, data}], fps, start, end}, the frames in time order and "
    "`t` in seconds. On the local server a video may instead be {id, path} for a video file on the "
    "user's machine, which the server samples into frames with ffmpeg at `fps` within `start` and "
    "`end`. Only frames reach the API: no video file, URL or sound. The API publishes its video "
    "limits in `models`. On the hosted server the base64 data of the images and frames of one call "
    "is limited to about 3 MiB in total."
)

BOOSTERS_DESCRIPTION = (
    "Specialist models that read the call's images before the model answers; their readings come back "
    "as `evidence`. Up to 4, each once, in your order: a booster name, or {name, images, require, "
    "options, timeout_ms} to read only some images, fail the call when the booster fails "
    "(`require: true`), set its options or its time budget. The names are free text, so a new booster "
    "works without an update. The boosters: `age` (the age of the person in a picture), `ai_generated` "
    "(whether a picture was made by a generator), `face_detection` (the faces in a picture and their "
    "boxes), `face_recognition` (faces matched in a face collection; options `collection`, required, "
    "and `threshold`), `ocr` (the text a picture shows; option `prompt`), `bynn_depth` (Depth Vision: "
    "meters to the nearest obstacle straight ahead, the floor left out, in 7 directions from left to "
    "right and as a grid; option `grid_size`, 2 to 16, default 8) and `bynn_segment` (Segment Vision: "
    "each match of what you name, with its label, score, box, outline, centre, the opening of a gate or "
    "doorway, and with depth on its distance and size in meters; options `prompt`, required, one "
    "phrase as a string or 1 to 4 as an array, each at most 120 characters, `max_instances` 1 to 16 "
    "(8), `min_score` 0 to 1 (0.25), `polygon_points` 3 to 64 (24) and `depth` (true); coordinates "
    "are fractions from the top left, and descriptive phrases match better). Each picture a booster "
    "reads is billed as one booster call. Distances are estimates from one picture, for decisions and "
    "planning, never a safety stop."
)

# The API also takes `noul`, an older name of the yes-or-no type. A question sent with it is
# accepted and asked as `truth`, so its answer comes back under the current name.
LEGACY_TRUTH = "noul"

# The question types that answer where something is in an image.
PLACE_TYPES = ("point", "box")
Units = Literal["pixels", "normalized"]
UNITS_DESCRIPTION = (
    "pixels (the default): pixels of the image as sent. normalized: fractions of the width and the "
    "height, from 0 to 1."
)

# A number as the caller wrote it: a whole number is sent as 1, never turned into 1.0.
# The bounds sit on each member, so the schema states them in JSON Schema's own words.
Number = Union[int, float]
PositiveNumber = Union[Annotated[int, Field(gt=0)], Annotated[float, Field(gt=0)]]
NonNegativeNumber = Union[Annotated[int, Field(ge=0)], Annotated[float, Field(ge=0)]]
Rate = Union[Annotated[int, Field(gt=0, le=4)], Annotated[float, Field(gt=0, le=4)]]


class ImageInput(BaseModel):
    """An image beside the state."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        description=(
            "Unique within the request; a short, meaningful name (`receipt`, not `2`). The model sees the "
            "image as `Picture <n> (<id>):`, so the instructions can refer to it by this id in backticks "
            "(recommended) or by its number."
        )
    )
    url: Optional[str] = Field(default=None, description="An https URL the API fetches.")
    data: Optional[str] = Field(default=None, description="The image bytes, base64-encoded.")
    media_type: Optional[Literal["image/jpeg", "image/png", "image/webp"]] = Field(
        default=None, description="The type of `data`."
    )
    path: Optional[str] = Field(
        default=None, description="A file on the user's machine. Local server only; the hosted server refuses it."
    )


class QuestionInput(BaseModel):
    """One question, exactly in the API's form."""

    model_config = ConfigDict(extra="allow")

    type: Literal["truth", "choice", "score", "number", "point", "box"] = Field(description="The question type.")
    instructions: Any = Field(
        default=None, description="The question or statement: a string, or a JSON value holding it."
    )
    criteria: Any = Field(
        default=None,
        description=(
            "truth: {\"true\": ..., \"false\": ...}, what a yes and a no mean (optional). "
            "choice: an object of option name to description (null when the name says enough), 1 to 255 options. "
            "score: the list of levels, lowest first, 1 to 10. number, point and box: not used."
        ),
    )
    min: Optional[Number] = Field(default=None, description="number only: the lowest possible answer (default 0).")
    max: Optional[Number] = Field(
        default=None,
        description=(
            "number: the highest possible answer (default 255). point and box: the most places to answer, "
            "an integer from 2 to 16; the answer then lists every match, up to that many."
        ),
    )
    step: Optional[PositiveNumber] = Field(
        default=None,
        description="number only: the grid of possible answers, 1 for whole numbers (the default), 0.5 for halves.",
    )
    image: Optional[str] = Field(
        default=None,
        description=(
            "point and box only: the id of the image to look in (a string), needed when the request has two "
            'or more images. A reference by number ("Picture 2") goes in the instructions.'
        ),
    )
    units: Optional[Units] = Field(default=None, description="point and box only: " + UNITS_DESCRIPTION)

    @field_validator("type", mode="before")
    @classmethod
    def _current_truth_name(cls, value: Any) -> Any:
        return "truth" if value == LEGACY_TRUTH else value

    def wire(self) -> Dict[str, Any]:
        out = self.model_dump(exclude_unset=True)
        # An optional field sent as null is the same as one left out.
        for key in ("min", "max", "step", "image", "units"):
            if key in out and out[key] is None:
                del out[key]
        return out


class ItemInput(BaseModel):
    """One state of a `decide_many` call."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Your id for this item; its result comes back under it.")
    state: Any = Field(default=None, description=STATE_DESCRIPTION)
    images: Optional[List[ImageInput]] = Field(
        default=None,
        description=(
            "This item's images, as `images` of the other tools: up to 16. A point or box question needs "
            "at least one. On the hosted server the base64 data of every item's images together is "
            "limited to about 3 MiB."
        ),
    )


class BoosterInput(BaseModel):
    """A booster with settings, in the API's form."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The booster's name, such as `bynn_depth`.")
    images: Optional[List[str]] = Field(
        default=None, description="The ids of the images it reads. Left out: every image of the call."
    )
    require: Optional[bool] = Field(
        default=None,
        description="true: the call fails when this booster fails or runs out of time. Otherwise the answer comes without its reading.",
    )
    options: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "The booster's settings: `face_recognition` {collection, threshold}; `ocr` {prompt}; "
            "`bynn_depth` {grid_size}; `bynn_segment` {prompt, max_instances, min_score, polygon_points, "
            "depth}. A booster without settings takes none; the API refuses a setting it does not know."
        ),
    )
    timeout_ms: Optional[int] = Field(
        default=None,
        description="This booster's time budget in milliseconds, within the booster's own maximum. Left out: its default.",
    )

    def wire(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


BoosterEntry = Union[str, BoosterInput]


class FrameInput(BaseModel):
    """One frame of a video."""

    model_config = ConfigDict(extra="forbid")

    t: NonNegativeNumber = Field(description="The frame's time in seconds, larger than the time of the frame before it.")
    media_type: Literal["image/jpeg", "image/png", "image/webp"] = Field(description="The type of `data`.")
    data: str = Field(description="The frame's image bytes, base64-encoded.")


class VideoInput(BaseModel):
    """A video, sent as frames with their times."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        description=(
            "Unique within the request, across images and videos. The model sees the video as "
            "`Video <n> (<id>):`, so the instructions can refer to it by this id (recommended) or by its "
            "number."
        )
    )
    frames: Optional[List[FrameInput]] = Field(
        default=None, description="The frames in time order, all of one size. Give `frames` or `path`."
    )
    path: Optional[str] = Field(
        default=None,
        description=(
            "A video file on the user's machine, sampled into frames there with ffmpeg; only the frames are "
            "sent. Local server only; the hosted server refuses it."
        ),
    )
    fps: Optional[Rate] = Field(
        default=None,
        description=(
            "The most frames a second the API reads (default 2, at most 4). With `path`, the rate the "
            "frames are taken at."
        ),
    )
    start: Optional[NonNegativeNumber] = Field(
        default=None,
        description="The start of the window read, in seconds on the frames' times (with `path`, from the video's first frame).",
    )
    end: Optional[PositiveNumber] = Field(default=None, description="The end of the window read, in seconds, not included.")
    max_frames: Optional[int] = Field(
        default=None,
        ge=1,
        le=128,
        description=(
            "With `path` only: the most frames taken (default 32). A window that holds more at `fps` is cut "
            "there, and the result's `warnings` say so."
        ),
    )


# --------------------------------------------------------------------------- outputs

_NUMBER_MAP = {"type": "object", "additionalProperties": {"type": "number"}}

SAFETY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": (
        "Present only when the safety check flagged a category for this key: each category's "
        "probability and the `flagged` categories."
    ),
    "properties": {
        "checked": {"type": "boolean"},
        "harm": {"type": ["number", "null"]},
        "self_harm": {"type": ["number", "null"]},
        "flagged": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": ["string", "null"]},
    },
    "additionalProperties": True,
}

COST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": (
        "What the call cost, as the API booked it, in USD: `usd` (the amount booked), split into "
        "`model_usd` and `boosters_usd`, and `free_tier` (true when the free tier covered the model's part)."
    ),
    "properties": {
        "usd": {"type": "number"},
        "model_usd": {"type": "number"},
        "boosters_usd": {"type": "number"},
        "free_tier": {"type": "boolean"},
    },
    "additionalProperties": True,
}

TRUTH_OUTPUT: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "truth": {"type": "number", "description": "The probability that the answer is yes, 0 to 1."},
        "cost": COST_SCHEMA,
        "safety": SAFETY_SCHEMA,
    },
    "required": ["truth"],
    "additionalProperties": True,
}

CHOICE_OUTPUT: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "choice": {"type": "string", "description": "The most likely option, by its name as sent."},
        "confidence": {"type": "number", "description": "0 means no better than a guess, 1 means certain."},
        "probabilities": {**_NUMBER_MAP, "description": "Each option's probability, by option name."},
        "cost": COST_SCHEMA,
        "safety": SAFETY_SCHEMA,
    },
    "required": ["choice", "confidence", "probabilities"],
    "additionalProperties": True,
}

SCORE_OUTPUT: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "description": "The expected level index, from 0 to the number of levels minus 1."},
        "confidence": {"type": "number", "description": "0 means spread over the scale, 1 means certain."},
        "legend": {"type": "object", "description": "Level index (\"0\", \"1\", ...) to the level as sent."},
        "probabilities": {**_NUMBER_MAP, "description": "Each level's probability, by level index."},
        "cost": COST_SCHEMA,
        "safety": SAFETY_SCHEMA,
    },
    "required": ["score", "confidence", "legend", "probabilities"],
    "additionalProperties": True,
}

NUMBER_OUTPUT: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "number": {"type": "number", "description": "The most probable value on the grid from min to max at step."},
        "expected": {
            "type": "number",
            "description": "The probability-weighted mean: a fractional estimate, also on a grid of whole numbers.",
        },
        "confidence": {"type": "number", "description": "The probability of `number`, 0 to 1."},
        "probabilities": {
            **_NUMBER_MAP,
            "description": "The probability of each value above 0.001, by the value as a string; they can sum to less than 1.",
        },
        "cost": COST_SCHEMA,
        "safety": SAFETY_SCHEMA,
    },
    "required": ["number", "expected", "confidence", "probabilities"],
    "additionalProperties": True,
}

_COORDINATE = {"type": "number"}
_PLACE_FIELDS: Dict[str, Any] = {
    "present": {
        "type": "number",
        "description": (
            "The probability that what the instructions describe is in the image at all. Under a low "
            "`present` the place is only a best guess for something probably not there."
        ),
    },
    "image": {"type": "string", "description": "The id of the image looked in."},
    "width": {"type": "integer", "description": "The image's width in pixels."},
    "height": {"type": "integer", "description": "The image's height in pixels."},
    "cost": COST_SCHEMA,
    "safety": SAFETY_SCHEMA,
}

POINT_OUTPUT: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "x": {**_COORDINATE, "description": "From the left edge: whole pixels, or a fraction of the width with normalized units."},
        "y": {**_COORDINATE, "description": "From the top edge: whole pixels, or a fraction of the height with normalized units."},
        "confidence": {"type": "number", "description": "The probability that the point lies on the thing, if it is there."},
        **_PLACE_FIELDS,
    },
    "required": ["x", "y", "present", "confidence", "image", "width", "height"],
    "additionalProperties": True,
}

BOX_OUTPUT: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "x1": {**_COORDINATE, "description": "The left edge of the box (its top-left corner's x)."},
        "y1": {**_COORDINATE, "description": "The top edge of the box (its top-left corner's y)."},
        "x2": {**_COORDINATE, "description": "The right edge of the box (its bottom-right corner's x)."},
        "y2": {**_COORDINATE, "description": "The bottom edge of the box (its bottom-right corner's y)."},
        "confidence": {
            "type": "number",
            "description": (
                "The probability that the box overlaps the true box with an intersection over union of 0.5 "
                "or more, if the thing is there."
            ),
        },
        **_PLACE_FIELDS,
    },
    "required": ["x1", "y1", "x2", "y2", "present", "confidence", "image", "width", "height"],
    "additionalProperties": True,
}

ANSWERS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": (
        "Answers by question id. Each has `type` and that type's fields: `truth`; `choice`, "
        "`confidence`, `probabilities`; `score`, `confidence`, `legend`, `probabilities`; "
        "`number`, `expected`, `confidence`, `probabilities`; or, for a point or a box, `x` and `y` "
        "or `x1`, `y1`, `x2`, `y2`, with `present`, `confidence`, `image`, `width`, `height`. A point or "
        "box asked with `max` answers `count`, `present`, `complete` (the probability that nothing was "
        "left out) and `points` or `boxes`, each with its own `confidence`, left to right, then `image`, "
        "`width` and `height`."
    ),
    "additionalProperties": {"type": "object"},
}

EVIDENCE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": (
        "Present when the call named boosters: booster name to its `version`, `status` (`ok`, `timeout`, "
        "`error` or `skipped`), `latency_ms`, `billed_calls` and `results`, one reading per picture as the "
        "API sent it."
    ),
    "additionalProperties": True,
}

DECIDE_OUTPUT: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answers": ANSWERS_SCHEMA,
        "model": {"type": ["string", "null"], "description": "The model version that answered."},
        "input_tokens": {"type": ["integer", "null"]},
        "evidence": EVIDENCE_SCHEMA,
        "cost": COST_SCHEMA,
        "safety": SAFETY_SCHEMA,
    },
    "required": ["answers", "model", "input_tokens"],
    "additionalProperties": True,
}

DECIDE_VIDEO_OUTPUT: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answers": ANSWERS_SCHEMA,
        "model": {"type": ["string", "null"], "description": "The model version that answered."},
        "input_tokens": {"type": ["integer", "null"], "description": "The tokens read, the videos' included."},
        "video_frames": {"type": ["integer", "null"], "description": "The frames the API read, over all videos."},
        "video_seconds": {"type": ["number", "null"], "description": "The seconds of the windows the API read, summed."},
        "cost": COST_SCHEMA,
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Present when the API read a video at a lower rate or a smaller size than sent, or when a "
                "video file held more frames in its window than were taken."
            ),
        },
        "safety": SAFETY_SCHEMA,
    },
    "required": ["answers", "model", "input_tokens"],
    "additionalProperties": True,
}

ERROR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "error_type": {"type": "string"},
        "message": {"type": ["string", "null"]},
        "reason": {"type": ["string", "null"]},
        "status": {"type": ["integer", "null"]},
        "request_id": {"type": ["string", "null"]},
    },
    "required": ["error_type", "message"],
    "additionalProperties": True,
}

DECIDE_MANY_OUTPUT: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "object",
            "description": "Results by item id: `answers` (with `model`), or `error` when that item failed.",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "answers": ANSWERS_SCHEMA,
                    "model": {"type": ["string", "null"]},
                    "cost": COST_SCHEMA,
                    "safety": SAFETY_SCHEMA,
                    "error": ERROR_SCHEMA,
                },
                "additionalProperties": True,
            },
        },
        "input_tokens": {"type": "integer", "description": "The tokens of the items that were answered, in all."},
    },
    "required": ["results", "input_tokens"],
    "additionalProperties": True,
}

MODELS_OUTPUT: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "models": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The id to pass as `model`."},
                    "description": {"type": ["string", "null"]},
                    "release_date": {"type": ["string", "null"]},
                },
                "required": ["name"],
                "additionalProperties": True,
            },
            "description": "Each model with the limits it publishes as extra fields.",
        }
    },
    "required": ["models"],
    "additionalProperties": True,
}

GUIDE_OUTPUT: Dict[str, Any] = {
    "type": "object",
    "properties": {"guide": {"type": "string", "description": "The guide, in Markdown."}},
    "required": ["guide"],
    "additionalProperties": False,
}
