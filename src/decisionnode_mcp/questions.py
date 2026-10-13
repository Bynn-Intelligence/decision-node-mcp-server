"""The one-question tools: `truth`, `choice`, `score`, `number`, `point` and `box`."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, Union

from decisionnode import choice as choice_question
from decisionnode import score as score_question
from decisionnode import truth as truth_question
from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from pydantic import Field

from . import api, schemas, texts
from .schemas import (
    IMAGES_DESCRIPTION,
    MODEL_DESCRIPTION,
    STATE_DESCRIPTION,
    UNITS_DESCRIPTION,
    ImageInput,
    Number,
    PositiveNumber,
    Units,
)

mcp = FastMCP("DecisionNode questions")

State = Annotated[Any, Field(description=STATE_DESCRIPTION)]
Images = Annotated[Optional[List[ImageInput]], Field(description=IMAGES_DESCRIPTION)]
Model = Annotated[Optional[str], Field(description=MODEL_DESCRIPTION)]
Instructions = Annotated[
    Any,
    Field(description="The question or statement to judge, such as \"Does the source support the claim?\"."),
]


def _images(images: Optional[List[ImageInput]]) -> Optional[List[Any]]:
    return api.images_for_api([i.model_dump() for i in images] if images else None)


async def _ask(question_id: str, question: Dict[str, Any], state: Any, images: Any, model: Optional[str]) -> ToolResult:
    async def call() -> Dict[str, Any]:
        if question["type"] in schemas.PLACE_TYPES:
            _check_place_image(question.get("image"), images)
        sdk_images = _images(images)
        async with api.client() as dn:
            decision = await api.decide(dn, state, {question_id: question}, sdk_images, model)
        return api.with_safety(api.with_cost(api.answer_fields(decision, question_id), decision), decision)

    return await api.run(call)


def _check_place_image(image: Optional[str], images: List[ImageInput]) -> None:
    """The API's rule for the image a point or a box looks in, checked before a request."""
    ids = [i.id for i in images]
    if image is None and len(ids) > 1:
        raise api.ToolFailure(
            "invalid_request_error",
            "With two or more images, 'image' names the one to look in: one of %s." % ", ".join(repr(i) for i in ids),
        )
    if image is not None and image not in ids:
        raise api.ToolFailure(
            "invalid_request_error",
            "'image' is %r, which is not the id of an image of the call: %s." % (image, ", ".join(repr(i) for i in ids)),
        )


@mcp.tool(
    name="truth",
    description=texts.TRUTH_DESCRIPTION,
    annotations=schemas.annotations("Truth"),
    output_schema=schemas.TRUTH_OUTPUT,
)
async def truth(
    instructions: Instructions,
    state: State = None,
    if_true: Annotated[Any, Field(description="What a yes means (optional).")] = None,
    if_false: Annotated[Any, Field(description="What a no means (optional).")] = None,
    images: Images = None,
    model: Model = None,
) -> ToolResult:
    return await _ask("check", dict(truth_question(instructions, if_true, if_false)), state, images, model)


@mcp.tool(
    name="choice",
    description=texts.CHOICE_DESCRIPTION,
    annotations=schemas.annotations("Choice"),
    output_schema=schemas.CHOICE_OUTPUT,
)
async def choice(
    instructions: Instructions,
    options: Annotated[
        Union[Annotated[List[str], Field(min_length=1)], Annotated[Dict[str, Any], Field(min_length=1)]],
        Field(
            description=(
                "The options, 1 to 255: a list of names, or an object of name to description. "
                "The answer names one of them exactly as given."
            )
        ),
    ],
    state: State = None,
    images: Images = None,
    model: Model = None,
) -> ToolResult:
    return await _ask("classify", dict(choice_question(instructions, options)), state, images, model)


@mcp.tool(
    name="score",
    description=texts.SCORE_DESCRIPTION,
    annotations=schemas.annotations("Score"),
    output_schema=schemas.SCORE_OUTPUT,
)
async def score(
    instructions: Instructions,
    levels: Annotated[
        List[Any],
        Field(min_length=1, description="The levels of the scale, lowest first, 1 to 10, such as [\"low\", \"medium\", \"high\"]."),
    ],
    state: State = None,
    images: Images = None,
    model: Model = None,
) -> ToolResult:
    return await _ask("score", dict(score_question(instructions, levels)), state, images, model)


@mcp.tool(
    name="number",
    description=texts.NUMBER_DESCRIPTION,
    annotations=schemas.annotations("Number"),
    output_schema=schemas.NUMBER_OUTPUT,
)
async def number(
    instructions: Annotated[
        Any, Field(description="The question, such as \"How many people are in the photo?\".")
    ],
    min: Annotated[Optional[Number], Field(description="The lowest possible answer (default 0).")] = None,  # noqa: A002
    max: Annotated[Optional[Number], Field(description="The highest possible answer (default 255).")] = None,  # noqa: A002
    step: Annotated[
        Optional[PositiveNumber],
        Field(description="The grid of possible answers: 1 for whole numbers (the default), 0.5 for halves."),
    ] = None,
    state: State = None,
    images: Images = None,
    model: Model = None,
) -> ToolResult:
    question: Dict[str, Any] = {"type": "number", "instructions": instructions}
    for key, value in (("min", min), ("max", max), ("step", step)):
        if value is not None:
            question[key] = value
    return await _ask("number", question, state, images, model)


PlaceImages = Annotated[
    List[ImageInput],
    Field(
        min_length=1,
        description=(
            "The images, at least one: the one to look in, and any others the instructions refer to. "
            + IMAGES_DESCRIPTION
        ),
    ),
]
PlaceImage = Annotated[
    Optional[str],
    Field(
        description=(
            "The id of the image to look in (a string). Needed with two or more images; with one, that one. "
            'A reference by number ("Picture 2") goes in the instructions.'
        )
    ),
]
PlaceUnits = Annotated[Optional[Units], Field(description=UNITS_DESCRIPTION)]


def _place(kind: str, instructions: Any, image: Optional[str], units: Optional[str]) -> Dict[str, Any]:
    question: Dict[str, Any] = {"type": kind, "instructions": instructions}
    if image is not None:
        question["image"] = image
    if units is not None:
        question["units"] = units
    return question


@mcp.tool(
    name="point",
    description=texts.POINT_DESCRIPTION,
    annotations=schemas.annotations("Point"),
    output_schema=schemas.POINT_OUTPUT,
)
async def point(
    instructions: Annotated[
        Any, Field(description="What to find, such as \"the yellow package closest to the robot arm\".")
    ],
    images: PlaceImages,
    image: PlaceImage = None,
    units: PlaceUnits = None,
    state: State = None,
    model: Model = None,
) -> ToolResult:
    return await _ask("point", _place("point", instructions, image, units), state, images, model)


@mcp.tool(
    name="box",
    description=texts.BOX_DESCRIPTION,
    annotations=schemas.annotations("Box"),
    output_schema=schemas.BOX_OUTPUT,
)
async def box(
    instructions: Annotated[
        Any, Field(description="What to draw the box around, such as \"the damaged package\".")
    ],
    images: PlaceImages,
    image: PlaceImage = None,
    units: PlaceUnits = None,
    state: State = None,
    model: Model = None,
) -> ToolResult:
    return await _ask("box", _place("box", instructions, image, units), state, images, model)
