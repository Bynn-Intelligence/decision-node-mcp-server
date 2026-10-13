"""The texts an agent reads: the server instructions, the tool descriptions and the guide.

They decide whether an agent uses the server well, so they are tested like code
(tests/test_texts.py): each tool description says when to use the tool first, what
comes back second, and whether it costs a request.
"""

from __future__ import annotations

INSTRUCTIONS = """\
DecisionNode answers narrow questions about a piece of content with calibrated \
probabilities: is this statement true (truth), which of these options fits (choice), \
where on this scale does it sit (score), how many or how much (number), where in this \
image is it (point, box). It judges only \
the `state` you send with each call and knows nothing else about this conversation. It \
does not write text, plan or call tools; use it beside your own reasoning for fast, \
consistent judgments with a usable probability. Call `guide` once before your first \
question: it costs nothing and explains how to pick a question type, write the state and \
read the numbers.\
"""

FLASH_NOTE = (
    "`model` is optional: leave it out for the default model, or pass "
    "`decisionnode-flash` for the faster, lower-cost tier."
)

GUIDE_DESCRIPTION = (
    "Use this once before your first DecisionNode question, to learn which tool and question "
    "type to pick, how to write the state and the question, and how to read a probability and a "
    "confidence. Returns the guide as text. Costs nothing: it makes no API call."
)

TRUTH_DESCRIPTION = (
    "Use this to ask one yes-or-no question about a state, such as \"is this claim supported by "
    "the source\" or \"does this message ask for a refund\". Returns `truth`, the probability "
    "from 0 to 1 that the answer is yes, with the call's `cost` in USD. Costs one request. " + FLASH_NOTE
)

CHOICE_DESCRIPTION = (
    "Use this to pick exactly one option from a set you define (1 to 255 options), such as "
    "routing a ticket to a team. Returns `choice`, the most likely option, with `confidence` "
    "and the `probabilities` of every option, with the call's `cost` in USD. Costs one request. " + FLASH_NOTE
)

SCORE_DESCRIPTION = (
    "Use this to place a state on an ordered scale you define (1 to 10 levels, lowest first), "
    "such as urgency or severity. Returns `score`, the expected level index from 0, with "
    "`confidence`, the `probabilities` of each level and the `legend` of level names, with the "
    "call's `cost` in USD. Costs one request. " + FLASH_NOTE
)

NUMBER_DESCRIPTION = (
    "Use this to ask for a count or a quantity within a range you set, such as \"how many people "
    "are in the photo\", \"how many line items does the invoice list\" or \"what year was the "
    "contract signed\". Returns `number`, the most probable value from `min` to `max` at `step`, "
    "with `expected` (the probability-weighted mean), `confidence` (the probability of `number`) "
    "and the `probabilities` of the likely values, with the call's `cost` in USD. Costs one request. " + FLASH_NOTE
)

PLACE_PATTERN = (
    "`present` is the probability that the thing is in the image at all and `confidence` the "
    "probability that the place is right if it is: act on the answer when both are high, and hand "
    "the image to a person when either is low."
)

POINT_DESCRIPTION = (
    "Use this to find one spot in an image, such as \"the yellow package closest to the robot "
    "arm\" or \"the empty slot on the second shelf\". Returns `x` and `y` from the image's top-left "
    "corner (`x` to the right, `y` down), `present`, `confidence` (that the point lies on the "
    "thing), and the `image` looked in with its `width` and `height`, with the call's `cost` in "
    "USD. Costs one request. "
    + PLACE_PATTERN
    + " Needs at least one image; with two or more, `image` names the one to look in by its id "
    '(a reference by number, "Picture 2", goes in the instructions). '
    + FLASH_NOTE
)

BOX_DESCRIPTION = (
    "Use this to draw a box around one thing in an image, such as \"the damaged package\" or "
    "\"the person climbing the fence\". Returns the top-left corner `x1`, `y1` and the "
    "bottom-right corner `x2`, `y2`, `present`, `confidence` (that the box overlaps the true box "
    "by at least half, an intersection over union of 0.5), and the `image` looked in with its "
    "`width` and `height`, with the call's `cost` in USD. Costs one request. "
    + PLACE_PATTERN
    + " Needs at least one image; with two or more, `image` names the one to look in by its id "
    '(a reference by number, "Picture 2", goes in the instructions). '
    + FLASH_NOTE
)

DECIDE_DESCRIPTION = (
    "Use this to ask several questions of any type about the same state in one call, for "
    "example a route, an urgency and an escalation flag for one ticket. Returns `answers` by "
    "question id, with the `model` that answered, the `input_tokens` counted and the call's "
    "`cost` in USD. A point or box question with `max` (2 to 16) finds every match, up to that "
    "many, such as each car in a picture. Costs one request, with the state counted once. " + FLASH_NOTE
)

DECIDE_MANY_DESCRIPTION = (
    "Use this to ask the same questions about many states (1 to 50 items), such as \"which of "
    "these twenty messages asks for a refund\". Returns `results` by item id, each with its "
    "`answers` and `cost`, or its `error` (one item's failure does not fail the others), and the total "
    "`input_tokens`. Costs one request per item. Each item may carry its own `images`, such as "
    "one photo per item for a point or a box. " + FLASH_NOTE
)

DECIDE_VIDEO_DESCRIPTION = (
    "Use this to ask questions of any type about one or more videos, sent as frames with their "
    "times, such as \"does a person fall\", \"how many cars pass the gate\" or \"does the conveyor "
    "stop\". Returns `answers` by question id, with the `model`, the `input_tokens`, the `cost`, the "
    "`video_frames` and `video_seconds` the API read, and `warnings` when it read fewer frames or "
    "a smaller size than sent. Costs one request. Only frames are sent: no video file, URL or "
    "sound reaches the API. " + FLASH_NOTE
)

MODELS_DESCRIPTION = (
    "Use this to see which DecisionNode models your key may use and their published limits "
    "(tokens per request, options, levels, images). Returns `models`, the list of model "
    "entries. Costs nothing."
)

GUIDE = """\
# Using DecisionNode well

DecisionNode answers narrow questions about a state with calibrated probabilities. \
It is fast and consistent, and every answer is a number your code can threshold. It \
does not write, plan, extract free text or call tools.

## Pick the tool

| You want | Tool | You get |
|---|---|---|
| Yes or no | `truth` | `truth`: the probability of yes, 0 to 1 |
| One option of a set | `choice` | `choice`, `confidence`, `probabilities` per option |
| A degree on an ordered scale | `score` | `score` (expected level index), `confidence`, `probabilities`, `legend` |
| A count or a quantity | `number` | `number` (the most probable value), `expected`, `confidence`, `probabilities` |
| One spot in an image | `point` | `x`, `y`, `present`, `confidence`, `image`, `width`, `height` |
| A box around one thing in an image | `box` | `x1`, `y1`, `x2`, `y2`, `present`, `confidence`, `image`, `width`, `height` |
| Several questions about one state | `decide` | `answers` by question id, in one request |
| The same questions about many states | `decide_many` | `results` by item id, one request per item |
| Questions about videos, sent as frames | `decide_video` | `answers` by question id, with `video_frames` and `video_seconds` |
| The models and their limits | `models` | `models` |

A question of degree ("how strong is the candidate in Python?") is a `score`, not a \
`truth`. A count or an amount ("how many people are in the photo?") is a `number`.

## Write the state

- Put everything the model needs in `state`. It sees the state, the question and any \
images, and nothing else: not this conversation, not earlier calls, not files you have \
read. Paste the passage, the message or the record itself.
- `state` is a string or any JSON value. A JSON object with named fields (`{"claim": ..., \
"source": ...}`) lets a question refer to a field by name.
- Leave `state` out only for a question that stands on its own ("Is a hotdog a sandwich?").

## Write the question

- Ask one narrow thing per question. Split "is it urgent and about billing?" into two.
- Say what the question is about: "Does the source support the claim?" rather than "Is it \
true?".
- For `truth`, `if_true` and `if_false` say what a yes and a no mean. They change the \
answer, so use them when the line between yes and no needs drawing.
- For `choice`, give each option a short description when its name is not obvious. A \
choice always picks one of the options, even when none fits: add an option such as \
`"none": "none of these apply"` when that can happen.
- For `score`, list the levels lowest first, two or more, each a few words.
- For `number`, set `min` and `max` as narrow as the question allows (the default range \
is 0 to 255) and `step` for the grid of answers (1 by default; 0.5 for halves).
- For `point` and `box`, say what to find, with any condition that picks one thing: "the \
leftmost yellow package", "the person who entered after closing".
- In `decide` and `decide_many`, questions are written exactly as the API takes them: \
`{"type": "truth" | "choice" | "score" | "number" | "point" | "box", "instructions": ..., \
"criteria": ...}`. `criteria` is `{"true": ..., "false": ...}` for a truth, an object of \
option name to description for a choice, and the list of levels for a score; a number \
takes `min`, `max` and `step` in its place, and a point or a box `image` and `units`. \
Question ids are yours; answers come back under the same ids.

## Read the numbers

- `truth` is the probability that the answer is yes. 0.5 means the model cannot tell, not \
that the statement is half true. Near 0 or 1 means it is sure.
- `confidence` (choice and score) runs from 0, no better than a guess, to 1, certain. A \
low confidence usually means the state does not say enough, or the question or options \
are ambiguous.
- `score` is the expected level index, counted from 0: 2.31 on four levels sits between \
the third and the fourth, nearer the third. `legend` maps each index to its level.
- `number` is the most probable value on the grid and its `confidence` is that value's \
probability. `expected` is the probability-weighted mean, a fractional estimate even on \
a grid of whole numbers. `probabilities` lists only the values above 0.001, by the value \
as a string, so they can sum to less than 1.
- Probabilities are passed through as the API sent them: a list can sum to 0.99 or 1.01, \
and an entry can read 0.8200000000000001.
- The thresholds are yours. Decide in advance what number is enough to act on.
- The same request always gets the same answer from the same model version.

## Images

Up to 16 images a request, within the model's token limit, 10 MB each, JPEG, PNG or \
WebP: `{"id", "url"}` with an https URL, or `{"id", "data", "media_type"}` with base64 \
data. The local server also takes `{"id", "path"}` for a file on the user's machine. \
Two ways to say which image a question is about: by id (recommended), giving each image \
a short, meaningful id and using it in the instructions in backticks ("Does the total on \
`receipt` match the amount in the claim?"); or by number, from 1 in the order of \
`images` ("image 2", "the second image", "Picture 2", "the last image"). The model sees \
each image as `Picture 1 (receipt):`, so both point to the same image. Prefer ids: they \
stay right when images are added or reordered; use names, not digits (`front`, not \
`2`). Videos work the same way, by id or "video 1", numbered separately in the order of \
`videos`. On the hosted server the base64 `data` of one \
call is limited to about 3 MiB in total (about 2.25 MB of image bytes); send larger \
images by https `url`.

## Points and boxes

`point` and `box` answer where something is in an image. A point is `x` and `y`; a box \
is its top-left corner `x1`, `y1` and its bottom-right corner `x2`, `y2`. Both are \
measured from the image's top-left corner, `x` to the right and `y` down, in pixels of \
the image as sent, or with `units: "normalized"` in fractions of the width and the \
height from 0 to 1.

- They need at least one image; with two or more, `image` names the one to look in, by \
its id; a reference by number ("Picture 2") goes in the instructions. The state and the \
other images are context the instructions can refer to ("the item that \
matches the order in the state").
- `present` is the probability that the thing is in the image at all. Under a low \
`present` the place is only a best guess for something probably not there.
- `confidence` is the probability that the place is right if the thing is there: for a \
point, that it lies on the thing; for a box, that it overlaps the true box by at least \
half (an intersection over union of 0.5).
- Act on an answer when both are high, and hand the image to a person when either is \
low. The thresholds are yours.
- When several things match, the answer is the best match: say which one you mean.
- Every field is always there, with `image`, `width` and `height` to draw or convert it.

## Video

`decide_video` asks questions of any type about videos, sent as frames with their times. \
The model reads the frames of a video as one video, in order and with their times, not as \
separate pictures.

- A video is `{"id", "frames": [{"t", "media_type", "data"}], "fps", "start", "end"}`: the \
frames in time order, all of one size, `t` in seconds and larger than the frame before. \
`fps` caps the frames read a second (2 by default, at most 4); `start` and `end` set a \
window on the frames' own times.
- The local server also takes `{"id", "path"}` for a video file on the user's machine and \
samples it into frames with ffmpeg, at `fps` within `start` and `end` (seconds from the \
first frame), at most `max_frames` of them (32 by default).
- Only frames are sent: no video file, URL or sound reaches the API.
- Send as few frames as the question needs: one or two for one item on a line, about one \
a second for an incident such as a fall or smoke, about two a second for counting what \
passes. The API publishes its video limits in `models`.
- A result carries `video_frames` and `video_seconds`, what the API read, and `warnings` \
when it read fewer frames or a smaller size than sent.

## Limits and errors

- Each model publishes its limits (tokens per request, options, levels, images) in \
`models`. A request over the token limit is refused with a reason that says by how much; \
the state is never cut to fit. Shorten it and ask again.
- Choice: 1 to 255 options. Score: 1 to 10 levels. Number: at most 4,096 values from \
`min` to `max` at `step`. `decide_many`: 1 to 50 items.
- A failed call returns an error with `error_type`, `message`, `reason` (the API's \
explanation in words), `status` and `request_id`. Fix the call from the reason rather \
than retrying it unchanged. Temporary failures are already retried for you.
- A request refused under the usage policy returns `error_type` `refusal_error` with its \
`category`. It is not retried, and asking again in other words will not change it.
- A result carries a `safety` field only when the safety check flagged a category for \
your key.
"""
