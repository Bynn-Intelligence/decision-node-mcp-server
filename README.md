# DecisionNode MCP server

[DecisionNode](https://decisionnode.com) is a decision engine: send it text, JSON or images with a question, and it answers with calibrated probabilities.

DecisionNode's judgments as tools for AI agents. An agent asks a narrow question about a
piece of content (is this claim supported, which team should take this ticket, how urgent
is it, how many line items does the invoice list, where in the photo is the damaged
package) and gets a calibrated answer back in one tool call.

Every tool call is a DecisionNode API request made with your own API key. The server
stores no key, state, question or answer.

## Install

You need a DecisionNode API key and [uv](https://docs.astral.sh/uv/). The server runs on
your machine and your MCP client starts it over stdio:

```sh
uvx bynn-decisionnode-mcp
```

Before the package's first release on PyPI, install it from this repository instead:

```sh
uvx --from git+https://github.com/Bynn-Intelligence/decision-node-mcp-server bynn-decisionnode-mcp
```

In the client setups below, the git form replaces `bynn-decisionnode-mcp` with
`--from git+https://github.com/Bynn-Intelligence/decision-node-mcp-server bynn-decisionnode-mcp`.

### Claude Code

```sh
claude mcp add decisionnode -e DECISIONNODE_API_KEY=<key> -- uvx bynn-decisionnode-mcp
```

### Claude Desktop

Add the server to `claude_desktop_config.json` (Settings, Developer, Edit Config) and
restart Claude Desktop:

```json
{
  "mcpServers": {
    "decisionnode": {
      "command": "uvx",
      "args": ["bynn-decisionnode-mcp"],
      "env": { "DECISIONNODE_API_KEY": "<key>" }
    }
  }
}
```

### Other MCP clients

Cursor, Windsurf and other clients configured by a JSON file take the same entry under
`mcpServers`; VS Code takes it under `servers`. Any client that launches a server over
stdio works: the command is `uvx`, the argument is `bynn-decisionnode-mcp`, and the key
goes in the server's environment as `DECISIONNODE_API_KEY`. The server name
(`decisionnode` above) is yours to choose.

### Over HTTP

The same package serves Streamable HTTP, for a team that runs one server for several
clients:

```sh
uvx bynn-decisionnode-mcp --transport http --host 0.0.0.0 --port 8000
```

The endpoint is the root path, stateless, with `GET /health`. Each client sends its own
key on every request, `Authorization: Bearer <key>`, and the server forwards it to the
API; it never uses a key of its own. Over HTTP a tool never reads a file path, and the
base64 image data of one call is limited to 3 MiB (see Images below).

```sh
claude mcp add --transport http decisionnode http://127.0.0.1:8000/ \
  --header "Authorization: Bearer $DECISIONNODE_API_KEY"
```

## Tools

All eleven are read-only and idempotent (`readOnlyHint` and `idempotentHint` true,
`destructiveHint` false). Each declares an output schema and returns its result twice: as
structured content, and as the same JSON in a text block.

| Tool | Use it to | Returns | Costs |
|---|---|---|---|
| `guide` | Learn how to use the other tools well | The guide, as text | Nothing |
| `truth` | Ask one yes-or-no question about a state | `{"truth": 0.94, "cost": {...}}` | A request |
| `choice` | Pick one option from a set you define (1 to 255) | `{"choice", "confidence", "probabilities", "cost"}` | A request |
| `score` | Place a state on an ordered scale (1 to 10 levels) | `{"score", "confidence", "legend", "probabilities", "cost"}` | A request |
| `number` | Ask for a count or a quantity within a range | `{"number", "expected", "confidence", "probabilities", "cost"}` | A request |
| `point` | Find one spot in an image | `{"x", "y", "present", "confidence", "image", "width", "height", "cost"}` | A request |
| `box` | Draw a box around one thing in an image | `{"x1", "y1", "x2", "y2", "present", "confidence", "image", "width", "height", "cost"}` | A request |
| `decide` | Ask several questions of any type about one state, with boosters if you like | `{"answers": {id: answer}, "model", "input_tokens", "cost"}`, and `evidence` with boosters | One request, plus one booster call per picture a booster reads |
| `decide_many` | Ask the same questions about many states (1 to 50 items) | `{"results": {item id: {"answers", "cost"} or {"error"}}, "input_tokens"}` | One request per item |
| `decide_video` | Ask questions of any type about videos sent as frames | `{"answers", "model", "input_tokens", "cost", "video_frames", "video_seconds"}`, and `warnings` | One request |
| `models` | List the models the key may use, with their limits | `{"models": [...]}` | Nothing |

Inputs:

| Tool | Arguments |
|---|---|
| `truth` | `instructions`, `state`, `if_true`, `if_false`, `images`, `model` |
| `choice` | `instructions`, `options` (a list of names, or an object of name to description), `state`, `images`, `model` |
| `score` | `instructions`, `levels` (lowest first), `state`, `images`, `model` |
| `number` | `instructions`, `min`, `max`, `step`, `state`, `images`, `model` |
| `point`, `box` | `instructions`, `images` (at least one), `image`, `units`, `state`, `model` |
| `decide` | `questions` (id to `{type, instructions, criteria}`, `{type, instructions, min, max, step}` for a number, or `{type, instructions, image, units}` for a point or a box, as the API takes them), `state`, `images`, `boosters`, `model` |
| `decide_many` | `items` (a list of `{id, state, images}`, `images` optional and each item's own), `questions`, `model` |
| `decide_video` | `questions`, `videos`, `state`, `images`, `model` |

`cost`, on every tool that asks a question, is what the call cost as the API booked it, in
USD: `usd`, split into `model_usd` and `boosters_usd`, and `free_tier`. It is passed
through as sent and appears when the API sends it.

`state` is a string or any JSON value and may be left out. `model` defaults to
`decisionnode-moe`; `decisionnode-flash` is the faster, lower-cost tier. Both mean the
current release; a result's `model` is the pinned id of the version that answered. The
question types are `truth`, `choice`, `score`, `number`, `point` and `box`.

Values are passed through exactly as the API sent them. A result has a `safety` field only
when the API's safety check flagged a category for the key.

## Images

Up to 16 images a call, within the model's token limit, 10 MB each, JPEG, PNG or WebP. An
image is `{id, url}`, `{id, data, media_type}` or, on your own machine, `{id, path}` for a
file. A `url` must start with `https://`; any other scheme is refused before a request is
made.

There are two ways to say which image a question is about:

1. **By id (recommended).** Give each image a short, meaningful id and use it in the
   instructions, in backticks: "Does the total on `receipt` match the amount in the claim?"
2. **By number.** Images are numbered from 1 in the order of `images`: "image 2", "the
   second image", "Picture 2" and "the last image" all work.

The model sees each image as `Picture 1 (receipt):`, `Picture 2 (claim_photo):`, so both
ways point to the same image. Prefer ids: they stay right when images are added or
reordered. Use names, not digits, as ids (`front`, not `2`). Videos work the same way, by
id or by number ("video 1"), numbered separately in the order of `videos`. The `image` of
a point or box question takes the id.

Over HTTP one call's body is limited to 4 MiB by the MCP transport, so the base64 `data`
of all images in a call (every item's, in `decide_many`) is limited to 3 MiB in total
(3,145,728 characters, about 2.25 MiB of image bytes). A call with more is refused with an
`invalid_request_error` tool result before any request is made, and a body over 4 MiB is
answered 413. Send larger images by https `url`; the API fetches them itself, up to the
model's own limit (10 MB an image).

## Number questions

A number question asks for a count or a quantity within a range: how many people are in
the photo, how many line items the invoice lists, the year a contract was signed, a
rating in halves. `min` and `max` bound the answer (0 and 255 when left out) and `step`
sets the grid of possible answers (1 when left out; 0.5 for halves). The grid holds at
most 4,096 values; keep the range as narrow as the question allows.

```json
{"type": "number", "instructions": "How many cars are visible?", "min": 0, "max": 50}
```

The answer:

```json
{"number": 7, "expected": 7.3, "confidence": 0.62, "probabilities": {"6": 0.21, "7": 0.62, "8": 0.14}}
```

`number` is the most probable value on the grid and `confidence` its probability.
`expected` is the probability-weighted mean, a fractional estimate even on a grid of
whole numbers. `probabilities` lists the values above 0.001, keyed by the value as a
string, so they can sum to less than 1. Number questions work on text, images and video
frames, alone with the `number` tool or beside other questions in `decide`,
`decide_many` and `decide_video`.

## Points and boxes

A point or box question asks where something is in an image: `point` gives one spot,
`box` a rectangle around the thing. The instructions say what to find, as a phrase ("the
yellow package"), a description with conditions ("the person standing nearest the
door") or a question ("Where is the crack?"), and can refer to the state and to the
other images ("the part that differs from the reference image").

```json
{"type": "box", "instructions": "Draw a box around the damaged package.", "image": "belt"}
```

| Field | |
|---|---|
| `instructions` | What to find. |
| `image` | The id of the image to look in (a string). Needed with two or more images; with one, that one is used. A reference by number ("Picture 2") goes in the instructions. |
| `units` | `pixels` (the default): pixels of the image as sent. `normalized`: fractions of the width and the height, from 0 to 1. |

A point or box question needs at least one image in the call. The `point` and `box` tools
require `images` and check `image` before any request is made.

The answers:

```json
{"x": 812, "y": 455, "present": 0.97, "confidence": 0.91, "image": "belt", "width": 1920, "height": 1080}
{"x1": 300, "y1": 120, "x2": 520, "y2": 410, "present": 0.95, "confidence": 0.88, "image": "belt", "width": 1920, "height": 1080}
```

| Field | |
|---|---|
| `x`, `y` | The point, from the image's top-left corner, `x` to the right and `y` down: whole pixels, or fractions with `normalized` units. |
| `x1`, `y1`, `x2`, `y2` | The box: its top-left corner (`x1`, `y1`) and bottom-right corner (`x2`, `y2`), in the same units. |
| `present` | The probability that what the instructions describe is in the image at all. Under a low `present` the place is only a best guess for something probably not there. |
| `confidence` | The probability that the place is right if the thing is there: for a point, that it lies on the thing; for a box, that it overlaps the true box by at least half (an intersection over union of 0.5 or more). |
| `image`, `width`, `height` | The image looked in and its size in pixels, to draw or convert the answer without another lookup. |

Every field is always there, so the shape never changes. Act on an answer when both
`present` and `confidence` are high, and hand the image to a person when either is low.
The thresholds are yours. When several things match ("the yellow package" with three on
the belt), the answer is the best match; the instructions can say which one ("the
leftmost yellow package"). Point and box questions mix with the other types in
`decide`, and in `decide_many` each item can carry its own images, such as one photo per
item.

## Boosters

`decide` takes `boosters`: specialist models that read the call's images before the
model answers. Their readings are evidence the model weighs, and they come back in the
result as `evidence`, one reading per picture. Up to 4 a call, each once: a name, or
`{"name", "images", "require", "options"}` to read only some images, to fail the call
when the booster fails (`"require": true`) or to set its options. Each picture a booster
reads is billed as one booster call; a booster that fails or runs out of time is free.

| Booster | Reads | Options |
|---|---|---|
| `age` | The age of the person in a picture | none |
| `ai_generated` | Whether a picture was made by a generator | none |
| `ocr` | The text a picture shows | `prompt` |
| `bynn_depth` (Depth Vision) | Distances in meters: `ahead` (the nearest obstacle straight ahead, the floor left out, or `null`), `directions` (7, left to right), `ground`, and a `grid` with `depth_min`, `depth_max` and `depth_median` | `grid_size`, 2 to 16 (8) |
| `bynn_segment` (Segment Vision) | What `prompt` names: per match `label`, `score`, `box`, `polygon`, `area`, `centroid`, `opening` (the hole of a gate, doorway or ring, or `null`), and with depth on its `depth` and `size_m` in meters | `prompt` (required: one phrase, or a list of 1 to 4 of at most 120 characters), `max_instances` 1 to 16 (8), `min_score` 0 to 1 (0.25), `polygon_points` 3 to 64 (24), `depth` (true) |

Depth Vision and Segment Vision are made for robots and drones. Segment Vision's
coordinates are fractions of the picture from its top left, and descriptive phrases
("orange H landing pad", "purple octagonal frame on a pole") match better than a single
word. Their distances and sizes are estimates from one picture, for decisions and
planning, never a safety stop: a robot or a drone uses stereo cameras or LiDAR for that.

```json
{
  "state": "Drone at 1.5 m, heading to the next gate of the course.",
  "images": [{"id": "front", "path": "~/frames/front.jpg"}],
  "boosters": [
    {"name": "bynn_depth", "options": {"grid_size": 8}},
    {"name": "bynn_segment", "options": {"prompt": ["purple gate", "orange H landing pad"]}}
  ],
  "questions": {
    "clear": {"type": "truth", "instructions": "Is the path straight ahead clear for at least 2 meters?"},
    "gate": {"type": "point", "instructions": "The centre of the opening of the purple gate.", "image": "front"}
  }
}
```

Ask the question your code acts on and read its answer; the readings in `evidence` are
there to check and log. Every booster DecisionNode offers is listed at
[decisionnode.com/docs/concepts/boosters](https://decisionnode.com/docs/concepts/boosters).

## Video (frames)

Only frames are sent: no video file, video URL or sound ever reaches the API. A video is
the frames a camera or a script took, each with its time, and the model reads them as
one video, in order and with their times:

```json
{
  "id": "dock",
  "fps": 2,
  "frames": [
    {"t": 30.0, "media_type": "image/jpeg", "data": "<base64>"},
    {"t": 30.5, "media_type": "image/jpeg", "data": "<base64>"}
  ]
}
```

| Field | |
|---|---|
| `id` | Unique within the call, across images and videos. The model sees the video as "Video <n>" (its place in `videos`, from 1) with this id beside it, so questions can refer to it by either. |
| `frames` | The frames in time order, all of one size: `t` in seconds, larger than the frame before; `media_type` `image/jpeg`, `image/png` or `image/webp`; `data` the image as base64. |
| `fps` | A cap on the frames read a second: 2 when left out, at most 4. |
| `start`, `end` | A window in seconds on the frames' own times; every frame when left out. |

On your own machine a video can also be a file,
`{"id": "dock", "path": "~/clips/dock.mp4", "fps": 2, "start": 30, "end": 45}`. The
server samples it into JPEG frames on that machine with ffmpeg, at `fps` within `start`
and `end` (seconds from the video's first frame), at most `max_frames` of them (32 when
left out, up to 128). A window that holds more frames at that rate is cut there, and the
result's `warnings` say so. The frames are then sent as above; the file itself and its
sound stay on the machine. ffmpeg and ffprobe must be on the `PATH` the client starts
the server with (set `PATH` in the server's `env` if they are not); without them a
`path` video is refused with an error that says so, and frames still work. Over HTTP the
server never reads a path.

Send as few frames as the question needs: one or two for one item on a sorting line,
about one a second for an incident such as a fall or smoke, about two a second for
counting what passes. The API publishes its video limits with the models (`models`).
Over HTTP the base64 data of the images and the frames of one call share the 3 MiB above.

A `decide_video` result carries `answers`, `model` and `input_tokens` as `decide` does,
plus `video_frames` and `video_seconds` (what the API read), and `warnings` when the API
read a video at a lower rate or a smaller size than sent, or a file's window was cut.

## Errors

A failed call is a tool result with `isError` true and the error as structured content:

```json
{
  "error_type": "invalid_request_error",
  "message": "Too many score levels. Must have at most 10 levels.",
  "reason": "question 'score': a score takes up to 10 levels, got 11",
  "status": 400,
  "request_id": "req_..."
}
```

`reason` is the API's explanation in words, so the agent can correct the call. A
request refused under the usage policy has `error_type` `refusal_error` and its
`category`. A 422 adds `entries`; a 429 or 529 adds `retry_after` in seconds.
Temporary failures (408, 429, 5xx, timeouts, dropped connections) are retried twice
before an error is reported; 400, 401, 402, 403, 404 and 422 are never retried.

## Configuration

| Variable | Default | |
|---|---|---|
| `DECISIONNODE_API_KEY` | none | The API key, on stdio. Over HTTP each caller's own key is used and never this one. |
| `DECISIONNODE_BASE_URL` | `https://api.decisionnode.com` | The API the tools call. Must be `https://`; plain `http://` is accepted only for a loopback address (`localhost`, `127.0.0.1`, `::1`), and any other is refused at startup. |
| `DECISIONNODE_MODEL` | `decisionnode-moe` | The default model. |
| `DECISIONNODE_MCP_RATE_LIMIT` | `120` over HTTP, `0` (off) on stdio | Requests a minute per key on each server process; `0` turns it off. A negative or non-numeric value is refused at startup. Several processes each count on their own. At most 10,000 keys are tracked per process; when the table is full, a new key replaces one whose minute has passed, and if every tracked key has called within the minute the new key's call is refused with the rate-limit error. |
| `MCP_TRANSPORT` | `stdio` | Or `http`; the `--transport` flag wins. |
| `MCP_HOST`, `MCP_PORT`, `MCP_PATH` | `127.0.0.1`, `8000`, `/` | HTTP only; the `--host`, `--port` and `--path` flags win. |

Over HTTP the server accepts any well-formed bearer value and refuses a request without
one (401). The API decides whether the key is valid. A key the API has just answered 401
for is refused for the next 60 seconds with the same error result, and no request is
made; the server holds only the key's SHA-256 digest, in memory. The key is never written
to a log.

## Development

Requires Python 3.10 or later and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/Bynn-Intelligence/decision-node-mcp-server
cd decision-node-mcp-server
uv sync
uv run bynn-decisionnode-mcp                      # stdio
uv run bynn-decisionnode-mcp --transport http     # http://127.0.0.1:8000/
```

The tests run against the mock API server of the
[DecisionNode SDK conformance suite](https://github.com/Bynn-Intelligence/decision-node-conformance)
and never call a real endpoint. Clone the suite beside this repository, or point
`DECISIONNODE_CONFORMANCE_DIR` at it:

```sh
git clone https://github.com/Bynn-Intelligence/decision-node-conformance ../decision-node-conformance
uv run pytest
```

The video file tests need ffmpeg and ffprobe and are skipped without them. After a
deliberate change to a tool's schema or result, regenerate the golden files with
`UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py`.

## License

MIT. See [LICENSE](LICENSE).
