# Changelog

What changed in each release of `bynn-decisionnode-mcp`.

## 0.1.3

- `decide` takes `boosters`: specialist models that read the call's images before the
  model answers. Up to 4, each a name or `{"name", "images", "require", "options",
  "timeout_ms"}`, sent as given; the names are free text, so a new booster works
  without an update. The result carries the boosters' `evidence` exactly as the API
  sent it.
- The tool's schema and the guide describe every booster: `age`, `ai_generated`,
  `face_detection`, `face_recognition`, `ocr`, and for robots and drones Depth Vision
  (`bynn_depth`) and Segment Vision (`bynn_segment`), with their options.
- An error of a call that named boosters carries `boosters`, what each one did, so a
  required booster that failed shows which; a refused setting keeps the API's
  `reason`.

## 0.1.2 - 2026-10-13

- The README and the package description call DecisionNode a decision engine.

## 0.1.1 - 2026-10-13

- The first release on PyPI: eleven tools (`guide`, `truth`, `choice`, `score`,
  `number`, `point`, `box`, `decide`, `decide_many`, `decide_video`, `models`) over
  stdio or Streamable HTTP.
