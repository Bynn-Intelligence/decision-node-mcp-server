"""Video files into frames, on the user's machine, for the local server.

The API takes a video only as frames with their times. For `{id, path}` the local server
samples the file with ffmpeg and ffprobe by the API's own rule: within the window
[start, end) a frame is taken when it is the first, or at least 1/fps seconds (less a
millisecond) after the frame taken before it. Each frame's `t` is its presentation time
in seconds from the video's first frame, the display rotation applied. Frames are written
as JPEG, the longer side at most 640 pixels (the API reads frames smaller than that).
Only these frames leave the machine: never the file, and never its sound.

ffmpeg reads the file through its `file` protocol only, so a playlist or a path that looks
like a URL cannot make it fetch anything from the network.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

SLACK = 0.001  # seconds: a frame this much early still counts as 1/fps after the one before it
MAX_SIDE = 640
JPEG_QUALITY = 85
PROBE_TIMEOUT = 30.0
SAMPLE_TIMEOUT = 120.0
INSTALL = "https://ffmpeg.org/download.html"

_PTS = re.compile(r"Parsed_showinfo_\d+ @ [^\]]*\] n:\s*\d+ pts:\s*(-?\d+)")
_TIME_BASE = re.compile(r"Parsed_showinfo_\d+ @ [^\]]*\] config in time_base:\s*(\d+)/(\d+)")


class FrameError(Exception):
    """A video file that cannot be sampled; the message says why, for the agent."""


@dataclass
class Sampled:
    frames: List[Tuple[float, bytes]]  # (t, the JPEG's bytes), in time order
    cut: bool  # the window held more frames at the rate than were taken


def programs() -> Tuple[str, str]:
    """The ffmpeg and ffprobe found on PATH; FrameError when either is missing."""
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        missing = " and ".join(name for name, found in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)) if found is None)
        raise FrameError(
            "Reading a video file needs ffmpeg and ffprobe on this machine, and %s %s not found on the "
            "PATH the server was started with. Install FFmpeg (%s), or send the video as frames."
            % (missing, "was" if " and " not in missing else "were", INSTALL)
        )
    return ffmpeg, ffprobe


def _run(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(list(command), capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise FrameError(
            "%s took longer than %d seconds on the file. Send a shorter window or a lower fps."
            % (os.path.basename(command[0]), timeout)
        ) from None


def _ticks(text: object) -> Optional[int]:
    try:
        return int(str(text))
    except (TypeError, ValueError):
        return None


def _input(path: str) -> str:
    # The file protocol by name: a path that starts with a protocol ("https:", "concat:")
    # or with "-" is read as a file name and never as a URL or an option.
    return "file:" + os.path.abspath(path)


def _probe(path: str, ffprobe: str) -> Dict[str, object]:
    """The video stream read (the first that is not cover art), its time base, its first
    frame's time (the zero of every `t`) and the container's start time."""
    source = _input(path)
    done = _run(
        [ffprobe, "-v", "error", "-protocol_whitelist", "file",
         "-show_entries", "stream=index,codec_type,time_base,start_pts:stream_disposition=attached_pic:format=start_time",
         "-of", "json", source],
        PROBE_TIMEOUT,
    )
    if done.returncode != 0:
        raise FrameError("ffprobe cannot read the file: %s" % done.stderr.decode("utf-8", "replace").strip()[:300])
    info = json.loads(done.stdout or b"{}")
    videos = [s for s in info.get("streams") or () if s.get("codec_type") == "video"]
    streams = [s for s in videos if (s.get("disposition") or {}).get("attached_pic") != 1]
    if not streams:
        raise FrameError("the file holds no video stream")
    stream = streams[0]
    number = videos.index(stream)
    num, _, den = str(stream.get("time_base") or "").partition("/")
    if not num.isdigit() or not den.isdigit() or int(num) <= 0 or int(den) <= 0:
        raise FrameError("the video stream's time base cannot be read")
    # The first frame: the smallest presentation time among the first packets the container
    # does not discard (an edit list's leading packets are marked D), or the stream's start.
    done = _run(
        [ffprobe, "-v", "error", "-protocol_whitelist", "file", "-select_streams", "v:%d" % number,
         "-show_entries", "packet=pts,dts,flags", "-read_intervals", "%+#256", "-of", "csv=p=0", source],
        PROBE_TIMEOUT,
    )
    first = None
    for line in done.stdout.decode("ascii", "replace").splitlines():
        parts = line.strip().split(",")
        if len(parts) < 3 or "D" in parts[2]:
            continue
        t = _ticks(parts[0]) if _ticks(parts[0]) is not None else _ticks(parts[1])
        if t is not None and (first is None or t < first):
            first = t
    start_pts = _ticks(stream.get("start_pts"))
    if first is None and start_pts is None:
        raise FrameError("the file holds no frame that can be read")
    zero = min(t for t in (first, start_pts) if t is not None)
    try:
        container_start = float((info.get("format") or {}).get("start_time") or 0.0)
    except ValueError:
        container_start = 0.0
    return {
        "stream": number,
        "time_base": (int(num), int(den)),
        "zero": zero * int(num) / int(den),
        "container_start": container_start,
    }


def sample(path: str, *, fps: float, start: float = 0.0, end: float = math.inf, max_frames: int) -> Sampled:
    """The frames of `path` taken by the rule at `fps` within [start, end) (seconds from the
    video's first frame), at most `max_frames`. Blocking: run it off the event loop."""
    if not os.path.isfile(path):
        raise FrameError("no video file at %s" % path)
    if not 0 < fps <= 4:
        raise FrameError("fps must be above 0 and at most 4")
    if start < 0 or end <= start:
        raise FrameError("the window needs 0 <= start < end")
    ffmpeg, ffprobe = programs()
    info = _probe(path, ffprobe)
    num, den = info["time_base"]  # type: ignore[misc]
    zero = float(info["zero"])  # type: ignore[arg-type]
    low, high = zero + start, zero + end
    gap = 1.0 / fps - SLACK
    before_end = "lt(t\\,%.9f)*" % high if math.isfinite(high) else ""
    rule = "select='gte(t\\,%.9f)*%s(isnan(prev_selected_t)+gte(t-prev_selected_t\\,%.6f))'" % (low, before_end, gap)
    scale = "scale='if(gte(iw,ih),min(iw,{0}),-2)':'if(gte(iw,ih),-2,min(ih,{0}))':flags=bicubic".format(MAX_SIDE)
    graph = ",".join((rule, "showinfo=checksum=0", scale))
    quality = str(max(2, min(31, round(31 - JPEG_QUALITY * 0.29))))
    seek: List[str] = []
    at = low - float(info["container_start"]) - 1.0  # type: ignore[arg-type]
    if start > 2.0 and at > 0:
        # A second before the first frame wanted; the decoder starts at the keyframe before it.
        seek = ["-noaccurate_seek", "-ss", "%.6f" % at]
    with tempfile.TemporaryDirectory(prefix="decisionnode-frames-") as directory:
        pattern = os.path.join(directory, "%06d.jpg")
        base = [ffmpeg, "-nostdin", "-hide_banner", "-nostats", "-v", "info", "-protocol_whitelist", "file"]
        base += seek + ["-copyts", "-i", _input(path), "-map", "0:v:%d" % info["stream"], "-an", "-sn", "-dn", "-vf", graph]
        tail = ["-frames:v", str(max_frames + 1), "-f", "image2", "-c:v", "mjpeg", "-pix_fmt", "yuvj420p", "-q:v", quality, pattern]
        done = _run(base + ["-fps_mode", "passthrough"] + tail, SAMPLE_TIMEOUT)
        if done.returncode != 0 and b"fps_mode" in done.stderr:  # an ffmpeg older than 5.1
            done = _run(base + ["-vsync", "passthrough"] + tail, SAMPLE_TIMEOUT)
        if done.returncode != 0:
            raise FrameError("ffmpeg failed on the file: %s" % done.stderr.decode("utf-8", "replace").strip()[-300:])
        stderr = done.stderr.decode("utf-8", "replace")
        found = _TIME_BASE.search(stderr)
        tick = int(found.group(1)) / int(found.group(2)) if found and int(found.group(2)) > 0 else num / den
        times = [int(m) * tick - zero for m in _PTS.findall(stderr)]
        names = sorted(n for n in os.listdir(directory) if n.endswith(".jpg"))
        if len(names) > len(times):  # frames past -frames:v are filtered, never written
            raise FrameError("ffmpeg wrote %d frames and reported %d times" % (len(names), len(times)))
        frames = []
        for name, t in zip(names, times):
            with open(os.path.join(directory, name), "rb") as f:
                frames.append((round(t, 6), f.read()))
    if not frames:
        raise FrameError("no frame of the file lies in the window [%g, %g) s" % (start, end))
    return Sampled(frames=frames[:max_frames], cut=len(frames) > max_frames)


def video_object(video_id: str, sampled: Sampled, fps: float) -> Dict[str, object]:
    """The frames form the API takes. `fps` is sent so the API reads every frame taken."""
    return {
        "id": video_id,
        "fps": fps,
        "frames": [
            {"t": t, "media_type": "image/jpeg", "data": base64.b64encode(blob).decode("ascii")}
            for t, blob in sampled.frames
        ],
    }
