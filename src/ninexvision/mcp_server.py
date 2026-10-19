"""MCP server: the same tools, returning images inline so any MCP client can see them."""
from __future__ import annotations

import json
from pathlib import Path

import functools

try:  # mcp >= 2
    from mcp.server.mcpserver import Image, MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server, Image
    from mcp.server.fastmcp.exceptions import ToolError

from . import tools
from .watch import watch as run_watch

mcp = _Server("9xvision", instructions=(
    "Perception tools. Videos: call watch(source) first: it returns INDEX.md (structure, flow, text, timings) "
    "and the first overview sheet. Look at sheets before zooming. Use zoom/frame/diff only where you need "
    "more detail, and cite mm:ss.s timestamps in answers. Images: image(source) for one picture (use a region "
    "for fine detail), images(sources) for a folder. Live camera/mic/screen: live_start, then live_now, "
    "live_snapshot (turns the last seconds into a watch), live_stop."))


def _img(path: str) -> Image:
    return Image(path=path)


def _explained(fn):
    """mcp 2.x hides unexpected exceptions behind 'Error executing tool'; pass the reason to the agent."""
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"{type(e).__name__}: {e}") from e
    return wrapper


@mcp.tool()
@_explained
def watch(source: str, mode: str = "auto", force: bool = False, start: str | None = None, end: str | None = None,
          cookies_from_browser: str | None = None) -> list:
    """Analyse a local video file or URL (YouTube, Vimeo, TikTok, X, Instagram...). mode: auto | film | screen
    (screen = app/screen recordings). start/end ("12:30", "1:02:00", or seconds) watch only a window, and a URL
    downloads only that part: use it for long videos when the question is about one stretch. Times stay in
    full-video time. cookies_from_browser ("chrome", "safari", ...) is for login-gated URLs, only with the
    user's OK. Returns INDEX.md plus the first overview contact sheet. First run takes ~1-2 s per minute."""
    wd = run_watch(source, mode=mode, force=force, start=start, end=end, cookies_from_browser=cookies_from_browser)
    m = json.loads((wd / "watch.json").read_text())
    out: list = [(wd / "INDEX.md").read_text()]
    if m.get("sheets"):
        out.append(_img(str(wd / m["sheets"][0]["path"])))
    return out


@mcp.tool()
@_explained
def look(watch_id: str, what: str) -> list:
    """Show an artefact of a watch: sheet:N, timeline:N, view:N (full-res screen), shot:N (keyframe),
    page:N (stitched long page), colour (colour script)."""
    p = tools.look(watch_id, what)["path"]
    return [f"{what}: {p}", _img(p)]


@mcp.tool()
@_explained
def zoom(watch_id: str, start: str, end: str, n: int = 12) -> list:
    """Dense timestamped grid of frames between start and end (mm:ss.s or seconds), up to 30 frames."""
    r = tools.zoom(watch_id, start, end, n)
    return [json.dumps({k: r[k] for k in ("times", "tokens", "note") if k in r}), _img(r["path"])]


@mcp.tool()
@_explained
def frame(watch_id: str, at: str, crop: str | None = None) -> list:
    """Full-resolution frame at a time. crop = "x,y,w,h" in pixels or 0-1 fractions, for reading small detail."""
    r = tools.frame(watch_id, at, crop)
    return [json.dumps({k: r[k] for k in ("t", "size", "source_px", "note") if k in r}), _img(r["path"])]


@mcp.tool()
@_explained
def diff(watch_id: str, a: str, b: str) -> list:
    """Compare two moments: side-by-side plus highlighted changed regions (pixel boxes in source coordinates)."""
    r = tools.diff(watch_id, a, b)
    return [json.dumps({k: r[k] for k in ("changed_pct", "regions", "times", "note") if k in r}), _img(r["path"])]


@mcp.tool()
@_explained
def search(query: str, watch_id: str | None = None, visual: bool = True) -> list:
    """Find moments by on-screen text (OCR), spoken words, AND what is in the picture ("a red car at night",
    "a pie chart", "someone hugging"). Text hits cover every watch; visual moments cover watch_id (default:
    the most recent watch) and come with a contact sheet of the best frames; for an album, the best images.
    Visual scores: strong >= 0.5, possible >= 0.05; all-weak means it is probably not there. No negation
    ("an empty room"): search for what is there. The first visual search on a watch indexes it (seconds)."""
    r = tools.search(query, watch_id, visual=visual)
    out: list = [json.dumps(r, indent=1, ensure_ascii=False)]
    sheet = (r.get("visual") or {}).get("sheet")
    if sheet:
        out.append(_img(sheet["path"]))
    return out


@mcp.tool()
@_explained
def qa(source: str, start: str | None = None, end: str | None = None, fov_deg: float | None = None) -> list:
    """Capture QA for a game, XR (VR/AR headset) or app recording: frame pacing (hitches = frames the build
    failed to deliver while the picture moved, with times and ms; effective fps per second), side-by-side /
    top-bottom stereo detection, comfort (fast camera turns; pass the headset's horizontal fov_deg for deg/s)
    and flashes above 3 per second (photosensitivity triage). source = file, URL or watch id.
    Returns a summary plus a chart."""
    r = tools.capture_qa(source, start, end, fov_deg)
    out: list = [r["summary"]]
    if r.get("chart"):
        out.append(_img(r["chart"]))
    return out


@mcp.tool()
@_explained
def image(source: str, region: str | None = None) -> list:
    """One image (file or URL). Without region: an overview with a lettered grid (when the image is larger than
    you can see at full resolution), plus EXIF, palette and OCR text with grid cells. With region ("B2", "B2:C3",
    or "x,y,w,h" in px or 0-1 fractions): that part at native resolution, enlarged when small. Use it to read small
    text in big screenshots and photos instead of guessing."""
    from .images import inspect
    r = inspect(source, region)
    return [json.dumps({k: v for k, v in r.items() if k != "path"}, ensure_ascii=False), _img(r["path"])]


@mcp.tool()
@_explained
def images(sources: list[str], recursive: bool = False) -> list:
    """Index images (folders, files, globs, URLs) into an album: thumbnails, dates, cameras, OCR text, palettes,
    near-duplicate groups and a visual index. Returns INDEX.md and the first contact sheet. Then search(query,
    album_id) finds images by content or text, look(album_id, "image:N") gives an original, image(path, region)
    reads detail."""
    from .images import album
    d = album(sources, recursive=recursive)
    m = json.loads((d / "watch.json").read_text())
    out: list = [(d / "INDEX.md").read_text()]
    if m.get("sheets"):
        out.append(_img(str(d / m["sheets"][0]["path"])))
    return out


@mcp.tool()
@_explained
def live_start(source: str = "camera", watch_for: list[str] | None = None, hands: bool = False, pose: bool = False,
               mic: str = "auto", osc: str | None = None) -> str:
    """Start live perception in the background: source = "camera" (or "camera:N"/name), "screen" ("screen:N"), a
    stream URL (rtsp/srt/YouTube live) or a file played in real time. It tracks motion, cuts, colour, sound (level,
    beat, voice pitch, speech), faces (52 blendshapes, head pose: expression, not feelings; no identities), optional
    hands (gesture, pinch) and body, affect (energy, arousal vs this session, expression valence, engagement) and
    visual triggers (watch_for: ["someone waving"]). It also streams to a monitor page and to OSC if given.
    Only start the camera or microphone when the user asked for it. Then use live_now, live_snapshot, live_stop."""
    from . import live
    args = [source, "--mic", mic]
    if watch_for:
        args += ["--watch-for", *watch_for]
    if hands:
        args.append("--hands")
    if pose:
        args.append("--pose")
    if osc:
        args += ["--osc", osc]
    s = live.start_background(args)
    return json.dumps({"started": True, "capture": s["capture"],
                       "monitor": f"http://127.0.0.1:{s['port']}/?k={s['token']}", "dir": s["dir"]})


@mcp.tool()
@_explained
def live_now() -> list:
    """What the running live session sees and hears right now: a summary, the latest frame, and recent events."""
    from . import live
    from .cli import now_text
    r = live.request({"cmd": "status"})
    f = live.request({"cmd": "frame"})
    return [now_text(r, f), _img(f["path"])]


@mcp.tool()
@_explained
def live_snapshot(seconds: float = 30) -> list:
    """Turn the last `seconds` of the live session (up to the buffer, 120 s by default) into a normal watch: shots,
    text, speech, sound, plus a 'Live signals' section (affect, smiles, nods, gestures, triggers). All watch tools
    (zoom, frame, diff, search) then work on it."""
    from . import live
    r = live.request({"cmd": "snapshot", "seconds": seconds}, timeout=900)
    out: list = [r.get("index") or json.dumps(r)]
    if r.get("watch"):
        m = json.loads((Path(r["watch"]) / "watch.json").read_text())
        if m.get("sheets"):
            out.append(_img(str(Path(r["watch"]) / m["sheets"][0]["path"])))
    return out


@mcp.tool()
@_explained
def live_watch_for(add: list[str] | None = None, remove: list[str] | None = None) -> str:
    """Change the running session's visual triggers (descriptions of things to watch for)."""
    from . import live
    return json.dumps(live.request({"cmd": "watch_for", "add": add or [], "remove": remove or []}))


@mcp.tool()
@_explained
def live_stop() -> str:
    """Stop the live session (camera, microphone and streams close)."""
    from . import live
    return json.dumps(live.request({"cmd": "stop"}))


@mcp.tool()
@_explained
def list_watches() -> str:
    """Previously watched videos (cached; re-watching is instant)."""
    return json.dumps(tools.list_watches(), indent=1, ensure_ascii=False)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
