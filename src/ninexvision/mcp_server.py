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
    "Video perception tools. Call watch(source) first: it returns INDEX.md (structure, flow, text, timings) "
    "and the first overview sheet. Look at sheets before zooming. Use zoom/frame/diff only where you need "
    "more detail, and cite mm:ss.s timestamps in answers."))


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
    return [json.dumps({k: r[k] for k in ("times", "tokens")}), _img(r["path"])]


@mcp.tool()
@_explained
def frame(watch_id: str, at: str, crop: str | None = None) -> list:
    """Full-resolution frame at a time. crop = "x,y,w,h" in pixels or 0-1 fractions, for reading small detail."""
    r = tools.frame(watch_id, at, crop)
    return [json.dumps({k: r[k] for k in ("t", "size", "source_px")}), _img(r["path"])]


@mcp.tool()
@_explained
def diff(watch_id: str, a: str, b: str) -> list:
    """Compare two moments: side-by-side plus highlighted changed regions (pixel boxes in source coordinates)."""
    r = tools.diff(watch_id, a, b)
    return [json.dumps({k: r[k] for k in ("changed_pct", "regions")}), _img(r["path"])]


@mcp.tool()
@_explained
def search(query: str, watch_id: str | None = None, visual: bool = True) -> list:
    """Find moments by on-screen text (OCR), spoken words, AND what is in the picture ("a red car at night",
    "a pie chart", "someone hugging"). Text hits cover every watch; visual moments cover watch_id (default:
    the most recent watch) and come with a contact sheet of the best frames. Visual scores: strong >= 0.10;
    all-weak means it is probably not in the video. The first visual search on a watch indexes it (seconds)."""
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
def list_watches() -> str:
    """Previously watched videos (cached; re-watching is instant)."""
    return json.dumps(tools.list_watches(), indent=1, ensure_ascii=False)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
