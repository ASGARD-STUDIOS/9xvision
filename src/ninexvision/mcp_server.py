"""MCP server: the same tools, returning images inline so any MCP client can see them."""
from __future__ import annotations

import json
from pathlib import Path

try:  # mcp >= 2
    from mcp.server.mcpserver import Image, MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server, Image

from . import tools
from .watch import watch as run_watch

mcp = _Server("9xvision", instructions=(
    "Video perception tools. Call watch(source) first: it returns INDEX.md (structure, flow, text, timings) "
    "and the first overview sheet. Look at sheets before zooming. Use zoom/frame/diff only where you need "
    "more detail, and cite mm:ss.s timestamps in answers."))


def _img(path: str) -> Image:
    return Image(path=path)


@mcp.tool()
def watch(source: str, mode: str = "auto", force: bool = False) -> list:
    """Analyse a local video file or URL (YouTube, Vimeo, TikTok, X...). mode: auto | film | screen
    (screen = app/screen recordings). Returns INDEX.md plus the first overview contact sheet."""
    wd = run_watch(source, mode=mode, force=force)
    m = json.loads((wd / "watch.json").read_text())
    out: list = [(wd / "INDEX.md").read_text()]
    if m.get("sheets"):
        out.append(_img(str(wd / m["sheets"][0]["path"])))
    return out


@mcp.tool()
def look(watch_id: str, what: str) -> list:
    """Show an artefact of a watch: sheet:N, timeline:N, view:N (full-res screen), shot:N (keyframe),
    page:N (stitched long page), colour (colour script)."""
    p = tools.look(watch_id, what)["path"]
    return [f"{what}: {p}", _img(p)]


@mcp.tool()
def zoom(watch_id: str, start: str, end: str, n: int = 12) -> list:
    """Dense timestamped grid of frames between start and end (mm:ss.s or seconds), up to 30 frames."""
    r = tools.zoom(watch_id, start, end, n)
    return [json.dumps({k: r[k] for k in ("times", "tokens")}), _img(r["path"])]


@mcp.tool()
def frame(watch_id: str, at: str, crop: str | None = None) -> list:
    """Full-resolution frame at a time. crop = "x,y,w,h" in pixels or 0-1 fractions, for reading small detail."""
    r = tools.frame(watch_id, at, crop)
    return [json.dumps({k: r[k] for k in ("t", "size", "source_px")}), _img(r["path"])]


@mcp.tool()
def diff(watch_id: str, a: str, b: str) -> list:
    """Compare two moments: side-by-side plus highlighted changed regions (pixel boxes in source coordinates)."""
    r = tools.diff(watch_id, a, b)
    return [json.dumps({k: r[k] for k in ("changed_pct", "regions")}), _img(r["path"])]


@mcp.tool()
def search(query: str, watch_id: str | None = None) -> str:
    """Find on-screen text (OCR) and spoken words across watched videos, with timestamps."""
    return json.dumps(tools.search(query, watch_id), indent=1, ensure_ascii=False)


@mcp.tool()
def list_watches() -> str:
    """Previously watched videos (cached; re-watching is instant)."""
    return json.dumps(tools.list_watches(), indent=1, ensure_ascii=False)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
