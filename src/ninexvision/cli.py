"""9xvision command line."""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__, tools
from .watch import watch


def _print(obj) -> None:
    print(json.dumps(obj, indent=1, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="9xvision", description="Video perception toolkit for AI agents.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("watch", help="Analyse a video file or URL into a watch folder; prints INDEX.md")
    w.add_argument("source")
    w.add_argument("--mode", choices=["auto", "film", "screen"], default="auto")
    w.add_argument("--out")
    w.add_argument("--force", action="store_true")
    w.add_argument("--transcribe", choices=["auto", "on", "off"], default="auto")
    w.add_argument("--no-ocr", action="store_true")
    w.add_argument("--sheet-cells", type=int)
    w.add_argument("--quiet", action="store_true")

    z = sub.add_parser("zoom", help="Dense timestamped grid of an interval")
    z.add_argument("watch"); z.add_argument("start"); z.add_argument("end"); z.add_argument("-n", type=int, default=12)

    f = sub.add_parser("frame", help="Full-resolution frame at a time, optional crop x,y,w,h (px or 0-1 fractions)")
    f.add_argument("watch"); f.add_argument("at"); f.add_argument("--crop")

    d = sub.add_parser("diff", help="What changed between two times (side by side + highlighted regions)")
    d.add_argument("watch"); d.add_argument("a"); d.add_argument("b")

    s = sub.add_parser("search", help="Search on-screen text and speech across watches")
    s.add_argument("query"); s.add_argument("--watch")

    lk = sub.add_parser("look", help="Path of an artefact: sheet:N timeline:N view:N shot:N page:N colour")
    lk.add_argument("watch"); lk.add_argument("what")

    sub.add_parser("list", help="List watches")
    sub.add_parser("mcp", help="Run the MCP server on stdio")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "watch":
            log = (lambda m: None) if a.quiet else (lambda m: print(f"[9xvision] {m}", file=sys.stderr))
            wd = watch(a.source, mode=a.mode, out=a.out, force=a.force, transcribe=a.transcribe,
                       ocr=not a.no_ocr, sheet_cells=a.sheet_cells, log=log)
            print((wd / "INDEX.md").read_text())
        elif a.cmd == "zoom":
            _print(tools.zoom(a.watch, a.start, a.end, a.n))
        elif a.cmd == "frame":
            _print(tools.frame(a.watch, a.at, a.crop))
        elif a.cmd == "diff":
            _print(tools.diff(a.watch, a.a, a.b))
        elif a.cmd == "search":
            _print(tools.search(a.query, a.watch))
        elif a.cmd == "look":
            _print(tools.look(a.watch, a.what))
        elif a.cmd == "list":
            _print(tools.list_watches())
        elif a.cmd == "mcp":
            from .mcp_server import main as mcp_main
            mcp_main()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"9xvision: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
