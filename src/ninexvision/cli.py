"""9xvision command line."""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__, tools
from .watch import watch


def _print(obj) -> None:
    print(json.dumps(obj, indent=1, ensure_ascii=False))


def _live(a) -> int:
    from . import live
    cmd = a.live_cmd or "start"
    if cmd == "devices":
        for kind, items in live.devices().items():
            print(f"{kind}:")
            for it in items:
                print(f"  {it['index']}: {it['name']}")
        return 0
    if cmd == "start":
        if a.live_cmd is None:      # bare `9xvision live`
            a = argparse.Namespace(source="camera", mic="auto", fps=15, width=640, no_face=False, hands=False, pose=False,
                                   mirrored=False, watch_for=[], watch_threshold=0.1, buffer=120, osc=None, port=8899,
                                   host="127.0.0.1", allow_origin=[], jsonl=None, state_every=1, duration=None,
                                   loop=False, beat_lead=0.0, open=False, quiet=False)
        if live.session():
            raise RuntimeError("a live session is already running (9xvision live stop)")
        log = (lambda m: None) if a.quiet else (lambda m: print(f"[9xvision] {m}", file=sys.stderr))
        if a.host not in ("127.0.0.1", "localhost", "::1"):
            log(f"listening on {a.host}: other machines on this network can read the live signals")
        lv = live.Live(a.source, mic=a.mic, fps=a.fps, width=a.width, face=not a.no_face, hands=a.hands, pose=a.pose,
                       mirrored=a.mirrored, watch_for=a.watch_for, watch_threshold=a.watch_threshold, buffer_s=a.buffer,
                       osc=a.osc, ws_port=a.port or None, host=a.host, allow_origins=a.allow_origin, jsonl=a.jsonl,
                       state_every=a.state_every, loop=a.loop, beat_lead=a.beat_lead, log=log)
        log(f"source: {lv.cap.info} at {lv.cap.size[0]}x{lv.cap.size[1]}, {a.fps:g} fps")
        if lv.monitor_url():
            log(f"monitor: {lv.monitor_url()}")
            log(f"WebSocket: ws://{a.host}:{a.port}/ (states + events as JSON)" + (f"; OSC -> {a.osc}" if a.osc else ""))
            if a.open:
                import webbrowser
                webbrowser.open(lv.monitor_url())
        if not a.quiet and a.jsonl != "-":
            def ticker():
                import time as _t
                while not lv.stop_flag.is_set():
                    _t.sleep(1)
                    s, af = lv.latest, (lv.latest.get("affect") or {})
                    au = s.get("audio") or {}
                    print(f"\r[9xvision] {s.get('t', 0):7.1f}s  {lv.stats['fps']:4.1f} fps  energy {af.get('energy', 0):.2f}  "
                          f"arousal {af.get('arousal', 0):.2f}  faces {(s.get('face') or {}).get('count', 0)}  "
                          f"{(str(au.get('bpm')) + ' BPM') if au.get('bpm') else ''}   ", end="", file=sys.stderr, flush=True)
            import threading
            threading.Thread(target=ticker, daemon=True).start()
        try:
            lv.run(a.duration)
        except KeyboardInterrupt:
            lv.close()
        if not a.quiet:
            print("", file=sys.stderr)
        return 0
    if cmd == "now":
        r = live.request({"cmd": "status"})
        f = live.request({"cmd": "frame"})
        if a.json:
            _print({**r, "frame": f})
        else:
            print(now_text(r, f))
        return 0
    if cmd == "snapshot":
        r = live.request({"cmd": "snapshot", "seconds": a.seconds}, timeout=900)
        print(r.get("index") or json.dumps(r, indent=1))
        return 0
    if cmd == "watch-for":
        req = {"cmd": "watch_for", "add": a.queries, "remove": a.remove}
        if a.clear:
            req["set"] = []
        _print(live.request(req))
        return 0
    if cmd == "stop":
        _print(live.request({"cmd": "stop"}))
        return 0
    return 2


def now_text(r: dict, frame: dict | None = None) -> str:
    """A readable account of a live status reply."""
    from .media import fmt_t
    s = r.get("state") or {}
    af, au, v, f = s.get("affect") or {}, s.get("audio") or {}, s.get("video") or {}, s.get("face")
    L = [f"live: {r.get('source')} at {fmt_t(s.get('t', 0))} into the session; {r['stats'].get('fps')} fps, "
         f"{r['stats'].get('proc_ms')} ms per frame"]
    L.append(f"- affect: energy {af.get('energy')}, arousal {af.get('arousal')} (0.5 = typical for this session)"
             + (f", face expression valence {af['expression_valence']:+.2f}" if af.get("expression_valence") is not None else "")
             + (f", engagement {af.get('engagement')}" if f else "") + (" [warming up]" if af.get("warming_up") else ""))
    if au:
        L.append(f"- sound: {au.get('db')} dBFS" + (f", {au['bpm']} BPM" if au.get("bpm") else "")
                 + (", speaking" if au.get("speaking") else "") + (f", pitch {au['pitch_hz']} Hz" if au.get("pitch_hz") else ""))
    L.append(f"- picture: motion {v.get('motion')}, brightness {v.get('brightness')}, palette {' '.join(v.get('palette') or [])}")
    if f:
        top = sorted(((k, x) for k, x in (f.get("blend") or {}).items() if not k.startswith("eyeLook")), key=lambda kv: -kv[1])[:5]
        L.append(f"- faces: {f.get('count')}; head yaw {f.get('yaw')} pitch {f.get('pitch')}; smile {f.get('smile')}; "
                 "strongest: " + ", ".join(f"{k} {x:.2f}" for k, x in top))
    else:
        L.append("- faces: none")
    if s.get("hands"):
        L.append("- hands: " + ", ".join(f"{h['hand']} {h.get('gesture') or '-'} pinch {h.get('pinch')}" for h in s["hands"]))
    if s.get("watch"):
        L.append("- watching for: " + ", ".join(f"{q} {x:.3f}" for q, x in s["watch"].items()))
    if r.get("watch_for_error"):
        L.append(f"- {r['watch_for_error']}")
    ev = [e for e in r.get("events", []) if e["kind"] not in ("beat", "blink")][-12:]
    if ev:
        L.append("- recent events: " + ", ".join(f"{e['kind']}"
                 + (f"({e.get('gesture') or e.get('what')})" if e.get("gesture") or e.get("what") else "")
                 + f" @{fmt_t(e['t'])}" for e in ev))
    if frame:
        L.append(f"- frame: {frame['path']}")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="9xvision", description="Perception for AI agents: video, live input and images.")
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
    w.add_argument("--start", help="watch only from here (mm:ss, h:mm:ss or seconds); URLs download only the window")
    w.add_argument("--end", help="watch only up to here")
    w.add_argument("--cookies-from-browser", metavar="BROWSER",
                   help="for login-gated URLs: let yt-dlp use this browser's cookies (chrome, safari, firefox, ...)")

    z = sub.add_parser("zoom", help="Dense timestamped grid of an interval")
    z.add_argument("watch"); z.add_argument("start"); z.add_argument("end"); z.add_argument("-n", type=int, default=12)

    f = sub.add_parser("frame", help="Full-resolution frame at a time, optional crop x,y,w,h (px or 0-1 fractions)")
    f.add_argument("watch"); f.add_argument("at"); f.add_argument("--crop")

    d = sub.add_parser("diff", help="What changed between two times (side by side + highlighted regions)")
    d.add_argument("watch"); d.add_argument("a"); d.add_argument("b")

    s = sub.add_parser("search", help="Search on-screen text and speech across watches")
    s.add_argument("query"); s.add_argument("--watch")
    s.add_argument("--text-only", action="store_true", help="skip visual search (OCR text and speech only)")
    s.add_argument("--json", action="store_true", help="print the full JSON")

    lk = sub.add_parser("look", help="Path of an artefact: sheet:N timeline:N view:N shot:N page:N colour")
    lk.add_argument("watch"); lk.add_argument("what")

    q = sub.add_parser("qa", help="Capture QA for games/XR/apps: frame pacing, hitches, stereo, fast turns, flashes")
    q.add_argument("source", help="video file, URL or watch id")
    q.add_argument("--start"); q.add_argument("--end")
    q.add_argument("--fov", type=float, help="horizontal field of view in degrees (turn speeds in deg/s)")
    q.add_argument("--json", action="store_true", help="print the full JSON instead of the summary")
    im = sub.add_parser("image", help="One image: overview with a grid, or a region at full resolution (B2, B2:C3, x,y,w,h)")
    im.add_argument("source", help="image file or URL")
    im.add_argument("--region", help="grid cell (B2), range (B2:C3) or x,y,w,h in px or 0-1 fractions")
    im.add_argument("--no-ocr", action="store_true")
    ims = sub.add_parser("images", help="Index images (folders, files, globs, URLs) into an album; prints its INDEX.md")
    ims.add_argument("sources", nargs="+")
    ims.add_argument("--recursive", "-r", action="store_true")
    ims.add_argument("--no-ocr", action="store_true")
    ims.add_argument("--force", action="store_true")
    ims.add_argument("--quiet", action="store_true")

    lv = sub.add_parser("live", help="Live perception from a camera, mic, screen, stream or file (OSC / WebSocket / JSON)",
                        description="Actions: devices | start [SOURCE] | now | snapshot [SECONDS] | watch-for QUERY... | stop. "
                                    "SOURCE: camera[:N|name] (default), screen[:N], a stream URL (rtsp/srt/YouTube live...), "
                                    "or a file/URL played in real time.")
    lsub = lv.add_subparsers(dest="live_cmd")
    lsub.add_parser("devices", help="List cameras, microphones and screens")
    ls = lsub.add_parser("start", help="Start a live session (foreground; Ctrl-C stops it)")
    ls.add_argument("source", nargs="?", default="camera")
    ls.add_argument("--mic", default="auto", help="microphone index or name, or 'none' (default: the built-in one)")
    ls.add_argument("--fps", type=float, default=15, help="analysis frames per second (default 15)")
    ls.add_argument("--width", type=int, default=640, help="analysis width in pixels (default 640)")
    ls.add_argument("--no-face", action="store_true", help="skip face tracking")
    ls.add_argument("--hands", action="store_true", help="track hands: gesture, position, pinch")
    ls.add_argument("--pose", action="store_true", help="track the body: key points, movement energy, hands up")
    ls.add_argument("--mirrored", action="store_true", help="the picture is a mirrored selfie view (for left/right hands)")
    ls.add_argument("--watch-for", nargs="+", default=[], metavar="QUERY", help='visual triggers, e.g. "someone waving"')
    ls.add_argument("--watch-threshold", type=float, default=0.1)
    ls.add_argument("--buffer", type=float, default=120, help="seconds kept in memory for snapshots (default 120)")
    ls.add_argument("--osc", metavar="HOST:PORT", help="send OSC here (TouchDesigner, Max, Ableton, Resolume...)")
    ls.add_argument("--port", type=int, default=8899, help="WebSocket + monitor page port (0 = off; default 8899)")
    ls.add_argument("--host", default="127.0.0.1", help="listen address (default local only)")
    ls.add_argument("--allow-origin", nargs="+", default=[], metavar="URL", help="web origins allowed to connect, e.g. http://localhost:5173")
    ls.add_argument("--jsonl", metavar="PATH", help="also write every message as JSON lines ('-' = stdout)")
    ls.add_argument("--state-every", type=int, default=1, help="send every Nth state (events always go out)")
    ls.add_argument("--duration", type=float, help="stop after this many seconds")
    ls.add_argument("--loop", action="store_true", help="loop a file source")
    ls.add_argument("--beat-lead", type=float, default=0.0, help="fire beat events this many seconds early")
    ls.add_argument("--open", action="store_true", help="open the monitor page in the browser")
    ls.add_argument("--quiet", action="store_true")
    ln = lsub.add_parser("now", help="What the running session sees now (summary + a frame)")
    ln.add_argument("--json", action="store_true")
    lsn = lsub.add_parser("snapshot", help="Turn the last N seconds into a normal watch (prints its INDEX.md)")
    lsn.add_argument("seconds", nargs="?", type=float, default=30)
    lw = lsub.add_parser("watch-for", help="Add visual triggers to the running session")
    lw.add_argument("queries", nargs="*")
    lw.add_argument("--remove", nargs="+", default=[])
    lw.add_argument("--clear", action="store_true")
    lsub.add_parser("stop", help="Stop the running session")
    sub.add_parser("list", help="List watches")
    sub.add_parser("mcp", help="Run the MCP server on stdio")
    sub.add_parser("doctor", help="Check dependencies and which agents have 9xvision plugged in")
    su = sub.add_parser("setup", help="Install missing dependencies (asks first) and plug 9xvision into Claude Code, "
                                      "Codex, Cursor and Claude Desktop")
    su.add_argument("--yes", "-y", action="store_true", help="install missing pieces without asking")
    su.add_argument("--dry-run", action="store_true", help="show what would happen, change nothing")
    su.add_argument("--remove", action="store_true", help="unregister 9xvision from the agents")
    su.add_argument("--only", nargs="+", metavar="AGENT", help="claude | codex | cursor | desktop")
    sub.add_parser("skill", help="Print the agent instructions (SKILL.md) for agents without plugin support")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "watch":
            log = (lambda m: None) if a.quiet else (lambda m: print(f"[9xvision] {m}", file=sys.stderr))
            wd = watch(a.source, mode=a.mode, out=a.out, force=a.force, transcribe=a.transcribe,
                       ocr=not a.no_ocr, sheet_cells=a.sheet_cells, start=a.start, end=a.end,
                       cookies_from_browser=a.cookies_from_browser, log=log)
            print((wd / "INDEX.md").read_text())
        elif a.cmd == "zoom":
            _print(tools.zoom(a.watch, a.start, a.end, a.n))
        elif a.cmd == "frame":
            _print(tools.frame(a.watch, a.at, a.crop))
        elif a.cmd == "diff":
            _print(tools.diff(a.watch, a.a, a.b))
        elif a.cmd == "search":
            r = tools.search(a.query, a.watch, visual=not a.text_only,
                             log=lambda m: print(f"[9xvision] {m}", file=sys.stderr))
            if a.json:
                _print(r)
            else:
                print(tools.search_text(r))
        elif a.cmd == "look":
            _print(tools.look(a.watch, a.what))
        elif a.cmd == "qa":
            r = tools.capture_qa(a.source, a.start, a.end, a.fov)
            if a.json:
                _print(r)
            else:
                print(r["summary"])
        elif a.cmd == "image":
            from .images import inspect
            _print(inspect(a.source, a.region, ocr=not a.no_ocr))
        elif a.cmd == "images":
            from .images import album
            log = (lambda m: None) if a.quiet else (lambda m: print(f"[9xvision] {m}", file=sys.stderr))
            d = album(a.sources, recursive=a.recursive, ocr=not a.no_ocr, force=a.force, log=log)
            print((d / "INDEX.md").read_text())
        elif a.cmd == "live":
            return _live(a)
        elif a.cmd == "list":
            _print(tools.list_watches())
        elif a.cmd == "doctor":
            from .setup import doctor
            return doctor()
        elif a.cmd == "setup":
            from .setup import setup
            only = [{"desktop": "claude desktop"}.get(x.lower(), x) for x in a.only] if a.only else None
            return setup(yes=a.yes, dry=a.dry_run, remove=a.remove, only=only)
        elif a.cmd == "skill":
            from .setup import _skill_text
            print(_skill_text())
        elif a.cmd == "mcp":
            from .mcp_server import main as mcp_main
            mcp_main()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"9xvision: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
