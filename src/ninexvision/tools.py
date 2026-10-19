"""Drill-down tools on an existing watch: zoom, frame, diff, search."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import media, render
from .watch import load, watches_root


def zoom(watch: str, start, end, n: int = 12) -> dict:
    d, m = load(watch)
    t0, t1 = media.parse_t(start), media.parse_t(end)
    dur = m["probe"]["duration"]
    t0, t1 = max(0.0, min(t0, dur)), max(0.0, min(t1, dur))
    if t1 <= t0:
        t1 = min(dur, t0 + 1.0)
    n = max(2, min(int(n), 30))
    times = list(np.linspace(t0, t1, n))
    frames = [media.grab_frame(m["file"], t, max_edge=720) for t in times]
    p = d / "zoom" / f"zoom_{t0:.2f}-{t1:.2f}_{n}.jpg"
    size = render.contact_sheet(frames, [media.fmt_t(t) for t in times], p,
                                title=f"zoom {media.fmt_t(t0)}-{media.fmt_t(t1)}  step {(t1 - t0) / (n - 1):.2f}s")
    return {"path": str(p), "size": list(size), "tokens": render.token_estimate(*size), "times": [round(t, 2) for t in times]}


def frame(watch: str, at, crop: str | None = None, max_edge: int = render.MAX_EDGE) -> dict:
    d, m = load(watch)
    t = media.parse_t(at)
    fr = media.grab_frame(m["file"], t)
    H, W = fr.shape[:2]
    tag = ""
    if crop:
        x, y, w, h = [float(v) for v in crop.split(",")]
        if max(x, y, w, h) <= 1.0:  # fractions
            x, y, w, h = x * W, y * H, w * W, h * H
        x, y = int(max(0, x)), int(max(0, y))
        w, h = int(min(W - x, w)), int(min(H - y, h))
        fr = fr[y:y + h, x:x + w]
        tag = f"_crop{x}-{y}-{w}-{h}"
    p = d / "frames" / f"frame_{t:.2f}{tag}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    im = Image.fromarray(fr)
    if max(im.size) > max_edge:
        s = max_edge / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    im.save(p, optimize=True)
    return {"path": str(p), "size": list(im.size), "source_px": [W, H], "t": round(t, 2),
            "tokens": render.token_estimate(*im.size)}


def diff(watch: str, a, b) -> dict:
    d, m = load(watch)
    ta, tb = media.parse_t(a), media.parse_t(b)
    fa = media.grab_frame(m["file"], ta, max_edge=1000)
    fb = media.grab_frame(m["file"], tb, max_edge=1000)
    p = d / "diffs" / f"diff_{ta:.2f}_{tb:.2f}.jpg"
    res = render.diff_image(fa, fb, p, (media.fmt_t(ta), media.fmt_t(tb)))
    scale = m["probe"]["width"] / fa.shape[1]
    for r in res["regions"]:
        for k in ("x", "y", "w", "h"):
            r[k] = int(r[k] * scale)
    res["tokens"] = render.token_estimate(*res["size"])
    return res


def search(query: str, watch: str | None = None, limit: int = 30) -> list[dict]:
    """Case-insensitive text search over OCR'd screen text and transcripts."""
    q = query.lower()
    dirs = [load(watch)[0]] if watch else sorted(p.parent for p in watches_root().glob("*/watch.json"))
    hits = []
    for d in dirs:
        m = json.loads((d / "watch.json").read_text())
        for v in m.get("views", []):
            for ln in v.get("text", []):
                if q in ln["text"].lower():
                    hits.append({"watch": m["id"], "kind": "screen-text", "view": v["id"], "t": v["first_t"],
                                 "text": ln["text"], "image": v["image"]})
        for sh in m.get("shots", []):
            for line in sh.get("text", []):
                if q in line.lower():
                    hits.append({"watch": m["id"], "kind": "screen-text", "shot": sh["id"], "t": sh["key_t"],
                                 "text": line, "image": sh["keyframe"]})
        tp = d / "transcript.json"
        if tp.exists():
            for s in json.loads(tp.read_text()):
                if q in s["text"].lower():
                    hits.append({"watch": m["id"], "kind": "speech", "t": s["start"], "text": s["text"]})
        if len(hits) >= limit:
            break
    return hits[:limit]


def list_watches() -> list[dict]:
    out = []
    for p in sorted(watches_root().glob("*/watch.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        m = json.loads(p.read_text())
        out.append({"id": m["id"], "title": m["title"], "mode": m["mode"], "duration": m["probe"]["duration"],
                    "dir": str(p.parent)})
    return out


def look(watch: str, what: str) -> dict:
    """Resolve a named artefact (sheet:N, timeline:N, view:N, shot:N, page:N, colour) to a file path."""
    d, m = load(watch)
    kind, _, num = what.partition(":")
    k = int(num) if num.isdigit() else 1
    if kind == "sheet":
        rel = m["sheets"][k - 1]["path"]
    elif kind == "timeline":
        rel = m["timeline_sheets"][k - 1]["path"]
    elif kind == "view":
        rel = next(v["image"] for v in m["views"] if v["id"] == k)
    elif kind == "shot":
        rel = next(s["keyframe"] for s in m["shots"] if s["id"] == k)
    elif kind == "page":
        rel = next(p["stitched"] for p in m["pages"] if p["id"] == k and p.get("stitched"))
    elif kind in ("colour", "color", "colour_script"):
        rel = m["colour_script"]
    else:
        raise ValueError("what must be sheet:N, timeline:N, view:N, shot:N, page:N or colour")
    return {"path": str(Path(d) / rel)}
