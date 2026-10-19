"""Drill-down tools on an existing watch: zoom, frame, diff, search."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import media, render
from .watch import load, watches_root


def zoom(watch: str, start, end, n: int = 12) -> dict:
    d, m = load(watch)
    a0, a1 = media.parse_t(start), media.parse_t(end)
    lo = m["probe"].get("offset", 0.0)
    hi = lo + m["probe"]["duration"]
    t0, t1 = max(lo, min(a0, hi)), max(lo, min(a1, hi))
    if t1 <= t0:
        t0, t1 = max(lo, min(t0, hi - 1.0)), hi if t0 >= hi - 1.0 else min(hi, t0 + 1.0)
    n = max(2, min(int(n), 30))
    times = list(np.linspace(t0, t1, n))
    frames = [media.grab_frame(m["file"], t, max_edge=720) for t in times]
    p = d / "zoom" / f"zoom_{t0:.2f}-{t1:.2f}_{n}.jpg"
    size = render.contact_sheet(frames, [media.fmt_t(t) for t in times], p,
                                title=f"zoom {media.fmt_t(t0)}-{media.fmt_t(t1)}  step {(t1 - t0) / (n - 1):.2f}s")
    out = {"path": str(p), "size": list(size), "tokens": render.token_estimate(*size), "times": [round(t, 2) for t in times]}
    if abs(a0 - t0) > 0.06 or abs(a1 - t1) > 0.06:
        out["note"] = (f"asked for {_t(a0)}-{_t(a1)}; the video covers {media.fmt_t(lo)}-{media.fmt_t(hi)}, "
                       f"so this shows {media.fmt_t(t0)}-{media.fmt_t(t1)}")
    return out


def _t(x: float) -> str:
    return media.fmt_t(x) if x >= 0 else f"{x:g} s"


def frame(watch: str, at, crop: str | None = None, max_edge: int = render.MAX_EDGE) -> dict:
    d, m = load(watch)
    asked = media.parse_t(at)
    lo = m["probe"].get("offset", 0.0)
    hi = lo + m["probe"]["duration"]
    t = min(max(asked, lo), max(lo, hi - 0.05))
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
    out = {"path": str(p), "size": list(im.size), "source_px": [W, H], "t": round(t, 2),
           "tokens": render.token_estimate(*im.size)}
    if abs(asked - t) > 0.06:
        out["note"] = f"asked for {_t(asked)}, which is outside {media.fmt_t(lo)}-{media.fmt_t(hi)}; showing {media.fmt_t(t)}"
    return out


def diff(watch: str, a, b) -> dict:
    d, m = load(watch)
    aa, ab = media.parse_t(a), media.parse_t(b)
    lo = m["probe"].get("offset", 0.0)
    hi = lo + m["probe"]["duration"] - 0.05
    ta, tb = min(max(aa, lo), hi), min(max(ab, lo), hi)
    fa = media.grab_frame(m["file"], ta, max_edge=1000)
    fb = media.grab_frame(m["file"], tb, max_edge=1000)
    p = d / "diffs" / f"diff_{ta:.2f}_{tb:.2f}.jpg"
    res = render.diff_image(fa, fb, p, (media.fmt_t(ta), media.fmt_t(tb)))
    scale = m["probe"]["width"] / fa.shape[1]
    for r in res["regions"]:
        for k in ("x", "y", "w", "h"):
            r[k] = int(r[k] * scale)
    res["tokens"] = render.token_estimate(*res["size"])
    res["times"] = [round(ta, 2), round(tb, 2)]
    if abs(aa - ta) > 0.06 or abs(ab - tb) > 0.06:
        res["note"] = f"asked for {_t(aa)} and {_t(ab)}; the video covers {media.fmt_t(lo)}-{media.fmt_t(hi)}, so this compares {media.fmt_t(ta)} and {media.fmt_t(tb)}"
    return res


def search(query: str, watch: str | None = None, limit: int = 30, visual: bool = True, log=lambda s: None) -> dict:
    """Text hits (OCR'd screen text, speech) across watches, plus visual moments (what is in the picture)
    in one watch: the given one, or the most recent if none is given."""
    out: dict = {"query": query, "text_hits": text_search(query, watch, limit)}
    if visual:
        from . import semantic
        try:
            d, m = load(watch) if watch else load(list_watches()[0]["id"])
        except (FileNotFoundError, IndexError):
            return out
        try:
            if m.get("kind") == "album":
                from .images import album_search
                out["visual"] = {"watch": m["id"], "album": True, **album_search(d, m, query)}
            else:
                out["visual"] = {"watch": m["id"], **semantic.search(d, m, query, log=log)}
        except ImportError as e:
            out["visual"] = {"error": f"visual search needs onnxruntime + tokenizers ({e})"}
    return out


def search_text(r: dict) -> str:
    """Readable form of search() for people and shell agents."""
    L = [f"text hits for {r['query']!r}: {len(r['text_hits'])}"]
    for h in r["text_hits"][:15]:
        where = f"#{h['image']}" if h["kind"] == "image-text" else media.fmt_t(h["t"])
        L.append(f"  {where:>8}  {h['kind']:<11} {h['text'][:90]}  [{h['watch']}]")
    v = r.get("visual")
    if v and v.get("error"):
        L.append(f"visual: {v['error']}")
    elif v and v.get("album"):
        L.append(f"visual matches in album {v['watch']}: " + " · ".join(f"#{x['id']} {Path(x['file']).name} {x['match']} {x['score']:.2g}"
                                                                     for x in v["results"][:10]))
        if v.get("sheet"):
            L.append(f"  best images: {v['sheet']['path']}")
        if v["results"] and all(x["match"] == "weak" for x in v["results"]):
            L.append("  all weak: probably not in these images")
    elif v:
        L.append(f"visual moments in {v['watch']}: " + " · ".join(f"{mo['at']} {mo['match']} {mo['score']:.2g}" for mo in v["moments"]))
        if v.get("sheet"):
            L.append(f"  best frames: {v['sheet']['path']}")
        if v["moments"] and all(mo["match"] == "weak" for mo in v["moments"]):
            L.append("  all weak: probably not in the video")
        if "negation" in v.get("note", ""):
            L.append("  note: " + v["note"].split(". Search")[0] + ". Search for what IS there instead.")
    return "\n".join(L)


def text_search(query: str, watch: str | None = None, limit: int = 30) -> list[dict]:
    """Case-insensitive text search over OCR'd screen text and transcripts."""
    q = query.lower()
    dirs = [load(watch)[0]] if watch else sorted(p.parent for p in watches_root().glob("*/watch.json"))
    hits = []
    for d in dirs:
        m = json.loads((d / "watch.json").read_text())
        for it in m.get("images", []):
            for line in it.get("text", []):
                if q in line.lower():
                    hits.append({"watch": m["id"], "kind": "image-text", "image": it["id"], "t": 0.0, "text": line,
                                 "file": it["file"]})
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
        row = {"id": m["id"], "title": m["title"], "mode": m["mode"], "dir": str(p.parent)}
        if m.get("kind") == "album":
            row["images"] = len(m.get("images", []))
        else:
            row["duration"] = m["probe"]["duration"]
        out.append(row)
    return out


def look(watch: str, what: str) -> dict:
    """Resolve a named artefact (sheet:N, timeline:N, view:N, shot:N, page:N, colour) to a file path."""
    d, m = load(watch)
    kind, _, num = what.partition(":")
    k = int(num) if num.isdigit() else 1
    table = {"sheet": [s["path"] for s in m.get("sheets", [])],
             "timeline": [s["path"] for s in m.get("timeline_sheets", [])],
             "view": {v["id"]: v["image"] for v in m.get("views", [])},
             "shot": {s["id"]: s["keyframe"] for s in m.get("shots", [])},
             "page": {p["id"]: p["stitched"] for p in m.get("pages", []) if p.get("stitched")},
             "image": {it["id"]: it["file"] for it in m.get("images", [])}}
    if kind in ("colour", "color", "colour_script"):
        if not m.get("colour_script"):
            raise ValueError("this watch has no colour script (screen mode); try sheet:1 or view:N")
        return {"path": str(Path(d) / m["colour_script"])}
    if kind not in table:
        raise ValueError("what must be sheet:N, timeline:N, view:N, shot:N, page:N, image:N (albums) or colour")
    items = table[kind]
    if not items:
        raise ValueError(f"this {m['mode']}-mode watch has no {kind}s"
                         + ("; film watches have shot:N and sheet:N" if kind in ("view", "timeline", "page") else
                            "; screen watches have view:N, timeline:N and sheet:N" if kind == "shot" else ""))
    rel = items[k - 1] if isinstance(items, list) and 1 <= k <= len(items) else items.get(k) if isinstance(items, dict) else None
    if rel is None:
        ids = list(range(1, len(items) + 1)) if isinstance(items, list) else sorted(items)
        raise ValueError(f"no {kind}:{k}; this watch has {kind}:{ids[0]}..{ids[-1]}" if len(ids) > 1 else f"no {kind}:{k}; only {kind}:{ids[0]}")
    return {"path": str(Path(d) / rel)}


def capture_qa(source: str, start=None, end=None, fov_deg: float | None = None) -> dict:
    """Frame pacing, hitches, stereo, fast turns and flashes for a game / XR / app capture (file, URL or watch id)."""
    from . import qa
    from .watch import HOME, watch as run_watch
    p = Path(source).expanduser()
    s = media.parse_t(start) if start not in (None, "") else None
    e = media.parse_t(end) if end not in (None, "") else None
    if media.is_url(source):   # download only (no full watch)
        from .watch import download_dir
        section = (s or 0.0, e if e is not None else 10 ** 7) if (s is not None or e is not None) else None
        dl = download_dir(source, section)
        f, _ = media.fetch(source, dl, section=section)
        f, out = str(f), dl / "qa"
    elif p.exists() and p.is_file():
        f, out = str(p), HOME / "qa" / p.stem
    else:
        d, m = load(source)
        f, out = m["file"], d / "qa"
    r = qa.analyse(f, s, e, fov_deg, out_dir=out)
    r["summary"] = qa.summary_md(r)
    return r
