"""The `watch` pipeline: source -> watch folder (INDEX.md, watch.json, sheets, keyframes)."""
from __future__ import annotations

import json
import os
from glob import escape as glob_escape
import re
import shutil
import time
from pathlib import Path

import numpy as np

from . import __version__, extract, film, media, render, screen, signal

HOME = Path(os.environ.get("NINEXVISION_HOME", Path.home() / ".9xvision"))


def watches_root() -> Path:
    return HOME / "watches"


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return (s[:40] or "video").strip("-")


def resolve(watch: str) -> Path:
    """Accept a watch dir path or a watch id."""
    p = Path(watch).expanduser()
    if (p / "watch.json").exists():
        return p
    q = watches_root() / watch
    if (q / "watch.json").exists():
        return q
    hits = sorted(watches_root().glob(f"*{watch}*/watch.json"))
    if len(hits) == 1:
        return hits[0].parent
    raise FileNotFoundError(f"No watch named {watch!r} (looked in {watches_root()})")


def load(watch: str) -> tuple[Path, dict]:
    d = resolve(watch)
    return d, json.loads((d / "watch.json").read_text())


def watch(source: str, mode: str = "auto", out: str | None = None, force: bool = False,
          transcribe: str = "auto", ocr: bool = True, sheet_cells: int | None = None,
          log=lambda m: None) -> Path:
    t0 = time.time()
    info: dict = {}
    if media.is_url(source):
        dl = HOME / "downloads" / _slug(re.sub(r"^https?://(www\.)?", "", source))
        log(f"fetching {source}")
        path, info = media.fetch(source, dl)
    else:
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
    pr = media.probe(path)
    title = info.get("title") or pr.title
    wid = f"{_slug(title)}-{media.file_fingerprint(path)}"
    wdir = Path(out).expanduser() if out else watches_root() / wid
    meta_path = wdir / "watch.json"
    if meta_path.exists() and not force:
        prev = json.loads(meta_path.read_text())
        if prev.get("version") == __version__ and (mode == "auto" or prev.get("mode") == mode):
            log(f"cached: {wdir}")
            return wdir
    if wdir.exists() and force:
        for sub in ("sheets", "keyframes", "views", "pages"):
            shutil.rmtree(wdir / sub, ignore_errors=True)
    wdir.mkdir(parents=True, exist_ok=True)

    log(f"analysing {pr.width}x{pr.height} {pr.duration:.1f}s")
    sig = signal.compute(str(path), pr, "auto" if mode == "auto" else mode)
    chosen = signal.guess_mode(pr, sig) if mode == "auto" else mode
    log(f"mode: {chosen}")

    meta: dict = {
        "version": __version__, "id": wdir.name, "dir": str(wdir), "source": source, "file": str(path),
        "title": title, "mode": chosen, "probe": pr.to_dict(), "info": info,
        "analysis": {"fps": sig.fps, "width": sig.width, "mask_top": sig.mask_top, "mask_bottom": sig.mask_bottom},
    }

    # ---- audio
    segs: list[dict] = []
    meta["audio"] = {"present": pr.has_audio}
    if pr.has_audio and transcribe != "off":
        # captions: yt-dlp's for URLs, or a sidecar <name>*.vtt next to a local file
        vtts = sorted(path.parent.glob("source*.vtt")) if info else sorted(path.parent.glob(f"{glob_escape(path.stem)}*.vtt"))
        level = media.audio_level(path)
        meta["audio"]["mean_db"] = level
        if vtts:
            segs = extract.parse_vtt(vtts[0])
            meta["audio"]["transcript"] = f"captions:{vtts[0].name}"
        elif level is not None and level < -55:
            meta["audio"]["transcript"] = "skipped: near-silent audio"
        elif transcribe == "on" or (transcribe == "auto" and pr.duration <= 20 * 60):
            log("transcribing locally")
            res = extract.whisper_transcribe(path, wdir, pr.duration)
            if res:
                segs, backend = res
                meta["audio"]["transcript"] = backend
            else:
                meta["audio"]["transcript"] = "unavailable: install whisper.cpp (brew install whisper-cpp) + a ggml model"
        else:
            meta["audio"]["transcript"] = "skipped: long video (use --transcribe on)"
    if segs:
        (wdir / "transcript.json").write_text(json.dumps(segs, indent=1, ensure_ascii=False))
        (wdir / "transcript.txt").write_text(
            "\n".join(f"[{media.fmt_t(s['start'])}] {s['text']}" for s in segs), encoding="utf-8")

    if chosen == "screen":
        _screen(meta, sig, path, pr, wdir, segs, ocr, sheet_cells, log)
    else:
        _film(meta, sig, path, pr, wdir, segs, ocr, sheet_cells, log)

    meta["elapsed_s"] = round(time.time() - t0, 1)
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    (wdir / "INDEX.md").write_text(index_md(meta), encoding="utf-8")
    log(f"done in {meta['elapsed_s']}s -> {wdir}")
    return wdir


# ---------------------------------------------------------------- film

def _film(meta, sig, path, pr, wdir, segs, do_ocr, sheet_cells, log):
    shots = film.shots(sig, pr.duration)
    log(f"{len(shots)} shots")
    key_edge = 1280
    for s in shots:
        fr = media.grab_frame(path, s["key_t"], max_edge=key_edge)
        kp = wdir / "keyframes" / f"shot_{s['id']:03d}_{s['key_t']:.2f}.jpg"
        render.save_jpg(fr, kp, max_edge=key_edge)
        s["keyframe"] = str(kp.relative_to(wdir))
        s["palette"] = extract.palette(fr)
        if do_ocr:
            s["text"] = [ln["text"] for ln in extract.ocr(fr) if ln.get("conf", 1) >= 0.4]
        if segs:
            s["speech"] = extract.segments_between(segs, s["start"], s["end"])
    meta["shots"] = shots
    meta["rhythm"] = film.rhythm(shots, pr.duration)
    cs = wdir / "sheets" / "colour_script.jpg"
    render.colour_script(shots, pr.duration, cs)
    meta["colour_script"] = str(cs.relative_to(wdir))
    meta["sheets"] = _sheets(wdir, path, [(s["key_t"], f"#{s['id']} {media.fmt_t(s['start'])}", s["id"]) for s in shots],
                             pr, sheet_cells, "shot")


# ---------------------------------------------------------------- screen

def _screen(meta, sig, path, pr, wdir, segs, do_ocr, sheet_cells, log):
    states = screen.find_states(sig)
    screen.cluster_views(sig, states)
    transitions = [screen.classify_transition(sig, a, b) for a, b in zip(states, states[1:])]
    page_of = screen.group_pages(states, transitions)
    log(f"{len(states)} states, {len({s['view'] for s in states})} views")
    sx = pr.width
    sy = pr.height
    views: dict[int, dict] = {}
    out_states = []
    for k, s in enumerate(states):
        t_a, t_b = float(sig.t[s["a"]]), float(sig.t[s["b"]])
        rep_t = float(sig.t[s["rep"]])
        v = s["view"]
        st = {"id": k + 1, "start": round(t_a, 2), "end": round(t_b, 2), "dur": round(t_b - t_a, 2),
              "rep_t": round(rep_t, 2), "view": v, "page": page_of.get(v), "moving": bool(s.get("moving"))}
        if v not in views:
            fr = media.grab_frame(path, rep_t)
            vp = wdir / "views" / f"view_{v:02d}_{rep_t:.2f}.png"
            vp.parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            Image.fromarray(fr).save(vp, optimize=True)
            view = {"id": v, "page": page_of.get(v), "moving": bool(s.get("moving")), "first_t": round(rep_t, 2), "image": str(vp.relative_to(wdir)),
                    "background": extract.background_colour(fr), "palette": extract.palette(fr, k=6), "visits": []}
            if do_ocr:
                y0 = int(sy * sig.mask_top)
                lines = extract.ocr(fr[y0:])
                for ln in lines:  # back to full-frame coordinates
                    ln["box"][1] = round((ln["box"][1] * (sy - y0) + y0) / sy, 4)
                    ln["box"][3] = round(ln["box"][3] * (sy - y0) / sy, 4)
                view["text"] = lines
            views[v] = view
        views[v]["visits"].append([st["start"], st["end"]])
        if segs:
            st["speech"] = extract.segments_between(segs, t_a, t_b)
        out_states.append(st)
    for k, tr in enumerate(transitions):
        tr["from_state"], tr["to_state"] = k + 1, k + 2
        if tr.get("region"):
            x, y, w, h = tr["region"]
            # thumbs cover the masked band only -> map back to full-frame pixels
            band = 1 - sig.mask_top - sig.mask_bottom
            tr["region_px"] = [int(x * sx), int((sig.mask_top + y * band) * sy), int(w * sx), int(h * band * sy)]
    meta["states"] = out_states
    meta["transitions"] = transitions
    meta["views"] = [views[v] for v in sorted(views)]
    meta["flow"] = _flow(out_states, transitions)

    # long pages: stitch the first downward scroll chain per page
    pages = []
    for pid in sorted({s["page"] for s in out_states}):
        chain_states = []
        for k, st in enumerate(out_states):
            if st["page"] != pid:
                continue
            if chain_states and k > 0 and out_states[k - 1]["page"] == pid and transitions[k - 1]["type"] == "scroll" \
                    and transitions[k - 1]["direction"] == "down" and chain_states[-1]["id"] == st["id"] - 1:
                chain_states.append(st)
            elif not chain_states:
                chain_states = [st]
            elif len(chain_states) < 2:
                chain_states = [st]
        page = {"id": pid, "views": sorted({s["view"] for s in out_states if s["page"] == pid})}
        if len(chain_states) >= 2:
            frames = [media.grab_frame(path, s["rep_t"]) for s in chain_states]
            img, offs = screen.stitch(frames, top_skip_frac=sig.mask_top)
            if img is not None:
                pp = wdir / "pages" / f"page_{pid:02d}_stitched.jpg"
                size = render.save_jpg(img, pp, max_edge=None, quality=85)
                page["stitched"] = str(pp.relative_to(wdir))
                page["stitched_size"] = list(size)
                page["from_states"] = [s["id"] for s in chain_states]
        pages.append(page)
    meta["pages"] = pages
    items = [(views[v]["first_t"], f"V{v}{'~' if views[v]['moving'] else ''} {media.fmt_t(views[v]['first_t'])}", v) for v in sorted(views)]
    meta["sheets"] = _sheets(wdir, path, items, pr, sheet_cells, "view")
    st_items = [(s["rep_t"], f"S{s['id']} V{s['view']}{'~' if s['moving'] else ''} {media.fmt_t(s['start'])}", s["id"]) for s in out_states]
    meta["timeline_sheets"] = _sheets(wdir, path, st_items, pr, sheet_cells, "state", prefix="timeline")


def _flow(states: list[dict], transitions: list[dict]) -> list[str]:
    lines = []
    for k, tr in enumerate(transitions):
        a, b = states[k], states[k + 1]
        extra = ""
        if tr["type"] in ("scroll", "swipe", "slide"):
            extra = f" {tr['direction']} {tr['distance_screens']}x"
        elif tr.get("region_px"):
            extra = f" region={tr['region_px']}"
        lines.append(f"{media.fmt_t(tr['from_t'])}  V{a['view']} -> V{b['view']}  [{tr['type']}{extra}]")
    return lines


# ---------------------------------------------------------------- sheets

def _sheets(wdir, path, items, pr, sheet_cells, kind, prefix="overview"):
    if not items:
        return []
    portrait = pr.height > pr.width
    per = sheet_cells or (12 if portrait else 20)
    out = []
    for n, k in enumerate(range(0, len(items), per)):
        chunk = items[k:k + per]
        frames = [media.grab_frame(path, t, max_edge=720) for t, _, _ in chunk]
        sp = wdir / "sheets" / f"{prefix}_{n + 1:02d}.jpg"
        size = render.contact_sheet(frames, [lab for _, lab, _ in chunk], sp)
        out.append({"path": str(sp.relative_to(wdir)), kind + "s": [i for _, _, i in chunk],
                    "size": list(size), "tokens": render.token_estimate(*size)})
    return out


# ---------------------------------------------------------------- INDEX.md

def index_md(m: dict) -> str:
    pr = m["probe"]
    L = []
    L.append(f"# 9xvision watch: {m['title']}")
    L.append("")
    L.append(f"- source: {m['source']}")
    L.append(f"- {pr['width']}x{pr['height']}, {media.fmt_t(pr['duration'])} ({pr['duration']:.1f}s), {pr['fps']} fps, mode **{m['mode']}**")
    au = m.get("audio", {})
    L.append(f"- audio: {'yes' if au.get('present') else 'no'}" + (f", transcript: {au.get('transcript')}" if au.get("transcript") else ""))
    if m.get("info", {}).get("uploader"):
        L.append(f"- uploader: {m['info']['uploader']}, uploaded {m['info'].get('upload_date')}")
    L.append(f"- watch dir: {m['dir']}  (all paths below are relative to it)")
    L.append("")
    L.append("## How to look (coarse to fine)")
    sheets = m.get("sheets", [])
    if m["mode"] == "screen":
        L.append(f"1. Distinct screens: " + ", ".join(f"`{s['path']}` (~{s['tokens']} tok)" for s in sheets) + ". Each cell = one distinct view (V#), labelled with first time seen.")
        L.append("2. Full-resolution screens: `views/view_NN_*.png` (read these for small text). OCR text for each is below.")
        L.append("3. The flow below gives every transition in order. Timeline sheets (every state in order): " + ", ".join(f"`{s['path']}`" for s in m.get("timeline_sheets", [])))
    else:
        L.append(f"1. Shot overview: " + ", ".join(f"`{s['path']}` (~{s['tokens']} tok)" for s in sheets) + ". Each cell = one shot, labelled #shot start-time.")
        L.append(f"2. Colour script: `{m.get('colour_script')}` (one band per shot, width = duration).")
        L.append("3. Keyframes per shot: `keyframes/shot_NNN_<t>.jpg`.")
    L.append("4. Need more detail? `9xvision zoom <dir> START END` (dense timestamped grid), `9xvision frame <dir> T [--crop x,y,w,h]`, `9xvision diff <dir> T1 T2`.")
    L.append("Times are mm:ss.s from the start of the video. Cite them.")
    L.append("")
    if m["mode"] == "screen":
        views = m.get("views", [])
        states = m.get("states", [])
        groups = [p for p in m.get("pages", []) if len(p["views"]) > 1]
        moving = sum(1 for v in views if v.get("moving"))
        L.append(f"## Summary: {len(views)} distinct views ({moving} captured while moving, marked ~), {len(states)} states")
        L.append("")
        L.append("## Flow")
        L.append("```")
        if states:
            L.append(f"{media.fmt_t(states[0]['start'])}  start on V{states[0]['view']}")
        L.extend(m.get("flow", []))
        L.append("```")
        if groups:
            L.append("")
            L.append("## Scroll groups (views that are one page, scrolled)")
            for p in groups:
                line = f"- group {p['id']}: views {', '.join('V' + str(v) for v in p['views'])}"
                if p.get("stitched"):
                    line += f"; stitched full page `{p['stitched']}` ({p['stitched_size'][0]}x{p['stitched_size'][1]})"
                L.append(line)
        L.append("")
        L.append("## Views")
        seen_text: list[tuple[int, list[str]]] = []
        for v in views:
            visits = ", ".join(f"{media.fmt_t(a)}-{media.fmt_t(b)}" for a, b in v["visits"])
            pal = " ".join(p["hex"] for p in v["palette"][:5])
            grp = next((g["id"] for g in groups if v["id"] in g["views"]), None)
            head = f"### V{v['id']}{' (moving)' if v.get('moving') else ''}{f' [scroll group {grp}]' if grp else ''} `{v['image']}`"
            L.append(head)
            L.append(f"- seen: {visits}")
            L.append(f"- background {v['background']}; palette {pal}")
            txt = [ln["text"] for ln in v.get("text", []) if ln.get("conf", 1) >= 0.3]
            if txt:
                best = None
                for vid, prev in seen_text:
                    common = len(set(txt) & set(prev)) / max(len(set(txt) | set(prev)), 1)
                    if common >= 0.6 and (best is None or common > best[1]):
                        best = (vid, common, prev)
                if best:
                    plus = [t for t in txt if t not in best[2]]
                    minus = [t for t in best[2] if t not in txt]
                    if not plus and not minus:
                        L.append(f"- text: same as V{best[0]} (difference is visual: use diff)")
                    else:
                        L.append(f"- text: as V{best[0]}" + (f"; added: {' | '.join(plus)[:400]}" if plus else "")
                                 + (f"; gone: {' | '.join(minus)[:300]}" if minus else ""))
                else:
                    joined = " | ".join(txt)
                    L.append(f"- text: {joined[:900]}{' ...' if len(joined) > 900 else ''}")
                seen_text.append((v["id"], txt))
        if any(s.get("speech") for s in states):
            L.append("")
            L.append("## Speech by state")
            for s in states:
                if s.get("speech"):
                    L.append(f"- S{s['id']} {media.fmt_t(s['start'])}: {s['speech'][:300]}")
    else:
        r = m.get("rhythm", {})
        L.append(f"## Summary: {r.get('shots')} camera shots, average shot {r.get('asl')}s (median {r.get('median_shot')}s, longest {r.get('longest')}s)")
        if r.get("content_beats"):
            L.append(f"plus {r['content_beats']} content beats (the picture changed and held inside a shot: slides, pages, scene changes in one take). Segments below = shots + beats.")
        L.append(f"cuts per minute: {r.get('cuts_per_minute')}")
        L.append("")
        L.append("## Shots")
        has_text = any(s.get("text") for s in m.get("shots", []))
        L.append("| # | start | dur | via | motion | palette |" + (" on-screen text |" if has_text else "") + " speech |")
        L.append("|---|---|---|---|---|---|" + ("---|" if has_text else "") + "---|")
        for s in m.get("shots", []):
            pal = " ".join(p["hex"] for p in s.get("palette", [])[:3])
            sp = (s.get("speech") or "").replace("|", "/")[:140]
            via = {"cut": "cut", "content": "beat", "start": "-"}.get(s.get("boundary"), "cut")
            tx = (" / ".join(s.get("text") or []).replace("|", "/")[:160] + " |") if has_text else ""
            L.append(f"| {s['id']} | {media.fmt_t(s['start'])} | {s['dur']} | {via} | {s['motion']['class']} | {pal} | {tx} {sp} |")
    if (Path(m["dir"]) / "transcript.txt").exists():
        L.append("")
        L.append("Full transcript with timestamps: `transcript.txt`")
    L.append("")
    L.append(f"_9xvision {m['version']}, built in {m.get('elapsed_s', '?')}s_")
    return "\n".join(L) + "\n"
