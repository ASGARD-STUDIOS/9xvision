"""The `watch` pipeline: source -> watch folder (INDEX.md, watch.json, sheets, keyframes)."""
from __future__ import annotations

import hashlib
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


def download_dir(url: str, section: tuple[float, float] | None = None) -> Path:
    """One folder per exact URL (+ section): a readable slug plus a hash of the full, case-sensitive URL."""
    key = url.strip() + (f"#{section[0]:.3f}-{section[1]:.3f}" if section else "")
    slug = _slug(re.sub(r"^https?://(www\.)?", "", url) + (f"-{section[0]:.0f}-{min(section[1], 10**6):.0f}" if section else ""))
    return HOME / "downloads" / f"{slug}-{hashlib.sha1(key.encode()).hexdigest()[:8]}"


def resolve(watch: str) -> Path:
    """Accept a watch dir path or a watch id."""
    p = Path(watch).expanduser()
    if (p / "watch.json").exists():
        return p
    q = watches_root() / watch
    if (q / "watch.json").exists():
        return q
    hits = sorted(watches_root().glob(f"*{glob_escape(watch)}*/watch.json"))
    if len(hits) == 1:
        return hits[0].parent
    base = {re.sub(r"-\d+-\d+s$", "", h.parent.name) for h in hits}
    if len(base) == 1:   # one video, several windows: the full watch if there is one, else the most recent
        full = [h for h in hits if not re.search(r"-\d+-\d+s$", h.parent.name)]
        return max(full or hits, key=lambda h: h.stat().st_mtime).parent
    if hits:
        raise FileNotFoundError(f"{watch!r} matches {len(hits)} watches: " + ", ".join(h.parent.name for h in hits[:6]))
    raise FileNotFoundError(f"No watch named {watch!r} (looked in {watches_root()}; `9xvision list` shows them)")


def load(watch: str) -> tuple[Path, dict]:
    d = resolve(watch)
    return d, json.loads((d / "watch.json").read_text())


def watch(source: str, mode: str = "auto", out: str | None = None, force: bool = False,
          transcribe: str = "auto", ocr: bool = True, sheet_cells: int | None = None,
          start: str | float | None = None, end: str | float | None = None,
          cookies_from_browser: str | None = None, log=lambda m: None) -> Path:
    """start/end (mm:ss, h:mm:ss or seconds) watch only that window; for a URL only that part is downloaded.
    Every time reported stays in original-video time."""
    clock = time.time()
    info: dict = {}
    w_start = media.parse_t(start) if start not in (None, "") else None
    w_end = media.parse_t(end) if end not in (None, "") else None
    if w_start is not None and w_end is not None and w_end <= w_start:
        raise ValueError(f"end ({end}) must be after start ({start})")
    if media.is_url(source):
        section = None
        if w_start is not None or w_end is not None:
            section = (w_start or 0.0, w_end if w_end is not None else 10 ** 7)
        dl = download_dir(source, section)
        log(f"fetching {source}" + (f" ({media.fmt_t(section[0])}-{media.fmt_t(section[1]) if section[1] < 10**6 else 'end'})" if section else ""))
        path, info = media.fetch(source, dl, section=section, cookies_from_browser=cookies_from_browser)
    given = None
    if not media.is_url(source):
        given = Path(source).expanduser().absolute()
        path = given.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
    pr = media.probe(path)
    t0 = max(pr.offset, w_start) if w_start is not None else pr.offset
    t1 = min(pr.end, w_end) if w_end is not None else pr.end
    if t1 - t0 < 0.2:
        raise ValueError(f"the window {media.fmt_t(t0)}-{media.fmt_t(t1)} is empty; the video covers "
                         f"{media.fmt_t(pr.offset)}-{media.fmt_t(pr.end)}")
    windowed = t0 > pr.offset + 0.05 or t1 < pr.end - 0.05
    full = float(info.get("duration") or 0) or pr.end
    title = info.get("title") or (given.stem if given and pr.title == path.stem else pr.title)
    if ";" in title and len(set(title.split(";"))) == 1:   # some muxers repeat the title per stream
        title = title.split(";")[0]
    wid = f"{_slug(title)}-{media.file_fingerprint(path)}" + (f"-{t0:.0f}-{t1:.0f}s" if windowed else "")
    wdir = Path(out).expanduser() if out else watches_root() / wid
    meta_path = wdir / "watch.json"
    settings = {"ocr": bool(ocr), "transcribe": transcribe, "sheet_cells": sheet_cells}
    if meta_path.exists() and not force:
        prev = json.loads(meta_path.read_text())
        same = prev.get("settings", settings) == settings
        if prev.get("version") == __version__ and (mode == "auto" or prev.get("mode") == mode) and same:
            log(f"cached: {wdir}")
            return wdir
    if wdir.exists() and force:
        for sub in ("sheets", "keyframes", "views", "pages"):
            shutil.rmtree(wdir / sub, ignore_errors=True)
    wdir.mkdir(parents=True, exist_ok=True)

    log(f"analysing {pr.width}x{pr.height} {media.fmt_t(t0)}-{media.fmt_t(t1)}")
    sig = signal.compute(str(path), pr, "auto" if mode == "auto" else mode,
                         start=t0 if windowed else None, end=t1 if windowed else None)
    chosen = signal.guess_mode(pr, sig) if mode == "auto" else mode
    log(f"mode: {chosen}")

    meta: dict = {
        "version": __version__, "id": wdir.name, "dir": str(wdir), "source": source, "file": str(path),
        "title": title, "mode": chosen, "probe": pr.to_dict(), "info": info,
        "window": [round(t0, 2), round(t1, 2)], "full_duration": round(full, 2), "settings": settings,
        "analysis": {"fps": sig.fps, "width": sig.width, "mask_top": sig.mask_top, "mask_bottom": sig.mask_bottom},
    }

    # ---- audio
    segs: list[dict] = []
    meta["audio"] = {"present": pr.has_audio}
    if pr.has_audio and transcribe != "off":
        # captions: yt-dlp's for URLs, or a sidecar <name>*.vtt next to a local file
        if info:
            vtts = sorted(path.parent.glob("source*.vtt"))
        else:   # next to the path as given (it may be a symlink) or next to the real file
            vtts = sorted(given.parent.glob(f"{glob_escape(given.stem)}*.vtt")) or \
                   sorted(path.parent.glob(f"{glob_escape(path.stem)}*.vtt"))
        level = media.audio_level(path)
        meta["audio"]["mean_db"] = level
        if vtts:
            segs = [s for s in extract.parse_vtt(vtts[0]) if s["end"] > t0 and s["start"] < t1]
            meta["audio"]["transcript"] = f"captions:{vtts[0].name}"
        elif level is not None and level < -55:
            meta["audio"]["transcript"] = "skipped: near-silent audio"
        elif transcribe == "on" or (transcribe == "auto" and t1 - t0 <= 20 * 60):
            log("transcribing locally")
            res = extract.whisper_transcribe(path, wdir, t1 - t0,
                                             window=(t0 - pr.offset, t1 - pr.offset) if windowed else None)
            if res:
                segs, backend = res
                segs = [{**s, "start": round(s["start"] + pr.offset, 2), "end": round(s["end"] + pr.offset, 2)} for s in segs]
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
        _film(meta, sig, path, pr, wdir, segs, ocr, sheet_cells, log, span=(t0, t1))

    meta["elapsed_s"] = round(time.time() - clock, 1)
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    (wdir / "INDEX.md").write_text(index_md(meta), encoding="utf-8")
    log(f"done in {meta['elapsed_s']}s -> {wdir}")
    return wdir


# ---------------------------------------------------------------- film

def _film(meta, sig, path, pr, wdir, segs, do_ocr, sheet_cells, log, span=None):
    t0, t1 = span or (pr.offset, pr.end)
    shots = film.shots(sig, t1, start=t0)
    log(f"{len(shots)} shots")
    cut_idx = [k for k, s in enumerate(shots) if s["boundary"] == "cut"]
    if cut_idx:   # frame-accurate cut times (analysis runs at ~12 fps)
        for k, t in zip(cut_idx, film.refine_cuts(path, [shots[k]["start"] for k in cut_idx], sig.fps, pr.fps)):
            if shots[k - 1]["start"] < t < shots[k]["end"]:
                shots[k]["start"] = shots[k - 1]["end"] = round(t, 3)
        for s in shots:
            s["dur"] = round(s["end"] - s["start"], 3)
            s["key_t"] = round(min(max(s["key_t"], s["start"]), s["end"] - 0.02), 2)
    key_edge = 1280
    for s, fr in zip(shots, media.grab_frames(path, [s["key_t"] for s in shots], max_edge=key_edge)):
        kp = wdir / "keyframes" / f"shot_{s['id']:03d}_{s['key_t']:.2f}.jpg"
        render.save_jpg(fr, kp, max_edge=key_edge)
        s["keyframe"] = str(kp.relative_to(wdir))
        s["palette"] = extract.palette(fr)
        if do_ocr:
            s["text"] = [ln["text"] for ln in extract.ocr(fr) if ln.get("conf", 1) >= 0.4]
        if segs:
            s["speech"] = extract.segments_between(segs, s["start"], s["end"])
    meta["shots"] = shots
    meta["rhythm"] = film.rhythm(shots, t1, start=t0)
    if pr.has_audio and (meta["audio"].get("mean_db") is None or meta["audio"]["mean_db"] > -55):
        from . import audio
        log("listening (tempo, beats, loudness)")
        snd = audio.analyse(path, t0, t1, [s["start"] for s in shots if s["boundary"] == "cut"], pr.fps)
        if snd.get("present"):
            (wdir / "sound.json").write_text(json.dumps(snd))
            beats = np.array(snd.get("beats") or [])
            tol = (snd.get("cuts_vs_beats") or {}).get("tolerance_s")
            if len(beats) and tol:
                for s in shots:
                    if s["boundary"] == "cut":
                        s["on_beat"] = bool(np.min(np.abs(beats - s["start"])) <= tol)
            meta["sound"] = {k: v for k, v in snd.items() if k not in ("beats", "loudness_curve")}
            meta["sound"]["file"] = "sound.json"
    cs = wdir / "sheets" / "colour_script.jpg"
    render.colour_script(shots, t1, cs, start=t0)
    meta["colour_script"] = str(cs.relative_to(wdir))
    meta["sheets"] = _sheets(wdir, path, [(s["key_t"], f"#{s['id']} {media.fmt_t(s['start'])}", s["id"]) for s in shots],
                             pr, sheet_cells, "shot", images={s["id"]: wdir / s["keyframe"] for s in shots})


# ---------------------------------------------------------------- screen

def _screen(meta, sig, path, pr, wdir, segs, do_ocr, sheet_cells, log):
    states = screen.find_states(sig)
    import cv2
    y0, y1 = int(pr.height * sig.mask_top), pr.height - int(pr.height * sig.mask_bottom)
    sharp, sharp_rgb = [], []
    for fr in media.grab_frames(path, [float(sig.t[s["rep"]]) for s in states], max_edge=480):
        h = fr.shape[0]
        band_rgb = fr[int(h * y0 / pr.height):int(h * y1 / pr.height)]
        sharp_rgb.append(band_rgb)
        sharp.append(cv2.cvtColor(band_rgb, cv2.COLOR_RGB2GRAY))
    screen.cluster_views(sig, states, sharp)
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
            im = Image.fromarray(fr)
            if max(im.size) > render.MAX_EDGE:   # what a model would see anyway; frame --crop gives full resolution
                s_ = render.MAX_EDGE / max(im.size)
                im = im.resize((round(im.width * s_), round(im.height * s_)), Image.LANCZOS)
            im.save(vp, optimize=True)
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
        if do_ocr and (tr["type"] in ("local change", "content change") or tr["type"].startswith("partial")):
            # name what changed: the text in the same row as each changed blob (a toggled option, a picker value)
            band = 1 - sig.mask_top - sig.mask_bottom
            lines = views[states[k]["view"]].get("text", []) + views[states[k + 1]["view"]].get("text", [])
            at: list[str] = []
            for blob in screen.change_blobs(sharp[k], sharp[k + 1]):
                bx, by, bw, bh = blob
                sign = {1: "+", -1: "−"}.get(screen.colour_shift(sharp_rgb[k], sharp_rgb[k + 1], blob), "")
                def changed(lx, ly, lw, lh, k=k):   # the label itself changed (recoloured, greyed, swapped)
                    return screen.box_changed(sharp_rgb[k], sharp_rgb[k + 1], lx, (ly - sig.mask_top) / band, lw, lh / band)
                for lab in screen.labels_at([bx, sig.mask_top + by * band, bw, bh * band], lines, changed=changed):
                    if not any(a.lstrip("+−") == lab for a in at):
                        at.append(sign + lab)
            if at:
                tr["at"] = at[:6]
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
            extra = (" at " + " · ".join(f"{t[0]}'{t[1:]}'" if t[:1] in "+−" else f"'{t}'" for t in tr["at"])
                     if tr.get("at") else "") + f" region={tr['region_px']}"
        lines.append(f"{media.fmt_t(tr['from_t'])}  V{a['view']} -> V{b['view']}  [{tr['type']}{extra}]")
    return lines


# ---------------------------------------------------------------- sheets

def _sheets(wdir, path, items, pr, sheet_cells, kind, prefix="overview", images=None):
    if not items:
        return []
    portrait = pr.height > pr.width
    per = sheet_cells or (12 if portrait else 20)
    out = []
    for n, k in enumerate(range(0, len(items), per)):
        chunk = items[k:k + per]
        if images:   # reuse frames already saved (keyframes) instead of decoding again
            from PIL import Image
            frames = []
            for _, _, i in chunk:
                im = Image.open(images[i]).convert("RGB")
                im.thumbnail((720, 720))
                frames.append(np.asarray(im))
        else:
            frames = list(media.grab_frames(path, [t for t, _, _ in chunk], max_edge=720))
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
    w0, w1 = m.get("window") or (0.0, pr["duration"])
    full = m.get("full_duration") or pr["duration"]
    L.append(f"- {pr['width']}x{pr['height']}, {media.fmt_t(full)} ({full:.1f}s), {pr['fps']} fps, mode **{m['mode']}**")
    if w0 > 0.05 or w1 < full - 0.05:
        L.append(f"- **window: {media.fmt_t(w0)}-{media.fmt_t(w1)} only** (all times are from the start of the full video)")
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
        L.append("`at` names the text nearest each change: `+` = that area gained colour (often: selected / turned on), "
                 "`−` = lost colour (often: deselected). It is a pointer to where to look, not a verdict: "
                 "confirm states with `diff` before you report them.")
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
        snd = m.get("sound")
        if snd:
            L.append("## Sound")
            if snd.get("bpm"):
                L.append(f"- music: ~{snd['bpm']:g} BPM (pulse clarity {snd['pulse_clarity']}); beat times in `sound.json`")
                cb = snd.get("cuts_vs_beats") or {}
                if cb:
                    off = f"; median offset {cb['median_offset_frames']:+g} frames" if "median_offset_frames" in cb else ""
                    L.append(f"- edit vs music: **{cb['verdict']}**: {cb['on_beat']} of {cb['cuts']} cuts within "
                             f"±{cb['tolerance_s'] * 1000:.0f} ms of a beat ({cb['share']:.0%}; chance would be {cb['chance']:.0%}){off}. "
                             "♪ in the table marks cuts on the beat.")
            else:
                L.append(f"- no steady beat (pulse clarity {snd['pulse_clarity']}): speech, ambience or free-time music")
            if snd.get("level_dbfs"):
                L.append(f"- level: mean {snd['level_dbfs']['mean']} dBFS, peak {snd['level_dbfs']['max']} dBFS")
            if snd.get("jumps"):
                L.append("- sudden rises (drops, hits, entries): " + ", ".join(f"{media.fmt_t(j['t'])} (+{j['rise_db']:g} dB)" for j in snd["jumps"]))
            if snd.get("silences"):
                L.append("- silences: " + ", ".join(f"{media.fmt_t(a)}-{media.fmt_t(b)}" for a, b in snd["silences"][:8]))
            L.append("")
        L.append("## Shots")
        shots_ = m.get("shots", [])
        counts: dict[str, int] = {}
        for s in shots_:
            for line in set(s.get("text") or []):
                counts[line] = counts.get(line, 0) + 1
        boiler = {t for t, c in counts.items() if len(shots_) >= 5 and c >= 0.6 * len(shots_)}
        if boiler:   # logos, sidebars, sponsor bars: say them once, keep the table for what changes
            L.append("On screen in most segments (left out of the table): " + " | ".join(sorted(boiler))[:400])
            L.append("")
        has_text = any(set(s.get("text") or []) - boiler for s in shots_)
        L.append("| # | start | dur | via | motion | palette |" + (" on-screen text |" if has_text else "") + " speech |")
        L.append("|---|---|---|---|---|---|" + ("---|" if has_text else "") + "---|")
        for s in m.get("shots", []):
            pal = " ".join(p["hex"] for p in s.get("palette", [])[:3])
            sp = (s.get("speech") or "").replace("|", "/")[:140]
            via = {"cut": "cut", "content": "beat", "start": "-"}.get(s.get("boundary"), "cut") + (" ♪" if s.get("on_beat") else "")
            tx = (" / ".join(x for x in (s.get("text") or []) if x not in boiler).replace("|", "/")[:200] + " |") if has_text else ""
            L.append(f"| {s['id']} | {media.fmt_t(s['start'])} | {s['dur']} | {via} | {s['motion']['class']} | {pal} | {tx} {sp} |")
    if (Path(m["dir"]) / "transcript.txt").exists():
        L.append("")
        L.append("Full transcript with timestamps: `transcript.txt`")
    L.append("")
    L.append(f"_9xvision {m['version']}, built in {m.get('elapsed_s', '?')}s_")
    return "\n".join(L) + "\n"
