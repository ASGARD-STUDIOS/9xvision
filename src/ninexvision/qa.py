"""Capture QA for games, XR and apps: what a playtest recording says about the build.

- frame pacing: repeated frames while the picture is moving (the game missed a refresh), hitches with times and
  lengths, effective frame rate per second, and timestamp gaps in variable-frame-rate captures;
- stereo: side-by-side or top-bottom stereo captures are detected, and analysis runs on the left eye;
- comfort: fast camera turns (screen widths per second, or degrees per second when the horizontal FOV is known);
- flashes: general flashes per second against the WCAG 2.3.1 / Harding-style limit of 3 (an approximation for
  triage, not a certified photosensitivity test).

Every frame is decoded at its native rate (small and grey, so it stays fast)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import media, render

W = 160                 # analysis width (px)
REPEAT = 0.25           # mean abs grey diff below this = the same frame shown again
MOVING = 1.5            # mean abs diff across a held run above this = the picture kept moving (so the hold is a hitch)
FLASH = 0.1             # relative luminance swing that counts as a flash transition (WCAG: 10%)


def _stereo(path, pr) -> str | None:
    """'sbs' / 'tb' if the two halves are the same scene (small disparity allowed), else None."""
    votes = {"sbs": 0, "tb": 0}
    ts = np.linspace(pr.offset + pr.duration * 0.15, pr.offset + pr.duration * 0.85, 5)
    for fr in media.grab_frames(path, list(ts), max_edge=640):
        g = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = g.shape
        for kind, (a, b) in {"sbs": (g[:, :w // 2], g[:, w // 2:w // 2 * 2]), "tb": (g[:h // 2], g[h // 2:h // 2 * 2])}.items():
            if a.std() < 8:
                continue
            best = max(_ncc(a, b, dx) for dx in range(-int(a.shape[1] * 0.06), int(a.shape[1] * 0.06) + 1, 2))
            if best > 0.85:
                votes[kind] += 1
    kind = max(votes, key=votes.get)
    return kind if votes[kind] >= 3 else None


def _ncc(a, b, dx):
    if dx > 0:
        a, b = a[:, dx:], b[:, :-dx]
    elif dx < 0:
        a, b = a[:, :dx], b[:, -dx:]
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def _pts(path, pr, start, end) -> np.ndarray:
    """Presentation times of every video packet in [start, end] (original-video time)."""
    fs, fe = start - pr.offset, end - pr.offset
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", f"{max(0, fs - 1):.3f}%{fe + 1:.3f}",
                          "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    t = np.array(sorted(float(x) for x in out.stdout.split() if x.replace(".", "", 1).replace("-", "", 1).isdigit()))
    return t[(t >= fs - 1e-3) & (t <= fe + 1e-3)] + pr.offset


def _frames(path, pr, start, end, crop: str | None):
    """Every frame (no resampling), small grey, plus linear luminance means."""
    fs = max(0.0, start - pr.offset)
    vf = (f"{crop}," if crop else "") + f"scale={W}:-2:flags=area,format=gray"
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "auto", "-ss", f"{fs:.3f}", "-t", f"{end - start:.3f}", "-i", str(path),
           "-an", "-fps_mode", "passthrough", "-vf", vf, "-f", "rawvideo", "-"]
    w = W
    sw, sh = (pr.width // 2, pr.height) if crop and "iw/2" in crop else (pr.width, pr.height // 2) if crop else (pr.width, pr.height)
    h = int(round(sh * w / sw / 2) * 2)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frames = []
    while True:
        buf = proc.stdout.read(w * h)
        if len(buf) < w * h:
            break
        frames.append(np.frombuffer(buf, np.uint8).reshape(h, w))
    proc.wait()
    return frames


def analyse(path, start: float | None = None, end: float | None = None, fov_deg: float | None = None,
            out_dir: Path | None = None) -> dict:
    pr = media.probe(path)
    t0 = pr.offset if start is None else max(pr.offset, start)
    t1 = pr.end if end is None else min(pr.end, end)
    stereo = _stereo(path, pr)
    crop = {"sbs": "crop=iw/2:ih:0:0", "tb": "crop=iw:ih/2:0:0"}.get(stereo)
    frames = _frames(path, pr, t0, t1, crop)
    if len(frames) < 3:
        raise ValueError("too few frames decoded for capture QA")
    pts = _pts(path, pr, t0, t1)
    if abs(len(pts) - len(frames)) <= max(2, len(frames) // 100):
        n = min(len(pts), len(frames))
        frames, t = frames[:n], pts[:n]
    else:   # timestamps unusable: assume constant rate
        t = t0 + np.arange(len(frames)) / (pr.fps or 30)
    n = len(frames)
    dt = np.diff(t)
    nominal = float(np.median(dt)) if len(dt) else 1 / (pr.fps or 30)
    fps = 1 / nominal

    f32 = [f.astype(np.int16) for f in frames]
    diff = np.array([0.0] + [float(np.abs(f32[i] - f32[i - 1]).mean()) for i in range(1, n)])
    repeat = diff < REPEAT
    repeat[0] = False

    win = cv2.createHanningWindow((frames[0].shape[1], frames[0].shape[0]), cv2.CV_32F)

    def shift(a: int, b: int) -> tuple[float, float]:
        (dx, dy), r = cv2.phaseCorrelate(frames[a].astype(np.float32), frames[b].astype(np.float32), win)
        return float(np.hypot(dx, dy)), float(r)

    # hitches: runs of repeated frames after which the motion catches up (the game missed refreshes).
    # A pause (menu, cutscene hold) does not catch up, so it is not a hitch.
    hitches = []
    i = 1
    while i < n:
        if repeat[i]:
            j = i
            while j + 1 < n and repeat[j + 1]:
                j += 1
            before, after = i - 1, j + 1
            if after < n and before >= 1 and diff[before] > REPEAT:
                held_n = after - before
                d_across, r_across = shift(before, after)
                d_step, r_step = shift(before - 1, before)
                if r_across > 0.05 and r_step > 0.05 and d_step > 0.3:
                    caught_up = d_across >= 1.6 * d_step and held_n >= 2
                else:   # not a translation (animation, particles): fall back to how much the picture changed
                    across = float(np.abs(f32[after] - f32[before]).mean())
                    caught_up = diff[before] > MOVING and across >= 1.6 * diff[before]
                if caught_up:
                    held = float(t[after] - t[before])
                    hitches.append({"t": round(float(t[i]), 3), "held_ms": round(held * 1000, 1),
                                    "missed_frames": int(round(held / nominal)) - 1})
            i = j + 1
        else:
            i += 1
    gaps = [{"t": round(float(t[k]), 3), "gap_ms": round(float(dt[k]) * 1000, 1)}
            for k in np.where(dt > 1.5 * nominal)[0]]

    # effective fps per second, over moving seconds only (a static menu repeats frames legitimately)
    sec = np.floor(t - t0).astype(int)
    eff = []
    for s in range(int(sec.max()) + 1 if n else 0):
        idx = np.where(sec == s)[0]
        if len(idx) < 2:
            continue
        moving = (diff[idx] > REPEAT).mean() > 0.3
        if moving:
            span = (t[idx[-1]] - t[idx[0]]) + nominal
            missed = sum(h["missed_frames"] for h in hitches if t[idx[0]] <= h["t"] <= t[idx[-1]])
            eff.append([round(t0 + s, 1), round(max(0.0, (len(idx) - missed) / span), 1)])

    # comfort: horizontal turn speed from phase correlation between consecutive distinct frames
    speed = np.zeros(n)
    last = 0
    for k in range(1, n):
        if not repeat[k]:
            (dx, dy), r = cv2.phaseCorrelate(frames[last].astype(np.float32), frames[k].astype(np.float32), win)
            if r > 0.05:   # widths per second, over the time since the last distinct frame
                speed[k] = abs(dx) / W / max(float(t[k] - t[last]), 1e-3)
            last = k
    sm = np.convolve(speed, np.ones(max(1, int(fps * 0.2))) / max(1, int(fps * 0.2)), mode="same")
    unit = "deg/s" if fov_deg else "screen widths/s"
    scale = fov_deg or 1.0
    limit = (90.0 / scale) if fov_deg else 0.6     # ~90 deg/s sustained yaw is a common VR comfort ceiling
    turns = []
    k = 0
    while k < n:
        if sm[k] > limit:
            j = k
            while j + 1 < n and sm[j + 1] > limit:
                j += 1
            if t[j] - t[k] >= 0.15:
                turns.append({"start": round(float(t[k]), 2), "end": round(float(t[j]), 2),
                              "peak": round(float(sm[k:j + 1].max()) * scale, 2)})
            k = j + 1
        else:
            k += 1

    # flashes: opposing luminance swings of >= 10 % (relative), counted per sliding second
    lum = np.array([float(((f / 255.0) ** 2.2).mean()) for f in frames])
    ext = [0]
    for k in range(1, n - 1):
        if (lum[k] - lum[k - 1]) * (lum[k + 1] - lum[k]) < 0:
            ext.append(k)
    ext.append(n - 1)
    swings = [(t[b], abs(lum[b] - lum[a])) for a, b in zip(ext, ext[1:]) if abs(lum[b] - lum[a]) >= FLASH]
    flash_t = np.array([s for s, _ in swings])
    risky = []
    for s in flash_t:
        cnt = int(((flash_t >= s) & (flash_t < s + 1.0)).sum()) / 2   # a flash = a pair of opposing transitions
        if cnt > 3 and (not risky or s - risky[-1]["end"] > 0.5):
            risky.append({"start": round(float(s), 2), "end": round(min(float(s) + 1.0, t1), 2), "flashes_per_s": cnt})
        elif cnt > 3:
            risky[-1]["end"] = round(min(float(s) + 1.0, t1), 2)
            risky[-1]["flashes_per_s"] = max(risky[-1]["flashes_per_s"], cnt)

    effs = np.array([e for _, e in eff]) if eff else np.array([])
    res = {
        "file": str(path), "window": [round(t0, 2), round(t1, 2)], "frames": n,
        "capture_fps": round(fps, 2), "stereo": stereo or "mono",
        "pacing": {
            "repeated_frames_while_moving": int(sum(h["missed_frames"] for h in hitches)),
            "hitches": hitches[:200], "hitch_count": len(hitches),
            "worst_hitch_ms": max((h["held_ms"] for h in hitches), default=0.0),
            "timestamp_gaps": gaps[:100],
            "effective_fps": {"median": round(float(np.median(effs)), 1), "p5": round(float(np.percentile(effs, 5)), 1),
                              "min": round(float(effs.min()), 1)} if len(effs) else None,
            "effective_fps_per_second": eff,
        },
        "comfort": {"unit": unit, "limit": round(limit * scale, 2), "fast_turns": turns[:100],
                    "peak_turn": round(float(sm.max()) * scale, 2),
                    "note": "image rotation. In a headset mirror capture this includes the player's own head turns "
                            "(comfortable); it matters for artificial turning/locomotion, cutscenes and flat-screen viewers"},
        "flashes": {"risky_windows": risky[:50], "limit_per_s": 3,
                    "note": "approximation of WCAG 2.3.1 general flash; red flashes and area limits are not checked"},
    }
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{t0:.0f}-{t1:.0f}"
        (out_dir / f"qa_{tag}.json").write_text(json.dumps(res, indent=1))
        chart = out_dir / f"qa_{tag}.png"
        _chart(res, t, sm * scale, chart)
        res["chart"] = str(chart)
    return res


def _chart(res: dict, t: np.ndarray, turn: np.ndarray, path: Path) -> None:
    """Effective fps per second (top), turn speed (bottom), hitches and flash windows marked."""
    from PIL import Image, ImageDraw
    Wd, H, pad = 1400, 420, 50
    im = Image.new("RGB", (Wd, H), (18, 18, 18))
    d = ImageDraw.Draw(im)
    f = render.font(14)
    t0, t1 = res["window"]
    span = max(t1 - t0, 1e-6)
    x = lambda tt: pad + (tt - t0) / span * (Wd - 2 * pad)
    fps = res["capture_fps"]
    top = (20, 200)
    d.text((pad, 2), f"effective fps (capture {fps:g}); red ticks = hitches; yellow boxes = flashes > 3/s", fill=(220, 220, 220), font=f)
    pts = [(x(s + 0.5), top[1] - (e / fps) * (top[1] - top[0])) for s, e in res["pacing"]["effective_fps_per_second"]]
    if len(pts) > 1:
        d.line(pts, fill=(90, 200, 120), width=2)
    for h in res["pacing"]["hitches"]:
        d.line([x(h["t"]), top[1] - 30, x(h["t"]), top[1]], fill=(230, 70, 60), width=2)
    for r in res["flashes"]["risky_windows"]:
        d.rectangle([x(r["start"]), top[0], x(r["end"]), top[1]], outline=(240, 200, 40))
    bot = (230, 390)
    d.text((pad, bot[0] - 18), f"turn speed ({res['comfort']['unit']}); grey line = comfort limit", fill=(220, 220, 220), font=f)
    mx = max(float(turn.max()) if len(turn) else 1.0, res["comfort"]["limit"] * 1.5, 1e-6)
    step = max(1, len(t) // 1400)
    tp = [(x(float(t[k])), bot[1] - turn[k] / mx * (bot[1] - bot[0])) for k in range(0, len(t), step)]
    if len(tp) > 1:
        d.line(tp, fill=(110, 170, 240), width=1)
    ly = bot[1] - res["comfort"]["limit"] / mx * (bot[1] - bot[0])
    d.line([pad, ly, Wd - pad, ly], fill=(120, 120, 120))
    for k in range(6):
        tt = t0 + span * k / 5
        d.text((x(tt) - 20, H - 22), media.fmt_t(tt), fill=(200, 200, 200), font=f)
    im.save(path)


def summary_md(r: dict) -> str:
    p, c, fl = r["pacing"], r["comfort"], r["flashes"]
    L = [f"# capture QA: {Path(r['file']).name} {media.fmt_t(r['window'][0])}-{media.fmt_t(r['window'][1])}",
         f"- {r['frames']} frames at {r['capture_fps']:g} fps capture; {'stereo ' + r['stereo'] + ' (left eye analysed)' if r['stereo'] != 'mono' else 'mono'}"]
    e = p["effective_fps"]
    L.append(f"- effective fps while moving: median {e['median']}, 5th percentile {e['p5']}, worst second {e['min']}" if e
             else "- effective fps: the picture never moved enough to measure")
    L.append(f"- hitches: {p['hitch_count']} (repeated frames while the picture moved), worst {p['worst_hitch_ms']} ms"
             + ("; at " + ", ".join(f"{media.fmt_t(h['t'])} ({h['held_ms']:g} ms)" for h in sorted(p["hitches"], key=lambda h: -h["held_ms"])[:8]) if p["hitches"] else ""))
    if p["timestamp_gaps"]:
        L.append(f"- timestamp gaps (variable-rate capture): {len(p['timestamp_gaps'])}; " +
                 ", ".join(f"{media.fmt_t(g['t'])} ({g['gap_ms']:g} ms)" for g in p["timestamp_gaps"][:6]))
    L.append(f"- comfort: peak turn {c['peak_turn']} {c['unit']}; {len(c['fast_turns'])} turns above {c['limit']} {c['unit']}"
             + ("; at " + ", ".join(f"{media.fmt_t(x['start'])}-{media.fmt_t(x['end'])} (peak {x['peak']:g})" for x in c["fast_turns"][:6]) if c["fast_turns"] else "")
             + f" ({c['note']})")
    L.append(f"- flashes: " + ("; ".join(f"{media.fmt_t(x['start'])}-{media.fmt_t(x['end'])} ({x['flashes_per_s']:g}/s)" for x in fl["risky_windows"][:6])
                                + " exceed 3 per second" if fl["risky_windows"] else "none above 3 per second") + f" ({fl['note']})")
    if r.get("chart"):
        L.append(f"- chart: `{r['chart']}`")
    return "\n".join(L) + "\n"
