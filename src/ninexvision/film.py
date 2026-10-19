"""Film mode: shot boundaries, keyframes, camera motion, rhythm."""
from __future__ import annotations

import numpy as np

from .signal import Signal

ADAPTIVE_RATIO = 3.0     # PySceneDetect AdaptiveDetector defaults, applied at a fixed analysis rate
MIN_CONTENT = 15.0
HARD_CONTENT = 55.0
BLACK_LUMA = 12.0


def detect_cuts(sig: Signal, min_shot: float = 0.25) -> list[float]:
    c = sig.content
    n = len(c)
    cuts: list[tuple[int, float]] = []
    w = 2
    for i in range(1, n):
        lo, hi = max(1, i - w), min(n, i + w + 1)
        neigh = np.concatenate([c[lo:i], c[i + 1:hi]])
        avg = float(neigh.mean()) if len(neigh) else 0.0
        ratio = c[i] / (avg + 1e-6)
        if (c[i] >= MIN_CONTENT and ratio >= ADAPTIVE_RATIO) or c[i] >= HARD_CONTENT:
            cuts.append((i, float(c[i])))
    # fades: entering / leaving black are boundaries too
    black = sig.luma < BLACK_LUMA
    for i in range(1, n):
        if black[i] != black[i - 1]:
            cuts.append((i, 999.0))
    cuts.sort()
    min_frames = max(1, int(round(min_shot * sig.fps)))
    kept: list[tuple[int, float]] = []
    for i, s in cuts:
        if kept and i - kept[-1][0] < min_frames:
            if s > kept[-1][1]:
                kept[-1] = (i, s)
            continue
        kept.append((i, s))
    return [float(sig.t[i]) for i, _ in kept]


def motion_class(sig: Signal, a: int, b: int) -> dict:
    """Camera/subject motion inside frames [a, b) (first frame after a cut excluded)."""
    idx = np.arange(a + 1, b)
    if len(idx) < 2:
        return {"class": "STILL", "energy": 0.0, "pan": 0.0}
    energy = float(np.median(sig.changed[idx]))
    good = idx[sig.resp[idx] > 0.08]
    if len(good):
        vx = sig.dx[good] / sig.width * sig.fps
        vy = sig.dy[good] / sig.width * sig.fps
        pan = float(np.median(np.hypot(vx, vy)))
        steadiness = float(abs(vx.mean()) + abs(vy.mean())) / (float(np.hypot(vx, vy).mean()) + 1e-6)
    else:
        pan, steadiness = 0.0, 0.0
    if energy < 0.03 and pan < 0.03:
        cls = "STILL"
    elif energy > 0.45 or pan > 0.6:
        cls = "KINETIC"
    elif steadiness > 0.6 and pan >= 0.03:
        cls = "DRIFT"
    else:
        cls = "HANDHELD" if pan >= 0.03 else "SUBJECT"
    return {"class": cls, "energy": round(energy, 3), "pan": round(pan, 3)}


BEAT_CHANGE = 0.04      # fraction of thumb pixels that must differ from the beat's anchor
BEAT_HOLD = 1.0         # the new content must hold this long (s) to count
BEAT_MIN = 2.0          # no beats closer than this (s)


def content_beats(sig: Signal, cuts: list[float], duration: float, start: float = 0.0) -> list[float]:
    """Boundaries inside long shots where the picture content changes and stays changed
    (slides, screen-share pages, scene changes in a single take)."""
    import cv2
    bounds = [start] + cuts + [duration]
    hold = max(1, int(round(BEAT_HOLD * sig.fps)))
    beats: list[float] = []
    for s, e in zip(bounds, bounds[1:]):
        if e - s < 2 * BEAT_MIN:
            continue
        a = int(np.searchsorted(sig.t, s)) + 1
        b = int(np.searchsorted(sig.t, e)) - 1
        anchor = a
        last = s
        i = a + 1
        while i < b - hold:
            d = float((cv2.absdiff(sig.thumbs[anchor], sig.thumbs[i]) > 20).mean())
            if d > BEAT_CHANGE and sig.t[i] - last >= BEAT_MIN:
                d2 = float((cv2.absdiff(sig.thumbs[anchor], sig.thumbs[i + hold]) > 20).mean())
                d3 = float((cv2.absdiff(sig.thumbs[i], sig.thumbs[i + hold]) > 20).mean())
                if d2 > BEAT_CHANGE and d3 < BEAT_CHANGE:   # changed, and the new picture holds
                    beats.append(float(sig.t[i]))
                    last = float(sig.t[i])
                    anchor = i + hold
                    i = anchor + 1
                    continue
            i += 1
    return beats


def shots(sig: Signal, duration: float, min_shot: float = 0.25, beats: bool = True, start: float = 0.0) -> list[dict]:
    """Segments between start and duration (= end time; both in original-video seconds)."""
    cuts = [t for t in detect_cuts(sig, min_shot) if start < t < duration]
    beat_ts = content_beats(sig, cuts, duration, start) if beats else []
    kinds = {round(t, 3): "cut" for t in cuts}
    kinds.update({round(t, 3): "content" for t in beat_ts})
    bounds = [start] + sorted(cuts + beat_ts) + [duration]
    out = []
    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1]
        if e - s <= 0.01:
            continue
        a = int(np.searchsorted(sig.t, s))
        b = max(a + 1, int(np.searchsorted(sig.t, e)))
        lo = a + int(0.2 * (b - a))
        hi = max(lo + 1, a + int(0.8 * (b - a)))
        hi = min(hi, len(sig.t))
        best = lo + int(np.argmax(sig.sharp[lo:hi])) if hi > lo else a
        key_t = float(sig.t[min(best, len(sig.t) - 1)])
        out.append({
            "id": len(out) + 1,
            "boundary": "start" if k == 0 else kinds.get(round(s, 3), "cut"),
            "start": round(s, 2), "end": round(e, 2), "dur": round(e - s, 2),
            "key_t": round(min(max(key_t, s), e - 0.02), 2),
            "motion": motion_class(sig, a, b),
            "black": bool((sig.luma[a:b] < BLACK_LUMA).mean() > 0.8) if b > a else False,
            "brightness": round(float(sig.luma[a:b].mean()) / 255, 2) if b > a else 0.0,
        })
    return out


def rhythm(segs: list[dict], duration: float, start: float = 0.0) -> dict:
    """Editing rhythm counts camera cuts only; content beats are not cuts."""
    shot_list = _merge_beats(segs)
    durs = np.array([s["dur"] for s in shot_list]) if shot_list else np.array([duration])
    per_min = []
    for m in range(int(np.ceil((duration - start) / 60))):
        a = start + m * 60
        per_min.append(sum(1 for s in shot_list if a <= s["start"] < a + 60))
    return {
        "shots": len(shot_list),
        "asl": round(float(durs.mean()), 2),
        "median_shot": round(float(np.median(durs)), 2),
        "longest": round(float(durs.max()), 2),
        "cuts_per_minute": per_min,
        "content_beats": sum(1 for s in segs if s.get("boundary") == "content"),
    }


def _merge_beats(segs: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for s in segs:
        if merged and s.get("boundary") == "content":
            merged[-1] = {**merged[-1], "end": s["end"], "dur": round(s["end"] - merged[-1]["start"], 2)}
        else:
            merged.append(dict(s))
    return merged


def refine_cuts(path, times: list[float], analysis_fps: float, video_fps: float) -> list[float]:
    """Cut times are found at the analysis rate (~12 fps). Re-decode a short window around each at the
    native frame rate and move the cut to the first frame of the new shot (frame-accurate)."""
    from concurrent.futures import ThreadPoolExecutor
    from .media import stream_frames
    import cv2

    fps = min(max(video_fps, 1.0), 120.0)

    def one(t: float) -> float:
        frames = list(stream_frames(path, fps, 96, pix="gray", start=max(0.0, t - 1.5 / analysis_fps), end=t + 0.5 / analysis_fps))
        if len(frames) < 3:
            return t
        d = [float(cv2.absdiff(a[1], b[1]).mean()) for a, b in zip(frames, frames[1:])]
        k = int(np.argmax(d))
        return round(frames[k + 1][0], 3)

    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, times))
