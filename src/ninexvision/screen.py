"""Screen mode: stable UI states, transitions, distinct views, pages, scroll stitching."""
from __future__ import annotations

import cv2
import numpy as np

from .signal import Signal

MIN_STATE = 0.3          # seconds a UI must hold still to count as a state
SAME_VIEW = 0.006        # fraction of changed thumb pixels under which two states are the same view
SAME_VIEW_SHARP = 0.0015 # same, measured on 480 px reps
MOTION_GAP = 0.8         # a moving stretch longer than this (s) becomes a "moving" state of its own
MOTION_STEP = 2.0        # long moving stretches get one representative frame per this many seconds


def quiet_threshold(sig: Signal) -> float:
    moving = sig.changed[1:]
    base = float(np.median(moving)) if len(moving) else 0.0
    return max(0.002, 1.5 * base)


def find_states(sig: Signal) -> list[dict]:
    tau = quiet_threshold(sig)
    quiet = sig.changed < tau
    quiet[0] = True
    min_frames = max(2, int(round(MIN_STATE * sig.fps)))
    states = []
    i, n = 0, len(quiet)
    while i < n:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and quiet[j + 1]:
            j += 1
        a = max(0, i - 1) if i > 0 else 0   # frame i-1 already looks like frame i
        if (j - a + 1) >= min_frames or (a == 0) or (j == n - 1):
            rep = a + int(round(0.8 * (j - a)))
            states.append({"a": a, "b": j, "rep": rep})
        i = j + 1
    # merge neighbours that are visually identical (a flicker between them)
    merged: list[dict] = []
    for s in states:
        if merged and _thumb_diff(sig.thumbs[merged[-1]["rep"]], sig.thumbs[s["rep"]]) < 0.0008:
            merged[-1]["b"] = s["b"]
            merged[-1]["rep"] = merged[-1]["a"] + int(round(0.8 * (s["b"] - merged[-1]["a"])))
        else:
            merged.append(dict(s))
    return _fill_motion(sig, merged)


def _fill_motion(sig: Signal, states: list[dict]) -> list[dict]:
    """Screens that never hold still (auto-scrolling carousels, video ads, loaders) still get seen."""
    out: list[dict] = []
    n = len(sig.t)
    bounds = [(-1, -1)] + [(s["a"], s["b"]) for s in states] + [(n, n)]
    k = 0
    for (_, pb), (na, _) in zip(bounds, bounds[1:]):
        lo, hi = pb + 1, na - 1
        if hi >= lo and (sig.t[hi] - sig.t[lo]) >= MOTION_GAP:
            span = sig.t[hi] - sig.t[lo]
            parts = max(1, int(span // MOTION_STEP))
            edges = np.linspace(lo, hi + 1, parts + 1).astype(int)
            for e0, e1 in zip(edges, edges[1:]):
                e1 = max(e0, e1 - 1)
                out.append({"a": int(e0), "b": int(e1), "rep": int((e0 + e1) // 2), "moving": True})
        if k < len(states):
            out.append(states[k])
            k += 1
    return out


def _thumb_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return float((cv2.absdiff(a, b) > 20).mean())


def _thumb_bbox(a: np.ndarray, b: np.ndarray) -> tuple[float, list[float] | None]:
    d = (cv2.absdiff(a, b) > 20).astype(np.uint8)
    area = float(d.mean())
    if area == 0:
        return 0.0, None
    ys, xs = np.nonzero(d)
    H, W = d.shape
    return area, [round(xs.min() / W, 3), round(ys.min() / H, 3), round((xs.max() + 1 - xs.min()) / W, 3), round((ys.max() + 1 - ys.min()) / H, 3)]


def _overlaps(a: np.ndarray, b: np.ndarray, dy: float) -> bool:
    """True if b is a vertically shifted a (same page scrolled), not a new screen sliding in."""
    best = 1.0
    for s in {int(round(dy)) + k for k in range(-2, 3)}:
        if s == 0 or abs(s) >= a.shape[0] - 8:
            continue
        if s < 0:   # content moved up
            pa, pb = a[-s:], b[:a.shape[0] + s]
        else:
            pa, pb = a[:a.shape[0] - s], b[s:]
        best = min(best, float((cv2.absdiff(pa, pb) > 20).mean()))
    return best < 0.06


def classify_transition(sig: Signal, s0: dict, s1: dict) -> dict:
    lo, hi = s0["b"] + 1, s1["a"] + 1
    idx = np.arange(lo, max(lo, hi))
    area, bbox = _thumb_bbox(sig.thumbs[s0["rep"]], sig.thumbs[s1["rep"]])
    out = {"from_t": round(float(sig.t[s0["b"]]), 2), "to_t": round(float(sig.t[min(s1["a"] + 1, len(sig.t) - 1)]), 2),
           "changed_area": round(area, 3), "region": bbox}
    if len(idx):
        good = idx[(sig.resp[idx] > 0.15) & (sig.changed[idx] > 0)]
        cum_dx = float(sig.dx[good].sum()) if len(good) else 0.0
        cum_dy = float(sig.dy[good].sum()) if len(good) else 0.0
        vert = good[(np.abs(sig.dy[good]) > 0.5) & (np.abs(sig.dx[good]) < 0.5)] if len(good) else good
        horiz = good[(np.abs(sig.dx[good]) > 0.5) & (np.abs(sig.dy[good]) < 0.5)] if len(good) else good
        frac_v = len(vert) / len(idx)
        frac_h = len(horiz) / len(idx)
        h_px = sig.height * (1 - sig.mask_top - sig.mask_bottom)
        if frac_v >= 0.4 and abs(cum_dy) >= 0.03 * h_px and abs(cum_dx) < 0.3 * abs(cum_dy):
            direction = "down" if cum_dy < 0 else "up"
            if _overlaps(sig.thumbs[s0["rep"]], sig.thumbs[s1["rep"]], cum_dy / sig.width * 64):
                out.update(type="scroll", direction=direction, distance_screens=round(abs(cum_dy) / h_px, 2))
            else:
                out.update(type="slide", direction="up" if cum_dy < 0 else "down",
                           distance_screens=round(abs(cum_dy) / h_px, 2))
            return out
        if frac_h >= 0.4 and abs(cum_dx) >= 0.05 * sig.width and abs(cum_dy) < 0.3 * abs(cum_dx):
            out.update(type="swipe", direction="left" if cum_dx < 0 else "right",
                       distance_screens=round(abs(cum_dx) / sig.width, 2))
            return out
    box_area = bbox[2] * bbox[3] if bbox else 0.0
    if s0.get("moving") or s1.get("moving"):
        out["type"] = "animating"
    elif box_area < 0.25:
        out["type"] = "local change"
    elif area < 0.12:
        out["type"] = "content change"
    elif bbox and bbox[1] > 0.3 and bbox[3] < 0.75:
        out["type"] = "partial change (sheet/modal/keyboard?)"
    else:
        out["type"] = "screen change"
    return out


def change_blobs(a: np.ndarray, b: np.ndarray, max_blobs: int = 4) -> list[list[float]]:
    """Separate changed regions between two grayscale frames, largest first, as normalised x, y, w, h."""
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    d = (cv2.absdiff(a, b) > 20).astype(np.uint8)
    d = cv2.dilate(d, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(d)
    H, W = d.shape
    blobs = [st for st in stats[1:] if st[4] >= 0.0005 * H * W]
    blobs.sort(key=lambda st: -st[4])
    return [[st[0] / W, st[1] / H, st[2] / W, st[3] / H] for st in blobs[:max_blobs]]


def colour_shift(a_rgb: np.ndarray, b_rgb: np.ndarray, box: list[float]) -> int:
    """+1 if the changed pixels in box gained colour (a control turned on / got selected), -1 if they lost it,
    0 if unclear. Measured on saturation of the pixels that actually changed."""
    H, W = a_rgb.shape[:2]
    x0, y0 = int(box[0] * W), int(box[1] * H)
    x1, y1 = max(x0 + 1, int((box[0] + box[2]) * W)), max(y0 + 1, int((box[1] + box[3]) * H))
    a, b = a_rgb[y0:y1, x0:x1], b_rgb[y0:y1, x0:x1]
    if b.shape != a.shape or a.size == 0:
        return 0
    changed = (np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2) > 20)
    if changed.sum() < 20:
        return 0
    sa = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)[..., 1][changed].astype(float)
    sb = cv2.cvtColor(b, cv2.COLOR_RGB2HSV)[..., 1][changed].astype(float)
    d = sb.mean() - sa.mean()
    return 1 if d > 25 else -1 if d < -25 else 0


def box_changed(a: np.ndarray, b: np.ndarray, x: float, y: float, w: float, h: float) -> bool:
    """Did pixels change on or right around this text box? Band coordinates (0-1)."""
    H, W = a.shape[:2]
    r0, r1 = max(0, int((y - 0.5 * h) * H)), min(H, int((y + 1.5 * h) * H) + 1)
    c0, c1 = max(0, int(x * W) - 4), min(W, int((x + w) * W) + 5)
    if r1 <= r0 or c1 <= c0:
        return False
    d = cv2.absdiff(a[r0:r1, c0:c1], b[r0:r1, c0:c1])
    if d.ndim == 3:
        d = d.max(axis=2)
    return float((d > 20).mean()) > 0.02


def labels_at(box: list[float], lines: list[dict], k: int = 2, changed=None) -> list[str]:
    """On-screen text in the same row as a changed box (a checkbox's label, a tapped option, a picker value).
    changed(x, y, w, h) -> bool: prefer labels that changed themselves (a recoloured option); if none did
    (a switch next to a fixed label), fall back to the nearest label in the row."""
    x, y, w, h = box
    scored = []
    for ln in lines:
        lx, ly, lw, lh = ln["box"]
        overlap = min(y + h, ly + lh) - max(y, ly)
        if overlap < 0.3 * min(h, lh) or ln.get("conf", 1) < 0.4 or len(ln["text"].strip()) < 2:
            continue
        gap = max(0.0, max(x, lx) - min(x + w, lx + lw))
        scored.append((gap, -overlap, ln["text"].strip(), ln["box"]))
    scored.sort(key=lambda r: r[:3])
    if changed is not None:
        own = [r for r in scored if changed(*r[3])]
        scored = own if own else scored[:1]
    out = []
    for _, _, t, _ in scored:
        if t not in out:
            out.append(t)
    return out[:k]


def cluster_views(sig: Signal, states: list[dict], imgs: list[np.ndarray] | None = None) -> None:
    """Give every state a view id; revisits of an identical screen share one.
    imgs: sharper grayscale reps (one per state) so small controls (toggles, checkmarks) count."""
    reps: list[tuple[int, np.ndarray]] = []
    for k, s in enumerate(states):
        th = imgs[k] if imgs is not None else sig.thumbs[s["rep"]]
        best, best_d = None, 1.0
        for vid, rth in reps:
            d = _thumb_diff(rth, th)
            if d < best_d:
                best, best_d = vid, d
        if best is not None and best_d < (SAME_VIEW_SHARP if imgs is not None else SAME_VIEW):
            s["view"] = best
        else:
            s["view"] = len(reps) + 1
            reps.append((s["view"], th))


def group_pages(states: list[dict], transitions: list[dict]) -> dict[int, int]:
    """Views joined by scrolling are one page. Returns view -> page id."""
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s in states:
        find(s["view"])
    for k, tr in enumerate(transitions):
        if tr["type"] == "scroll":
            a, b = find(states[k]["view"]), find(states[k + 1]["view"])
            if a != b:
                parent[max(a, b)] = min(a, b)
    roots = sorted({find(v) for v in parent})
    ids = {r: i + 1 for i, r in enumerate(roots)}
    return {v: ids[find(v)] for v in parent}


# ---------------------------------------------------------------- stitching

def _fixed_bands(a: np.ndarray, b: np.ndarray, top_skip: int) -> tuple[int, int]:
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.int16)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.int16)
    rows = np.abs(ga - gb).mean(axis=1)
    H = len(rows)
    top = top_skip
    while top < H // 2 and rows[top] < 2.0:
        top += 1
    bot = H - 1
    while bot > H // 2 and rows[bot] < 2.0:
        bot -= 1
    return top, H - 1 - bot


def stitch(frames: list[np.ndarray], top_skip_frac: float = 0.0) -> tuple[np.ndarray | None, list[int]]:
    """Stitch consecutive downward-scroll frames into one long page. Returns (image, offsets)."""
    if len(frames) < 2:
        return None, []
    H, W = frames[0].shape[:2]
    top_skip = int(H * top_skip_frac)
    header, footer = H, H
    for a, b in zip(frames, frames[1:]):
        t, f = _fixed_bands(a, b, top_skip)
        header, footer = min(header, t), min(footer, f)
    header = max(header, top_skip)
    mids = [fr[header:H - footer] for fr in frames]
    L = mids[0].shape[0]
    if L < 80:
        return None, []
    canvas = [frames[0][:H - footer]]
    offsets = []
    for a, b in zip(mids, mids[1:]):
        strip_h = max(40, min(L // 3, 240))
        tpl = cv2.cvtColor(b[:strip_h], cv2.COLOR_RGB2GRAY)
        src = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        res = cv2.matchTemplate(src, tpl, cv2.TM_SQDIFF_NORMED)
        _, _, min_loc, _ = cv2.minMaxLoc(res)
        score = float(res[min_loc[1], 0]) if res.shape[1] == 1 else float(res.min())
        s = int(min_loc[1])
        if score > 0.06 or s <= 0:
            return None, []
        offsets.append(s)
        new_rows = b[L - s:]
        canvas.append(new_rows)
    canvas.append(frames[-1][H - footer:])
    return np.vstack(canvas), offsets
