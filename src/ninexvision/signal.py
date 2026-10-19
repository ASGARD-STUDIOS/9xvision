"""One decode pass -> per-frame features shared by film and screen analysis."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .media import Probe, stream_frames


@dataclass
class Signal:
    fps: float
    width: int
    height: int
    mask_top: float
    mask_bottom: float
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    content: np.ndarray = field(default_factory=lambda: np.zeros(0))   # HSV mean abs delta vs previous frame
    changed: np.ndarray = field(default_factory=lambda: np.zeros(0))   # fraction of (masked) pixels that changed
    dx: np.ndarray = field(default_factory=lambda: np.zeros(0))        # phase-correlation shift vs previous (analysis px)
    dy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    resp: np.ndarray = field(default_factory=lambda: np.zeros(0))      # phase-correlation peak response
    luma: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sharp: np.ndarray = field(default_factory=lambda: np.zeros(0))
    identical: np.ndarray = field(default_factory=lambda: np.zeros(0))  # changed == 0 (static screen frames)
    thumbs: list = field(default_factory=list)                          # tiny masked grayscale thumbs


def default_masks(pr: Probe, mode: str) -> tuple[float, float]:
    """Mask OS chrome (clock, battery, home indicator) so it never registers as a UI change."""
    if mode == "film":
        return 0.0, 0.0
    if pr.height / max(1, pr.width) > 1.5:      # phone portrait
        return 0.07, 0.025
    return 0.035, 0.0                            # desktop menu bar / tablet status bar


def analysis_fps(pr: Probe, mode: str, span: float | None = None) -> float:
    base = 12.0
    if (span or pr.duration) > 3600:
        base /= 2
    return float(min(base, pr.fps or base))


def compute(path: str, pr: Probe, mode: str, fps: float | None = None,
            start: float | None = None, end: float | None = None) -> Signal:
    """start/end (original-video seconds) limit the analysis to a window."""
    span = (end if end is not None else pr.end) - (start if start is not None else pr.offset)
    fps = fps or analysis_fps(pr, mode, span)
    width = 240 if pr.height > pr.width else 320
    mt, mb = default_masks(pr, mode)
    sig = Signal(fps=fps, width=width, height=0, mask_top=mt, mask_bottom=mb)
    ts, content, changed, dxs, dys, resps, lumas, sharps = [], [], [], [], [], [], [], []
    prev_hsv = prev_g = None
    win = None
    for t, rgb in stream_frames(path, fps, width, start=start, end=end):
        h = rgb.shape[0]
        sig.height = h
        y0, y1 = int(h * mt), h - int(h * mb)
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gm = g[y0:y1]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[y0:y1].astype(np.int16)
        if win is None:
            win = cv2.createHanningWindow((gm.shape[1], gm.shape[0]), cv2.CV_32F)
        if prev_g is None:
            c, ch, dx, dy, r = 0.0, 0.0, 0.0, 0.0, 1.0
        else:
            c = float(np.abs(hsv - prev_hsv).mean())
            ch = float((cv2.absdiff(gm, prev_g) > 14).mean())
            if ch > 0:
                (dx, dy), r = cv2.phaseCorrelate(prev_g.astype(np.float32), gm.astype(np.float32), win)
            else:
                dx, dy, r = 0.0, 0.0, 1.0
        ts.append(t); content.append(c); changed.append(ch); dxs.append(dx); dys.append(dy); resps.append(r)
        lumas.append(float(g.mean()))
        sharps.append(float(cv2.Laplacian(gm, cv2.CV_32F).var()))
        sig.thumbs.append(cv2.resize(gm, (64, max(8, int(64 * gm.shape[0] / gm.shape[1]))), interpolation=cv2.INTER_AREA))
        prev_g, prev_hsv = gm, hsv
    if not ts:
        raise RuntimeError("decoded zero frames")
    sig.t = np.array(ts); sig.content = np.array(content); sig.changed = np.array(changed)
    sig.dx = np.array(dxs); sig.dy = np.array(dys); sig.resp = np.array(resps)
    sig.luma = np.array(lumas); sig.sharp = np.array(sharps)
    sig.identical = sig.changed == 0
    return sig


def guess_mode(pr: Probe, sig: Signal) -> str:
    """Screen recordings sit still most of the time and have phone/desktop geometry."""
    still = float((sig.changed[1:] < 0.0015).mean()) if len(sig.changed) > 1 else 0.0
    exact = float(sig.identical[1:].mean()) if len(sig.identical) > 1 else 0.0
    phone = pr.height / max(1, pr.width) > 1.9
    if exact > 0.25 or (still > 0.5 and phone) or still > 0.7:
        return "screen"
    return "film"
