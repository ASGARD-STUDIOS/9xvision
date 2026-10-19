"""Image outputs sized for vision models: labelled contact sheets, colour scripts, diffs."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Most frontier vision APIs downscale anything past ~1568 px on the long edge,
# so sheets are built at that size: every pixel sent is a pixel the model sees.
MAX_EDGE = 1568


def font(size: int) -> ImageFont.ImageFont:
    for name in ("/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "DejaVuSansMono-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def token_estimate(w: int, h: int) -> int:
    """Rough image token cost (Claude-style w*h/750, after the long-edge cap)."""
    s = min(1.0, MAX_EDGE / max(w, h))
    return int(w * s * h * s / 750)


def save_jpg(img: np.ndarray | Image.Image, path: Path, max_edge: int | None = MAX_EDGE, quality: int = 88) -> tuple[int, int]:
    im = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    if max_edge and max(im.size) > max_edge:
        s = max_edge / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.convert("RGB").save(path, "JPEG", quality=quality, optimize=True)
    return im.size


def label(im: Image.Image, text: str, xy=(0, 0), size: int = 18, corner: str = "tl") -> None:
    d = ImageDraw.Draw(im)
    f = font(size)
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    pad = max(3, size // 4)
    w, h = r - l + 2 * pad, b - t + 2 * pad
    x, y = xy
    if corner == "bl":
        y = im.height - h
    d.rectangle([x, y, x + w, y + h], fill=(0, 0, 0))
    d.text((x + pad - l, y + pad - t), text, fill=(255, 235, 59), font=f)


def grid_shape(n: int, cell_aspect: float, max_edge: int = MAX_EDGE, max_cols: int = 8) -> tuple[int, int, int, int]:
    """Choose cols/rows so the sheet fits max_edge and cells stay as large as possible."""
    best = None
    for cols in range(1, min(n, max_cols) + 1):
        rows = math.ceil(n / cols)
        # cell width w, height w/cell_aspect; fit cols*w <= E and rows*w/a <= E
        w = min(max_edge / cols, max_edge * cell_aspect / rows)
        if best is None or w > best[0]:
            best = (w, cols, rows)
    w, cols, rows = best
    cw = int(w)
    ch = int(w / cell_aspect)
    return cols, rows, cw, ch


def contact_sheet(frames: list[np.ndarray], labels: list[str], path: Path, max_edge: int = MAX_EDGE,
                  gap: int = 4, title: str | None = None) -> tuple[int, int]:
    if not frames:
        raise ValueError("no frames")
    h0, w0 = frames[0].shape[:2]
    aspect = w0 / h0
    title_h = 30 if title else 0
    cols, rows, cw, ch = grid_shape(len(frames), aspect, max_edge - title_h)
    cw -= gap
    ch = int(cw / aspect)
    W = cols * (cw + gap) + gap
    H = rows * (ch + gap) + gap + title_h
    sheet = Image.new("RGB", (W, H), (24, 24, 24))
    if title:
        label(sheet, title, (0, 0), size=18)
    lab_size = max(12, min(22, cw // 11))
    for i, (fr, lab) in enumerate(zip(frames, labels)):
        r, c = divmod(i, cols)
        cell = Image.fromarray(fr).resize((cw, ch), Image.LANCZOS)
        label(cell, lab, size=lab_size)
        sheet.paste(cell, (gap + c * (cw + gap), title_h + gap + r * (ch + gap)))
    return save_jpg(sheet, path, max_edge=max_edge)


def colour_script(shots: list[dict], duration: float, path: Path, width: int = MAX_EDGE, height: int = 160) -> tuple[int, int]:
    """One band per shot, width proportional to duration, stacked palette swatches."""
    im = Image.new("RGB", (width, height + 26), (16, 16, 16))
    d = ImageDraw.Draw(im)
    for s in shots:
        x0 = int(s["start"] / duration * width)
        x1 = max(x0 + 1, int(s["end"] / duration * width))
        y = 0
        pal = s.get("palette") or [{"hex": "#000000", "share": 1.0}]
        for sw in pal:
            if y >= height:
                break
            hgt = max(1, int(round(sw["share"] * height)))
            d.rectangle([x0, y, x1, min(height, y + hgt)], fill=sw["hex"])
            y += hgt
    f = font(14)
    for k in range(0, int(duration) + 1, max(10, int(duration / 10) // 10 * 10 or 10)):
        x = int(k / duration * width)
        d.line([x, height, x, height + 6], fill=(200, 200, 200))
        m, s_ = divmod(k, 60)
        d.text((x + 2, height + 6), f"{m}:{s_:02d}", fill=(200, 200, 200), font=f)
    return save_jpg(im, path, max_edge=None)


def diff_image(a: np.ndarray, b: np.ndarray, path: Path, labels: tuple[str, str]) -> dict:
    """A | B | B-with-changes-highlighted, plus bounding boxes of changed regions."""
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    delta = cv2.absdiff(ga, gb)
    mask = (delta > 24).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    H, W = mask.shape
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 0.0004 * W * H:
            continue
        boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area_pct": round(100 * area / (W * H), 2)})
    boxes.sort(key=lambda bx: -bx["area_pct"])
    hl = b.copy()
    tint = hl.copy()
    tint[mask > 0] = (255, 0, 80)
    hl = cv2.addWeighted(hl, 0.55, tint, 0.45, 0)
    for bx in boxes[:12]:
        cv2.rectangle(hl, (bx["x"], bx["y"]), (bx["x"] + bx["w"], bx["y"] + bx["h"]), (255, 235, 59), max(2, W // 300))
    panels = [Image.fromarray(x) for x in (a, b, hl)]
    for p, t in zip(panels, (labels[0], labels[1], "changed")):
        label(p, t, size=max(14, W // 30))
    gap = 6
    sheet = Image.new("RGB", (3 * W + 4 * gap, H + 2 * gap), (24, 24, 24))
    for i, p in enumerate(panels):
        sheet.paste(p, (gap + i * (W + gap), gap))
    size = save_jpg(sheet, path)
    changed = float((mask > 0).mean() * 100)
    return {"path": str(path), "size": size, "changed_pct": round(changed, 2), "regions": boxes[:12]}
