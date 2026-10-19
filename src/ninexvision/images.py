"""Still images: inspect one large image at native resolution, or index a folder of photos and screenshots into an
album that search, look and list work on like a watch.

Agents usually receive images shrunk to about 1,500 px, so small text in a 4,000 px screenshot is lost. inspect()
returns an overview with a labelled grid; asking for a cell (B2), a range (B2:C3) or a box returns that part at full
resolution (enlarged when small), so nothing has to be guessed."""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from . import extract, media, render

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".gif", ".avif"}
MAX_ALBUM = 3000


def home() -> Path:
    from .watch import HOME
    return HOME


# ---------------------------------------------------------------- loading


def fetch(url: str) -> Path:
    """Download an image URL once (cached by URL)."""
    d = home() / "images" / "downloads"
    d.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(url.encode()).hexdigest()[:12]
    hit = next(d.glob(key + ".*"), None)
    if hit:
        return hit
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 9xvision"})
    with urllib.request.urlopen(req, timeout=60) as r:
        ctype = r.headers.get("Content-Type", "")
        if not ctype.startswith("image/"):
            raise ValueError(f"{url} is not an image (content type {ctype or 'unknown'}); for videos use watch")
        data = r.read(80 * 1024 * 1024 + 1)
    if len(data) > 80 * 1024 * 1024:
        raise ValueError(f"{url}: image larger than 80 MB")
    ext = "." + ctype.split("/")[1].split(";")[0].replace("jpeg", "jpg").replace("svg+xml", "svg")
    p = d / (key + ext)
    p.write_bytes(data)
    return p


def _decode_other(path: Path) -> Image.Image:
    """HEIC/AVIF and anything Pillow can't open: macOS sips, else ffmpeg, to a temporary PNG."""
    tmp = home() / "images" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    out = tmp / (hashlib.sha1(str(path).encode()).hexdigest()[:12] + ".png")
    if sys.platform == "darwin" and shutil.which("sips"):
        subprocess.run(["sips", "-s", "format", "png", str(path), "--out", str(out)], capture_output=True)
    if not out.exists():
        subprocess.run([media.require("ffmpeg"), "-v", "error", "-y", "-i", str(path), "-frames:v", "1", str(out)], capture_output=True)
    if not out.exists():
        raise ValueError(f"can't read {path.name} as an image")
    im = Image.open(out)
    im.load()
    out.unlink(missing_ok=True)
    return im


Image.MAX_IMAGE_PIXELS = 400_000_000          # big scans and panoramas are the point; 400 MP is ~1.2 GB as RGB


def open_image(path: str | Path) -> tuple[np.ndarray, Image.Image]:
    """RGB array (EXIF orientation applied) and the PIL image (for EXIF)."""
    path = Path(path)
    try:
        im = Image.open(path)
        im.load()
    except Image.DecompressionBombError as e:
        raise ValueError(f"{path.name}: {e} (over {Image.MAX_IMAGE_PIXELS // 1_000_000} MP); crop or downscale it first") from None
    except Exception:
        im = _decode_other(path)
    raw = im
    im = ImageOps.exif_transpose(im)
    if im.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):     # 16-bit / float greyscale (scans, depth maps)
        a = np.asarray(im, dtype=np.float64)
        lo, hi = float(a.min()), float(a.max())
        a = a / 257 if im.mode.startswith("I;16") or (hi > 255 and hi <= 65535 and lo >= 0) else (a - lo) / max(hi - lo, 1e-9) * 255
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "L")
    if im.mode not in ("RGB",):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
    return np.asarray(im), raw


def exif(raw: Image.Image) -> dict:
    out: dict = {}
    try:
        ex = raw.getexif()
    except Exception:
        return out
    if not ex:
        return out
    sub = ex.get_ifd(0x8769) if hasattr(ex, "get_ifd") else {}
    taken = sub.get(0x9003) or ex.get(0x0132)
    if taken:
        out["taken"] = str(taken).replace(":", "-", 2)
    cam = " ".join(str(x).strip() for x in (ex.get(0x010F), ex.get(0x0110)) if x)
    if cam:
        out["camera"] = cam
    gps = ex.get_ifd(0x8825) if hasattr(ex, "get_ifd") else {}
    try:
        if gps and 2 in gps and 4 in gps:
            dms = lambda v: float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
            lat = dms(gps[2]) * (-1 if gps.get(1) == "S" else 1)
            lon = dms(gps[4]) * (-1 if gps.get(3) == "W" else 1)
            out["gps"] = [round(lat, 5), round(lon, 5)]
    except Exception:
        pass
    return out


def _resize(rgb: np.ndarray, edge: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = edge / max(h, w)
    return cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else rgb


def dhash(rgb: np.ndarray) -> int:
    g = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


# ---------------------------------------------------------------- one image: overview + native-resolution regions


def _cache_dir(path: Path) -> Path:
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:12]
    d = home() / "images" / f"{re.sub(r'[^A-Za-z0-9]+', '-', path.stem)[:30].strip('-').lower()}-{key}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _grid(w: int, h: int) -> tuple[int, int]:
    """Columns and rows so each cell is at most ~1000 native px (and at least 2x2 for big images)."""
    if max(w, h) <= render.MAX_EDGE:
        return 1, 1
    return max(2, math.ceil(w / 1000)), max(2, math.ceil(h / 1000))


def _cell_box(ref: str, w: int, h: int, cols: int, rows: int) -> tuple[int, int, int, int]:
    m = re.fullmatch(r"([A-Za-z])(\d+)(?::([A-Za-z])(\d+))?", ref.strip())
    if not m:
        raise ValueError(f"region {ref!r}: use a grid cell like B2, a range like B2:C3, or x,y,w,h")
    c0, r0 = ord(m.group(1).upper()) - 65, int(m.group(2)) - 1
    c1, r1 = (ord(m.group(3).upper()) - 65, int(m.group(4)) - 1) if m.group(3) else (c0, r0)
    c0, c1, r0, r1 = min(c0, c1), max(c0, c1), min(r0, r1), max(r0, r1)
    if not (0 <= c0 and c1 < cols and 0 <= r0 and r1 < rows):
        raise ValueError(f"region {ref!r} is outside the grid (columns A-{chr(64 + cols)}, rows 1-{rows})")
    cw, ch = w / cols, h / rows
    return int(c0 * cw), int(r0 * ch), int((c1 + 1) * cw) - int(c0 * cw), int((r1 + 1) * ch) - int(r0 * ch)


def _box(region: str, w: int, h: int, cols: int, rows: int) -> tuple[int, int, int, int]:
    if "," not in region:
        return _cell_box(region, w, h, cols, rows)
    try:
        v = [float(x) for x in region.split(",")]
    except ValueError:
        raise ValueError(f"region {region!r}: use x,y,w,h in pixels or 0-1 fractions") from None
    if len(v) != 4:
        raise ValueError(f"region {region!r}: needs four numbers x,y,w,h")
    if v[2] <= 0 or v[3] <= 0 or v[0] < 0 or v[1] < 0:
        raise ValueError(f"region {region!r}: width and height must be positive and x,y not negative")
    if all(0 <= x <= 1 for x in v):
        v = [v[0] * w, v[1] * h, v[2] * w, v[3] * h]
    x, y = int(max(0, min(v[0], w - 1))), int(max(0, min(v[1], h - 1)))
    return x, y, int(max(1, min(v[2], w - x))), int(max(1, min(v[3], h - y)))


def inspect(source: str, region: str | None = None, ocr: bool = True) -> dict:
    """Overview with a labelled grid (no region), or a region at native resolution."""
    path = fetch(source) if media.is_url(source) else Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{source}: no such image")
    rgb, raw = open_image(path)
    h, w = rgb.shape[:2]
    cols, rows = _grid(w, h)
    d = _cache_dir(path)
    if region:
        x, y, bw, bh = _box(region, w, h, cols, rows)
        crop = rgb[y:y + bh, x:x + bw]
        scale = 1.0
        if max(bw, bh) < 768:      # enlarge small regions so fine text is legible
            scale = min(4.0, 768 / max(bw, bh))
            crop = cv2.resize(crop, (round(bw * scale), round(bh * scale)), interpolation=cv2.INTER_LANCZOS4)
        elif max(bw, bh) > render.MAX_EDGE:     # never hand an agent more than it can see
            scale = render.MAX_EDGE / max(bw, bh)
            crop = cv2.resize(crop, (round(bw * scale), round(bh * scale)), interpolation=cv2.INTER_AREA)
        p = d / f"region-{x}-{y}-{bw}-{bh}.png"
        Image.fromarray(crop).save(p)
        out = {"path": str(p), "box": [x, y, bw, bh], "source_px": [w, h], "scale": round(scale, 2),
               "size": list(crop.shape[1::-1]), "tokens": render.token_estimate(*crop.shape[1::-1])}
        if scale < 1:
            out["note"] = (f"this region is {bw}x{bh} px, shown at {scale:.2f}x; ask for a region under "
                           f"{render.MAX_EDGE} px to see it at native resolution")
        if ocr:
            lines = extract.ocr(rgb[y:y + bh, x:x + bw])
            if lines:
                out["text"] = [ln["text"] for ln in lines]
        return out
    ov = Image.fromarray(_resize(rgb, render.MAX_EDGE))
    s = ov.width / w
    if cols * rows > 1:
        dr = ImageDraw.Draw(ov, "RGBA")
        f = render.font(max(14, ov.width // 60))
        for c in range(cols):
            for r in range(rows):
                x0, y0 = c * ov.width / cols, r * ov.height / rows
                dr.rectangle([x0, y0, x0 + ov.width / cols, y0 + ov.height / rows], outline=(255, 210, 0, 150), width=1)
                lab = f"{chr(65 + c)}{r + 1}"
                dr.rectangle([x0 + 2, y0 + 2, x0 + 8 + f.getlength(lab), y0 + 6 + f.size], fill=(0, 0, 0, 150))
                dr.text((x0 + 5, y0 + 3), lab, fill=(255, 230, 80, 255), font=f)
    p = d / "overview.jpg"
    ov.save(p, quality=88)
    out = {"path": str(p), "source_px": [w, h], "shown_at": round(s, 3), "size": [ov.width, ov.height],
           "tokens": render.token_estimate(ov.width, ov.height), "file": str(path), **exif(raw),
           "palette": [c["hex"] for c in extract.palette(_resize(rgb, 256), k=5)]}
    if cols * rows > 1:
        out["grid"] = {"cols": cols, "rows": rows, "cell_px": [round(w / cols), round(h / rows)],
                       "how": "pass region='B2' (or 'B2:C3', or 'x,y,w,h' in px or 0-1) for that part at full resolution"}
    if ocr:
        lines = extract.ocr(_resize(rgb, 2600))
        if lines:
            def cell(b):
                cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
                return f"{chr(65 + min(cols - 1, int(cx * cols)))}{min(rows - 1, int(cy * rows)) + 1}"
            out["text"] = [{"text": ln["text"], "cell": cell(ln["box"])} if cols * rows > 1 else ln["text"] for ln in lines[:400]]
    return out


# ---------------------------------------------------------------- albums


def collect(sources: list[str], recursive: bool = False) -> list[Path]:
    files: list[Path] = []
    for s in sources:
        if media.is_url(s):
            files.append(fetch(s))
            continue
        p = Path(s).expanduser()
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            files += sorted(f for f in it if f.is_file() and f.suffix.lower() in EXTS and not f.name.startswith("."))
        elif p.is_file():
            files.append(p)
        else:
            matched = sorted(Path().glob(s)) if not Path(s).is_absolute() else sorted(Path("/").glob(s.lstrip("/")))
            if not matched:
                raise FileNotFoundError(f"{s}: no such file, folder, glob or URL")
            files += [f for f in matched if f.is_file() and f.suffix.lower() in EXTS]
    seen, out = set(), []
    for f in files:
        k = str(f.resolve())
        if k not in seen:
            seen.add(k)
            out.append(f)
    if len(out) > MAX_ALBUM:
        raise ValueError(f"{len(out)} images; albums are capped at {MAX_ALBUM} (split the folder)")
    if not out:
        raise FileNotFoundError("no images found (" + ", ".join(sorted(EXTS)) + ")")
    return out


def album(sources: list[str], recursive: bool = False, ocr: bool = True, force: bool = False, log=lambda m: None) -> Path:
    """Index images into an album folder (watches/<name>-album-<hash>): thumbnails, text, palettes, dates, cameras,
    near-duplicate groups, contact sheets and visual-search embeddings. Re-running is instant."""
    from . import semantic
    from .watch import _slug, watches_root
    files = collect(sources, recursive)
    fp = hashlib.sha1("|".join(f"{f.resolve()}:{f.stat().st_size}:{f.stat().st_mtime_ns}" for f in files).encode()).hexdigest()[:8]
    first = Path(sources[0]).expanduser()
    name = _slug(first.name if first.is_dir() else first.stem if len(files) == 1 else first.parent.name or "images")
    wid = f"{name}-album-{fp}"
    d = watches_root() / wid
    if (d / "watch.json").exists() and not force:
        m = json.loads((d / "watch.json").read_text())
        if m.get("settings", {}).get("ocr") == ocr or not ocr:
            return d
    t_start = time.time()
    d.mkdir(parents=True, exist_ok=True)
    (d / "thumbs").mkdir(exist_ok=True)
    items, thumbs = [], []
    for i, f in enumerate(files, 1):
        try:
            rgb, raw = open_image(f)
        except Exception as e:
            log(f"skip {f.name}: {e}")
            continue
        h, w = rgb.shape[:2]
        th = _resize(rgb, 384)
        rel = f"thumbs/{i:04d}.jpg"
        Image.fromarray(th).save(d / rel, quality=85)
        it = {"id": i, "file": str(f.resolve()), "name": f.name, "px": [w, h], "bytes": f.stat().st_size, "thumb": rel,
              **exif(raw), "palette": [c["hex"] for c in extract.palette(_resize(rgb, 256), k=4)], "dhash": f"{dhash(rgb):016x}"}
        if ocr:
            it["text"] = [ln["text"] for ln in extract.ocr(_resize(rgb, 2600))]
        items.append(it)
        thumbs.append(th)
        if i % 25 == 0:
            log(f"{i}/{len(files)} images")
    # near duplicates: dHash distance <= 6 (same picture, re-encoded / resized / lightly edited)
    groups, parent = [], list(range(len(items)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    hs = [int(it["dhash"], 16) for it in items]
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            if bin(hs[a] ^ hs[b]).count("1") <= 6:
                parent[root(b)] = root(a)
    by: dict[int, list[int]] = {}
    for a in range(len(items)):
        by.setdefault(root(a), []).append(items[a]["id"])
    groups = [g for g in by.values() if len(g) > 1]
    sheets = []
    per = 48
    for k in range(0, len(thumbs), per):
        p = d / f"sheet_{k // per + 1:02d}.jpg"
        render.contact_sheet(thumbs[k:k + per], [f"#{it['id']} {it['name'][:22]}" for it in items[k:k + per]], p,
                             title=f"{wid}: images {k + 1}-{min(len(items), k + per)}")
        sheets.append({"path": p.name, "first": k + 1, "last": min(len(items), k + per)})
    m = {"id": wid, "kind": "album", "mode": "album", "title": name, "images": items, "duplicates": groups,
         "sheets": sheets, "sources": sources, "settings": {"ocr": ocr}, "probe": {"duration": 0},
         "built_s": round(time.time() - t_start, 1)}
    try:      # visual search is optional (the model downloads on first use)
        e = semantic._Model.get().images(thumbs)
        np.savez_compressed(d / "embeddings.npz", t=np.array([it["id"] for it in items], np.float32), e=e.astype(np.float16))
        m["visual_index"] = "embeddings.npz"
    except Exception as ex:
        log(f"visual index skipped: {ex}")
    (d / "watch.json").write_text(json.dumps(m, indent=1, ensure_ascii=False))
    (d / "INDEX.md").write_text(album_index(m))
    return d


def album_index(m: dict) -> str:
    items = m["images"]
    L = [f"# {m['title']} (album, {len(items)} images)", "",
         f"- id: `{m['id']}`; sheets: " + ", ".join(f"sheet:{i + 1} (#{s['first']}-{s['last']})" for i, s in enumerate(m["sheets"])),
         "- tools: look(id, 'sheet:N') for thumbnails, look(id, 'image:N') for the original, image(path, region) for "
         "full-resolution detail, search(query, id) for what is in the pictures or written on them"]
    dated = sorted(it["taken"] for it in items if it.get("taken"))
    if dated:
        L.append(f"- taken: {dated[0]} to {dated[-1]} ({len(dated)} dated)")
    cams = sorted({it["camera"] for it in items if it.get("camera")})
    if cams:
        L.append(f"- cameras: {', '.join(cams[:8])}")
    if m["duplicates"]:
        L.append("- near-duplicates: " + "; ".join(" = ".join(f"#{i}" for i in g) for g in m["duplicates"][:20]))
    L += ["", "| # | file | px | taken | text | palette |", "|---|---|---|---|---|---|"]
    for it in items:
        txt = " / ".join(it.get("text", []))[:90].replace("|", "/")
        L.append(f"| {it['id']} | {it['name'][:40]} | {it['px'][0]}x{it['px'][1]} | {it.get('taken', '')[:16]} | {txt} | "
                 f"{' '.join(it['palette'][:3])} |")
    L.append(f"\n_9xvision album, built in {m.get('built_s')}s_")
    return "\n".join(L)


def album_search(d: Path, m: dict, query: str, k: int = 12) -> dict:
    from .semantic import LOGIT_BIAS, LOGIT_SCALE, _Model, band, query_note
    z = np.load(d / "embeddings.npz")
    ids, e = z["t"].astype(int), z["e"].astype(np.float32)
    q = _Model.get().texts([query])[0]
    score = 1 / (1 + np.exp(-(LOGIT_SCALE * (e @ q) + LOGIT_BIAS)))
    by_id = {it["id"]: it for it in m["images"]}
    med = float(np.median(score))
    res = []
    for i in np.argsort(-score)[:k]:
        it = by_id[int(ids[i])]
        s = float(score[i])
        res.append({"id": it["id"], "file": it["file"], "score": round(s, 4),
                    "match": band(s, med)})
    sheet = None
    if res:
        p = d / "search" / f"visual_{hashlib.sha1(query.encode()).hexdigest()[:10]}.jpg"
        thumbs = [np.asarray(Image.open(d / by_id[r["id"]]["thumb"]).convert("RGB")) for r in res]
        size = render.contact_sheet(thumbs, [f"#{r['id']} {r['match']} {r['score']:.2f}" for r in res], p,
                                    title=f"visual search: {query}"[:90])
        sheet = {"path": str(p), "tokens": render.token_estimate(*size)}
    return {"query": query, "results": res, "sheet": sheet, "note": query_note(query)}
