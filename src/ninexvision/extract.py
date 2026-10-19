"""Per-image extraction: palettes, OCR, transcripts."""
from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def palette(rgb: np.ndarray, k: int = 5) -> list[dict]:
    """Dominant colours by k-means in Lab space, largest share first."""
    small = cv2.resize(rgb, (96, max(1, int(96 * rgb.shape[0] / rgb.shape[1]))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    k = max(1, min(k, len(np.unique(lab, axis=0))))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(lab, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    rgbc = cv2.cvtColor(centers.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB).reshape(-1, 3)
    out = []
    for i in np.argsort(-counts):
        r, g, b = (int(v) for v in rgbc[i])
        out.append({"hex": f"#{r:02X}{g:02X}{b:02X}", "share": round(float(counts[i] / counts.sum()), 3)})
    return out


def background_colour(rgb: np.ndarray) -> str:
    """Most common colour along the left/right edges (UI page background)."""
    h, w = rgb.shape[:2]
    edge = np.concatenate([rgb[int(h * .1):int(h * .9), :max(2, w // 40)].reshape(-1, 3),
                           rgb[int(h * .1):int(h * .9), -max(2, w // 40):].reshape(-1, 3)])
    q = (edge // 8) * 8
    vals, counts = np.unique(q, axis=0, return_counts=True)
    r, g, b = (int(v) + 4 for v in vals[counts.argmax()])
    return f"#{min(r,255):02X}{min(g,255):02X}{min(b,255):02X}"


# ---------------------------------------------------------------- OCR

def ocr_backend() -> str | None:
    if sys.platform == "darwin":
        try:
            import ocrmac  # noqa: F401
            return "apple-vision"
        except Exception:
            pass
    if shutil.which("tesseract"):
        return "tesseract"
    return None


OCR_TIMEOUT = 30.0   # seconds per image before the worker is killed and tesseract takes over


class _OcrWorker:
    """Apple Vision in a child process: it can deadlock on long batches, and a hang must never stall a watch."""

    def __init__(self) -> None:
        self.p = subprocess.Popen([sys.executable, "-m", "ninexvision._ocr_worker"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def run(self, rgb: np.ndarray, timeout: float) -> list[dict]:
        import select
        ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        data = png.tobytes()
        self.p.stdin.write(len(data).to_bytes(4, "big") + data)
        self.p.stdin.flush()
        ready, _, _ = select.select([self.p.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("ocr worker exited")
        return json.loads(line)

    def kill(self) -> None:
        self.p.kill()


_worker: _OcrWorker | None = None


def ocr(rgb: np.ndarray) -> list[dict]:
    """Text lines with normalised boxes (x, y, w, h from top-left), top-to-bottom."""
    global _worker
    backend = ocr_backend()
    if backend == "apple-vision":
        try:
            if _worker is None:
                _worker = _OcrWorker()
            return _worker.run(rgb, OCR_TIMEOUT)
        except Exception:
            if _worker is not None:
                _worker.kill()
                _worker = None
            backend = "tesseract" if shutil.which("tesseract") else None
    if backend is None:
        return []
    return _ocr_direct(rgb, backend)


def _ocr_direct(rgb: np.ndarray, backend: str) -> list[dict]:
    from PIL import Image
    im = Image.fromarray(rgb)
    lines: list[dict] = []
    if backend == "apple-vision":
        from ocrmac import ocrmac
        res = ocrmac.OCR(im, recognition_level="accurate", language_preference=None).recognize()
        for text, conf, (x, y, w, h) in res:
            # Vision boxes are bottom-left origin
            lines.append({"text": text, "conf": round(float(conf), 2), "box": [round(x, 4), round(1 - y - h, 4), round(w, 4), round(h, 4)]})
    else:
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            im.save(tmp.name)
            out = subprocess.run(["tesseract", tmp.name, "-", "--psm", "11", "tsv"], capture_output=True, text=True)
        rows: dict[tuple, list] = {}
        for row in out.stdout.splitlines()[1:]:
            p = row.split("\t")
            if len(p) < 12 or not p[11].strip() or float(p[10]) < 40:
                continue
            key = (p[2], p[3], p[4])
            rows.setdefault(key, []).append(p)
        W, H = im.size
        for words in rows.values():
            xs = [int(w[6]) for w in words]; ys = [int(w[7]) for w in words]
            xe = [int(w[6]) + int(w[8]) for w in words]; ye = [int(w[7]) + int(w[9]) for w in words]
            lines.append({"text": " ".join(w[11] for w in words), "conf": round(min(float(w[10]) for w in words) / 100, 2),
                          "box": [round(min(xs) / W, 4), round(min(ys) / H, 4), round((max(xe) - min(xs)) / W, 4), round((max(ye) - min(ys)) / H, 4)]})
    lines.sort(key=lambda l: (round(l["box"][1], 2), l["box"][0]))
    return lines


# ---------------------------------------------------------------- transcripts

_TS = re.compile(r"(\d+):(\d{2}):(\d{2})[.,](\d{3})\s+-->\s+(\d+):(\d{2}):(\d{2})[.,](\d{3})")


def parse_vtt(path: Path) -> list[dict]:
    segs: list[dict] = []
    cur = None
    for raw in path.read_text(errors="ignore").splitlines():
        m = _TS.search(raw)
        if m:
            g = [int(x) for x in m.groups()]
            cur = {"start": g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                   "end": g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000, "text": ""}
            segs.append(cur)
        elif cur is not None and raw.strip() and not raw.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            txt = html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()
            if txt:
                cur["text"] = (cur["text"] + " " + txt).strip()
    # YouTube auto-captions repeat rolling lines; keep only new text
    clean: list[dict] = []
    prev = ""
    for s in segs:
        t = s["text"]
        if not t:
            continue
        if prev and t.startswith(prev):
            t = t[len(prev):].strip()
        if t and (not clean or t != clean[-1]["text"]):
            clean.append({"start": round(s["start"], 2), "end": round(s["end"], 2), "text": t})
        prev = s["text"]
    return clean


def _whisper_cpp_model(duration: float) -> Path | None:
    env = os.environ.get("NINEXVISION_WHISPER_MODEL")
    if env and Path(env).exists():
        return Path(env)
    dirs = [Path("/opt/homebrew/share/whisper-cpp"), Path("/usr/local/share/whisper-cpp"),
            Path.home() / ".cache" / "whisper.cpp", Path.home() / ".9xvision" / "models"]
    prefs = (["ggml-medium.en.bin", "ggml-medium.bin"] if duration <= 600 else []) + \
            ["ggml-small.en.bin", "ggml-small.bin", "ggml-base.en.bin", "ggml-base.bin",
             "ggml-medium.en.bin", "ggml-large-v3-turbo.bin"]
    for name in prefs:
        for d in dirs:
            if (d / name).exists():
                return d / name
    return None


def whisper_transcribe(media: Path, workdir: Path, duration: float = 0.0,
                       window: tuple[float, float] | None = None) -> tuple[list[dict], str] | None:
    """Local Whisper: whisper.cpp first (fast, Metal), then the openai-whisper CLI. Returns (segments, backend).
    window = (start, end) in file seconds: transcribe only that part; segment times stay file times."""
    cli = shutil.which("whisper-cli") or shutil.which("whisper-cpp")
    model = _whisper_cpp_model(duration) if cli else None
    w0 = window[0] if window else 0.0
    cut = ["-ss", f"{window[0]:.3f}", "-t", f"{window[1] - window[0]:.3f}"] if window else []
    if cli and model:
        wav = workdir / "audio16k.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", *cut, "-i", str(media), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                       capture_output=True)
        base = workdir / "whisper"
        subprocess.run([cli, "-m", str(model), "-f", str(wav), "-l", "auto" if ".en" not in model.name else "en",
                        "-oj", "-of", str(base), "-np"], capture_output=True, text=True)
        wav.unlink(missing_ok=True)
        js = base.with_suffix(".json")
        if js.exists():
            data = json.loads(js.read_text(errors="ignore"))
            segs = []
            for s in data.get("transcription", []):
                txt = s.get("text", "").strip()
                if txt and txt not in ("[Music]", "[MUSIC]", "(music)") and txt.strip("♪♫ ."):
                    segs.append({"start": round(w0 + s["offsets"]["from"] / 1000, 2),
                                 "end": round(w0 + s["offsets"]["to"] / 1000, 2), "text": txt})
            js.unlink(missing_ok=True)
            return segs, f"whisper.cpp:{model.stem.replace('ggml-', '')}"
    if shutil.which("whisper"):
        model = "small"
        src = media
        if window:
            src = workdir / "audio_window.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", *cut, "-i", str(media), "-vn", "-ac", "1", "-ar", "16000", str(src)],
                           capture_output=True)
        out = subprocess.run(["whisper", str(src), "--model", model, "--output_format", "json",
                              "--output_dir", str(workdir), "--verbose", "False", "--fp16", "False"],
                             capture_output=True, text=True)
        js = workdir / (src.stem + ".json")
        if window:
            src.unlink(missing_ok=True)
        if js.exists():
            data = json.loads(js.read_text())
            segs = [{"start": round(w0 + s["start"], 2), "end": round(w0 + s["end"], 2), "text": s["text"].strip()}
                    for s in data.get("segments", [])]
            return segs, f"openai-whisper:{model}"
        return None
    return None


def segments_between(segs: list[dict], start: float, end: float) -> str:
    return " ".join(s["text"] for s in segs if s["end"] > start and s["start"] < end).strip()
