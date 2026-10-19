"""Source handling: probe, fetch, decode streams and single frames via ffmpeg."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


class MediaError(RuntimeError):
    pass


def require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise MediaError(f"`{binary}` not found on PATH. Install it (macOS: brew install {binary}).")
    return path


def is_url(source: str) -> bool:
    return bool(re.match(r"^https?://", source.strip()))


@dataclass
class Probe:
    path: str
    duration: float
    width: int  # display width (after rotation)
    height: int
    fps: float
    has_audio: bool
    codec: str
    title: str

    def to_dict(self) -> dict:
        return asdict(self)


def probe(path: str | Path) -> Probe:
    require("ffprobe")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise MediaError(f"ffprobe failed on {path}: {out.stderr.strip()[:400]}")
    data = json.loads(out.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"No video stream in {path}")
    w, h = int(video.get("width", 0)), int(video.get("height", 0))
    rotation = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(float(sd["rotation"]))
    if "rotate" in (video.get("tags") or {}):
        rotation = int(float(video["tags"]["rotate"]))
    if abs(rotation) % 180 == 90:
        w, h = h, w
    fps = _ratio(video.get("avg_frame_rate")) or _ratio(video.get("r_frame_rate")) or 30.0
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0.0)
    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    title = (data.get("format", {}).get("tags") or {}).get("title") or Path(path).stem
    return Probe(str(path), duration, w, h, round(fps, 3), has_audio, video.get("codec_name", "?"), title)


def _ratio(value: str | None) -> float:
    if not value or value in ("0/0", "0"):
        return 0.0
    if "/" in value:
        a, b = value.split("/")
        return float(a) / float(b) if float(b) else 0.0
    return float(value)


def file_fingerprint(path: str | Path) -> str:
    p = Path(path)
    st = p.stat()
    h = hashlib.sha1(f"{st.st_size}".encode())
    with p.open("rb") as fh:
        h.update(fh.read(1 << 20))
        if st.st_size > (2 << 20):
            fh.seek(-(1 << 20), 2)
            h.update(fh.read(1 << 20))
    return h.hexdigest()[:10]


def fetch(url: str, dest_dir: Path, max_height: int = 1080) -> tuple[Path, dict]:
    """Download a URL with yt-dlp (single item, captions if present). Returns (file, info)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(dest_dir.glob("source.*"))
    existing = [p for p in existing if p.suffix not in (".vtt", ".json", ".part")]
    info_path = dest_dir / "source.info.json"
    if existing and info_path.exists():
        return existing[0], json.loads(info_path.read_text())
    ytdlp = shutil.which("yt-dlp")
    cmd = [ytdlp] if ytdlp else ["python3", "-m", "yt_dlp"]
    cmd += [
        "--no-playlist", "--no-progress", "--quiet", "--no-warnings",
        "-f", f"bv*[height<={max_height}]+ba/b[height<={max_height}]/bv*+ba/b",
        "--merge-output-format", "mp4",
        "--write-info-json", "--write-subs", "--write-auto-subs",
        "--sub-langs", "en.*,en", "--sub-format", "vtt",
        "-o", str(dest_dir / "source.%(ext)s"),
        url,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    files = [p for p in dest_dir.glob("source.*") if p.suffix not in (".vtt", ".json", ".part")]
    if not files:
        raise MediaError(f"yt-dlp could not fetch {url}: {(res.stderr or res.stdout).strip()[-600:]}")
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    slim = {k: info.get(k) for k in ("title", "uploader", "upload_date", "webpage_url", "duration", "description")}
    if slim.get("description"):
        slim["description"] = slim["description"][:1500]
    return files[0], slim


def stream_frames(path: str | Path, fps: float, width: int, pix: str = "rgb24", start: float | None = None,
                  end: float | None = None):
    """Yield (t, frame ndarray) decoded at a fixed analysis fps and width."""
    require("ffmpeg")
    channels = 3 if pix == "rgb24" else 1
    pr = probe(path)
    height = int(round(pr.height * width / pr.width / 2) * 2)
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if end is not None:
        cmd += ["-t", f"{max(0.05, end - (start or 0)):.3f}"]
    cmd += ["-an", "-vf", f"fps={fps},scale={width}:{height}:flags=area", "-pix_fmt", pix, "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = width * height * channels
    i = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            arr = np.frombuffer(buf, np.uint8).reshape((height, width, channels) if channels == 3 else (height, width))
            yield (start or 0.0) + i / fps, arr
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()


def grab_frame(path: str | Path, t: float, max_edge: int | None = None) -> np.ndarray:
    """Exact-ish RGB frame at time t (accurate seek)."""
    require("ffmpeg")
    pr = probe(path)
    t = min(max(0.0, t), max(0.0, pr.duration - 0.05))
    w, h = pr.width, pr.height
    if max_edge and max(w, h) > max_edge:
        s = max_edge / max(w, h)
        w, h = int(round(w * s / 2) * 2), int(round(h * s / 2) * 2)
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1", "-an",
           "-vf", f"scale={w}:{h}:flags=lanczos", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    out = subprocess.run(cmd, capture_output=True)
    need = w * h * 3
    if len(out.stdout) < need:
        # seeking past the last decodable frame: back off a little
        if t > 0.2:
            return grab_frame(path, t - 0.2, max_edge)
        raise MediaError(f"Could not decode frame at {t:.2f}s: {out.stderr.decode()[:300]}")
    return np.frombuffer(out.stdout[:need], np.uint8).reshape((h, w, 3)).copy()


def audio_level(path: str | Path) -> float | None:
    """Mean volume in dB (None if no audio)."""
    res = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-vn", "-af", "volumedetect", "-f", "null", "-"],
                         capture_output=True, text=True)
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", res.stderr)
    return float(m.group(1)) if m else None


def fmt_t(t: float) -> str:
    t = max(0.0, t)
    m, s = divmod(t, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{int(h)}:{int(m):02d}:{s:04.1f}"
    return f"{int(m):02d}:{s:04.1f}"


def parse_t(value: str | float | int) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    v = value.strip()
    if ":" in v:
        parts = [float(p) for p in v.split(":")]
        total = 0.0
        for p in parts:
            total = total * 60 + p
        return total
    return float(v.rstrip("s"))
