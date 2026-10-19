"""Source handling: probe, fetch, decode streams and single frames via ffmpeg."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
import re
import shutil
import subprocess
import sys
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
    offset: float = 0.0   # time in the original video at this file's t=0 (a downloaded section)

    @property
    def end(self) -> float:
        return self.offset + self.duration

    def to_dict(self) -> dict:
        return asdict(self)


def probe(path: str | Path) -> Probe:
    p = Path(path)
    st = p.stat() if p.exists() else None
    return _probe((str(path), st.st_mtime_ns if st else 0, st.st_size if st else 0))


@lru_cache(maxsize=64)
def _probe(key: tuple) -> Probe:
    path = key[0]
    require("ffprobe")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise MediaError(f"ffprobe failed on {path}: {out.stderr.strip()[:400]}")
    data = json.loads(out.stdout)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    fmt = data.get("format", {}).get("format_name", "")
    if video is None or fmt in ("tty", "txt") or video.get("codec_name") in ("ansi", "bintext", "xbin", "idf"):
        raise MediaError(f"{path} is not a video (ffprobe sees {fmt or 'no video stream'})")
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
    side = Path(path).with_suffix(".offset")
    offset = float(side.read_text().strip()) if side.exists() else 0.0
    return Probe(str(path), duration, w, h, round(fps, 3), has_audio, video.get("codec_name", "?"), title, offset)


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


def _ytdlp_commands() -> list[list[str]]:
    """Every yt-dlp available (the one installed with the kit, and one on PATH), newest first:
    sites break old extractors, so a stale copy should not win."""
    found = {}
    for cmd in ([shutil.which("yt-dlp")] if shutil.which("yt-dlp") else [], [sys.executable, "-m", "yt_dlp"]):
        if not cmd:
            continue
        try:
            v = subprocess.run(cmd + ["--version"], capture_output=True, text=True, timeout=60).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            v = ""
        if v and v not in found:
            found[v] = cmd
    if not found:
        raise MediaError("yt-dlp is missing. Reinstall the kit (`uv tool install --force git+https://github.com/ASGARD-STUDIOS/9xvision`).")
    return [found[v] for v in sorted(found, reverse=True)]


def _ytdlp_hint(err: str) -> str:
    """Turn a yt-dlp failure into the next step for the agent."""
    e = err.lower()
    if any(k in e for k in ("sign in", "log in", "login", "cookies", "private video", "members-only", "confirm your age",
                            "rate-limit", "not a bot", "authentication")):
        return ("the site wants a logged-in session. Retry with cookies_from_browser='chrome' (or safari, firefox, edge, brave): "
                "yt-dlp then uses that browser's cookies on this machine. Only do this if the user agrees.")
    if "unsupported url" in e:
        return "yt-dlp has no extractor for this site. Download the video another way and pass the file path."
    if "404" in e or "does not exist" in e or "video unavailable" in e or "has been removed" in e:
        return "the video is unavailable (404, removed or region-locked). Check the URL."
    if any(k in e for k in ("unable to extract", "nsig", "signature", "player response", "http error 403")):
        return ("yt-dlp is probably out of date for this site. Update it: `uv tool upgrade ninexvision` "
                "(or `brew upgrade yt-dlp` / `pip install -U yt-dlp`), then retry.")
    if any(k in e for k in ("timed out", "connection", "network", "name or service", "getaddrinfo")):
        return "network error. Retry; if it persists, check the connection or proxy."
    return "see the yt-dlp error above."


def _run_ytdlp(args: list[str], url: str, cookies_from_browser: str | None) -> subprocess.CompletedProcess:
    extra = ["--cookies-from-browser", cookies_from_browser] if cookies_from_browser else []
    last = None
    for cmd in _ytdlp_commands():
        last = subprocess.run(cmd + ["--no-warnings", "--socket-timeout", "30", "--retries", "5",
                                     "--fragment-retries", "5", *extra, *args, url], capture_output=True, text=True)
        if last.returncode == 0:
            return last
    return last


def fetch(url: str, dest_dir: Path, max_height: int = 1080, section: tuple[float, float] | None = None,
          cookies_from_browser: str | None = None) -> tuple[Path, dict]:
    """Download a URL with yt-dlp: one video (or only `section`, in seconds), plus captions if present.
    Returns (file, info). A section download gets a `.offset` sidecar so all times stay in full-video time."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    info_path = dest_dir / "source.info.json"
    existing = [p for p in sorted(dest_dir.glob("source.*")) if p.suffix not in (".vtt", ".json", ".part", ".offset", ".ytdl")]
    if existing and info_path.exists():
        return existing[0], _slim(json.loads(info_path.read_text()))

    res = _run_ytdlp(["-J", "--flat-playlist", "--no-playlist"], url, cookies_from_browser)
    if res.returncode != 0:
        _rmdir(dest_dir)
        err = (res.stderr or res.stdout).strip()[-600:]
        raise MediaError(f"yt-dlp could not read {url}: {err}\nNext step: {_ytdlp_hint(err)}")
    info = json.loads(res.stdout)
    if info.get("_type") == "playlist":
        _rmdir(dest_dir)
        entries = [e.get("url") or e.get("id") for e in (info.get("entries") or [])][:5]
        raise MediaError(f"{url} is a playlist ({info.get('playlist_count') or len(info.get('entries') or [])} videos). "
                         f"Watch one video at a time, e.g. {', '.join(map(str, entries))}")
    if info.get("is_live") and not section:
        _rmdir(dest_dir)
        raise MediaError(f"{url} is a live stream. Pass start and end to capture a section of it, or wait for the replay.")
    dur = float(info.get("duration") or 0)
    if not section and dur > 3600 and max_height > 720:
        max_height = 720    # long video: smaller download, same analysis quality
    lang = (info.get("language") or "").split("-")[0]
    subs = "en.*,en" + (f",{lang}.*,{lang}" if lang and lang != "en" else "")
    sub_args = ["--write-subs", "--write-auto-subs", "--sub-langs", subs, "--sub-format", "vtt"]
    # -S res:N limits the SHORT edge, so portrait phone videos still come down at 1080x1920
    args = ["--no-playlist", "--no-progress", "--quiet", "-f", "bv*+ba/b", "-S", f"res:{max_height}",
            "--merge-output-format", "mp4", "--write-info-json", "-o", str(dest_dir / "source.%(ext)s")]
    if section:
        a, b = section
        args += ["--download-sections", f"*{a:.3f}-{b:.3f}", "--force-keyframes-at-cuts"]
    res = _run_ytdlp(args + sub_args, url, cookies_from_browser)
    files = [p for p in dest_dir.glob("source.*") if p.suffix not in (".vtt", ".json", ".part", ".offset", ".ytdl")]
    if not files and "subtitle" in (res.stderr or "").lower():   # captions are a bonus: never let them sink the video
        res = _run_ytdlp(args, url, cookies_from_browser)
        files = [p for p in dest_dir.glob("source.*") if p.suffix not in (".vtt", ".json", ".part", ".offset", ".ytdl")]
    if not files:
        for p in dest_dir.glob("source.*"):
            p.unlink(missing_ok=True)
        _rmdir(dest_dir)
        err = (res.stderr or res.stdout).strip()[-600:]
        raise MediaError(f"yt-dlp could not download {url}: {err}\nNext step: {_ytdlp_hint(err)}")
    if section:
        files[0].with_suffix(".offset").write_text(f"{section[0]:.3f}")
    if not info_path.exists():
        info_path.write_text(json.dumps(info))
    return files[0], _slim(json.loads(info_path.read_text()))


def _rmdir(d: Path) -> None:
    try:
        d.rmdir()
    except OSError:
        pass


def _slim(info: dict) -> dict:
    slim = {k: info.get(k) for k in ("title", "uploader", "upload_date", "webpage_url", "duration", "description", "language")}
    if slim.get("description"):
        slim["description"] = slim["description"][:1500]
    return slim


def stream_frames(path: str | Path, fps: float, width: int, pix: str = "rgb24", start: float | None = None,
                  end: float | None = None):
    """Yield (t, frame ndarray) decoded at a fixed analysis fps and width. start/end and t are in
    original-video time (a downloaded section's offset is applied)."""
    require("ffmpeg")
    channels = 3 if pix == "rgb24" else 1
    pr = probe(path)
    height = int(round(pr.height * width / pr.width / 2) * 2)
    fs = max(0.0, (start if start is not None else pr.offset) - pr.offset)
    cmd = ["ffmpeg", "-v", "error"]
    if fs:
        cmd += ["-ss", f"{fs:.3f}"]
    cmd += ["-i", str(path)]
    if end is not None:
        cmd += ["-t", f"{max(0.05, end - pr.offset - fs):.3f}"]
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
            yield pr.offset + fs + i / fps, arr
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()


def grab_frame(path: str | Path, t: float, max_edge: int | None = None) -> np.ndarray:
    """Exact-ish RGB frame at time t (original-video time; accurate seek)."""
    require("ffmpeg")
    pr = probe(path)
    t = min(max(0.0, t - pr.offset), max(0.0, pr.duration - 0.05))
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
            return grab_frame(path, pr.offset + t - 0.2, max_edge)
        raise MediaError(f"Could not decode frame at {t:.2f}s: {out.stderr.decode()[:300]}")
    return np.frombuffer(out.stdout[:need], np.uint8).reshape((h, w, 3)).copy()


def grab_frames(path: str | Path, times: list[float], max_edge: int | None = None, workers: int = 8):
    """grab_frame for many times in parallel (each is an independent fast seek). Yields frames in order."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        yield from ex.map(lambda t: grab_frame(path, t, max_edge), times)


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
