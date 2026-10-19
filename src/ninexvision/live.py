"""Live perception: a camera, microphone, screen, stream URL or file played in real time becomes a stream of
features and events (motion, cuts, colour, sound, beats, voice, face, hands, body, affect, visual triggers) sent to
OSC / WebSocket / JSON lines, with the last minutes kept in memory so an agent can turn them into a normal watch.

Nothing is written to disk unless asked for (snapshot, frame, --jsonl)."""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import extract, media
from .audio import SR
from .live_audio import StreamingAudio

CHUNK = 1024                      # audio samples read at a time (~46 ms)


def home() -> Path:
    return Path(os.environ.get("NINEXVISION_HOME", Path.home() / ".9xvision"))


def session_file() -> Path:
    return home() / "live" / "session.json"


# ---------------------------------------------------------------- devices and sources


def devices() -> dict:
    """Cameras, microphones and screens ffmpeg can open here."""
    out: dict = {"cameras": [], "microphones": [], "screens": []}
    if sys.platform == "darwin":
        r = subprocess.run([media.require("ffmpeg"), "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                           capture_output=True, text=True)
        kind = None
        for line in r.stderr.splitlines():
            if "video devices" in line:
                kind = "video"
            elif "audio devices" in line:
                kind = "audio"
            m = re.search(r"\[(\d+)\] (.+)$", line)
            if m and kind:
                i, name = int(m.group(1)), m.group(2).strip()
                if kind == "audio":
                    out["microphones"].append({"index": i, "name": name})
                elif name.startswith("Capture screen"):
                    out["screens"].append({"index": i, "name": name})
                else:
                    out["cameras"].append({"index": i, "name": name})
    elif sys.platform.startswith("linux"):
        out["cameras"] = [{"index": int(p.name[5:]), "name": str(p)} for p in sorted(Path("/dev").glob("video*"))
                          if p.name[5:].isdigit()]
        out["microphones"] = [{"index": 0, "name": "default"}]
        if os.environ.get("DISPLAY"):
            out["screens"] = [{"index": 0, "name": os.environ["DISPLAY"]}]
    return out


def _pick(items: list[dict], sel: str | None, prefer: tuple[str, ...] = ()) -> dict | None:
    if not items:
        return None
    if sel in (None, "", "default", "auto"):
        for p in prefer:
            for it in items:
                if p.lower() in it["name"].lower():
                    return it
        return items[0]
    if sel.isdigit():
        return next((it for it in items if it["index"] == int(sel)), None)
    return next((it for it in items if sel.lower() in it["name"].lower()), None)


def build_input(source: str, mic: str | None, fps: float, loop: bool = False, log=print) -> tuple[list[str], bool, dict]:
    """ffmpeg input arguments for a source, whether audio is present, and a description."""
    kind, _, sel = source.partition(":")
    mac = sys.platform == "darwin"
    if kind in ("camera", "cam", "screen"):
        dev = devices()
        if kind == "screen":
            it = _pick(dev["screens"], sel or "0") if sel.isdigit() or not sel else _pick(dev["screens"], sel)
            if it is None and dev["screens"]:
                it = next((s for s in dev["screens"] if s["name"].endswith(sel or "0")), None)
        else:
            it = _pick(dev["cameras"], sel, prefer=("MacBook", "FaceTime", "Built-in"))
        if it is None:
            raise ValueError(f"no {kind} matching {sel!r}; available: {[d['name'] for d in dev['cameras' if kind != 'screen' else 'screens']]}")
        m = None if mic in ("none", "off") else _pick(dev["microphones"], mic, prefer=("MacBook", "Built-in", "Microphone"))
        if mac:
            spec = f"{it['index']}:{m['index'] if m else 'none'}"
            args = ["-f", "avfoundation", "-framerate", "30", "-pixel_format", "nv12"]
            if kind == "screen":
                args += ["-capture_cursor", "1"]
            return args + ["-i", spec], m is not None, {"kind": kind, "device": it["name"], "mic": m["name"] if m else None}
        if kind == "screen":
            args = ["-f", "x11grab", "-framerate", "30", "-i", it["name"]]
        else:
            args = ["-f", "v4l2", "-i", it["name"]]
        if m:
            args += ["-f", "pulse" if shutil.which("pactl") else "alsa", "-i", "default"]
        return args, m is not None, {"kind": kind, "device": it["name"], "mic": "default" if m else None}
    if re.match(r"^(rtsp|rtmp|rtmps|srt|udp|tcp)://", source):
        return ["-fflags", "nobuffer", "-flags", "low_delay", "-i", source], True, {"kind": "stream", "url": source}
    if media.is_url(source):
        cmd = media._ytdlp_commands()[0] + ["--no-warnings", "--no-playlist", "-f", "b/bv*+ba", "-O", "is_live", "-O", "urls", source]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        lines = [x for x in r.stdout.splitlines() if x.strip()]
        if r.returncode or len(lines) < 2:
            raise RuntimeError(f"couldn't open {source}: {media._ytdlp_hint(r.stderr) or r.stderr.strip()[-300:]}")
        live = lines[0] == "True"
        urls = lines[1:3]
        args: list[str] = []
        for u in urls:
            args += ([] if live else ["-re"]) + ["-i", u]
        return args, True, {"kind": "live stream" if live else "url (played in real time)", "url": source}
    p = Path(source).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"{source}: not a file, URL, 'camera[:N|name]' or 'screen[:N]' (see: 9xvision live devices)")
    pr = media.probe(p)
    return (["-re"] + (["-stream_loop", "-1"] if loop else []) + ["-i", str(p)]), pr.has_audio, {"kind": "file", "path": str(p)}


# ---------------------------------------------------------------- capture


def _read_exact(f, n: int) -> bytes | None:
    """n bytes from an unbuffered pipe (a single read may return less), or None at the end."""
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = f.readinto(view[got:])
        if not k:
            return None
        got += k
    return bytes(buf)


class Capture:
    """One ffmpeg process: scaled RGB frames on stdout, mono float audio on a second pipe."""

    def __init__(self, source: str, mic: str | None = "auto", fps: float = 15, width: int = 640, loop: bool = False):
        args, has_audio, self.info = build_input(source, mic, fps, loop)
        self.fps, self.has_audio = fps, has_audio
        self.ar, self.aw = os.pipe() if has_audio else (None, None)
        cmd = [media.require("ffmpeg"), "-hide_banner", "-nostdin", "-loglevel", "info", *args,
               "-map", "0:v:0", "-vf", f"fps={fps},scale={width}:-2", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
        if has_audio:
            amap = ("1:a:0" if args.count("-i") > 1 and "avfoundation" not in args else "0:a:0") + "?"
            cmd += ["-map", amap, "-ac", "1", "-ar", str(SR), "-f", "f32le", f"pipe:{self.aw}"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                     pass_fds=(self.aw,) if has_audio else (), bufsize=0)
        if has_audio:
            os.close(self.aw)
        self.size = None
        self.err: deque = deque(maxlen=40)
        self._size_ready = threading.Event()
        threading.Thread(target=self._stderr, daemon=True).start()
        if not self._size_ready.wait(20) or self.size is None:
            self.close()
            raise RuntimeError("ffmpeg could not open the source:\n  " + "\n  ".join(list(self.err)[-6:]) +
                               ("\n(macOS: allow camera/microphone/screen access for this terminal in System Settings > "
                                "Privacy & Security)" if sys.platform == "darwin" and self.info["kind"] in ("camera", "screen") else ""))

    def _stderr(self) -> None:
        out_section = False
        for raw in iter(self.proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            self.err.append(line)
            if line.startswith("Output #0"):
                out_section = True
            m = re.search(r"Video: rawvideo.*?, (\d+)x(\d+)", line) if out_section else None
            if m and self.size is None:
                self.size = (int(m.group(1)), int(m.group(2)))
                self._size_ready.set()
        self._size_ready.set()

    def frames(self):
        w, h = self.size
        n = w * h * 3
        while True:
            buf = _read_exact(self.proc.stdout, n)
            if buf is None:
                return
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)

    def audio_chunks(self):
        if not self.has_audio:
            return
        with os.fdopen(self.ar, "rb", buffering=0) as f:
            while True:
                b = _read_exact(f, CHUNK * 4)
                if b is None:
                    return
                yield np.frombuffer(b, np.float32)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# ---------------------------------------------------------------- visual triggers


class WatchFor:
    """Visual triggers by description ("someone waving"), scored with SigLIP on the newest frame a few times a second.
    A trigger turns on after two scores >= threshold and off after two below a third of it."""

    def __init__(self, queries: list[str], threshold: float = 0.1, every: float = 0.5):
        self.queries, self.threshold, self.every = list(queries), threshold, every
        self.scores: dict[str, float] = {}
        self.on: dict[str, bool] = {}
        self.runs: dict[str, int] = {}
        self.events: list[dict] = []
        self.ms = 0.0
        self.error: str | None = None
        self._frame = None
        self._lock = threading.Lock()
        self._text: dict[str, np.ndarray] = {}
        self._alive = True
        threading.Thread(target=self._loop, daemon=True).start()

    def set(self, queries: list[str]) -> None:
        with self._lock:
            self.queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
            for d in (self.scores, self.on, self.runs):
                for k in list(d):
                    if k not in self.queries:
                        d.pop(k)

    def offer(self, rgb: np.ndarray, t: float) -> None:
        self._frame = (rgb, t)

    def _loop(self) -> None:
        from .semantic import LOGIT_BIAS, LOGIT_SCALE, _Model, available
        model = None
        while self._alive:
            time.sleep(self.every)
            with self._lock:
                qs = list(self.queries)
            if not qs or self._frame is None:
                continue
            if model is None:
                if not available():      # no surprise 210 MB download in the middle of a session
                    self.error = "visual triggers need the visual search model: run 9xvision setup"
                    continue
                try:
                    model = _Model.get()
                except Exception as e:
                    self.error = f"visual triggers unavailable: {e}"
                    continue
                self.error = None
            rgb, t = self._frame
            t0 = time.perf_counter()
            e = model.images([rgb])[0]
            new = [q for q in qs if q not in self._text]
            if new:
                for q, v in zip(new, model.texts(new)):
                    self._text[q] = v
            for q in qs:
                s = float(1 / (1 + np.exp(-(LOGIT_SCALE * float(e @ self._text[q]) + LOGIT_BIAS))))
                self.scores[q] = round(s, 4)
                hit = s >= self.threshold if not self.on.get(q) else s >= self.threshold / 3
                if hit != self.on.get(q, False):
                    self.runs[q] = self.runs.get(q, 0) + 1
                    if self.runs[q] >= 2:
                        self.on[q] = hit
                        self.runs[q] = 0
                        self.events.append({"kind": "seen_start" if hit else "seen_end", "t": round(t, 3), "what": q,
                                            "score": round(s, 3)})
                else:
                    self.runs[q] = 0
            self.ms = (time.perf_counter() - t0) * 1000

    def close(self) -> None:
        self._alive = False


# ---------------------------------------------------------------- the engine


class Live:
    def __init__(self, source: str = "camera", mic: str | None = "auto", fps: float = 15, width: int = 640,
                 face: bool = True, hands: bool = False, pose: bool = False, mirrored: bool = False,
                 watch_for: list[str] | None = None, watch_threshold: float = 0.1, buffer_s: float = 120,
                 osc: str | None = None, ws_port: int | None = 8899, host: str = "127.0.0.1",
                 allow_origins: list[str] | None = None, jsonl: str | None = None, state_every: int = 1,
                 loop: bool = False, beat_lead: float = 0.0, log=print):
        self.log = log
        self.source = source
        self.cap = Capture(source, mic, fps, width, loop)
        self.fps = fps
        self.t0 = time.monotonic()
        self.started = datetime.now()
        self.dir = home() / "live" / self.started.strftime("%Y%m%d-%H%M%S")
        self.token = secrets.token_urlsafe(16)
        self.audio = StreamingAudio(lead=beat_lead) if self.cap.has_audio else None
        self.audio_t0 = None
        self.origin = 0.0
        self.first_frame = threading.Event()
        self.audio_ring: deque = deque()          # (t_start, samples)
        from .affect import Affect
        self.affect = Affect()
        self.tracker = None
        if face or hands or pose:
            from . import body
            try:
                self.tracker = body.Tracker(face=face, hands=hands, pose=pose, mirrored=mirrored, log=log)
            except Exception as e:      # face tracking is optional: keep the rest running
                log(f"face/hand/body tracking unavailable ({e}); continuing without it")
        self.watch = WatchFor(watch_for or [], watch_threshold) if watch_for is not None else WatchFor([], watch_threshold)
        self.ring: deque = deque(maxlen=int(buffer_s * fps))         # (t, jpeg)
        self.states: deque = deque(maxlen=int(buffer_s * fps))
        self.events: deque = deque(maxlen=2000)
        self.buffer_s = buffer_s
        self.latest: dict = {}
        self.latest_rgb = None
        self.stats = {"fps": 0.0, "proc_ms": 0.0}
        self.stop_flag = threading.Event()
        self.state_every = max(1, state_every)
        self.outs = []
        from .bus import JsonlOut, OscOut, WsServer
        if osc:
            self.outs.append(OscOut(osc))
        if jsonl:
            self.outs.append(JsonlOut(jsonl))
        self.ws = None
        if ws_port:
            self.ws = WsServer(host, ws_port, self.token, self.control, self._jpeg_latest, allow_origins)
            self.outs.append(self.ws)
        self.host, self.port = host, ws_port
        self._grey_prev = None
        self._diffs: deque = deque(maxlen=int(2 * fps))
        self._palette_t = -1.0
        self._palette = []
        self._win = None

    # -- publishing
    def _publish(self, msg: dict) -> None:
        for o in self.outs:
            o.send(msg)

    def _event(self, ev: dict) -> None:
        ev = {"type": "event", **ev}
        self.events.append(ev)
        self._publish(ev)

    # -- per-frame vision
    def _video(self, rgb: np.ndarray, t: float) -> dict:
        h, w = rgb.shape[:2]
        sw = 160
        grey = cv2.cvtColor(cv2.resize(rgb, (sw, max(2, int(sw * h / w))), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        v: dict = {}
        if self._grey_prev is not None and self._grey_prev.shape == grey.shape:
            d = float(cv2.absdiff(grey, self._grey_prev).mean())
            v["motion"] = round(d / 255, 4)
            med = float(np.median(self._diffs)) if len(self._diffs) >= 5 else d
            if d > max(25.0, 4 * med) and len(self._diffs) >= 5:
                self._event({"kind": "cut", "t": round(t, 3), "change": round(d, 1)})
            self._diffs.append(d)
            if self._win is None or self._win.shape != grey.shape[::-1]:
                self._win = cv2.createHanningWindow((grey.shape[1], grey.shape[0]), cv2.CV_32F)
            (dx, dy), r = cv2.phaseCorrelate(self._grey_prev.astype(np.float32), grey.astype(np.float32), self._win)
            if r > 0.1:
                v["pan"] = [round(dx / sw * self.fps, 3), round(dy / sw * self.fps, 3)]   # frame widths per second
        else:
            v["motion"] = 0.0
        self._grey_prev = grey
        small = cv2.resize(rgb, (64, max(2, int(64 * h / w))), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
        s = hsv[:, 1] / 255
        ang = hsv[:, 0] * 2 * np.pi / 180
        hue = float(np.degrees(np.arctan2((np.sin(ang) * s).sum(), (np.cos(ang) * s).sum())) % 360) if s.sum() > 1 else None
        f = small.reshape(-1, 3).astype(np.float32)
        luma = (0.2126 * f[:, 0] + 0.7152 * f[:, 1] + 0.0722 * f[:, 2]) / 255
        v.update(brightness=round(float(luma.mean()), 3), contrast=round(float(luma.std()), 3),
                 saturation=round(float(s.mean()), 3), warmth=round(float((f[:, 0] - f[:, 2]).mean() / 255), 3),
                 hue=None if hue is None else round(hue))
        if t - self._palette_t >= 1.0:
            self._palette = [p["hex"] for p in extract.palette(small, k=4)]
            self._palette_t = t
        v["palette"] = self._palette
        return v

    # -- audio thread
    def _audio_loop(self) -> None:
        # audio and video share one content clock: both streams start at the same input timestamp in ffmpeg, so
        # sample count / rate lines up with frame index / fps (arrival times through pipes jitter by 100s of ms)
        self.first_frame.wait()
        self.audio_t0 = 0.0
        for chunk in self.cap.audio_chunks():
            now = time.monotonic() - self.t0 - self.origin
            self.audio.feed(chunk)
            self.audio_ring.append((self.audio_t0 + (self.audio.n * 512 + len(self.audio.pending) - len(chunk)) / SR, chunk))
            while self.audio_ring and self.audio_ring[0][0] < now - self.buffer_s:
                self.audio_ring.popleft()
            evs, self.audio.events = self.audio.events, []
            for e in evs:
                e["t"] = round(e["t"] + self.audio_t0, 3)
                if "at" in e:
                    e["at"] = round(e["at"] + self.audio_t0, 3)
                self._event(e)

    # -- main loop
    def run(self, duration: float | None = None) -> None:
        self._write_session()
        if self.audio:
            threading.Thread(target=self._audio_loop, daemon=True).start()
        n, tick = 0, time.monotonic()
        idx = 0
        try:
            for rgb in self.cap.frames():
                wall = time.monotonic() - self.t0
                if idx == 0:
                    self.origin = wall                     # session time 0 = the first frame
                    self.first_frame.set()
                t = idx / self.fps                         # content time (see _audio_loop)
                idx += 1
                self.stats["lag_ms"] = round(max(0.0, wall - self.origin - t) * 1000)   # how far processing trails the source
                p0 = time.perf_counter()
                video = self._video(rgb, t)
                body = None
                if self.tracker:
                    self.tracker.submit(rgb, t)
                    body = self.tracker.latest
                self.watch.offer(rgb, t)
                aud = dict(self.audio.state) if self.audio else None
                aff = self.affect.update(t, aud, video, body)
                for e in self.affect.events:
                    self._event(e)
                self.affect.events = []
                for e in self.watch.events:
                    self._event(e)
                self.watch.events = []
                faces = (body or {}).get("faces") or []
                state = {"type": "state", "t": round(t, 3), "video": video, "audio": aud,
                         "face": None if not faces else {**aff.get("face", {}), "count": len(faces),
                                                          "yaw": faces[0].get("yaw"), "pitch": faces[0].get("pitch"),
                                                          "roll": faces[0].get("roll"), "box": faces[0].get("box"),
                                                          "blend": faces[0].get("blend")},
                         "hands": (body or {}).get("hands") if self.tracker else None,
                         "body": (body or {}).get("pose"),
                         "affect": aff["affect"],
                         "watch": dict(self.watch.scores) or None}
                self.latest, self.latest_rgb = state, rgb
                self.states.append(_compact(state))
                ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    self.ring.append((t, jpg.tobytes()))
                n += 1
                if n % self.state_every == 0:
                    self._publish(state)
                ms = (time.perf_counter() - p0) * 1000
                self.stats["proc_ms"] = round(0.9 * self.stats["proc_ms"] + 0.1 * ms, 1)
                if time.monotonic() - tick >= 1.0:
                    self.stats["fps"] = round(n / (time.monotonic() - tick), 1)
                    self.stats["body_ms"] = round(self.tracker.ms, 1) if self.tracker else None
                    self.stats["watch_ms"] = round(self.watch.ms, 1) if self.watch.queries else None
                    n, tick = 0, time.monotonic()
                if self.stop_flag.is_set() or (duration and t >= duration):
                    break
        finally:
            self.close()

    def close(self) -> None:
        self.stop_flag.set()
        self.cap.close()
        if self.tracker:
            self.tracker.close()
        self.watch.close()
        if self.ws:
            self.ws.close()
        try:
            sf = session_file()
            if sf.exists() and json.loads(sf.read_text()).get("pid") == os.getpid():
                sf.unlink()
        except (OSError, json.JSONDecodeError):
            pass

    def _write_session(self) -> None:
        sf = session_file()
        sf.parent.mkdir(parents=True, exist_ok=True)
        info = {"pid": os.getpid(), "host": self.host, "port": self.port, "token": self.token, "source": self.source,
                "capture": self.cap.info, "started": self.started.isoformat(timespec="seconds"), "dir": str(self.dir)}
        fd = os.open(sf, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(info, f)

    def monitor_url(self) -> str | None:
        return f"http://{'127.0.0.1' if self.host in ('0.0.0.0', '::') else self.host}:{self.port}/?k={self.token}" if self.ws else None

    # -- control (WebSocket requests, CLI and MCP)
    def _jpeg_latest(self) -> bytes | None:
        return self.ring[-1][1] if self.ring else None

    def control(self, req: dict) -> dict:
        cmd = req.get("cmd")
        if self.stop_flag.is_set():
            return {"error": "the live session is stopping"}
        if cmd == "status":
            return {"source": self.source, "capture": self.cap.info, "stats": self.stats, "state": self.latest,
                    "events": [e for e in list(self.events)[-60:] if e["kind"] not in ("onset",)][-40:],
                    "watch_for": self.watch.queries, "buffer_s": self.buffer_s,
                    **({"watch_for_error": self.watch.error} if self.watch.error else {})}
        if cmd == "frame":
            if self.latest_rgb is None:
                return {"error": "no frame yet"}
            self.dir.mkdir(parents=True, exist_ok=True)
            t = self.latest.get("t", 0.0)
            p = self.dir / f"frame-{t:09.2f}.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(self.latest_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            return {"path": str(p), "t": t, "size": list(self.latest_rgb.shape[1::-1])}
        if cmd == "snapshot":
            return snapshot(self, float(req.get("seconds") or 30), watch=req.get("watch", True))
        if cmd == "watch_for":
            q = list(self.watch.queries)
            if "set" in req:
                q = list(req["set"])
            q += [x for x in req.get("add", []) if x not in q]
            q = [x for x in q if x not in req.get("remove", [])]
            self.watch.set(q)
            return {"watch_for": self.watch.queries}
        if cmd == "stop":
            self.stop_flag.set()
            return {"stopping": True}
        return {"error": f"unknown cmd {cmd!r}; use status | frame | snapshot | watch_for | stop"}


def _compact(state: dict) -> dict:
    """What the buffer keeps per frame (for snapshots): no blendshapes or palettes."""
    a = state.get("audio") or {}
    f = state.get("face") or {}
    return {"t": state["t"], "energy": state["affect"]["energy"], "arousal": state["affect"]["arousal"],
            "valence": state["affect"]["expression_valence"], "engagement": state["affect"]["engagement"],
            "motion": state["video"].get("motion"), "db": a.get("db"), "speaking": a.get("speaking"),
            "smile": f.get("smile"), "faces": f.get("count", 0)}


# ---------------------------------------------------------------- snapshot -> watch


def snapshot(live: Live, seconds: float, watch: bool = True) -> dict:
    """The last `seconds` of the buffer as an mp4 (plus live.json), then a normal watch of it."""
    now = live.latest.get("t", 0.0)
    a = max(0.0, now - min(seconds, live.buffer_s))
    frames = [(t, j) for t, j in list(live.ring) if t >= a]
    if len(frames) < 2:
        return {"error": "nothing buffered yet"}
    live.dir.mkdir(parents=True, exist_ok=True)
    stamp = live.started.timestamp() + live.origin + a
    name = datetime.fromtimestamp(stamp).strftime("snap-%Y%m%d-%H%M%S")
    mp4 = live.dir / f"{name}.mp4"
    fps = live.fps
    grid = np.arange(frames[0][0], frames[-1][0], 1 / fps)
    times = np.array([t for t, _ in frames])
    # frames to a temporary MJPEG stream on the content-time grid, audio padded with silence: the video decides the length
    mjpeg = live.dir / f"{name}.mjpeg"
    with open(mjpeg, "wb") as f:
        for g in grid:
            f.write(frames[int(np.argmin(np.abs(times - g)))][1])
    inputs = ["-f", "mjpeg", "-framerate", f"{fps}", "-i", str(mjpeg)]
    raw = None
    chunks = [(t, c) for t, c in list(live.audio_ring) if t + len(c) / SR >= frames[0][0]]
    if chunks:
        samples = np.concatenate([c for _, c in chunks])
        skip = int(max(0.0, frames[0][0] - chunks[0][0]) * SR)
        raw = live.dir / f"{name}.f32"
        samples[skip:].astype(np.float32).tofile(raw)
        inputs += ["-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", str(raw)]
    cmd = [media.require("ffmpeg"), "-v", "error", "-y", *inputs, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
    cmd += (["-c:a", "aac", "-b:a", "128k", "-af", "apad", "-shortest"] if raw else []) + [str(mp4)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    mjpeg.unlink(missing_ok=True)
    if raw:
        raw.unlink(missing_ok=True)
    if r.returncode:
        raise RuntimeError(f"couldn't write the snapshot video: {r.stderr.strip()[-400:]}")
    t0 = frames[0][0]
    states = [s for s in list(live.states) if s["t"] >= t0]
    step = max(1, int(round(fps / 4)))
    rel = lambda x: round(x - t0, 3)
    info = {"source": live.source, "wall_start": datetime.fromtimestamp(stamp).isoformat(timespec="seconds"),
            "session_t0": round(t0, 3), "seconds": round(frames[-1][0] - t0, 2),
            "states": [{**s, "t": rel(s["t"])} for s in states[::step]],
            "events": [{**e, "t": rel(e["t"])} for e in list(live.events) if e["t"] >= t0 and e["kind"] != "onset"]}
    (live.dir / f"{name}.live.json").write_text(json.dumps(info, ensure_ascii=False))
    out = {"video": str(mp4), "seconds": info["seconds"], "wall_start": info["wall_start"]}
    if watch:
        from .watch import watch as run_watch
        wd = run_watch(str(mp4), log=lambda m: None)
        (wd / "live.json").write_text(json.dumps(info, ensure_ascii=False))
        idx = wd / "INDEX.md"
        idx.write_text(idx.read_text().rstrip() + "\n\n" + live_summary(info) + "\n")
        out["watch"] = str(wd)
        out["index"] = idx.read_text()
    return out


def live_summary(info: dict) -> str:
    """The '## Live signals' section of a snapshot's INDEX.md."""
    st, ev = info["states"], info["events"]
    L = [f"## Live signals ({info['wall_start']}, {info['seconds']:g} s from {info['source']})", ""]
    if st:
        def peak(key):
            vals = [(s[key], s["t"]) for s in st if s.get(key) is not None]
            return max(vals) if vals else None
        e, ar = peak("energy"), peak("arousal")
        mean = lambda k: np.mean([s[k] for s in st if s.get(k) is not None]) if any(s.get(k) is not None for s in st) else None
        L.append(f"- energy: mean {mean('energy'):.2f}, peak {e[0]:.2f} at {media.fmt_t(e[1])}" if e else "- energy: n/a")
        L.append(f"- arousal (vs this session's baseline, 0.5 = typical): mean {mean('arousal'):.2f}, peak {ar[0]:.2f} at "
                 f"{media.fmt_t(ar[1])}" if ar else "- arousal: n/a")
        v = mean("valence")
        if v is not None:
            L.append(f"- face expression valence (expression, not feeling): mean {v:+.2f}")
        smiles = [s["smile"] for s in st if s.get("smile") is not None]
        if smiles:
            L.append(f"- smiling (smile > 0.5) {sum(x > 0.5 for x in smiles) / len(smiles):.0%} of the time a face was visible")
        faces = max((s.get("faces") or 0) for s in st)
        L.append(f"- faces: up to {faces}; speaking {sum(1 for s in st if s.get('speaking')) / len(st):.0%} of the time")
    counts: dict[str, list] = {}
    for x in ev:
        counts.setdefault(x["kind"], []).append(x)
    for kind, label in (("smile_start", "smiles"), ("nod", "nods"), ("shake", "head shakes"), ("gesture_start", "gestures"),
                        ("seen_start", "visual triggers"), ("cut", "cuts"), ("look_away", "looked away")):
        if kind in counts:
            items = counts[kind][:12]
            extra = lambda x: f" {x.get('gesture') or x.get('what') or ''}".rstrip()
            L.append(f"- {label}: " + ", ".join(f"{media.fmt_t(x['t'])}{extra(x)}" for x in items))
    beats = counts.get("beat", [])
    if beats:
        L.append(f"- beat: ~{beats[-1].get('bpm')} BPM ({len(beats)} beats)")
    return "\n".join(L)


# ---------------------------------------------------------------- client side (CLI / MCP talk to a running session)


def session() -> dict | None:
    sf = session_file()
    if not sf.exists():
        return None
    info = json.loads(sf.read_text())
    try:
        os.kill(info["pid"], 0)
    except (ProcessLookupError, PermissionError, OSError):
        sf.unlink(missing_ok=True)
        return None
    return info


def request(cmd: dict, timeout: float = 300) -> dict:
    info = session()
    if not info:
        raise RuntimeError("no live session running; start one with: 9xvision live start [camera|screen|file|url]")
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect
    url = f"ws://127.0.0.1:{info['port']}/?k={info['token']}"
    try:
        return _request(url, cmd, timeout)
    except ConnectionClosed:
        raise RuntimeError("the live session closed the connection (did it stop? see ~/.9xvision/live/last.log)") from None


def _request(url: str, cmd: dict, timeout: float) -> dict:
    from websockets.sync.client import connect
    with connect(url, open_timeout=5, close_timeout=2, max_size=None) as ws:
        rid = secrets.token_hex(4)
        ws.send(json.dumps({**cmd, "id": rid}))
        end = time.time() + timeout
        while time.time() < end:
            msg = json.loads(ws.recv(timeout=max(0.1, end - time.time())))
            if msg.get("id") == rid:
                msg.pop("id", None)
                msg.pop("type", None)
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg
    raise RuntimeError("the live session did not answer in time")


def start_background(args: list[str], wait: float = 60) -> dict:
    """Start `9xvision live start ...` detached (for MCP); return the session once it is up."""
    if session():
        raise RuntimeError("a live session is already running; stop it first (live stop)")
    cmd = [sys.executable, "-m", "ninexvision.cli", "live", "start", *args, "--quiet"]   # this very install
    home().joinpath("live").mkdir(parents=True, exist_ok=True)
    logf = open(home() / "live" / "last.log", "w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=logf, stdin=subprocess.DEVNULL, start_new_session=True)
    end = time.time() + wait
    while time.time() < end:
        s = session()
        if s and s["pid"] == proc.pid:
            return s
        if proc.poll() is not None:
            logf.close()
            raise RuntimeError("live session failed to start:\n" + (home() / "live" / "last.log").read_text()[-1500:])
        time.sleep(0.3)
    raise RuntimeError("live session did not come up in time; see ~/.9xvision/live/last.log")
