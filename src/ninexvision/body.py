"""Face, hand and body tracking for live input, through MediaPipe in a separate environment (see _body_worker.py).

Faces give 51 blendshapes (the ARKit set minus tongueOut: smile, brow raise, jaw open, blinks, gaze...) and head pose; hands give a
gesture, position and pinch; the body gives a few key points. These describe what a face or body DOES, not what a
person feels. No identities: nothing here recognises who someone is."""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from importlib import resources
from pathlib import Path

import numpy as np

BASE = "https://storage.googleapis.com/mediapipe-models/"
MODELS = {"face_landmarker.task": "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
          "gesture_recognizer.task": "gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
          "pose_landmarker_lite.task": "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"}
PACKAGE = "mediapipe>=0.10.21"
SIZE_MB = 350


def home() -> Path:
    return Path(os.environ.get("NINEXVISION_HOME", Path.home() / ".9xvision"))


def runtime() -> Path:
    return home() / "runtimes" / "body"


def models() -> Path:
    return home() / "models" / "mediapipe"


def _python() -> Path:
    r = runtime()
    return r / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def available() -> bool:
    return _python().exists() and all((models() / f).exists() for f in MODELS)


def install(log=print) -> None:
    """Create the MediaPipe environment (uv when present, else venv + pip) and download the three models."""
    r = runtime()
    if not _python().exists() or subprocess.run([str(_python()), "-c", "import mediapipe"], capture_output=True).returncode:
        log(f"creating the face/hand/body runtime in {r} (about {SIZE_MB} MB)")
        shutil.rmtree(r, ignore_errors=True)
        r.parent.mkdir(parents=True, exist_ok=True)
        uv = shutil.which("uv")
        if uv:
            subprocess.run([uv, "venv", "-q", "--python", sys.executable, str(r)], check=True)
            subprocess.run([uv, "pip", "install", "-q", "--python", str(_python()), PACKAGE], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(r)], check=True)
            subprocess.run([str(_python()), "-m", "pip", "install", "-q", PACKAGE], check=True)
    d = models()
    d.mkdir(parents=True, exist_ok=True)
    for name, rel in MODELS.items():
        if not (d / name).exists():
            log(f"downloading {BASE}{rel}")
            tmp = d / (name + ".part")
            urllib.request.urlretrieve(BASE + rel, tmp)
            tmp.rename(d / name)


class Tracker:
    """Runs the worker. submit() hands it the newest frame when it is free (frames in between are skipped, so
    tracking never falls behind); latest holds the newest result and the stream time of its frame."""

    def __init__(self, face: bool = True, hands: bool = False, pose: bool = False, mirrored: bool = False, log=print):
        if not available():
            install(log)
        worker = resources.files("ninexvision").joinpath("_body_worker.py")
        env = {**os.environ, "GLOG_minloglevel": "2", "TF_CPP_MIN_LOG_LEVEL": "2", "PYTHONNOUSERSITE": "1"}
        self.errlog = runtime() / "worker.log"
        err = open(self.errlog, "wb")
        self.proc = subprocess.Popen([str(_python()), "-I", str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=err, env=env, cwd=str(runtime()))
        err.close()
        cfg = {"models": str(models()), "face": face, "hands": hands, "pose": pose, "mirrored": mirrored}
        self.proc.stdin.write((json.dumps(cfg) + "\n").encode())
        self.proc.stdin.flush()
        ready = self.proc.stdout.readline()
        if not ready:
            tail = self.errlog.read_text(errors="replace").strip().splitlines()[-3:] if self.errlog.exists() else []
            hint = ("on Linux MediaPipe needs EGL/GL libraries: sudo apt install libegl1 libgles2 libgl1"
                    if any("libEGL" in x or "libGL" in x for x in tail) else "try: 9xvision setup")
            raise RuntimeError("the face/hand/body worker did not start" + (": " + tail[-1].strip() if tail else "")
                               + f" ({hint})")
        self.latest: dict | None = None
        self.latest_t = None
        self.ms = 0.0               # worker time per frame
        self._busy = False
        self._next: tuple | None = None
        self._lock = threading.Condition()
        self._alive = True
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, rgb: np.ndarray, t: float) -> None:
        with self._lock:
            self._next = (rgb, t)
            self._lock.notify()

    def _loop(self) -> None:
        while self._alive:
            with self._lock:
                while self._next is None and self._alive:
                    self._lock.wait(0.5)
                if not self._alive:
                    return
                rgb, t = self._next
                self._next = None
            h, w = rgb.shape[:2]
            data = np.ascontiguousarray(rgb, np.uint8).tobytes()
            t0 = time.perf_counter()
            try:
                self.proc.stdin.write(struct.pack(">IHHI", len(data), w, h, int(t * 1000) & 0xFFFFFFFF) + data)
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
            except (BrokenPipeError, OSError):
                self._alive = False
                return
            if not line:
                self._alive = False
                return
            self.ms = 0.8 * self.ms + 0.2 * (time.perf_counter() - t0) * 1000 if self.ms else (time.perf_counter() - t0) * 1000
            self.latest, self.latest_t = json.loads(line), t

    @property
    def alive(self) -> bool:
        return self._alive and self.proc.poll() is None

    def close(self) -> None:
        self._alive = False
        with self._lock:
            self._lock.notify()
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()
