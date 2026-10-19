"""Music and sound: tempo, beat grid, loudness sections, and how the edit sits on the beat.

Plain numpy, no extra dependencies: a log-magnitude spectral-flux onset envelope, tempo from its
autocorrelation (with a prior around 120 BPM), and Ellis' dynamic-programming beat tracker."""
from __future__ import annotations

import subprocess

import numpy as np

from . import media

SR = 22050
HOP = 512
NFFT = 2048
FPS = SR / HOP          # onset-envelope frames per second (~43)
LATENCY = 0.75 * NFFT / SR   # flux peaks when a transient is 3/4 into the Hann window (measured on a click track)


def load(path, start: float, end: float) -> np.ndarray:
    """Mono float32 samples of [start, end] in original-video time."""
    pr = media.probe(path)
    fs = max(0.0, start - pr.offset)
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{fs:.3f}", "-t", f"{end - start:.3f}", "-i", str(path),
                          "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True)
    return np.frombuffer(out.stdout, np.float32)


def onset_envelope(y: np.ndarray) -> np.ndarray:
    """Spectral flux of log-compressed magnitude, in chunks so an hour of audio stays small in memory."""
    if len(y) < NFFT:
        return np.zeros(0, np.float32)
    win = np.hanning(NFFT).astype(np.float32)
    n = 1 + (len(y) - NFFT) // HOP
    env = np.zeros(n, np.float32)
    prev = None
    for a in range(0, n, 2048):
        b = min(n, a + 2048)
        frames = np.lib.stride_tricks.sliding_window_view(y, NFFT)[a * HOP:(b - 1) * HOP + 1:HOP] * win
        mag = np.log1p(100 * np.abs(np.fft.rfft(frames, axis=1)))
        if prev is not None:
            mag_full = np.vstack([prev[None], mag])
        else:
            mag_full = np.vstack([mag[:1], mag])
        env[a:b] = np.maximum(0, np.diff(mag_full, axis=0)).sum(axis=1)
        prev = mag[-1]
    k = int(FPS)   # remove the slow trend (local mean over ~1 s)
    env = env - np.convolve(env, np.ones(k) / k, mode="same")
    env = np.maximum(env, 0)
    return env / (env.std() + 1e-9)


def tempo(env: np.ndarray) -> tuple[float, float]:
    """(BPM, pulse clarity 0-1). Clarity is the autocorrelation peak in the 60-200 BPM range, normalised."""
    if len(env) < FPS * 4:
        return 0.0, 0.0
    e = env - env.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(e, 2 * len(e))) ** 2)[:len(e)]
    ac = ac / (ac[0] + 1e-9)
    lags = np.arange(int(FPS * 60 / 200), int(FPS * 60 / 60) + 1)
    bpm = 60 * FPS / lags
    prior = np.exp(-0.5 * (np.log2(bpm / 120) / 1.0) ** 2)
    best = lags[int(np.argmax(ac[lags] * prior))]
    # parabolic refinement of the peak lag
    y0, y1, y2 = ac[best - 1], ac[best], ac[best + 1]
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    lag = best + float(np.clip(d, -0.5, 0.5))
    return round(60 * FPS / lag, 1), round(float(max(0.0, ac[best])), 3)


def beats(env: np.ndarray, bpm: float, tightness: float = 100.0) -> np.ndarray:
    """Ellis (2007) dynamic-programming beat tracker. Returns beat times in seconds from the envelope start."""
    if bpm <= 0 or len(env) < 4:
        return np.zeros(0)
    period = FPS * 60 / bpm
    n = len(env)
    score = env.astype(np.float64).copy()
    back = np.full(n, -1)
    lo, hi = int(round(period / 2)), int(round(2 * period))
    offs = np.arange(lo, hi + 1)
    pen = -tightness * np.log(offs / period) ** 2
    for i in range(lo, n):
        js = i - offs
        ok = js >= 0
        cand = score[js[ok]] + pen[ok]
        k = int(np.argmax(cand))
        if cand[k] > 0:
            score[i] = env[i] + cand[k]
            back[i] = js[ok][k]
    tail = slice(max(0, n - int(period)), n)
    i = tail.start + int(np.argmax(score[tail]))
    out = []
    while i >= 0:
        out.append(i)
        i = back[i]
    return np.array(out[::-1]) / FPS


def loudness(y: np.ndarray, win: float = 0.5) -> np.ndarray:
    """RMS level in dBFS per window."""
    k = int(SR * win)
    if len(y) < k:
        return np.zeros(0)
    m = len(y) // k
    rms = np.sqrt((y[:m * k].reshape(m, k).astype(np.float64) ** 2).mean(axis=1))
    return 20 * np.log10(rms + 1e-9)


def analyse(path, start: float, end: float, cuts: list[float] | None = None, video_fps: float = 25.0) -> dict:
    """Everything above for [start, end] (original-video time). cuts: frame-accurate cut times to align."""
    y = load(path, start, end)
    if len(y) < SR:
        return {"present": False}
    env = onset_envelope(y)
    bpm, clarity = tempo(env)
    rhythmic = clarity >= 0.25 and bpm > 0
    if rhythmic and clarity < 0.35:
        out_weak = True
    else:
        out_weak = False
    out: dict = {"present": True, "bpm": bpm if rhythmic else None, "pulse_clarity": clarity, "weak_pulse": out_weak}
    lv = loudness(y)
    if len(lv):
        out["level_dbfs"] = {"mean": round(float(np.mean(lv)), 1), "max": round(float(np.max(lv)), 1)}
        out["loudness_curve"] = [[round(start + i * 0.5, 1), round(float(v), 1)] for i, v in enumerate(lv)]
        out["jumps"] = _jumps(lv, start)
        out["silences"] = _silences(lv, start)
    if rhythmic:
        bt = beats(env, bpm) + start + LATENCY
        out["beats"] = [round(float(b), 3) for b in bt]
        if cuts:
            out["cuts_vs_beats"] = align(cuts, bt, bpm, video_fps)
    return out


def align(cuts: list[float], bt: np.ndarray, bpm: float, video_fps: float) -> dict:
    """How many cuts land on a beat, against what chance would give."""
    tol = max(0.05, 1.5 / max(video_fps, 1))
    period = 60 / bpm
    near = [float(c - bt[int(np.argmin(np.abs(bt - c)))]) for c in cuts if len(bt) and bt[0] - period <= c <= bt[-1] + period]
    if not near:
        return {}
    on = [d for d in near if abs(d) <= tol]
    chance = min(1.0, 2 * tol / period)
    p = _binom_tail(len(on), len(near), chance)
    res = {"cuts": len(near), "on_beat": len(on), "share": round(len(on) / len(near), 2),
           "chance": round(chance, 2), "tolerance_s": round(tol, 3), "p_value": float(f"{p:.2g}")}
    if len(on) >= 3:
        res["median_offset_frames"] = round(float(np.median(on)) * video_fps, 1)
    if len(near) < MIN_CUTS:
        res["verdict"] = f"too few cuts to judge ({len(near)}; needs {MIN_CUTS})"
    elif p < 0.001 and res["share"] >= 2 * chance:
        res["verdict"] = "cut to the beat"
    elif p < 0.01:
        res["verdict"] = "partly on the beat"
    else:
        res["verdict"] = "not cut to the beat"
    return res


MIN_CUTS = 20


def _binom_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p): how likely this many on-beat cuts are by chance."""
    from math import comb
    return float(min(1.0, sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))))


def _jumps(lv: np.ndarray, start: float, k: int = 4) -> list[dict]:
    """Biggest sudden rises in level (drops, hits, a band coming in): +dB over ~1 s vs the 2 s before."""
    if len(lv) < 8:
        return []
    rise = np.array([lv[i:i + 2].mean() - lv[max(0, i - 4):i].mean() for i in range(4, len(lv) - 2)])
    out = []
    for i in np.argsort(-rise):
        if rise[i] < 6 or len(out) >= k:
            break
        t = start + (i + 4) * 0.5
        if all(abs(t - o["t"]) > 3 for o in out):
            out.append({"t": round(t, 1), "rise_db": round(float(rise[i]), 1)})
    return sorted(out, key=lambda o: o["t"])


def _silences(lv: np.ndarray, start: float, floor: float = -50.0, min_s: float = 1.0) -> list[list[float]]:
    quiet = lv < floor
    out, a = [], None
    for i, q in enumerate(list(quiet) + [False]):
        if q and a is None:
            a = i
        elif not q and a is not None:
            if (i - a) * 0.5 >= min_s:
                out.append([round(start + a * 0.5, 1), round(start + i * 0.5, 1)])
            a = None
    return out[:20]
