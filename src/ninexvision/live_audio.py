"""Streaming sound features for live input: level, onsets, tempo with a running beat phase, voice pitch, brightness.

Fed one hop (512 samples at 22050 Hz, ~23 ms) at a time. The beat grid is predicted forward from the recent onset
envelope, so a beat event can fire on the beat rather than after it (what synced visuals need)."""
from __future__ import annotations

from collections import deque

import numpy as np

from .audio import FPS, HOP, NFFT, SR, tempo

PEAK_LAG = 0.25 * NFFT / SR
WIN_S = 8.0                 # seconds of onset envelope used for tempo and phase
F0_MIN, F0_MAX = 70.0, 800.0


def yin(frame: np.ndarray, sr: int = SR, fmin: float = F0_MIN, fmax: float = F0_MAX, thresh: float = 0.15) -> tuple[float, float]:
    """(f0 Hz, voicing 0-1) by YIN (de Cheveigné & Kawahara 2002); f0 0 when unvoiced."""
    lo, hi = int(sr / fmax), int(sr / fmin)
    n = len(frame) - hi
    if n <= lo:
        return 0.0, 0.0
    x = frame.astype(np.float64)
    # d(tau) = sum_j (x_j - x_{j+tau})^2 over j < n, from energies and a cross-correlation done by FFT
    e = np.cumsum(np.r_[0.0, x * x])
    size = 1 << int(np.ceil(np.log2(len(x) + n)))
    r = np.fft.irfft(np.conj(np.fft.rfft(x[:n], size)) * np.fft.rfft(x, size), size)[:hi + 1]
    taus = np.arange(hi + 1)
    d = (e[n] - e[0]) + (e[taus + n] - e[taus]) - 2 * r
    d[0] = 0
    cmnd = d[1:] * np.arange(1, hi + 1) / (np.cumsum(d[1:]) + 1e-12)
    cmnd = np.r_[1.0, cmnd]
    below = np.where(cmnd[lo:hi] < thresh)[0]
    if not len(below):
        return 0.0, float(max(0.0, 1 - cmnd[lo:hi].min()))
    t = lo + below[0]
    while t + 1 < hi and cmnd[t + 1] < cmnd[t]:
        t += 1
    a, b, c = cmnd[t - 1], cmnd[t], cmnd[t + 1] if t + 1 <= hi else cmnd[t]
    shift = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
    return float(sr / (t + np.clip(shift, -0.5, 0.5))), float(max(0.0, 1 - b))


class StreamingAudio:
    """Push samples with feed(); read .state; collect events from .events (cleared by the caller)."""

    def __init__(self, lead: float = 0.0):
        self.lead = lead                # fire beat events this many seconds early (to cover output latency)
        self.win = np.hanning(NFFT).astype(np.float32)
        self.buf = np.zeros(NFFT, np.float32)
        self.pending = np.zeros(0, np.float32)
        self.prev_mag = None
        self.env = deque(maxlen=int(WIN_S * FPS))
        self.n = 0                      # hops processed
        self.bpm, self.clarity = 0.0, 0.0
        self.period = 0.0               # seconds
        self.next_beat = None           # predicted time of the next beat (stream seconds)
        self.beat_count = 0
        self.flux_hist = deque(maxlen=int(2 * FPS))
        self.onsets = deque()
        self.voiced_hist = deque(maxlen=int(0.3 * FPS / 2))   # voicing decisions over the last 0.3 s
        self.f0_hist = deque(maxlen=int(2 * FPS))
        self.level_floor = -60.0
        self.speaking = False
        self.quiet_run = 0
        self.events: list[dict] = []
        self.hist: deque = deque(maxlen=int(3 * FPS))    # (hop time, state) for reading the sound at a frame's time
        self.state: dict = {"db": -120.0, "onset": 0.0, "bpm": None, "beat_phase": None, "pitch_hz": None,
                            "voiced": False, "centroid_hz": None, "pitch_var": None, "onset_rate": 0.0}

    @property
    def t(self) -> float:
        return self.n * HOP / SR

    def feed(self, samples: np.ndarray) -> None:
        self.pending = np.r_[self.pending, samples.astype(np.float32)]
        while len(self.pending) >= HOP:
            hop, self.pending = self.pending[:HOP], self.pending[HOP:]
            self.buf = np.r_[self.buf[HOP:], hop]
            self._hop()

    def _hop(self) -> None:
        self.n += 1
        t = self.t
        x = self.buf * self.win
        mag = np.log1p(100 * np.abs(np.fft.rfft(x)))
        flux = 0.0 if self.prev_mag is None else float(np.maximum(0, mag - self.prev_mag).sum())
        self.prev_mag = mag
        self.env.append(flux)
        self.flux_hist.append(flux)
        rms = float(np.sqrt(np.mean(self.buf[-HOP * 2:] ** 2)))
        db = max(-120.0, 20 * np.log10(rms + 1e-9))
        lin = np.expm1(mag)
        centroid = float((lin * np.arange(len(lin))).sum() / (lin.sum() + 1e-9) * SR / NFFT)
        # onset strength against the recent median and its median absolute deviation: a robust z-score, so one
        # loud burst does not inflate the spread and hide the next two seconds of onsets
        h = np.array(self.flux_hist)
        if len(h) > 8:
            med = float(np.median(h))
            z = (flux - med) / (1.4826 * float(np.median(np.abs(h - med))) + 0.05 * med + 1e-3)
        else:
            z = 0.0
        st = self.state
        z = min(float(z), 10.0)                 # a click in digital silence has no spread to measure against
        st.update(db=round(db, 1), onset=round(float(max(0.0, z)), 2), centroid_hz=round(centroid) if db > -60 else None)
        if z > 2.5 and db > -50 and (not self.onsets or t - self.onsets[-1] > 0.08):
            self._event("onset", strength=round(float(z), 1))
            self.events[-1]["t"] = round(t - PEAK_LAG, 3)        # when the sound started, not when flux peaked
            self.onsets.append(t)
        while self.onsets and self.onsets[0] < t - 2.0:
            self.onsets.popleft()
        # voice pitch every other hop
        if self.n % 2 == 0:
            f0, conf = yin(self.buf[-NFFT // 2:], thresh=0.2) if db > -50 else (0.0, 0.0)
            voiced = f0 > 0 and conf > 0.6
            self.voiced_hist.append(voiced and db > -45)
            st.update(pitch_hz=round(f0, 1) if voiced else None, voiced=voiced)
            if voiced:
                self.f0_hist.append(np.log2(f0))
            st["pitch_var"] = round(float(np.std(self.f0_hist)) * 12, 2) if len(self.f0_hist) > 10 else None  # semitones
            self._speech(voiced, db)
        # tempo and beat phase, re-estimated twice a second
        if self.n % int(FPS / 2) == 0 and len(self.env) >= FPS * 4:
            self._tempo()
        if self.next_beat is not None and self.period:
            if t + self.lead >= self.next_beat:   # announce each beat when it is due (or `lead` seconds early)
                self.beat_count += 1
                self._event("beat", n=self.beat_count, bpm=self.bpm, at=round(self.next_beat, 3))
                self.next_beat += self.period
            st["beat_phase"] = round(float(1 - ((self.next_beat - t) / self.period) % 1.0), 3)
        st["onset_rate"] = round(len(self.onsets) / 2.0, 1)    # onsets per second over the last 2 s
        self.hist.append((t, dict(st)))

    def state_at(self, t: float) -> dict:
        """The sound state describing stream time t (the hop whose window is centred on t), with the beat phase
        read from the current grid at t. Pipes deliver audio and video with different delays, so the newest audio
        state can be from a different moment than the newest frame."""
        # a hop's spectrum and level cover the NFFT samples before its time, centred NFFT/2 earlier
        target = t + NFFT / 2 / SR
        st = self.state
        for ht, s in reversed(self.hist):
            if ht <= target:
                st = s
                break
        else:
            if self.hist:
                st = self.hist[0][1]
        st = dict(st)
        if self.next_beat is not None and self.period:
            st["beat_phase"] = round(float((t - self.next_beat) / self.period % 1.0), 3)
        return st

    def _tempo(self) -> None:
        env = np.log1p(np.array(self.env, np.float32))      # compressed, so a few loud bursts can't drown the pulse
        e = np.maximum(env - np.convolve(env, np.ones(int(FPS)) / int(FPS), mode="same"), 0)
        e = e / (e.std() + 1e-9)
        bpm, clarity = tempo(e)
        self.clarity = clarity
        if clarity < 0.25 or bpm <= 0:
            self.bpm, self.period, self.next_beat = 0.0, 0.0, None
            self.state.update(bpm=None, beat_phase=None)
            return
        if self.bpm and abs(bpm - self.bpm) / self.bpm < 0.04:
            bpm = 0.8 * self.bpm + 0.2 * bpm            # steady tempo: smooth it
        self.bpm, self.period = round(float(bpm), 1), 60.0 / bpm
        # phase: the comb over the last few seconds that collects the most onset energy
        p = self.period * FPS
        n = len(e)
        best, best_s = 0.0, -1.0
        for off in np.arange(0, p, 0.5):
            idx = (n - 1 - off - p * np.arange(int((n - 1 - off) / p) + 1)).round().astype(int)
            idx = idx[idx >= 0]
            s = float(e[idx].sum()) if len(idx) else 0.0
            if s > best_s:
                best, best_s = off, s
        # a transient peaks the flux when it sits 3/4 into the window, i.e. PEAK_LAG before the newest sample
        last_beat = self.t - best / FPS - PEAK_LAG
        nb = last_beat
        while nb <= self.t:
            nb += self.period
        if self.next_beat is None:
            self.next_beat = nb
        else:
            # compare phases, not beat numbers: a beat that is due right now and the measured next one are the
            # same grid, and treating them as a full-period drift used to skip that beat
            d = (nb - self.next_beat + self.period / 2) % self.period - self.period / 2
            self.next_beat += d if abs(d) > 0.25 * self.period else 0.25 * d
        self.state["bpm"] = self.bpm

    def _speech(self, voiced: bool, db: float) -> None:
        """Speaking on when half of the last 0.3 s is voiced; off after 0.6 s without voice."""
        self.quiet_run = 0 if voiced and db > -45 else self.quiet_run + 1
        hops2 = FPS / 2      # pitch is computed every second hop
        share = sum(self.voiced_hist) / max(1, self.voiced_hist.maxlen)
        recent = list(self.f0_hist)[-12:]                    # ~0.5 s of voiced pitch
        moving = len(recent) >= 6 and float(np.std(recent)) * 12 >= 0.25   # speech glides; a held pure tone does not
        if not self.speaking and share >= 0.5 and moving:
            self.speaking = True
            self._event("speech_start")
        elif self.speaking and self.quiet_run >= 0.6 * hops2:
            self.speaking = False
            self._event("speech_end")
        self.state["speaking"] = self.speaking

    def _event(self, kind: str, **kw) -> None:
        self.events.append({"kind": kind, "t": round(self.t, 3), **kw})
