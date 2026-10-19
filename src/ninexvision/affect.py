"""Affect signals from live features: what the room, the voice and the face are doing, as numbers a synth, a
visual or a character can follow.

- energy (0-1, absolute): loudness, onsets and motion. The same scene gives the same number in any session.
- arousal (0-1, relative): activation compared with THIS session so far (0.5 = typical for the session), from
  loudness, onset rate, pitch movement, motion, facial movement and body movement. Relative because people and
  rooms differ; the first ~10 s are a warm-up.
- expression_valence (-1..1): the face's own smile vs frown/brow-lower/sneer, against this person's resting face.
  It reads the EXPRESSION, not the feeling: people smile when uncomfortable and frown when concentrating.
- engagement (0-1): share of the last 5 s with a face turned toward the camera.

Events: smile_start/end, blink (with blinks per minute in the state), nod, shake, look_away/look_back,
face_enter/face_leave, gesture_start/end, hands_up. None of this identifies anyone."""
from __future__ import annotations

import math
from collections import deque

import numpy as np


class Baseline:
    """Running mean and spread with time constant tau (seconds), robust to the first few samples."""

    def __init__(self, tau: float = 30.0, floor: float = 1e-3):
        self.tau, self.floor, self.mean, self.dev, self.n = tau, floor, 0.0, 0.0, 0

    def update(self, x: float, dt: float) -> float:
        """Add x; return its z-score against the baseline before x (0 during warm-up)."""
        self.n += 1
        if self.n == 1:
            self.mean, self.dev = x, 0.0
            return 0.0
        z = (x - self.mean) / max(self.dev, self.floor) if self.n > 20 else 0.0
        a = min(1.0, dt / self.tau) if self.n > 20 else 1.0 / self.n
        self.mean += a * (x - self.mean)
        self.dev += a * (abs(x - self.mean) - self.dev)
        return float(np.clip(z, -3, 3))


def _swings(x: np.ndarray, amp: float) -> int:
    """Direction reversals of at least amp (a zigzag count): left-right-left is 2."""
    n, d, hi, lo = 0, 0, x[0], x[0]
    for v in x[1:]:
        if d == 0:
            hi, lo = max(hi, v), min(lo, v)
            if hi - v >= amp:
                d, lo = -1, v
            elif v - lo >= amp:
                d, hi = 1, v
        elif d == 1:
            if v > hi:
                hi = v
            elif hi - v >= amp:
                d, lo, n = -1, v, n + 1
        else:
            if v < lo:
                lo = v
            elif v - lo >= amp:
                d, hi, n = 1, v, n + 1
    return n


def _avg(b: dict, *keys: str) -> float:
    return sum(b.get(k, 0.0) for k in keys) / len(keys)


class Affect:
    def __init__(self):
        # floors: the smallest spread that counts, so a perfectly still signal does not make noise look huge
        self.base = {k: Baseline(floor=fl) for k, fl in (("db", 3.0), ("onset_rate", 0.5), ("pitch_var", 0.5),
                                                          ("motion", 0.01), ("expressivity", 0.05), ("body", 0.02))}
        self.face_base: dict[str, Baseline] = {}
        self.arousal = 0.5
        self.energy = 0.0
        self.valence = None
        self.events: list[dict] = []
        self.smiling = False
        self.smile_run = 0.0
        self.blinks: deque = deque()
        self.eyes_closed = False
        self.pose_hist: deque = deque()          # (t, yaw, pitch)
        self.last_head_event = -10.0
        self.head_settled = True
        self.seg_start = -10.0                   # head reversals count from here (last event or last stillness)
        self.facing_cand, self.facing_since = None, 0.0
        self.looking = deque()                   # (t, facing)
        self.facing = None
        self.face_present = False
        self.face_gone_at = None
        self.gestures: dict[str, str] = {}       # hand -> gesture held
        self.gesture_run: dict[str, tuple[str, float]] = {}
        self.gesture_votes: dict[str, deque] = {}
        self.hands_up = False
        self.prev_pose = None
        self.t = 0.0

    def update(self, t: float, audio: dict | None, video: dict, body: dict | None) -> dict:
        dt = max(1e-3, t - self.t) if self.t else 1 / 15
        self.t = t
        comp: dict[str, float] = {}
        if audio and audio.get("db", -120) > -90:
            comp["db"] = self.base["db"].update(audio["db"], dt)
            comp["onset_rate"] = self.base["onset_rate"].update(audio.get("onset_rate") or 0.0, dt)
            if audio.get("pitch_var") is not None:
                comp["pitch_var"] = self.base["pitch_var"].update(audio["pitch_var"], dt)
        comp["motion"] = self.base["motion"].update(video.get("motion", 0.0), dt)
        face = (body or {}).get("faces") or []
        f = face[0] if face else None
        out: dict = {}
        self._presence(t, f is not None)
        if f and f.get("blend"):
            b = f["blend"]
            dev = 0.0
            for k, v in b.items():
                bl = self.face_base.setdefault(k, Baseline(tau=20.0))
                if not k.startswith(("eyeBlink", "eyeLook")):     # blinks and gaze are not expression
                    dev += abs(v - bl.mean) if bl.n > 20 else 0.0
                bl.update(v, dt)
            comp["expressivity"] = self.base["expressivity"].update(dev, dt)
            smile = _avg(b, "mouthSmileLeft", "mouthSmileRight")
            neg = _avg(b, "mouthFrownLeft", "mouthFrownRight") + 0.5 * _avg(b, "browDownLeft", "browDownRight") \
                + 0.5 * _avg(b, "noseSneerLeft", "noseSneerRight")
            rest_smile = self._rest("mouthSmileLeft", "mouthSmileRight")
            rest_neg = self._rest("mouthFrownLeft", "mouthFrownRight") + 0.5 * self._rest("browDownLeft", "browDownRight") \
                + 0.5 * self._rest("noseSneerLeft", "noseSneerRight")
            v = math.tanh(2.0 * ((smile - rest_smile) - (neg - rest_neg)))
            self.valence = v if self.valence is None else self.valence + min(1.0, dt / 0.3) * (v - self.valence)
            self._smile(t, dt, smile)
            self._blink(t, _avg(b, "eyeBlinkLeft", "eyeBlinkRight"))
            out["face"] = {"smile": round(smile, 3), "brow_raise": round(_avg(b, "browInnerUp", "browOuterUpLeft", "browOuterUpRight"), 3),
                           "brow_lower": round(_avg(b, "browDownLeft", "browDownRight"), 3), "jaw_open": round(b.get("jawOpen", 0.0), 3),
                           "eyes_closed": round(_avg(b, "eyeBlinkLeft", "eyeBlinkRight"), 3)}
        else:
            self.valence = None
        if f and "yaw" in f:
            self._head(t, f["yaw"], f["pitch"])
        self._hands(t, dt, (body or {}).get("hands") or [])
        if (body or {}).get("pose"):
            comp["body"] = self.base["body"].update(self._pose(t, dt, body["pose"]), dt)
        # arousal: mean z of what is available, squashed; energy: absolute levels
        zs = [v for v in comp.values()]
        target = 1 / (1 + math.exp(-0.9 * (sum(zs) / len(zs)))) if zs else 0.5
        self.arousal += min(1.0, dt / 0.5) * (target - self.arousal)
        db = (audio or {}).get("db", -120)
        e = 0.45 * float(np.clip((db + 60) / 50, 0, 1)) + 0.25 * min(1.0, ((audio or {}).get("onset_rate") or 0) / 6) \
            + 0.30 * min(1.0, video.get("motion", 0.0) / 0.08)
        self.energy += min(1.0, dt / 0.3) * (e - self.energy)
        recent = [x for x in self.looking if x[0] >= t - 5]
        cut = t - 60
        while self.blinks and self.blinks[0] < cut:
            self.blinks.popleft()
        out["affect"] = {"energy": round(self.energy, 3), "arousal": round(self.arousal, 3),
                         "expression_valence": None if self.valence is None else round(self.valence, 3),
                         "engagement": round(sum(1 for x in recent if x[1]) / len(recent), 2) if recent else 0.0,
                         "blinks_per_min": round(len(self.blinks) * 60 / min(60.0, max(t, 1.0)), 1) if self.face_present else None,
                         "warming_up": t < 10, "components": {k: round(v, 2) for k, v in comp.items()}}
        return out

    def _rest(self, *keys: str) -> float:
        vals = [self.face_base[k].mean for k in keys if k in self.face_base and self.face_base[k].n > 20]
        return sum(vals) / len(vals) if vals else 0.0

    def _ev(self, kind: str, t: float, **kw) -> None:
        self.events.append({"kind": kind, "t": round(t, 3), **kw})

    def _presence(self, t: float, present: bool) -> None:
        if present and not self.face_present:
            self.face_present = True
            self._ev("face_enter", t)
        elif not present and self.face_present:
            if self.face_gone_at is None:
                self.face_gone_at = t
            elif t - self.face_gone_at > 0.5:      # a missed detection or two is not leaving
                self.face_present = False
                self.face_gone_at = None
                self._ev("face_leave", t)
            return
        if present:
            self.face_gone_at = None

    def _smile(self, t: float, dt: float, smile: float) -> None:
        """Smiles are absolute (a resting face scores near 0), unlike valence, which is against the resting face."""
        on = smile > 0.5 if not self.smiling else smile > 0.3
        self.smile_run = self.smile_run + dt if on != self.smiling else 0.0
        if self.smile_run >= 0.3:
            self.smiling = on
            self.smile_run = 0.0
            self._ev("smile_start" if on else "smile_end", t)

    def _blink(self, t: float, closed: float) -> None:
        if closed > 0.5 and not self.eyes_closed:
            self.eyes_closed = True
            self.blinks.append(t)
            self._ev("blink", t)
        elif closed < 0.3:
            self.eyes_closed = False

    def _head(self, t: float, yaw: float, pitch: float) -> None:
        facing = abs(yaw) < 25 and abs(pitch) < 20
        self.looking.append((t, facing))
        while self.looking and self.looking[0][0] < t - 5:
            self.looking.popleft()
        # looking away has to hold 0.4 s, so the swings of a head shake are not glances
        if facing != self.facing_cand:
            self.facing_cand, self.facing_since = facing, t
        if self.facing is None:
            self.facing = facing
        elif facing != self.facing and t - self.facing_since >= 0.4:
            self.facing = facing
            self._ev("look_back" if facing else "look_away", t)
        self.pose_hist.append((t, yaw, pitch))
        while self.pose_hist and self.pose_hist[0][0] < t - 1.2:
            self.pose_hist.popleft()
        recent = [p for p in self.pose_hist if p[0] >= t - 0.4]
        if len(recent) >= 4 and np.ptp([p[1] for p in recent]) < 4 and np.ptp([p[2] for p in recent]) < 3:
            self.seg_start, self.head_settled = t, True   # the head is still: the next movement starts fresh
        # one event per gesture; a movement that keeps going (nodding along) gives another every 1.5 s
        gap = t - self.last_head_event
        if gap < 0.5 or (not self.head_settled and gap < 1.5):
            return
        win = [p for p in self.pose_hist if p[0] >= self.seg_start]
        if len(win) < 6:
            return
        ys = np.array([p[1] for p in win])
        ps = np.array([p[2] for p in win])
        yr, pr = float(np.ptp(ys)), float(np.ptp(ps))
        back = lambda x, r: abs(x[-1] - x[0]) < 0.4 * r          # ended near where it started
        kind = None
        if _swings(ys, 4) >= 2 and yr >= 8 and pr < 0.6 * yr:
            kind, size = "shake", yr
        elif (_swings(ps, 3) >= 2 or (_swings(ps, 3) >= 1 and pr >= 8 and back(ps, pr))) and pr >= 6 and yr < 0.6 * pr:
            kind, size = "nod", pr
        if kind:
            self._ev(kind, t, size_deg=round(size, 1))
            self.last_head_event, self.seg_start, self.head_settled = t, t, False

    def _hands(self, t: float, dt: float, hands: list[dict]) -> None:
        # the recogniser flickers between a gesture and "None" frame to frame: vote over the last 0.5 s
        frame = {h["hand"]: h.get("gesture") if h.get("gesture") not in (None, "None") and h.get("score", 0) >= 0.5 else None
                 for h in hands}
        for side in set(frame) | set(self.gesture_votes):
            q = self.gesture_votes.setdefault(side, deque())
            q.append((t, frame.get(side)))
            while q and q[0][0] < t - 0.5:
                q.popleft()
        seen = {}
        for side, q in self.gesture_votes.items():
            votes = [g for _, g in q if g]
            if votes:
                top = max(set(votes), key=votes.count)
                if votes.count(top) >= 0.35 * len(q):
                    seen[side] = top
        for side in set(self.gestures) | set(seen) | set(self.gesture_run):
            g = seen.get(side)
            if g == self.gestures.get(side):
                self.gesture_run.pop(side, None)
                continue
            cand, run = self.gesture_run.get(side, (g, 0.0))
            run = run + dt if cand == g else 0.0
            self.gesture_run[side] = (g, run)
            if run >= 0.2:                                   # held 0.2 s: real, not a passing shape
                if self.gestures.get(side):
                    self._ev("gesture_end", t, hand=side, gesture=self.gestures[side])
                if g:
                    self._ev("gesture_start", t, hand=side, gesture=g)
                    self.gestures[side] = g
                else:
                    self.gestures.pop(side, None)
                self.gesture_run.pop(side, None)

    def _pose(self, t: float, dt: float, p: dict) -> float:
        pts = np.array([v[:2] for v in p.values()])
        energy = 0.0
        if self.prev_pose is not None and self.prev_pose.shape == pts.shape:
            energy = float(np.linalg.norm(pts - self.prev_pose, axis=1).mean() / dt)   # frame widths per second
        self.prev_pose = pts
        nose_y = p["nose"][1]
        up = all(p[w][2] > 0.5 and p[w][1] < nose_y for w in ("l_wrist", "r_wrist"))
        if up != self.hands_up:
            self.hands_up = up
            if up:
                self._ev("hands_up", t)
        return energy
