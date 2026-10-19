"""Live input, affect and image work."""
import json
import math
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ninexvision import affect, bus, images, live, render
from ninexvision.audio import SR
from ninexvision.live_audio import StreamingAudio, yin


def _osc_parse(b: bytes):
    def s(i):
        j = b.index(b"\0", i)
        return b[i:j].decode(), (j // 4 + 1) * 4
    addr, i = s(0)
    tags, i = s(i)
    args = []
    for t in tags[1:]:
        if t == "f":
            args.append(struct.unpack(">f", b[i:i + 4])[0]); i += 4
        elif t == "i":
            args.append(struct.unpack(">i", b[i:i + 4])[0]); i += 4
        elif t == "s":
            v, i = s(i); args.append(v)
    return addr, args


def test_osc_messages_and_addresses():
    addr, args = _osc_parse(bus.osc_message("/9xv/affect/arousal", 0.5, 3, "smile"))
    assert addr == "/9xv/affect/arousal" and args[0] == pytest.approx(0.5) and args[1:] == [3, "smile"]
    state = {"type": "state", "affect": {"arousal": 0.6, "components": {"db": 1}}, "face": {"blend": {"jawOpen": 0.2}},
             "hands": [{"hand": "Right", "pinch": 0.1, "gesture": "Victory"}], "watch": {"Someone waving!": 0.3}}
    got = dict(_osc_parse(m) for m in bus.osc_addresses(state))
    assert got["/9xv/affect/arousal"][0] == pytest.approx(0.6) and "/9xv/affect/components/db" not in got
    assert got["/9xv/face/blend/jawOpen"][0] == pytest.approx(0.2)
    assert got["/9xv/hands/right/gesture"] == ["Victory"] and "/9xv/watch/someone_waving" in got
    assert _osc_parse(bus.osc_addresses({"type": "event", "kind": "beat", "t": 1.0, "n": 4})[0]) == ("/9xv/event/beat", [4])


def _clicks(seconds=24, period=0.5):
    y = np.zeros(SR * seconds, np.float32)
    clicks = np.arange(0.5, seconds, period)
    for c in clicks:
        i = int(c * SR)
        y[i:i + 200] += np.hanning(200) * 0.8 * np.sin(2 * np.pi * 2000 * np.arange(200) / SR)
    return y, clicks


def test_streaming_beats_land_on_the_clicks():
    y, clicks = _clicks()
    a, beats = StreamingAudio(), []
    for k in range(0, len(y), 1024):
        a.feed(y[k:k + 1024])
        beats += [e["at"] for e in a.events if e["kind"] == "beat"]
        a.events.clear()
    late = np.array([b for b in beats if b > 10])
    err = np.array([b - clicks[np.argmin(abs(clicks - b))] for b in late])
    assert a.bpm == pytest.approx(120, abs=1.5) and len(late) >= 20
    assert abs(np.median(err)) < 0.03 and np.abs(err).max() < 0.05


def test_pitch_and_speech():
    t = np.arange(2048) / SR
    assert yin(0.5 * np.sin(2 * np.pi * 220 * t))[0] == pytest.approx(220, abs=1)
    assert yin(np.random.default_rng(0).normal(0, 0.3, 2048))[0] == 0.0
    s = np.zeros(SR * 4, np.float32)
    tt = np.arange(SR) / SR
    s[SR:2 * SR] = 0.3 * sum(np.sin(2 * np.pi * (140 + 20 * np.sin(2 * np.pi * 3 * tt)) * k * tt) / k for k in range(1, 6))
    a, ev = StreamingAudio(), []
    for k in range(0, len(s), 512):
        a.feed(s[k:k + 512])
        ev += [(e["kind"], e["t"]) for e in a.events if e["kind"].startswith("speech")]
        a.events.clear()
    assert [k for k, _ in ev] == ["speech_start", "speech_end"] and 1.0 <= ev[0][1] < 1.4 and 2.0 <= ev[1][1] < 2.8


def test_affect_events_and_arousal():
    a, fps = affect.Affect(), 15
    blend = lambda smile: {"mouthSmileLeft": smile, "mouthSmileRight": smile, "eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.05}
    ar = {}
    for i in range(fps * 40):
        t = i / fps
        yaw = 15 * math.sin(2 * math.pi * 2.5 * (t - 20)) if 20 < t < 21.2 else 40 if 30 < t < 33 else 0.0
        pitch = -12 * math.sin(2 * math.pi * 1.25 * (t - 25)) if 25 < t < 25.8 else 0.0
        faces = [] if 35 < t < 37 else [{"yaw": yaw, "pitch": pitch, "blend": blend(0.7 if 15 < t < 17 else 0.05)}]
        loud = 5 < t < 8
        out = a.update(t, {"db": -20 if loud else -45, "onset_rate": 4 if loud else 0.5}, {"motion": 0.01},
                       {"faces": faces, "hands": [{"hand": "Right", "gesture": "Thumb_Up", "score": 0.9}] if 12 < t < 13 else []})
        ar[round(t, 2)] = out["affect"]["arousal"]
    kinds = [e["kind"] for e in a.events]
    for k in ("smile_start", "smile_end", "shake", "nod", "look_away", "look_back", "face_leave", "gesture_start"):
        assert k in kinds, (k, kinds)
    assert kinds.count("shake") == 1 and kinds.count("nod") == 1
    assert ar[4.0] == pytest.approx(0.5, abs=0.05) and ar[7.0] > 0.7


def _sync_clip(path, seconds=10):
    """Grey frames with a 2-frame white flash and a click together at k + 0.5 s."""
    fps, W, H = 30, 320, 180
    v = path.with_suffix(".v.mp4")
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", str(fps),
                          "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(v)], stdin=subprocess.PIPE)
    for i in range(fps * seconds):
        t = i / fps
        p.stdin.write(np.full((H, W), 255 if 0.5 - 1e-6 <= t % 1.0 < 0.5 + 2 / fps - 1e-6 else 70, np.uint8).tobytes())
    p.stdin.close(); p.wait()
    sr = 48000
    a = np.zeros(sr * seconds, np.float32)
    for k in range(seconds):
        i = int((k + 0.5) * sr)
        a[i:i + 480] += 0.8 * np.hanning(480) * np.sin(2 * np.pi * 1500 * np.arange(480) / sr)
    raw = path.with_suffix(".f32")
    a.tofile(raw)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(v), "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", str(raw),
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


def test_live_engine_keeps_sound_and_picture_in_sync(tmp_path, monkeypatch):
    monkeypatch.setenv("NINEXVISION_HOME", str(tmp_path / "h"))
    clip = _sync_clip(tmp_path / "sync.mp4")
    out = tmp_path / "out.jsonl"
    lv = live.Live(str(clip), face=False, ws_port=None, jsonl=str(out), log=lambda m: None)
    lv.run(8)
    L = [json.loads(x) for x in out.read_text().splitlines()]
    states = [m for m in L if m["type"] == "state"]
    cuts = [e["t"] for e in L if e["type"] == "event" and e["kind"] == "cut"]
    ons = np.array([e["t"] for e in L if e["type"] == "event" and e["kind"] == "onset"])
    flash_on = [c for i, c in enumerate(cuts) if i == 0 or c - cuts[i - 1] > 0.5]
    gaps = [ons[np.argmin(abs(ons - c))] - c for c in flash_on if len(ons)]
    assert len(states) > 100 and len(flash_on) >= 6
    assert np.median(np.abs(gaps)) < 0.05      # sound and picture agree within ~a frame


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def test_live_server_security_and_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("NINEXVISION_HOME", str(tmp_path / "h"))
    clip = _sync_clip(tmp_path / "sync.mp4")
    port = _free_port()
    lv = live.Live(str(clip), face=False, ws_port=port, loop=True, log=lambda m: None)
    th = threading.Thread(target=lv.run, args=(40,), daemon=True)
    th.start()
    try:
        time.sleep(5)
        base = f"http://127.0.0.1:{port}"

        def code(url, headers=None):
            try:
                return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=5).status
            except urllib.error.HTTPError as e:
                return e.code
        assert code(base + "/") == 200
        assert code(base + "/frame.jpg") == 401 and code(f"{base}/frame.jpg?k={lv.token}") == 200
        assert code(base + "/", {"Origin": "https://evil.example"}) == 403
        assert code(base + "/", {"Host": "evil.example"}) == 403
        from websockets.sync.client import connect
        with connect(f"ws://127.0.0.1:{port}/", open_timeout=5) as ws:      # no token: data yes, control no
            ws.send(json.dumps({"cmd": "stop", "id": "x"}))
            reply = next(m for m in (json.loads(ws.recv(timeout=5)) for _ in range(200)) if m.get("id") == "x")
            assert "token" in reply["error"]
        assert live.request({"cmd": "watch_for", "add": ["a white square"]})["watch_for"] == ["a white square"]
        r = live.request({"cmd": "snapshot", "seconds": 4, "watch": True}, timeout=300)
        idx = r["index"]
        assert "## Live signals" in idx and "cuts:" in idx and 3 <= r["seconds"] <= 4.5
    finally:
        lv.stop_flag.set()
        th.join(10)
    assert live.session() is None


def test_image_grid_and_native_regions(tmp_path, monkeypatch):
    monkeypatch.setenv("NINEXVISION_HOME", str(tmp_path / "h"))
    im = Image.new("RGB", (4200, 2600), (245, 245, 242))
    ImageDraw.Draw(im).rectangle([3400, 2300, 3700, 2500], fill=(200, 30, 30))
    p = tmp_path / "big.png"
    im.save(p)
    r = images.inspect(str(p), ocr=False)
    assert r["grid"]["cols"] == 5 and r["grid"]["rows"] == 3 and max(r["size"]) <= render.MAX_EDGE
    reg = images.inspect(str(p), "E3", ocr=False)
    assert reg["box"] == [3360, 1733, 840, 867] and reg["scale"] == 1.0
    crop = np.asarray(Image.open(reg["path"]))
    assert crop[650, 200].tolist() == [200, 30, 30]          # native pixels, the red block at x=3560, y=2383
    small = images.inspect(str(p), "0.9,0.9,0.02,0.02", ocr=False)
    assert small["scale"] == 4.0 and max(small["size"]) == 336       # enlarged, capped at 4x
    for bad in ("Z9", "1,2,3", "B2:Q1"):
        with pytest.raises(ValueError):
            images.inspect(str(p), bad, ocr=False)


def test_album_duplicates_and_text(tmp_path, monkeypatch):
    monkeypatch.setenv("NINEXVISION_HOME", str(tmp_path / "h"))
    import ninexvision.watch as w
    monkeypatch.setattr(w, "HOME", tmp_path / "h")
    d = tmp_path / "pics"
    d.mkdir()
    rng = np.random.default_rng(1)
    for i in range(3):
        a = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
        a = np.asarray(Image.fromarray(a).resize((32, 24)).resize((320, 240)))   # smooth blobs, distinct per image
        Image.fromarray(a).save(d / f"img{i}.png")
    Image.open(d / "img1.png").resize((640, 480)).save(d / "img1_big.jpg", quality=85)
    (d / "notes.txt").write_text("not an image")
    ad = images.album([str(d)], ocr=False)
    m = json.loads((ad / "watch.json").read_text())
    assert len(m["images"]) == 4 and m["duplicates"] == [[2, 3]]
    assert "near-duplicates: #2 = #3" in (ad / "INDEX.md").read_text()
    from ninexvision import tools
    assert tools.look(m["id"], "image:1")["path"].endswith("img0.png")
    assert any(x.get("images") == 4 for x in tools.list_watches())


def test_gestures_from_mediapipe_samples(tmp_path):
    from ninexvision import body
    if not body.available():
        pytest.skip("face/hand/body runtime not installed (9xvision setup)")
    try:
        data = urllib.request.urlopen("https://storage.googleapis.com/mediapipe-assets/thumb_up.jpg", timeout=10).read()
    except Exception:
        pytest.skip("offline")
    p = tmp_path / "thumb.jpg"
    p.write_bytes(data)
    rgb = np.asarray(Image.open(p).convert("RGB"))
    tr = body.Tracker(face=False, hands=True)
    try:
        for i in range(4):
            tr.submit(rgb, i / 15)
            time.sleep(0.2)
        assert tr.latest["hands"][0]["gesture"] == "Thumb_Up"
    finally:
        tr.close()
