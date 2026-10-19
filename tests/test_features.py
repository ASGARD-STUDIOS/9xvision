"""Windows, sections, sound, capture QA, visual search and error handling (needs ffmpeg)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ninexvision import audio, media, qa, semantic, tools
from ninexvision.watch import download_dir, watch
from test_pipeline import film_video  # noqa: F401  (fixture: cuts at 2, 4, 6, 8 s)

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")


def test_window_keeps_full_video_time(film_video, tmp_path, monkeypatch):
    monkeypatch.setenv("NINEXVISION_HOME", str(tmp_path))
    wd = watch(str(film_video), mode="film", out=str(tmp_path / "w"), transcribe="off", ocr=False, start="2.5", end="7.5")
    m = json.loads((wd / "watch.json").read_text())
    assert m["window"] == [2.5, 7.5]
    starts = [s["start"] for s in m["shots"]]
    assert starts[0] == 2.5 and m["shots"][-1]["end"] == 7.5
    assert any(abs(s - 4) < 0.1 for s in starts) and any(abs(s - 6) < 0.1 for s in starts)
    assert "window: 00:02.5-00:07.5" in (wd / "INDEX.md").read_text()


def test_downloaded_section_offset(film_video, tmp_path, monkeypatch):
    """A clip cut from 4 s with a .offset sidecar reports cuts at 6 and 8, like the full video."""
    monkeypatch.setenv("NINEXVISION_HOME", str(tmp_path))
    clip = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "4", "-i", str(film_video), "-c:v", "libx264", str(clip)], check=True)
    clip.with_suffix(".offset").write_text("4.0")
    wd = watch(str(clip), mode="film", out=str(tmp_path / "w"), transcribe="off", ocr=False)
    m = json.loads((wd / "watch.json").read_text())
    cuts = [s["start"] for s in m["shots"] if s["boundary"] == "cut"]
    assert m["shots"][0]["start"] == pytest.approx(4.0, abs=0.05)
    assert any(abs(c - 6) < 0.1 for c in cuts) and any(abs(c - 8) < 0.1 for c in cuts)
    f = tools.frame(str(wd), "6.5")          # original-video time maps into the clip
    assert f["t"] == 6.5 and "note" not in f
    assert "note" in tools.frame(str(wd), "1.0")   # before the section: clamped, and says so


def test_cuts_are_frame_accurate(film_video, tmp_path, monkeypatch):
    monkeypatch.setenv("NINEXVISION_HOME", str(tmp_path))
    wd = watch(str(film_video), mode="film", out=str(tmp_path / "w"), transcribe="off", ocr=False)
    cuts = [s["start"] for s in json.loads((wd / "watch.json").read_text())["shots"] if s["boundary"] == "cut"]
    assert all(min(abs(c - t) for t in (2, 4, 6, 8)) <= 1 / 24 + 1e-3 for c in cuts)


def test_look_and_frame_errors_are_explained(film_video, tmp_path, monkeypatch):
    monkeypatch.setenv("NINEXVISION_HOME", str(tmp_path))
    wd = watch(str(film_video), mode="film", out=str(tmp_path / "w"), transcribe="off", ocr=False)
    with pytest.raises(ValueError, match="no views"):
        tools.look(str(wd), "view:1")
    with pytest.raises(ValueError, match="shot:1..5"):
        tools.look(str(wd), "shot:99")
    assert "outside" in tools.frame(str(wd), "99:00")["note"]


def test_not_a_video(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello\nworld\n")
    with pytest.raises(media.MediaError):
        media.probe(p)


def test_download_dirs_do_not_collide():
    a = download_dir("https://example.com/v/BigBuckBunny_360_10s_1MB.mp4")
    b = download_dir("https://example.com/v/BigBuckBunny_1080_10s_1MB.mp4")
    c = download_dir("https://youtu.be/AbCdEf")
    d = download_dir("https://youtu.be/abcdef")
    assert len({a, b, c, d}) == 4
    assert download_dir("https://youtu.be/x", (10, 20)) != download_dir("https://youtu.be/x")


def test_ytdlp_hints():
    assert "cookies_from_browser" in media._ytdlp_hint("ERROR: Sign in to confirm you're not a bot")
    assert "out of date" in media._ytdlp_hint("ERROR: Unable to extract nsig function")
    assert "unavailable" in media._ytdlp_hint("HTTP Error 404: Not Found")


def _clicks(bpm=120.0, dur=20.0, start=1.0, sr=audio.SR):
    y = np.zeros(int(sr * dur), np.float32)
    times = np.arange(start, dur - 0.5, 60 / bpm)
    n = np.arange(400)
    for c in times:
        i = int(c * sr)
        y[i:i + 400] += (np.sin(2 * np.pi * 1500 * n / sr) * np.exp(-n / 80)).astype(np.float32)
    return y, times


def test_tempo_and_beat_timing_on_a_click_track():
    y, clicks = _clicks(bpm=120)
    env = audio.onset_envelope(y)
    bpm, clarity = audio.tempo(env)
    assert abs(bpm - 120) < 1.5 and clarity > 0.5
    beats = audio.beats(env, bpm) + audio.LATENCY
    err = [beats[np.argmin(abs(beats - c))] - c for c in clicks[2:-2]]
    assert abs(float(np.median(err))) < 0.012


def test_cut_alignment_verdict():
    beats = np.arange(0, 60, 0.5)
    on = audio.align(list(beats[4:64:2] + 0.01), beats, 120, 24)
    off = audio.align(list(beats[4:64:2] + 0.25), beats, 120, 24)
    few = audio.align(list(beats[4:16:2] + 0.01), beats, 120, 24)
    assert on["verdict"] == "cut to the beat" and on["p_value"] < 0.001
    assert off["verdict"] == "not cut to the beat"
    assert few["verdict"].startswith("too few cuts")


def _capture(path: Path, stereo=False, fps=60, seconds=8):
    """Smooth pan with held frames at 2.0 s (3 missed) and 5.0 s (6 missed), a static menu at 3.5-4.5 s,
    a whip turn at 6.0-6.5 s and an 8 Hz strobe at 7.0-8.0 s."""
    W, H = 320, 180
    rng = np.random.default_rng(1)
    from PIL import Image
    world = np.array(Image.fromarray((rng.random((H // 4, W * 2)) * 255).astype(np.uint8)).resize((W * 8, H)))
    held = {int(2.0 * fps): 3, int(5.0 * fps): 6}
    x, hold, last, frames = 0.0, 0, None, []
    for i in range(fps * seconds):
        t = i / fps
        x += 0 if 3.5 <= t < 4.5 else 30 if 6.0 <= t < 6.5 else 2
        img = world[:, int(x) % (W * 7):int(x) % (W * 7) + W].copy()
        if 7.0 <= t < 8.0 and int(t * 16) % 2:
            img = np.clip(img.astype(int) + 150, 0, 255).astype(np.uint8)
        if hold > 0:
            img, hold = last, hold - 1
        if i in held:
            hold = held[i]
        frames.append(img)
        last = img
    w = W * (2 if stereo else 1)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{H}", "-r", str(fps),
                          "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write((np.hstack([f, np.roll(f, 4, axis=1)]) if stereo else f).tobytes())
    p.stdin.close()
    p.wait()
    return path


def test_capture_qa_finds_exactly_the_injected_hitches(tmp_path):
    r = qa.analyse(_capture(tmp_path / "cap.mp4"), out_dir=tmp_path / "qa")
    hitches = r["pacing"]["hitches"]
    assert [round(h["t"]) for h in hitches] == [2, 5], hitches          # the menu pause is not a hitch
    assert [h["missed_frames"] for h in hitches] == [3, 6]
    assert any(5.8 <= t["start"] <= 6.2 for t in r["comfort"]["fast_turns"])
    assert r["flashes"]["risky_windows"] and r["flashes"]["risky_windows"][0]["start"] >= 6.9
    assert r["stereo"] == "mono" and Path(r["chart"]).exists()


def test_capture_qa_detects_side_by_side_stereo(tmp_path):
    assert qa.analyse(_capture(tmp_path / "sbs.mp4", stereo=True, seconds=3))["stereo"] == "sbs"


@pytest.mark.skipif(not semantic.available(), reason="visual search model not downloaded (9xvision setup)")
def test_visual_search_finds_the_colour(film_video, tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "models").mkdir(parents=True)
    (home / "models" / "siglip-base").symlink_to(semantic.model_dir())
    monkeypatch.setenv("NINEXVISION_HOME", str(home))
    wd = watch(str(film_video), mode="film", out=str(tmp_path / "w"), transcribe="off", ocr=False)
    m = json.loads((wd / "watch.json").read_text())
    top = semantic.search(wd, m, "a solid blue image", k=3)["moments"][0]
    assert 4.0 <= top["t"] <= 6.0        # shot 3 (4-6 s) is blue


def _sprite_capture(path: Path, fps=60, seconds=6, cap=None):
    """A character moving over a static background (most of the frame never changes). Holds at 2.0 s (1 missed
    frame) and 4.0 s (2 missed). cap=30 renders at 30 fps inside the 60 fps capture (every frame shown twice)."""
    W, H = 320, 180
    from PIL import Image, ImageDraw
    bg = Image.new("L", (W, H), 90)
    ImageDraw.Draw(bg).rectangle([0, 140, W, H], fill=60)
    held = {int(2.0 * fps): 1, int(4.0 * fps): 2}
    frames, hold, last = [], 0, None
    for i in range(fps * seconds):
        k = i if not cap else (i // (fps // cap)) * (fps // cap)
        im = bg.copy()
        x = 20 + (k * 3) % (W - 60)
        ImageDraw.Draw(im).ellipse([x, 90, x + 30, 140], fill=220)
        img = np.asarray(im)
        if hold > 0:
            img, hold = last, hold - 1
        if i in held:
            hold = held[i] * (fps // cap if cap else 1)
        frames.append(img)
        last = img
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", str(fps),
                          "-i", "-", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    return path


def test_capture_qa_character_over_static_background(tmp_path):
    r = qa.analyse(_sprite_capture(tmp_path / "sprite.mp4"))
    assert [round(h["t"]) for h in r["pacing"]["hitches"]] == [2, 4], r["pacing"]["hitches"]


def test_capture_qa_capped_frame_rate_is_a_cadence_not_hitches(tmp_path):
    r = qa.analyse(_sprite_capture(tmp_path / "capped.mp4", cap=30))
    assert r["pacing"]["cadence"] and r["pacing"]["effective_fps"]["median"] == pytest.approx(30, abs=2)
    assert [round(h["t"]) for h in r["pacing"]["hitches"]] == [2, 4], r["pacing"]["hitches"]


def test_clamped_times_say_so(film_video, tmp_path, monkeypatch):
    monkeypatch.setenv("NINEXVISION_HOME", str(tmp_path))
    wd = watch(str(film_video), mode="film", out=str(tmp_path / "w"), transcribe="off", ocr=False)
    assert "note" in tools.zoom(str(wd), "90:00", "95:00", 4)
    assert "note" in tools.diff(str(wd), "00:01", "99:00")
    assert "-5 s" in tools.frame(str(wd), -5)["note"]
    with pytest.raises(ValueError, match="can't read the time"):
        tools.frame(str(wd), "abc")
    assert "window" not in (wd / "INDEX.md").read_text()      # a full watch is not labelled partial
