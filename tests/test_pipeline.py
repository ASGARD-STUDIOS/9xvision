"""End-to-end tests on synthetic videos with known structure (needs ffmpeg)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ninexvision import media, tools
from ninexvision.watch import watch

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")


def _encode(frames_dir: Path, fps: int, out: Path) -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(frames_dir / "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)], check=True)
    return out


@pytest.fixture(scope="module")
def film_video(tmp_path_factory) -> Path:
    """Five solid-colour 'shots' with noise texture, cuts at 2, 4, 6, 8 s."""
    d = tmp_path_factory.mktemp("film")
    rng = np.random.default_rng(0)
    colours = [(200, 40, 40), (40, 160, 60), (30, 60, 200), (230, 200, 40), (120, 30, 140)]
    fps, n = 24, 24 * 10
    for i in range(n):
        c = np.array(colours[i // 48], dtype=np.int16)
        img = np.clip(c + rng.integers(-12, 12, (180, 320, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(d / f"f{i:05d}.png")
    return _encode(d, fps, d / "film.mp4")


@pytest.fixture(scope="module")
def screen_video(tmp_path_factory) -> Path:
    """A fake phone app: screen A (2 s) -> A with a toggle on (2 s) -> screen B (2 s) -> back to A-on (2 s)."""
    d = tmp_path_factory.mktemp("screen")
    W, H = 360, 780

    def screen(title: str, toggle: bool) -> Image.Image:
        im = Image.new("RGB", (W, H), (250, 250, 248))
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, W, 40], fill=(20, 20, 20))
        dr.text((20, 80), title, fill=(10, 10, 10))
        if title == "Profile page":   # a genuinely different layout
            dr.ellipse([W // 2 - 60, 140, W // 2 + 60, 260], fill=(230, 150, 120))
            for k in range(3):
                dr.rectangle([40, 320 + k * 60, W - 40, 340 + k * 60], fill=(200, 205, 210))
            return im
        for k in range(5):
            y = 160 + k * 90
            dr.rounded_rectangle([20, y, W - 20, y + 70], 12, outline=(80, 160, 90), width=3)
            dr.text((40, y + 25), f"Row {k + 1}", fill=(30, 30, 30))
        if toggle:
            dr.ellipse([W - 80, 182, W - 50, 212], fill=(60, 170, 80))
        dr.rectangle([20, H - 90, W - 20, H - 40], fill=(70, 160, 90))
        return im

    seq = [screen("Settings", False)] * 40 + [screen("Settings", True)] * 40 + \
          [screen("Profile page", False)] * 40 + [screen("Settings", True)] * 40
    for i, im in enumerate(seq):
        im.save(d / f"f{i:05d}.png")
    return _encode(d, 20, d / "screen.mp4")


def test_film_cuts(film_video, tmp_path, monkeypatch):
    monkeypatch.setenv("NINEXVISION_HOME", str(tmp_path))
    wd = watch(str(film_video), mode="film", out=str(tmp_path / "w"), transcribe="off", ocr=False)
    m = json.loads((wd / "watch.json").read_text())
    starts = [s["start"] for s in m["shots"]]
    assert len(m["shots"]) == 5
    for want, got in zip([0, 2, 4, 6, 8], starts):
        assert abs(want - got) <= 0.15
    assert (wd / "INDEX.md").exists() and (wd / m["sheets"][0]["path"]).exists()


def test_screen_states_and_revisit(screen_video, tmp_path, monkeypatch):
    monkeypatch.setenv("NINEXVISION_HOME", str(tmp_path))
    wd = watch(str(screen_video), mode="screen", out=str(tmp_path / "w"), transcribe="off", ocr=False)
    m = json.loads((wd / "watch.json").read_text())
    views = [s["view"] for s in m["states"]]
    assert len(m["states"]) == 4
    assert len(set(views)) == 3            # A, A-on, B
    assert views[1] == views[3]            # the return to A-on is recognised as a revisit
    assert m["transitions"][0]["type"] == "local change"   # the toggle


@pytest.mark.skipif(not __import__("ninexvision.extract", fromlist=["x"]).ocr_backend(), reason="needs an OCR backend")
def test_change_labels_name_the_toggle(screen_video, tmp_path, monkeypatch):
    monkeypatch.setenv("NINEXVISION_HOME", str(tmp_path))
    wd = watch(str(screen_video), mode="screen", out=str(tmp_path / "w"), transcribe="off", ocr=True)
    m = json.loads((wd / "watch.json").read_text())
    at = m["transitions"][0].get("at", [])
    assert any(a.startswith("+") and "Row 1" in a for a in at), at   # the green toggle on row 1 turned on


def test_drilldown_tools(screen_video, tmp_path, monkeypatch):
    monkeypatch.setenv("NINEXVISION_HOME", str(tmp_path))
    wd = watch(str(screen_video), mode="screen", out=str(tmp_path / "w"), transcribe="off", ocr=False)
    z = tools.zoom(str(wd), "0:01", "0:03", 6)
    assert Path(z["path"]).exists() and len(z["times"]) == 6
    f = tools.frame(str(wd), 3.0, crop="0.5,0.2,0.5,0.1")
    assert Path(f["path"]).exists()
    d = tools.diff(str(wd), 1.0, 3.0)
    assert d["regions"], "toggle change should produce a region"
    r = d["regions"][0]
    assert r["x"] > 200 and 150 < r["y"] < 260   # where the toggle was drawn


def test_time_parsing():
    assert media.parse_t("01:23.5") == 83.5
    assert media.parse_t("1:00:00") == 3600
    assert media.parse_t("12s") == 12
    assert media.fmt_t(83.5) == "01:23.5"


def test_plugin_skill_matches_packaged_skill():
    root = Path(__file__).resolve().parents[1]
    assert (root / "skills/9xvision/SKILL.md").read_text() == (root / "src/ninexvision/SKILL.md").read_text()
