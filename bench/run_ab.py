#!/usr/bin/env python3
"""A/B benchmark: the same agent and model answer the same video questions with and without 9xvision.

Both arms get the whole machine (ffmpeg, ffprobe, whisper.cpp, tesseract, python3) and the same prompt.
The kit arm also gets `9xvision` on PATH and the skill text. Each run starts with an empty cache, so the
kit arm pays its own processing time. Usage:

    python3 bench/run_ab.py --bench ~/Studio/9xvision-bench --videos dailybean future figma --reps 2 --jobs 6
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]

TOOLS_NOTE = (
    "You cannot watch video directly, but you can use any command-line tool on this machine: ffmpeg, ffprobe, "
    "whisper-cli (whisper.cpp; ggml models in /opt/homebrew/share/whisper-cpp), tesseract, python3. "
    "Use the Read tool to look at images you produce."
)
KIT_NOTE = (
    "The `9xvision` video toolkit is installed (run `9xvision --help`). Follow the skill instructions below.\n\n"
)


def prompt_for(qs: dict, video_rel: str) -> str:
    if qs.get("kind") == "spec":
        return "\n".join([f"Video file: ./{video_rel}", f"Context: {qs['context']}", "", TOOLS_NOTE, "", qs["prompt"]])
    lines = [
        f"Video file: ./{video_rel}",
        f"Context: {qs['context']}",
        "",
        TOOLS_NOTE,
        "",
        "Answer every question below as precisely as you can, with mm:ss timestamps where relevant. "
        "Work efficiently. When done, end your reply with a fenced ```json block mapping each question id "
        "to your final answer string.",
        "",
    ]
    for q in qs["questions"]:
        lines.append(f"{q['id']}. {q['q']}")
    return "\n".join(lines)


def run_one(bench: Path, runs: Path, video: str, arm: str, rep: int, model: str, budget: float) -> dict:
    qs = json.loads((bench / "gt" / video / "questions.json").read_text())
    src = bench / qs["video"]
    rd = runs / f"{video}__{arm}__{rep}"
    if (rd / "result.json").exists():
        return json.loads((rd / "result.json").read_text())
    shutil.rmtree(rd, ignore_errors=True)
    (rd / "media").mkdir(parents=True)
    for f in src.parent.glob(src.stem + "*"):
        os.symlink(f, rd / "media" / f.name)
    video_rel = f"media/{src.name}"
    prompt = prompt_for(qs, video_rel)
    env = {k: v for k, v in os.environ.items() if not k.startswith("NINEXVISION")}
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", model,
           "--dangerously-skip-permissions", "--disable-slash-commands", "--strict-mcp-config",
           "--setting-sources", "project", "--max-budget-usd", str(budget)]
    if arm == "kit":
        kitbin = bench / "kitbin"   # holds only a symlink to the 9xvision binary
        env["PATH"] = f"{kitbin}:{env['PATH']}"
        env["NINEXVISION_HOME"] = str(rd / ".9xvision")
        skill = (KIT / "skills" / "9xvision" / "SKILL.md").read_text()
        skill = skill.split("---", 2)[2] if skill.startswith("---") else skill
        cmd += ["--append-system-prompt", KIT_NOTE + skill]
    (rd / "prompt.txt").write_text(prompt)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=rd, env=env, capture_output=True, text=True, timeout=3600)
    wall = time.time() - t0
    try:
        out = json.loads(res.stdout)
    except json.JSONDecodeError:
        out = {"result": res.stdout[-4000:], "error": res.stderr[-2000:]}
    rec = {
        "video": video, "arm": arm, "rep": rep, "model": model,
        "wall_s": round(wall, 1),
        "cost_usd": out.get("total_cost_usd"),
        "turns": out.get("num_turns"),
        "duration_ms": out.get("duration_ms"),
        "usage": out.get("usage"),
        "is_error": out.get("is_error"),
        "result": out.get("result"),
    }
    (rd / "result.json").write_text(json.dumps(rec, indent=1))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(Path.home() / "Studio" / "9xvision-bench"))
    ap.add_argument("--runs", default=str(Path.home() / "Studio" / "9xvision-runs"),
                    help="where agents work; kept outside --bench so the answer keys are out of reach")
    ap.add_argument("--videos", nargs="+", default=["dailybean", "future", "figma"])
    ap.add_argument("--arms", nargs="+", default=["baseline", "kit"])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--budget", type=float, default=8.0)
    a = ap.parse_args()
    bench = Path(a.bench).expanduser()
    runs = Path(a.runs).expanduser()
    kitbin = bench / "kitbin"
    kitbin.mkdir(exist_ok=True)
    link = kitbin / "9xvision"
    if not link.exists():
        link.symlink_to(KIT / ".venv" / "bin" / "9xvision")
    jobs = [(v, arm, r) for r in range(1, a.reps + 1) for v in a.videos for arm in a.arms]
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(run_one, bench, runs, v, arm, r, a.model, a.budget): (v, arm, r) for v, arm, r in jobs}
        for f in cf.as_completed(futs):
            v, arm, r = futs[f]
            try:
                rec = f.result()
                print(f"done {v}/{arm}/{r}: ${rec['cost_usd']} {rec['turns']} turns {rec['wall_s']}s", flush=True)
            except Exception as e:  # keep the batch going
                print(f"FAIL {v}/{arm}/{r}: {e}", flush=True)


if __name__ == "__main__":
    main()
