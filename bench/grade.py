#!/usr/bin/env python3
"""Blind grading: each run's answers are scored against the answer key by a separate model call
that never sees which arm produced them. Then results are aggregated per video and arm."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import subprocess
from pathlib import Path


def extract_answers(text: str) -> dict:
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text or "", re.S)
    for b in reversed(blocks):
        try:
            return json.loads(b)
        except json.JSONDecodeError:
            continue
    return {}


def grade_run(rd: Path, bench: Path, model: str) -> dict:
    out_path = rd / "grade.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    rec = json.loads((rd / "result.json").read_text())
    qs = json.loads((bench / "gt" / rec["video"] / "questions.json").read_text())
    answers = {} if qs.get("kind") == "spec" else extract_answers(rec.get("result") or "")
    if qs.get("kind") == "spec":
        answers = {"teardown": rec.get("result") or ""}
    elif not answers:
        answers = {"_unparsed_reply": (rec.get("result") or "")[-6000:]}
    blind = hashlib.sha1(str(rd).encode()).hexdigest()[:8]
    prompt = (
        "You are grading answers about a video against an answer key. Be strict and consistent. "
        "Use each question's grading rule; scores are between 0 and 1 in steps of 0.25 unless the rule says "
        "otherwise. Wrong extra claims (hallucinated screens, names or numbers) cost points. Timestamps must be "
        "within the stated tolerance. Return ONLY a JSON object: {\"scores\": {\"<id>\": number}, "
        "\"notes\": {\"<id>\": \"<10-word reason>\"}}.\n\n"
        f"ANSWER KEY:\n{json.dumps(qs['questions'], indent=1, ensure_ascii=False)}\n\n"
        f"CANDIDATE {blind} ANSWERS:\n{json.dumps(answers, indent=1, ensure_ascii=False)}\n"
    )
    res = subprocess.run(["claude", "-p", prompt, "--output-format", "json", "--model", model,
                          "--disable-slash-commands", "--strict-mcp-config", "--setting-sources", "project",
                          "--tools", ""], capture_output=True, text=True, timeout=900, cwd="/tmp")
    raw = json.loads(res.stdout).get("result", "")
    m = re.search(r"\{.*\}", raw, re.S)
    g = json.loads(m.group(0)) if m else {"scores": {}, "notes": {"_": raw[:300]}}
    g["total"] = round(sum(float(v) for v in g["scores"].values()), 2)
    g["max"] = len(qs["questions"])
    g.update({k: rec[k] for k in ("video", "arm", "rep", "cost_usd", "turns", "wall_s")})
    out_path.write_text(json.dumps(g, indent=1))
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(Path.home() / "Studio" / "9xvision-bench"))
    ap.add_argument("--runs", default=str(Path.home() / "Studio" / "9xvision-runs"))
    ap.add_argument("--model", default="opus")
    a = ap.parse_args()
    bench = Path(a.bench).expanduser()
    runs = sorted(p.parent for p in Path(a.runs).expanduser().glob("*/result.json"))
    with cf.ThreadPoolExecutor(6) as ex:
        grades = list(ex.map(lambda r: grade_run(r, bench, a.model), runs))
    table: dict = {}
    for g in grades:
        key = (g["video"], g["arm"])
        table.setdefault(key, []).append(g)
    print(f"{'video':10} {'arm':9} {'n':>2} {'score':>12} {'cost $':>8} {'turns':>6} {'wall s':>7}")
    summary = []
    for (v, arm), gs in sorted(table.items()):
        n = len(gs)
        sc = sum(g["total"] for g in gs) / n
        mx = gs[0]["max"]
        cost = sum(g["cost_usd"] or 0 for g in gs) / n
        turns = sum(g["turns"] or 0 for g in gs) / n
        wall = sum(g["wall_s"] or 0 for g in gs) / n
        print(f"{v:10} {arm:9} {n:>2} {sc:6.2f}/{mx:<2} {100*sc/mx:4.0f}% {cost:8.2f} {turns:6.1f} {wall:7.0f}")
        summary.append({"video": v, "arm": arm, "n": n, "score": round(sc, 2), "max": mx, "pct": round(100 * sc / mx, 1),
                        "cost_usd": round(cost, 3), "turns": round(turns, 1), "wall_s": round(wall)})
    (bench / "summary.json").write_text(json.dumps({"summary": summary, "runs": grades}, indent=1))


if __name__ == "__main__":
    main()
