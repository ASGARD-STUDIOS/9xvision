"""`9xvision doctor` and `9xvision setup`: check dependencies and plug the kit into the agents on this machine."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.request
from importlib import resources
from pathlib import Path

from . import body, extract, semantic

WHISPER_MODEL = "ggml-small.en.bin"
WHISPER_URL = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{WHISPER_MODEL}"
MODELS = Path.home() / ".9xvision" / "models"


def _ok(flag: bool) -> str:
    return "ok " if flag else "-- "


def _gl_ok() -> bool:
    """MediaPipe on Linux loads EGL and GLES even on the CPU path (headless servers often lack them)."""
    if not sys.platform.startswith("linux"):
        return True
    import ctypes.util
    return bool(ctypes.util.find_library("EGL") and ctypes.util.find_library("GLESv2"))


def checks() -> list[dict]:
    """What the kit needs (required) and what makes it better (optional), with how to get it."""
    mac = sys.platform == "darwin"
    brew = shutil.which("brew")
    pkg = "brew install" if mac and brew else "sudo apt install" if shutil.which("apt") else "install"
    ocr = extract.ocr_backend()
    whisper_cpp = shutil.which("whisper-cli") or shutil.which("whisper-cpp")
    whisper = whisper_cpp or shutil.which("whisper")   # openai-whisper fetches its own models
    model = extract._whisper_cpp_model(0) if whisper_cpp else None
    return [
        {"name": "ffmpeg + ffprobe", "required": True, "ok": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
         "fix": f"{pkg} ffmpeg", "auto": ["brew", "install", "ffmpeg"] if brew else None},
        {"name": f"OCR ({ocr or 'none'})", "required": False, "ok": ocr is not None,
         "why": "reads text on screen", "fix": f"{pkg} tesseract",
         "auto": ["brew", "install", "tesseract"] if brew else None},
        {"name": "speech-to-text (whisper)", "required": False, "ok": bool(whisper),
         "why": "transcribes speech when a video has no captions",
         "fix": "brew install whisper-cpp" if brew else "pipx install openai-whisper  (or build github.com/ggml-org/whisper.cpp)",
         "auto": ["brew", "install", "whisper-cpp"] if brew else None},
        *([{"name": "whisper model", "required": False, "ok": model is not None,
            "why": f"{WHISPER_MODEL}, 466 MB, into {MODELS}", "fix": "9xvision setup", "auto": "model"}]
          if whisper_cpp or brew else []),   # openai-whisper fetches its own models
        {"name": "visual search model", "required": False, "ok": semantic.available(),
         "why": f"SigLIP, {semantic.SIZE_MB} MB: find moments by what is in the picture", "fix": "9xvision setup",
         "auto": "siglip"},
        {"name": "face/hand/body tracking", "required": False, "ok": body.available() and _gl_ok(),
         "why": f"MediaPipe in its own environment, about {body.SIZE_MB} MB: faces, gestures and body for `9xvision live`",
         "fix": "9xvision setup" + ("" if _gl_ok() else " and: sudo apt install libegl1 libgles2 libgl1"), "auto": "body"},
    ]


def doctor() -> int:
    rows = checks()
    print("9xvision doctor")
    for r in rows:
        note = "" if r["ok"] else f"   fix: {r['fix']}" + (f"  ({r['why']})" if r.get("why") else "")
        print(f"  {_ok(r['ok'])}{r['name']}{'' if r['required'] else ' (optional)'}{note}")
    for name, found in _agents().items():
        print(f"  {_ok(found['present'])}agent: {name}" + (f"  9xvision {'registered' if found['registered'] else 'not registered (run: 9xvision setup)'}" if found["present"] else ""))
    return 0 if all(r["ok"] for r in rows if r["required"]) else 1


# ---------------------------------------------------------------- agents

def _mcp_command() -> list[str]:
    exe = shutil.which("9xvision-mcp")
    return [exe] if exe else [sys.executable, "-m", "ninexvision.mcp_server"]


def _skill_text() -> str:
    return resources.files("ninexvision").joinpath("SKILL.md").read_text()


def _json_cfg(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        return {"__unreadable__": True}


def _agents() -> dict[str, dict]:
    home = Path.home()
    desktop = home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    cursor = home / ".cursor" / "mcp.json"
    out = {}
    if shutil.which("claude"):
        r = subprocess.run(["claude", "mcp", "get", "9xvision"], capture_output=True, text=True, cwd=Path.home())
        out["Claude Code"] = {"present": True, "registered": r.returncode == 0}
    else:
        out["Claude Code"] = {"present": False, "registered": False}
    if shutil.which("codex"):
        r = subprocess.run(["codex", "mcp", "get", "9xvision"], capture_output=True, text=True, cwd=Path.home())
        out["Codex"] = {"present": True, "registered": r.returncode == 0}
    else:
        out["Codex"] = {"present": False, "registered": False}
    for name, p in (("Cursor", cursor), ("Claude Desktop", desktop)):
        present = p.parent.exists()
        out[name] = {"present": present, "registered": present and "9xvision" in _json_cfg(p).get("mcpServers", {}),
                     "config": p}
    return out


def _write_skill(dest: Path, dry: bool) -> str:
    target = dest / "9xvision" / "SKILL.md"
    if target.exists() and "name: 9xvision" not in target.read_text()[:200]:
        return f"skip skill: {target} exists and is not ours"
    if not dry:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_skill_text())
    return f"skill -> {target}"


def _merge_json(path: Path, cmd: list[str], remove: bool, dry: bool) -> str:
    cfg = _json_cfg(path)
    if cfg.get("__unreadable__"):
        return f"skip: {path} is not valid JSON, left untouched"
    servers = cfg.setdefault("mcpServers", {})
    if remove:
        if "9xvision" not in servers:
            return f"nothing to remove in {path}"
        servers.pop("9xvision")
    else:
        servers["9xvision"] = {"command": cmd[0], "args": cmd[1:]}
    if not dry:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            shutil.copy2(path, path.with_name(path.name + f".bak-9xvision-{time.strftime('%Y%m%d-%H%M%S')}"))
        path.write_text(json.dumps(cfg, indent=2) + "\n")
    return f"{'removed from' if remove else 'MCP ->'} {path} (backup kept)"


def _run(cmd: list[str], dry: bool) -> bool:
    print("  $ " + " ".join(cmd))
    return dry or subprocess.run(cmd, cwd=Path.home()).returncode == 0


def _download_model(dry: bool) -> None:
    dest = MODELS / WHISPER_MODEL
    print(f"  downloading {WHISPER_URL} -> {dest}")
    if dry:
        return
    MODELS.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(WHISPER_URL, tmp)
    tmp.rename(dest)


def setup(yes: bool = False, dry: bool = False, remove: bool = False, only: list[str] | None = None) -> int:
    """Install what is missing (asks first) and register 9xvision with every agent found here."""
    if not remove:
        missing = [r for r in checks() if not r["ok"] and r.get("auto")]
        if missing:
            print("Missing: " + ", ".join(r["name"] for r in missing))
            go = yes or dry or input("Install them now? [Y/n] ").strip().lower() in ("", "y", "yes")
            if go:
                for r in missing:
                    if r["auto"] == "model":
                        if shutil.which("whisper-cli") or shutil.which("whisper-cpp") or dry:
                            _download_model(dry)
                    elif r["auto"] == "body":
                        print(f"  creating the MediaPipe runtime (~{body.SIZE_MB} MB) -> {body.runtime()}")
                        if not dry:
                            try:
                                body.install(lambda m: print("  " + m))
                            except Exception as e:      # optional: live still runs without faces
                                print(f"  skipped: {e}")
                    elif r["auto"] == "siglip":
                        print(f"  downloading SigLIP ({semantic.SIZE_MB} MB) -> {semantic.model_dir()}")
                        if not dry:
                            semantic.download(lambda m: None)
                    else:
                        _run(r["auto"], dry)
        bad = [r["name"] for r in checks() if r["required"] and not r["ok"]] if not dry else []
        if bad:
            print("Still missing (required): " + ", ".join(bad) + ". See `9xvision doctor`.")
    cmd = _mcp_command()
    agents = _agents()
    want = {a.lower() for a in only} if only else None
    did = 0
    for name, st in agents.items():
        if want and name.lower().split()[0] not in want and name.lower() not in want:
            continue
        if not st["present"]:
            continue
        print(f"{name}:")
        if name == "Claude Code":
            if remove:
                _run(["claude", "mcp", "remove", "-s", "user", "9xvision"], dry)
            elif not st["registered"]:
                _run(["claude", "mcp", "add", "-s", "user", "9xvision", "--", *cmd], dry)
            else:
                print("  MCP already registered")
            if not remove:
                print("  " + _write_skill(Path.home() / ".claude" / "skills", dry))
        elif name == "Codex":
            if remove:
                _run(["codex", "mcp", "remove", "9xvision"], dry)
            elif not st["registered"]:
                _run(["codex", "mcp", "add", "9xvision", "--", *cmd], dry)
            else:
                print("  MCP already registered")
            if not remove:
                print("  " + _write_skill(Path.home() / ".codex" / "skills", dry))
        else:
            print("  " + _merge_json(st["config"], cmd, remove, dry))
        did += 1
    if not did:
        print("No agents found (Claude Code, Codex, Cursor, Claude Desktop). Any agent with a shell can still run "
              "`9xvision`; give it the skill text from `9xvision skill`.")
    elif dry:
        print("Dry run: nothing was changed.")
    elif not remove:
        print("Done. Restart the agent apps, then ask: \"watch <video> and tell me ...\"")
    return 0
