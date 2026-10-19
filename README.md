# 9xvision

Video perception for AI agents. Give any agent (Claude Code, Codex, Cursor, anything that speaks MCP or can run a CLI) a video file or URL, and it gets back structured, timestamped evidence instead of a handful of random frames:

- **Screen recordings** (apps, websites): every UI state in order, distinct views, the transition between each (tap, scroll, sheet, swipe), OCR of every screen, stitched full pages, and a flow graph. Built for teardown work like "rebuild this onboarding".
- **Everything else** (films, music videos, talks, demos): shots split at camera cuts *and* content beats (slide changes inside one take), camera motion, palettes, colour script, on-screen text and speech per segment.
- **Drill-down tools**: `zoom` (dense frames of an interval), `frame` (full-res crop), `diff` (what changed between two moments), `search` (on-screen text and speech across everything watched).

Everything runs locally. There are no API keys and nothing is uploaded.

## Benchmark

The same agent (Claude Code headless, Claude Sonnet 5.5) answered the same questions about the same videos **with and without** 9xvision. Both arms had the whole machine: ffmpeg, ffprobe, whisper.cpp, tesseract and python. The kit arm also had `9xvision` and the skill text, and started from an empty cache each run, so its processing time counts. Answers were graded blind by Claude Opus 5.5 against answer keys written and verified from raw frames before the runs. 3–4 runs per cell, October 2026.

| Video | Task | Accuracy without → with | Cost / run | Agent turns | Wall time |
|---|---|---|---|---|---|
| iPhone app onboarding, 97 s | 12 questions | 99% → 99% | $0.27 → $0.25 | 20 → 15 | 84 s → 109 s |
| same recording | full screen-by-screen teardown, 14-point rubric | 98% → 96% | $0.49 → $0.29 | 27 → 13 | 157 s → 128 s |
| music video, 2:37, 110 shots | 12 questions | 92% → 98% | $0.33 → $0.25 | 28 → 15 | 98 s → 99 s |
| keynote segment, 8 min, captions | 12 questions | 94% → 100% | $0.27 → $0.18 | 21 → 13 | 56 s → 68 s |
| full keynote, 74 min, captions | 8 needle questions (on-screen only) | 98% → 98% | $0.43 → $0.42 | 39 → 28 | 156 s → 287 s |
| **all** | | **96% → 98%** | **−22%** | **−38%** | |

What this means, without spin:

- A frontier agent with ffmpeg is already a strong video reader. The kit's gains come from **fewer misses on fast, visual-only detail** (0.5 s inserts in a music video, a file path on a slide), **fewer tokens and turns**, and **a reusable cache**: every later question about the same video is instant.
- It produces things the baseline never builds: the UI flow with what changed at each step (`local change at 'Work' · 'Self-Care'`), every screen's copy, palettes, a colour script, and stitched pages.
- **Cold start is the cost.** A 74-minute video takes about 100 s to process the first time, so on a single question about a long, captioned video, caption search plus a few ffmpeg grabs is faster.
- **Selection state still needs a look.** OCR text can't tell checked from unchecked; the flow says *where* a change happened, and `diff` or `frame` shows *what*.

The answer keys were corrected three times after inspecting failures; each time the key was wrong, not the run, and both arms were re-graded. Harness and grader: `bench/`.

## Install

```bash
uv tool install git+https://github.com/koto9x/9xvision     # CLI: 9xvision, MCP server: 9xvision-mcp
brew install ffmpeg                                          # required
brew install whisper-cpp tesseract                           # optional: local transcripts, non-macOS OCR
```

**Claude Code plugin** (skill + MCP server):

```
/plugin marketplace add koto9x/9xvision
/plugin install 9xvision@9x
```

**Any MCP client** (Codex, Cursor, Claude Desktop):

```json
{ "mcpServers": { "9xvision": { "command": "uvx", "args": ["--from", "git+https://github.com/koto9x/9xvision", "9xvision-mcp"] } } }
```

**Any agent with a shell:** put `skills/9xvision/SKILL.md` in its instructions and `9xvision` on PATH.

## Use

```bash
9xvision watch recording.mp4            # prints INDEX.md, caches everything (~/.9xvision)
9xvision watch "https://youtu.be/..."   # URLs via yt-dlp
9xvision zoom <watch> 01:20 01:30 -n 16
9xvision frame <watch> 00:43.2 --crop 0,0.6,1,0.4
9xvision diff <watch> 00:17.0 00:18.3
9xvision search "limited offer"
9xvision look <watch> view:12
```

The agent reads `INDEX.md`, looks at the overview contact sheets, and drills down only where a question needs it. Every image is capped at 1568 px on the long edge, so it is never silently downscaled, and the image's token cost is printed.

## How it works

One ffmpeg decode pass at 12 fps (6 fps over an hour) computes per-frame signals: HSV content change, changed-pixel fraction, phase-correlation shift, luma and sharpness. Status bars and home indicators are masked on phone recordings.

- **Screen mode** finds stable UI states with an adaptive quiet threshold and keeps screens that never hold still as "moving" states. It clusters revisits into views on sharper 480 px frames, so toggles and checkmarks count. Transitions are classified by motion and change area, scroll groups are merged, and pages are stitched.
- **Film mode** detects cuts with adaptive thresholds and fades. Content beats are added where the picture changes and holds without a cut. Keyframes are the sharpest frame in the middle of each shot.
- **Text**: Apple Vision OCR on macOS (in a watchdog process), tesseract elsewhere.
- **Speech**: captions first (yt-dlp or a sidecar `.vtt`), then whisper.cpp locally.

## Limits

- Scroll stitching is experimental and only runs when consecutive states share content.
- Audio is transcribed, but music, sound design and tone are not described.
- OCR quality follows the source resolution; on 720p screen shares, small UI text needs `frame --crop`.

## License

MIT © Koto Murai
