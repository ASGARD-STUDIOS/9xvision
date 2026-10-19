# 9xvision

Video perception for AI agents. Give any agent (Claude Code, Codex, Cursor, anything that speaks MCP or can run a CLI) a video file or URL, and it gets back structured, timestamped evidence instead of a handful of random frames:

- **Screen recordings** (apps, websites): every UI state in order, distinct views, the transition between each (tap, scroll, sheet, swipe), OCR of every screen, stitched full pages, and a flow graph. Built for teardown work like "rebuild this onboarding".
- **Everything else** (films, music videos, talks, demos): shots split at camera cuts *and* content beats (slide changes inside one take), camera motion, palettes, colour script, on-screen text and speech per segment.
- **Drill-down tools**: `zoom` (dense frames of an interval), `frame` (full-res crop), `diff` (what changed between two moments), `search` (on-screen text and speech across everything watched).

Everything runs locally. There are no API keys and nothing is uploaded.

## Benchmark

RESULTS_PLACEHOLDER

## Install

```bash
uv tool install git+https://github.com/<owner>/9xvision     # CLI: 9xvision, MCP server: 9xvision-mcp
brew install ffmpeg                                          # required
brew install whisper-cpp tesseract                           # optional: local transcripts, non-macOS OCR
```

**Claude Code plugin** (skill + MCP server):

```
/plugin marketplace add <owner>/9xvision
/plugin install 9xvision@9x
```

**Any MCP client** (Codex, Cursor, Claude Desktop):

```json
{ "mcpServers": { "9xvision": { "command": "uvx", "args": ["--from", "git+https://github.com/<owner>/9xvision", "9xvision-mcp"] } } }
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
