# 9xvision

Perception for AI agents: video, live input and images. Give any agent (Claude Code, Codex, Cursor, anything that speaks MCP or can run a CLI) a video, a camera, a microphone, a screen or a folder of images, and it gets back structured, timestamped evidence instead of a handful of random frames. Underneath is one substrate: everything becomes the same kind of searchable, cached record, so the same tools work on all of it.

**Video files and URLs**

- **Screen recordings** (apps, websites): every UI state in order, distinct views, the transition between each (tap, scroll, sheet, swipe), OCR of every screen, stitched full pages, and the flow.
- **Everything else** (films, music videos, talks, demos): shots split at cuts placed on the exact frame *and* content beats (slide changes inside one take), camera motion, palettes, colour script, on-screen text and speech per segment, plus the music: tempo, beat grid, and whether the edit is cut to the beat.
- **Visual search**: find moments by what is in the picture ("motorcycles with glowing wheels at night", "a pie chart"), not only by on-screen text or speech. Runs locally.
- **Capture QA for games, XR and apps**: frame pacing and hitches with timestamps, stereo detection, fast camera turns (comfort), and flashes per second (photosensitivity triage).
- **Drill-down tools**: `zoom` (dense frames of an interval), `frame` (full-res crop), `diff` (what changed between two moments), `search`, `qa`.
- **Long videos and URLs**: watch only `--start 1:02:00 --end 1:05:00` of a three-hour stream; for a URL only that part is downloaded. All times stay in full-video time.

**Live input** (camera, microphone, screen, a stream, or a file played in real time)
- Motion, cuts, colour and light; sound level, onsets, tempo with a beat phase, voice pitch and speech; faces (51 ARKit-style blendshapes, all but tongueOut, and head pose), hands (gesture, position, pinch) and body; affect signals; and visual triggers by description ("a dog").
- Streamed as **OSC** (TouchDesigner, Max, Ableton, Resolume), **WebSocket** (browsers, games) and **JSON lines**, with a monitor page. An agent can ask what is happening now, or turn the last minutes into a normal watch.

**Images**
- One big image: an overview with a lettered grid, and any cell or box at native resolution, so an agent can read the fine print instead of guessing.
- A folder: an album with dates, cameras, text, palettes, near-duplicate groups and visual search.

Everything runs locally. There are no API keys and nothing is uploaded.

## What you get

**App and screen recordings:** the flow, step by step, naming where each change happened (`+` = that area gained colour, often a selection; `−` = lost colour):

```
00:09.3  V7 -> V8    [content change at +'Get recommendations' · +'Female']
00:10.4  V8 -> V9    [screen change]
00:17.7  V13 -> V14  [local change at −'Work' · −'Self-Care']
00:18.4  V14 -> V15  [local change at −'Hobbies']
```

Every distinct screen is kept with its text, background and palette, and the list compresses repeats ("same as V9; added: 1986 | 1987 | 1988"). The `+`/`−` labels are a pointer to where to look, not a verdict: on some apps they name the wrong neighbour, so `diff` confirms.

**Everything else:** one labelled cell per shot (or slide / scene change), with camera motion, palette, on-screen text and speech in INDEX.md, plus a colour script of the whole film:

![contact sheet](docs/film_sheet.jpg)
![colour script](docs/colour_script.jpg)

<sub>Big Buck Bunny © Blender Foundation, CC BY 3.0.</sub>

And the sound, for a music video:

```
- music: ~128.5 BPM (pulse clarity 0.385); beat times in `sound.json`
- edit vs music: partly on the beat: 45 of 109 cuts within ±63 ms of a beat (41%; chance would be 27%, p=0.00074)
- sudden rises (drops, hits, entries): 00:06.5 (+6 dB), 00:16.0 (+9.8 dB), 02:16.0 (+9.2 dB)
```

The verdict needs at least 20 cuts and a binomial test against chance; with fewer it says "too few cuts to judge", and a weak pulse (speech, ambience) is flagged as a guess.

**Visual search**, on the same music video:

```
$ 9xvision search "motorcycles with glowing wheels at night"
visual moments: 01:07.0-01:08.3 strong 0.98 · 01:09.0-01:09.5 strong 0.93 · 01:13.5-01:14.0 strong 0.82 · ...
$ 9xvision search "a giraffe"
visual moments: 02:11.5-02:12.0 weak 0.0001 · 01:45.0-01:45.5 weak 0.0001 · 00:55.0-00:55.5 weak 0
  all weak: probably not in the video
```

Scores are SigLIP's calibrated match strength (strong ≥ 0.5, possible ≥ 0.05), so "nothing matches" is a real answer instead of a confident wrong top hit. Short, generic queries ("a cat") score low even on a hit, so a score far above the rest of the video or album also counts as possible. It is reliable for objects, people, clothing, places and colours, looser for actions ("jumping over a treadmill"), and it cannot do negation ("an empty room") — the tool warns when a query uses it. Every search also returns a contact sheet of its best frames to check.

**Capture QA**, on 40 s of a Half-Life: Alyx playthrough from YouTube (only that window is downloaded):

```
$ 9xvision qa "https://www.youtube.com/watch?v=9M0_SWkRcMk" --start 1:00:00 --end 1:00:40 --fov 108
- 2400 frames at 60 fps capture; mono
- effective fps while moving: median 60.0, 5th percentile 60.0, worst second 59.0
- hitches: 1 (repeated frames while the picture moved), worst 33.3 ms; at 1:00:31.6 (33.3 ms)
- comfort: peak turn 345.21 deg/s; 13 turns above 90.0 deg/s; at 1:00:13.8-1:00:14.3 ...
- flashes: none above 3 per second
```

A hitch is a frame shown again while the picture is changing on both sides: the build missed a refresh. A still scene does not count, and a hold of 250 ms or more is reported separately as a freeze (a stall or a deliberate pause). A game capped below the capture rate (30 in 60) is reported as a cadence, and only holds beyond it count. On a 30 s Overwatch capture it reports exactly the three single-frame repeats found by hand. Side-by-side and top-bottom stereo captures are detected and analysed on the left eye. Turn speed is image rotation: in a headset mirror capture it includes the player's own head turns, so it matters most for artificial turning, cutscenes and trailers. The flash check approximates WCAG 2.3.1 for triage. It is not a certified photosensitivity test.

**Live**: start a session on a camera (`camera`), the screen, a stream or a file played in real time. Here, a test clip where a hand gives a thumbs up from 3 s to 6 s (real output):

```
$ 9xvision live start hand-demo.mp4 --hands --watch-for "a thumbs up"
[9xvision] monitor: http://127.0.0.1:8899/?k=...
$ 9xvision live now
live: ./hand-demo.mp4 at 00:07.3 into the session; 15.0 fps, 1.4 ms per frame
- affect: energy 0.0, arousal 0.495 (0.5 = typical for this session) [warming up]
- picture: motion 0.0, brightness 0.428, palette #6B6F6C
- faces: none
- watching for: a thumbs up 0.000
- recent events: gesture_start(Thumb_Up) @00:03.2, seen_start(a thumbs up) @00:03.8, gesture_end(Thumb_Up) @00:06.5, seen_end(a thumbs up) @00:06.9
$ 9xvision live snapshot 30        # the last 30 s become a normal watch, plus a "Live signals" section
```

Session time starts at the first frame. A gesture registers after a 0.2 s hold (3.0 → 3.2) and ends after a 0.5 s vote (6.0 → 6.5); a visual trigger, scored by SigLIP, needs two checks half a second apart (3.8). The clip has no sound, so there is no sound line. On a MacBook camera with a face in view, a frame takes about 10 ms.

Every message is a JSON state (about 15 a second) or an event. Over OSC the same numbers arrive as addresses:

| OSC address | what |
|---|---|
| `/9xv/affect/energy`, `/arousal`, `/expression_valence`, `/engagement` | affect (below) |
| `/9xv/audio/db`, `/onset`, `/bpm`, `/beat_phase`, `/pitch_hz`, `/speaking` | sound |
| `/9xv/video/motion`, `/brightness`, `/saturation`, `/warmth`, `/hue`, `/pan` | picture |
| `/9xv/face/smile`, `/jaw_open`, `/brow_raise`, `/yaw`, `/pitch`, `/face/blend/<name>` | face, all 51 blendshapes |
| `/9xv/hands/left/x`, `/y`, `/pinch` (1 = thumb and index touching), `/gesture` | hands |
| `/9xv/face/present`, `/9xv/hands/left/present`, `/right/present` | 1 or 0: nothing else is sent for an absent face or hand, so gate on these |
| `/9xv/watch/<query>` | visual trigger scores |
| `/9xv/event/beat`, `/cut`, `/onset`, `/smile_start`, `/nod`, `/shake`, `/gesture_start`, `/seen_start`, ... | events |

Beat events can fire on the beat rather than after it (the grid is predicted ahead, `--beat-lead` covers output latency), and `beat_phase` runs 0→1 between beats for synced visuals.

**What the affect numbers mean.** `energy` is absolute (loudness, onsets, motion), so the same scene always gives the same value. `arousal` is activation compared with this session so far: 0.5 is typical for the session, because people and rooms differ. `expression_valence` is the face's smile against frown, brow-lowering and sneer, measured against that person's resting face, so someone who smiles the whole time reads near 0. It reads the expression, not the feeling: people smile when uncomfortable and frown when concentrating. Smiles, blinks per minute, nods, head shakes, looking away and gestures come as events; nodding along gives a nod about every 1.5 s. Nothing identifies anyone: there is no face recognition.

**Privacy.** Live input is processed in memory on your machine. Nothing is written to disk unless you ask: `live now` keeps its newest frames in the session folder until the session stops; `snapshot` and `--jsonl` files stay. The WebSocket listens on 127.0.0.1. Browsers can only connect from the monitor page or an origin you allow with `--allow-origin`, so a web page you happen to visit can't read your camera's signals. Requests with a non-local `Host` are refused (no DNS rebinding), and frames and control need the session's token. Agents are told to start the camera or microphone only when the user asked.

**Images**: a 4200 × 2600 screenshot, where the overview's OCR garbles the fine print:

```
$ 9xvision image settings.png
{"source_px": [4200, 2600], "grid": {"cols": 5, "rows": 3, "cell_px": [840, 867]},
 "text": [{"text": "Quarterly settings", "cell": "A1"}, {"text": "bulla /T3a-19 ' V2.4.1 (Tine prlnt)", "cell": "E3"}], ...}
$ 9xvision image settings.png --region D3:E3
{"box": [2520, 1733, 1680, 867], "text": ["build 7f3a-19 • v2.4.1 (fine print)"], ...}
$ 9xvision images ~/Pictures/trip           # album: INDEX.md with dates, cameras, text, duplicates
$ 9xvision search "a thumbs up" --watch trip
visual matches in album trip-album-3a57952a: #5 thumb_up.jpg strong 0.83 · #2 fist.jpg weak 0.0015 · ...
```

HEIC, AVIF, TIFF, WebP, PNG, JPEG and GIF are read, with EXIF orientation applied.

## Install

```bash
uv tool install git+https://github.com/ASGARD-STUDIOS/9xvision
9xvision setup
```

`setup` plugs 9xvision into every agent it finds: **Claude Code, Codex, Cursor and Claude Desktop**, as an MCP server plus the skill. It also offers the optional pieces: on macOS it installs them with Homebrew (it asks first); on Linux it prints the commands. Those pieces are local speech-to-text (whisper.cpp plus a 466 MB model), tesseract for OCR where Apple Vision is missing, the visual search model (SigLIP, 210 MB), and face/hand/body tracking for live input (MediaPipe in its own environment, about 350 MB). Then restart the agent and ask it about a video. `9xvision doctor` shows what's installed and connected; `9xvision setup --remove` undoes it. Update later with `uv tool upgrade ninexvision` (the package name; the command is `9xvision`).

Requires Python 3.11+ and ffmpeg. Tested on macOS (Apple Silicon), Ubuntu x86-64 and Linux ARM64, Python 3.11 to 3.14. On Linux, face/hand/body tracking also needs `sudo apt install libegl1 libgles2 libgl1`.

No `uv` yet? `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux). Manual routes: Claude Code `/plugin marketplace add ASGARD-STUDIOS/9xvision` then `/plugin install 9xvision@9x`; any MCP client: command `9xvision-mcp`. Any other agent with a shell: put `9xvision` on PATH and give it the text from `9xvision skill`.

## Use

```bash
9xvision watch recording.mp4                          # prints INDEX.md, caches everything (~/.9xvision)
9xvision watch "https://youtu.be/..." --start 12:00 --end 15:00
9xvision watch "https://www.instagram.com/reel/..." --cookies-from-browser chrome   # login-gated sites
9xvision search "a red car at night"                  # on-screen text, speech, and what's in the picture
9xvision zoom <watch> 01:20 01:30 -n 16
9xvision frame <watch> 00:43.2 --crop 0,0.6,1,0.4
9xvision diff <watch> 00:17.0 00:18.3
9xvision qa capture.mp4 --fov 110                     # game / XR / app capture QA
9xvision look <watch> view:12
9xvision live devices                                 # cameras, microphones, screens
9xvision live start camera --osc 127.0.0.1:9000       # stream to TouchDesigner / Max / Ableton
9xvision live start screen --no-face                  # watch the screen
9xvision live start "https://youtube.com/watch?v=<live id>"   # a live stream (or rtsp://, srt://)
9xvision live now | snapshot 30 | watch-for "a dog" | stop
9xvision image photo.heic --region B2                 # one part at native resolution
9xvision images ./screenshots                         # an album; then search / look image:N
```

The session also serves `http://127.0.0.1:8899/mirror`: a cartoon face driven by the blendshapes, coloured by energy and arousal, pulsing on the beat. It is about 90 lines of plain JavaScript, meant as a template for your own affective visuals (`src/ninexvision/live_mirror.html`).

On macOS, the first camera, microphone or screen use asks for permission for the terminal (or agent app) running 9xvision. A web game or app can subscribe with `new WebSocket("ws://127.0.0.1:8899/")` once its dev origin is allowed: `--allow-origin http://localhost:5173`.

The agent reads `INDEX.md`, looks at the overview contact sheets, and drills down only where a question needs it. Every image an agent is shown is capped at 1568 px on the long edge, so the model never silently downscales it, and its token cost is printed; `frame --crop` gives full resolution.

Processing takes about 20–40 s for a short clip, then about 1.5 s per minute for long videos (a 74-minute keynote: ~100 s), once. After that every question is instant.

URLs go through yt-dlp: YouTube, Vimeo, TikTok, X, Instagram and about 1,800 other sites. Playlists are refused with the list of video URLs, live streams need `--start/--end`, and a failure comes back with the next step (update yt-dlp, use browser cookies, check the URL).

## Benchmark

The same agent (Claude Code headless, Claude Sonnet 5.5) answered the same questions about the same videos **with and without** 9xvision. Both arms had the whole machine: ffmpeg, ffprobe, whisper.cpp, tesseract and python. The kit arm also had `9xvision` (v0.1) and the skill text, and started from an empty cache each run, so its processing time counts. Answers were graded blind by Claude Opus 5.5 against answer keys written and verified from raw frames before the runs. 3–4 runs per cell, October 2026.

| Video | Task | Cost / run | Agent turns | Accuracy without → with | Wall time |
|---|---|---|---|---|---|
| iPhone app onboarding, 97 s | 12 questions | $0.27 → $0.25 | 20 → 15 | 99% → 99% | 84 s → 109 s |
| same recording | full screen-by-screen teardown, 14-point rubric | $0.49 → $0.29 | 27 → 13 | 98% → 96% | 157 s → 128 s |
| music video, 2:37, 110 shots | 12 questions | $0.33 → $0.25 | 28 → 15 | 92% → 98% | 98 s → 99 s |
| keynote segment, 8 min, captions | 12 questions | $0.27 → $0.18 | 21 → 13 | 94% → 100% | 56 s → 68 s |
| full keynote, 74 min, captions | 8 needle questions (on-screen only) | $0.43 → $0.42 | 39 → 28 | 98% → 98% | 156 s → 287 s |
| **all** | | **−22%** | **−38%** | 96% → 98% | |

What this means, without spin:

- **The clear win is cost and effort:** 22% cheaper and 38% fewer agent turns for the same or better answers, and a cache that makes every later question about the same video instant.
- **Accuracy:** a frontier agent with ffmpeg is already a strong video reader. 96% → 98% is within noise at 3–4 runs per cell. The misses the kit avoids are on fast, visual-only detail (0.5 s inserts in a music video, a file path on a slide).
- **Wall time** is often worse on a single question, because the first watch processes the whole video. Use `--start/--end` for a long video when the question is about one stretch.
- It builds things the baseline never does: the UI flow, every screen's copy, palettes, a colour script, stitched pages, beat alignment, capture QA.
- The visual search, sound and capture QA added in v0.3 are not in this benchmark yet. They are tested against known answers (a click track, synthetic captures with injected hitches, answer-keyed shots) in `tests/`.

The answer keys were corrected three times after inspecting failures; each time the key was wrong, not the run, and both arms were re-graded. Harness and grader: `bench/`. The videos and keys are not in the repo: one recording is private and the others are copyrighted.

## How it works

One ffmpeg decode pass at 12 fps (6 fps over an hour) computes per-frame signals: HSV content change, changed-pixel fraction, phase-correlation shift, luma and sharpness. Status bars and home indicators are masked on phone recordings.

- **Screen mode** finds stable UI states with an adaptive quiet threshold and keeps screens that never hold still as "moving" states. It clusters revisits into views on sharper 480 px frames, so toggles and checkmarks count. Transitions are classified by motion and change area, scroll groups are merged, and pages are stitched.
- **Film mode** detects cuts with adaptive thresholds and fades, then re-decodes a short window around each at the native frame rate to place it on the exact frame (102 of 103 cuts on a music video, checked against native frame differences; a dissolve or whip keeps its analysis time). Content beats are added where the picture changes and holds without a cut. Keyframes are the sharpest frame in the middle of each shot.
- **Sound**: spectral-flux onsets, tempo from autocorrelation, Ellis' dynamic-programming beat tracker (beat times within 2 ms on a click track; an outside check found 87–99% of beats within 50 ms of librosa's on pop songs, and Rick Astley at 113.2 BPM vs the published 113), RMS loudness.
- **Visual search**: SigLIP base (Google, Apache-2.0) on onnxruntime, CPU. Frames every 0.5–3 s plus every shot keyframe and screen, embedded once per watch.
- **Capture QA**: every frame at its native rate, small and grey; repeated frames, phase-correlation catch-up, packet timestamps, luminance swings.
- **Text**: Apple Vision OCR on macOS (in a watchdog process), tesseract elsewhere.
- **Live**: one ffmpeg process delivers frames and audio together, and both are stamped on the same content clock, so sound events land within ±17 ms of the true time (a click track), and picture events land at the next analysis frame (66 ms at 15 fps). Audio runs per 23 ms hop: spectral flux, YIN pitch, tempo by autocorrelation every 0.5 s, and a beat grid predicted forward (predicted beats land 11 ms after the clicks, range 6–18). Faces, hands and body run through MediaPipe in a separate process that always takes the newest frame, so tracking never falls behind (about 4–5 ms per frame for faces on an M4 Max; 32 ms with hands and body too). A face too small for MediaPipe's short-range detector is found by scanning tiles, then followed with a crop around it. Processing a 640 px frame takes about 10 ms, at 15 fps. Visual triggers run SigLIP on the newest frame twice a second and need two scores in a row, so they react in about 0.7 s.
- **Images**: Pillow, with macOS `sips` or ffmpeg for HEIC/AVIF; the grid keeps each cell under about 1,000 native pixels; small regions are enlarged up to 4× for legibility; duplicates by 64-bit difference hash.
- **Speech**: captions first (yt-dlp or a sidecar `.vtt`), then whisper.cpp locally.

## Limits

- The screen-mode `+`/`−` labels are a colour heuristic: right on a journaling app's onboarding toggles, wrong on some iOS settings sheets. Confirm with `diff`.
- Scroll stitching is experimental and only runs when consecutive states share content.
- Sound covers tempo, beats, loudness and speech. It does not describe instruments, mood or sound design.
- OCR quality follows the source resolution; on 720p screen shares, small UI text needs `frame --crop`. Stylised titles are often missed.
- Visual search is as good as SigLIP base: strong on objects, scenes and colours; looser on actions; no negation, counting, fine text or small details.
- Capture QA works from the video alone, so it sees what the viewer saw: a hitch hidden by the capture tool, or added by its encoder, is invisible or indistinguishable. Engine frame-time logs are the ground truth when you have them.
- Live affect is expression and activity, not emotion: it can't tell what someone feels, and it should not be used to judge people (hiring, school, policing; the EU AI Act bans emotion recognition at work and in education). It is built for art, music, games and your own sessions.
- Live picture analysis runs at 15 fps by default, so a one-frame event in 30 or 60 fps footage can fall between samples (`--fps 30` if it matters). Faces need to be roughly 50 px wide in the analysis frame; for a room or stage camera use `--width 1280`.
- Hands are labelled as the person's own left/right for a normal (unmirrored) camera; pass `--mirrored` for a selfie view. Head pitch direction follows MediaPipe's convention and has not been checked against a measured head.
- Visual triggers (`--watch-for`) judge single frames: use them for things you can see in a still ("a dog", "a red jacket", "a laptop"). Actions such as nodding or waving are guesses for them (an outside test saw "a person nodding" fire on a head shake); use the face and hand events for those.
- Hand gestures are MediaPipe's seven (Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou); recognition drops with hand angle and distance (an outside test: a peace sign recognised in about 1 frame in 20 at a side angle).
- Live cut detection is a light per-frame check (in an outside test it found 69% of the cuts `watch` found, 75% precise); `snapshot` the stretch and use its shots for exact cuts.
- Live sources need a picture: an audio-only file or stream is refused for now.
- Live input is tested on macOS (camera, microphone, screen, files, streams) and with files on Linux. Linux cameras (v4l2) and Windows are untested.

## License

MIT © Koto Murai
