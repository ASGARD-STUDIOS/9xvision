---
name: 9xvision
description: Use whenever a task involves a video (a local file, a URL from YouTube, Vimeo, TikTok, X or Instagram, a screen recording, a game or VR/XR capture, a talk, a music video, footage to edit), live input (the camera, microphone, screen or a stream, e.g. "what am I doing", "react to my face", "drive visuals from the room"), or images (a big screenshot or photo with small detail, a folder of photos or screenshots to search or sort). Turns them into structured, timestamped evidence (scenes or UI states, contact sheets, OCR text, transcript, flow, beats, visual search, capture QA, live affect and events, full-resolution image regions) so you can answer precisely instead of guessing.
---

# 9xvision: watch video properly

You cannot take video input directly. 9xvision converts a video into things you can read: a written index, labelled contact sheets, full-resolution frames, OCR text, a transcript and sound analysis. Everything carries mm:ss.s timestamps in full-video time.

## Protocol (coarse to fine)

1. **Watch once.** `9xvision watch <file-or-url>` (MCP: `watch`). It prints `INDEX.md` and caches everything, so a second call is instant. Mode is picked automatically:
   - **screen**: app or website recordings. You get distinct views (V#), every UI state in order (S#), a transition-by-transition flow, OCR of every view, and stitched full pages where the user scrolled.
   - **film**: everything else. You get shots split at frame-accurate cuts *and* content beats (slide changes, new pages in a screen share, scene changes inside one take), camera motion, palettes, a colour script, on-screen text and speech per segment, and a Sound section (tempo, whether the edit is cut to the beat, loudness jumps, silences).
   A short clip takes 20–40 s the first time; long videos about 1.5 s per minute (a 74-minute keynote: ~100 s). Give the command a timeout long enough to finish (e.g. 10 minutes) instead of abandoning it.
   **Long video, question about one stretch?** Add `--start 1:02:00 --end 1:05:00` (MCP: `start`, `end`). For a URL only that part is downloaded. Times stay in full-video time.
2. **Read INDEX.md completely** before looking at any image. It already holds the structure, the text on screen, the speech and the sound, and those are usually where the answer is.
3. **Search before scanning** when the question names a thing: `9xvision search "<words>"` (MCP: `search`). It returns on-screen text and speech hits across every watch, and visual moments in the watch: what is *in the picture* ("a red car at night", "someone hugging", "a pie chart"), with a contact sheet of the best frames. Visual scores: `strong` ≥ 0.5, `possible` ≥ 0.05; when every moment is `weak`, the thing is probably not in the video. Say so instead of guessing. It is reliable for objects, people, clothing, places and colours, looser for actions, and it cannot do negation ("an empty room", "no people"): search for what IS there. Check the sheet before you report a hit.
4. **Look at the overview sheets.** Each cell is one segment or view, labelled with its id and start time. For videos under ~10 minutes, view every sheet. For longer ones, view the colour script and the first sheet, then target.
5. **Drill down only where the question needs it:**
   - `9xvision frame <watch> 01:23.4 --crop x,y,w,h`: full-resolution pixels, for small text, numbers, prices, UI copy. Crop takes pixels or 0-1 fractions.
   - `9xvision zoom <watch> 01:20 01:30 -n 16`: a dense timestamped grid of an interval. Use it for anything fast (a 0.5 s insert shot, a toggle, a tap), or to check what happens *between* two segments.
   - `9xvision diff <watch> 00:17.0 00:18.3`: what changed between two moments, with highlighted boxes. Use it for UI state changes (a checkbox, a selection, a value).
   - `9xvision look <watch> view:12 | shot:40 | page:3 | sheet:2 | colour`: path of any artefact.
6. **Answer with timestamps.** Say what you saw and when ("at 01:35.3 a green-lit reclining Buddha, about 0.5 s"). If something is ambiguous, zoom before you answer. Don't guess.

`<watch>` is the watch id or directory printed at the top of INDEX.md.

## Game, VR/XR and app captures

`9xvision qa <file|url|watch> [--start --end] [--fov DEG]` (MCP: `qa`) checks a playtest or capture:
- **hitches**: a frame shown again while the picture changes on both sides (the build missed a refresh), with times and milliseconds; effective fps per second. Still scenes are not counted; holds of 250 ms or more are listed as **freezes** (a stall or a deliberate pause: look at them); a game capped below the capture rate is reported as a **cadence**, and only holds beyond it count.
- **stereo**: side-by-side / top-bottom captures are detected and analysed on the left eye.
- **comfort**: fast camera turns, in deg/s when you pass the headset's horizontal FOV. In a headset mirror capture this includes the player's own head turns; it matters for artificial turning, cutscenes and trailers.
- **flashes**: more than 3 per second (WCAG 2.3.1 style triage, not a certified test).
It writes a chart; look at it, then `zoom` on the reported times to see each hitch.

## Reading the outputs

- `~` after a view id means it was captured while the screen was moving (carousel, video, animation). Zoom there if it matters.
- OCR is Apple Vision (or Tesseract). It is very good, but it misreads stylised text, so check important strings against the frame.
- In screen mode the flow names the text nearest each change: `local change at −'Work' · −'Self-Care'` means that area lost colour (often: deselected); `+'Female'` gained colour (often: selected). It is a pointer to where to look, not a verdict: confirm states with `diff` or `frame` before you report them.
- In screen mode, "views" with the same text but a different look are real UI states (checked vs unchecked, selected vs not). Use `diff` on their times to see exactly what changed.
- In film mode, "via cut" is an edit and "via beat" is a content change inside a shot. ♪ marks cuts within a frame or so of a musical beat. Rhythm stats (ASL, shots per minute) count cuts only. The edit-vs-music verdict is a binomial test against chance and needs 20+ cuts; "weak pulse" means the tempo is a guess.
- Text that appears in most segments (logos, sidebars, sponsor bars) is listed once above the shots table.
- If audio exists, `transcript.txt` has timestamped speech: captions when available, otherwise local whisper.cpp.

## URLs

yt-dlp handles YouTube, Vimeo, TikTok, X, Instagram and ~1,800 other sites. A playlist URL is refused with its video URLs: pick one. A live stream needs start/end. When a fetch fails, the error ends with "Next step": follow it. If the site wants a login, ask the user before retrying with `--cookies-from-browser chrome` (MCP: `cookies_from_browser`), which reads that browser's cookies on this machine.

## Recipes

**App or site recording → product spec / clone brief**
1. List the views in flow order with their headline (from OCR) and purpose.
2. For each view, capture the copy verbatim from OCR, the components (buttons, pickers, cards, toggles, sheets) and the background and accent colours (palette in INDEX).
3. Use `diff` on the state changes to document interactions (what is selectable, what is the default, what changes on tap).
4. Write the flow as a list or mermaid graph with transition types (screen change, sheet/modal, scroll, swipe).
5. Note monetisation (paywalls, prices, ads), permission prompts and rewards with their timestamps.

**Edit / cinematography breakdown**
Use the shots table for rhythm (ASL, cuts per minute by minute), motion classes, the Sound section for tempo and cut-to-beat, the colour script for the grade arc, and the sheets for composition, locations, wardrobe and recurring motifs. Zoom on very short shots, which are often the inserts that carry meaning.

**Talk, tutorial or demo → notes**
Use the beats as the slide and section index. Pair each beat's on-screen text with the speech in the same row, and grab full frames of the slides that matter.

**Playtest → bug list**
Run `qa`, then `watch` the same capture. For each hitch and flash window, `zoom` on its time to see what was on screen (a spawn, a level load, an effect), and report it with the time and duration.

## Live input (camera, microphone, screen, streams)

Only start the camera or microphone when the user asked for it. Then:

1. `9xvision live start camera` (or MCP `live_start`; `screen`, `screen:1`, `camera:1`, a stream URL, or a file played in real time). Add `--hands`/`--pose` for gestures and body, `--watch-for "a dog"` for visual triggers, `--osc host:port` to drive TouchDesigner/Max/Ableton. It prints a monitor URL the user can open.
2. `9xvision live now` (MCP `live_now`) returns a summary, the latest frame and recent events. Look at the frame before describing the scene.
3. `9xvision live snapshot 30` (MCP `live_snapshot`) turns the last 30 s into a normal watch, with a "Live signals" section. Use zoom/frame/search on it for anything detailed.
4. `9xvision live stop` (MCP `live_stop`) when done.

Reading the signals: `energy` is absolute; `arousal` is relative to this session (0.5 = typical; the first 10 s are warm-up); `expression_valence` is the face's expression against that person's resting face, not their feelings. Say "smiling", "looked away", "nodded", not "happy", "bored", "agreed". There is no face recognition: never claim who someone is from these tools. Visual triggers judge single frames: use them for visible things ("a dog", "a laptop"), and the nod/shake/gesture events for actions.

## Images

- One image with small detail (a 4K screenshot, a document photo): `9xvision image <file|url>` (MCP `image`) gives an overview with a lettered grid when the image is bigger than you can see at full resolution, plus OCR text with grid cells. Then ask for a cell (`--region B2`), a range (`B2:C3`) or a box (`x,y,w,h`, px or 0-1) at native resolution. Quote fine print from the region, not the overview.
- A folder: `9xvision images <folder>` (MCP `images`) builds an album: INDEX.md with dates, cameras, OCR text, palettes and near-duplicates, plus contact sheets. `search "query" --watch <album>` finds images by content or text; `look <album> image:N` gives an original.

## Install (if `9xvision` is not on PATH)

`uv tool install git+https://github.com/ASGARD-STUDIOS/9xvision` then `9xvision setup`. Requires `ffmpeg`. Optional (setup offers them): `whisper-cpp` with a ggml model for local transcripts, the SigLIP model for visual search and visual triggers, `tesseract` for OCR where Apple Vision is missing, and the MediaPipe runtime for live faces, hands and body (on Linux also `sudo apt install libegl1 libgles2 libgl1`).
