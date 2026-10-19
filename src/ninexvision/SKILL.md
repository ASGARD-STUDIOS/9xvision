---
name: 9xvision
description: Use whenever a task involves a video - a local file, a URL (YouTube, Vimeo, TikTok, X, Instagram), a screen recording of an app or website, a talk, a music video, or footage to edit. Turns the video into structured, timestamped evidence (scenes or UI states, contact sheets, OCR text, transcript, flow) so you can answer precisely and cite times instead of guessing from a few frames.
---

# 9xvision: watch video properly

You cannot take video input directly. 9xvision converts a video into things you can read: a written index, labelled contact sheets, full-resolution frames, OCR text and a transcript. Everything carries mm:ss.s timestamps.

## Protocol (coarse to fine)

1. **Watch once.** `9xvision watch <file-or-url>` (MCP: `watch`). It prints `INDEX.md` and caches everything, so a second call is instant. Mode is picked automatically:
   - **screen**: app or website recordings. You get distinct views (V#), every UI state in order (S#), a transition-by-transition flow, OCR of every view, and stitched full pages where the user scrolled.
   - **film**: everything else. You get shots split at camera cuts *and* at content beats (slide changes, new pages in a screen share, scene changes inside one take), plus camera motion, palettes, a colour script, on-screen text and speech per segment.
   Processing takes about 1–2 s per minute of video (a 74-minute keynote: ~100 s), once. Give the command a timeout long enough to finish (e.g. 10 minutes) instead of abandoning it; everything after is instant.
2. **Read INDEX.md completely** before looking at any image. It already holds the structure, the text on screen and the speech, and those are usually where the answer is. For a long video with a specific question (a name, a number, a slide), `9xvision search "<word>"` first: it searches every OCR'd frame and the transcript, and returns times and keyframes to open.
3. **Look at the overview sheets.** Each cell is one segment or view, labelled with its id and start time. For videos under ~10 minutes, view every sheet. For longer ones, view the colour script and the first sheet, then target.
4. **Drill down only where the question needs it:**
   - `9xvision frame <watch> 01:23.4 --crop x,y,w,h`: full-resolution pixels, for small text, numbers, prices, UI copy. Crop takes pixels or 0-1 fractions.
   - `9xvision zoom <watch> 01:20 01:30 -n 16`: a dense timestamped grid of an interval. Use it for anything fast (a 0.5 s insert shot, a toggle, a tap), or to check what happens *between* two segments.
   - `9xvision diff <watch> 00:17.0 00:18.3`: what changed between two moments, with highlighted boxes. Use it for UI state changes (a checkbox, a selection, a value).
   - `9xvision search "<text>"`: find on-screen text or spoken words across everything watched, with times.
   - `9xvision look <watch> view:12 | shot:40 | page:3 | sheet:2 | colour`: path of any artefact.
5. **Answer with timestamps.** Say what you saw and when ("at 01:35.3 a green-lit reclining Buddha, about 0.5 s"). If something is ambiguous, zoom before you answer. Don't guess.

`<watch>` is the watch id or directory printed at the top of INDEX.md.

## Reading the outputs

- `~` after a view id means it was captured while the screen was moving (carousel, video, animation). Zoom there if it matters.
- OCR is Apple Vision (or Tesseract). It is very good, but it can misspell stylised text, so check important strings against the frame.
- In screen mode the flow names what changed: `local change at −'Work' · −'Self-Care'` means those options lost colour (deselected); `+'Female'` gained colour (selected). It is a colour heuristic, so confirm important states with `diff` or `frame`.
- In screen mode, "views" with the same text but a different look are real UI states (checked vs unchecked, selected vs not). Use `diff` on their times to see exactly what changed.
- In film mode, "via cut" is an edit and "via beat" is a content change inside a shot. Rhythm stats (ASL, cuts per minute) count cuts only.
- If audio exists, `transcript.txt` has timestamped speech: captions when available, otherwise local whisper.cpp.

## Recipes

**App or site recording → product spec / clone brief**
1. List the views in flow order with their headline (from OCR) and purpose.
2. For each view, capture the copy verbatim from OCR, the components (buttons, pickers, cards, toggles, sheets) and the background and accent colours (palette in INDEX).
3. Use `diff` on the state changes to document interactions (what is selectable, what is the default, what changes on tap).
4. Write the flow as a list or mermaid graph with transition types (screen change, sheet/modal, scroll, swipe).
5. Note monetisation (paywalls, prices, ads), permission prompts and rewards with their timestamps.

**Edit / cinematography breakdown**
Use the shots table for rhythm (ASL, cuts per minute by minute), motion classes, the colour script for the grade arc, and the sheets for composition, locations, wardrobe and recurring motifs. Zoom on very short shots, which are often the inserts that carry meaning.

**Talk, tutorial or demo → notes**
Use the beats as the slide and section index. Pair each beat's on-screen text with the speech in the same row, and grab full frames of the slides that matter.

## Install (if `9xvision` is not on PATH)

`uv tool install git+https://github.com/koto9x/9xvision` (or `uv tool install <local checkout>`). Requires `ffmpeg`. Optional: `whisper-cpp` with a ggml model for local transcripts. OCR uses Apple Vision on macOS and `tesseract` elsewhere.
