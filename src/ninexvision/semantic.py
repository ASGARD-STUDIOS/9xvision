"""Visual search: find moments by what is in the picture ("a man on a motorcycle", "a pie chart"), not only by
OCR text or speech. SigLIP base (Google, Apache-2.0) via onnxruntime: CPU, local, about 210 MB of weights.

Frames are sampled across the watch, embedded once (cached as embeddings.npz in the watch folder), and a query
is scored with SigLIP's own sigmoid calibration, so a score is a probability-like match strength: low scores
mean "probably not in the video", not merely "ranked last"."""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from . import media, render

REPO = "https://huggingface.co/Xenova/siglip-base-patch16-224/resolve/main/"
FILES = {"vision_model_quantized.onnx": "onnx/vision_model_quantized.onnx",
         "text_model_quantized.onnx": "onnx/text_model_quantized.onnx",
         "tokenizer.json": "tokenizer.json"}
SIZE_MB = 210
LOGIT_SCALE, LOGIT_BIAS = float(np.exp(4.764997)), -12.932437   # google/siglip-base-patch16-224
SEQ = 64
STRONG, WEAK = 0.10, 0.01    # match-strength bands (sigmoid scores) used to label hits


def model_dir() -> Path:
    return Path(os.environ.get("NINEXVISION_HOME", Path.home() / ".9xvision")) / "models" / "siglip-base"


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    return all((model_dir() / f).exists() for f in FILES)


def download(log=print) -> Path:
    d = model_dir()
    d.mkdir(parents=True, exist_ok=True)
    for name, rel in FILES.items():
        dest = d / name
        if dest.exists():
            continue
        log(f"downloading {REPO}{rel}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(REPO + rel, tmp)
        tmp.rename(dest)
    return d


class _Model:
    _inst = None

    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        d = model_dir()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.vision = ort.InferenceSession(str(d / "vision_model_quantized.onnx"), opts, providers=["CPUExecutionProvider"])
        self.text = ort.InferenceSession(str(d / "text_model_quantized.onnx"), opts, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_padding(length=SEQ, pad_id=1, pad_token="</s>")
        self.tok.enable_truncation(SEQ)

    @classmethod
    def get(cls) -> "_Model":
        if cls._inst is None:
            if not available():
                download(lambda m: None)
            cls._inst = cls()
        return cls._inst

    def images(self, frames: list[np.ndarray], batch: int = 16) -> np.ndarray:
        from PIL import Image
        out = []
        for k in range(0, len(frames), batch):
            x = np.stack([np.asarray(Image.fromarray(f).convert("RGB").resize((224, 224), Image.BICUBIC), np.float32)
                          for f in frames[k:k + batch]])
            x = ((x / 255.0 - 0.5) / 0.5).transpose(0, 3, 1, 2).astype(np.float32)
            out.append(self.vision.run(["pooler_output"], {"pixel_values": x})[0])
        return _norm(np.concatenate(out))

    def texts(self, queries: list[str]) -> np.ndarray:
        ids = np.array([e.ids for e in self.tok.encode_batch([q.lower().strip() for q in queries])], np.int64)
        return _norm(self.text.run(["pooler_output"], {"input_ids": ids})[0])


def _norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def sample_step(span: float) -> float:
    return 0.5 if span <= 300 else 1.0 if span <= 1200 else 2.0 if span <= 3600 else 3.0


def index(wdir: Path, m: dict, log=lambda s: None) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings of frames sampled across the watch (cached). Returns (times, unit vectors)."""
    cache = wdir / "embeddings.npz"
    if cache.exists():
        z = np.load(cache)
        return z["t"], z["e"]
    t0, t1 = m.get("window") or (0.0, m["probe"]["duration"])
    step = sample_step(t1 - t0)
    model = _Model.get()
    log(f"indexing frames every {step:g}s for visual search")
    ts, embs, buf, bt = [], [], [], []
    for t, fr in media.stream_frames(m["file"], 1.0 / step, 384, start=t0 + step / 2, end=t1):
        buf.append(fr)
        bt.append(t)
        if len(buf) == 32:
            embs.append(model.images(buf)); ts += bt; buf, bt = [], []
    if buf:
        embs.append(model.images(buf)); ts += bt
    # every shot keyframe and every distinct screen too, so a 0.3 s insert is never skipped
    from PIL import Image
    extra = [(s["key_t"], s["keyframe"]) for s in m.get("shots", [])] + [(v["first_t"], v["image"]) for v in m.get("views", [])]
    for k in range(0, len(extra), 32):
        chunk = extra[k:k + 32]
        embs.append(model.images([np.asarray(Image.open(wdir / rel).convert("RGB")) for _, rel in chunk]))
        ts += [float(tt) for tt, _ in chunk]
    order = np.argsort(ts, kind="stable")
    t = np.array(ts, np.float32)[order]
    e = np.concatenate(embs)[order] if embs else np.zeros((0, 768), np.float32)
    np.savez_compressed(cache, t=t, e=e.astype(np.float16))
    return t, e


def search(wdir: Path, m: dict, query: str, k: int = 8, log=lambda s: None) -> dict:
    """Top moments for a visual description, merged into time ranges, plus a contact sheet of the best frames."""
    t, e = index(wdir, m, log)
    if not len(t):
        return {"query": query, "moments": []}
    q = _Model.get().texts([query])[0]
    score = 1 / (1 + np.exp(-(LOGIT_SCALE * (e.astype(np.float32) @ q) + LOGIT_BIAS)))
    order = np.argsort(-score)
    t0, t1 = m.get("window") or (0.0, m["probe"]["duration"])
    step = sample_step(t1 - t0)
    moments: list[dict] = []
    for i in order:
        if len(moments) >= k or score[i] < score[order[0]] * 0.2:
            break
        ti = float(t[i])
        near = next((mo for mo in moments if mo["start"] - 1.5 * step <= ti <= mo["end"] + 1.5 * step), None)
        if near:
            near["start"], near["end"] = min(near["start"], ti), max(near["end"], ti)
            continue
        moments.append({"t": round(ti, 2), "start": ti, "end": ti, "score": round(float(score[i]), 4)})
    for mo in moments:   # extend each moment over neighbouring samples that still match
        for d in (-1, 1):
            j = int(np.argmin(np.abs(t - mo["t"])))
            while 0 <= j + d < len(t) and score[j + d] >= max(mo["score"] * 0.5, WEAK):
                j += d
            mo["start" if d < 0 else "end"] = (min if d < 0 else max)(mo["start" if d < 0 else "end"], float(t[j]))
        mo["start"], mo["end"] = round(max(t0, mo["start"] - step / 2), 2), round(min(t1, mo["end"] + step / 2), 2)
    merged: list[dict] = []   # extended ranges can overlap: keep one moment per range, best score
    for mo in sorted(moments, key=lambda x: -x["score"]):
        if not any(o["start"] <= mo["t"] <= o["end"] for o in merged):
            merged.append(mo)
    moments = merged
    for mo in moments:
        mo["match"] = "strong" if mo["score"] >= STRONG else "possible" if mo["score"] >= WEAK else "weak"
        mo["at"] = f"{media.fmt_t(max(0, mo['start']))}-{media.fmt_t(mo['end'])}"
    sheet = None
    if moments:
        frames = list(media.grab_frames(m["file"], [mo["t"] for mo in moments], max_edge=720))
        p = wdir / "search" / f"visual_{hashlib.sha1(query.encode()).hexdigest()[:10]}.jpg"
        size = render.contact_sheet(frames, [f"{media.fmt_t(mo['t'])} {mo['match']} {mo['score']:.2f}" for mo in moments], p,
                                    title=f"visual search: {query}"[:90])
        sheet = {"path": str(p), "tokens": render.token_estimate(*size)}
    return {"query": query, "sampled_every_s": step, "moments": moments, "sheet": sheet,
            "note": "score = SigLIP match strength (0-1). strong >= 0.10, possible >= 0.01; "
                    "all-weak results usually mean it is not in the video. Confirm with frame/zoom."}


def store_meta(wdir: Path) -> None:
    """Record in watch.json that a visual index exists (for list/INDEX)."""
    p = wdir / "watch.json"
    m = json.loads(p.read_text())
    m["visual_index"] = "embeddings.npz"
    p.write_text(json.dumps(m, indent=1, ensure_ascii=False))
