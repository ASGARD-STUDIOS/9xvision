"""Child process for face, hand and body tracking (MediaPipe). Runs in its own environment (~/.9xvision/runtimes/body)
so MediaPipe's OpenCV build never meets the kit's. Imports nothing from the kit.

stdin: one JSON config line, then frames: >IHHI header (payload bytes, width, height, timestamp ms) + RGB bytes.
stdout: one JSON line per frame."""
import json
import math
import struct
import sys

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

PALM = (0, 5, 9, 13, 17)
POSE_KEYS = {"nose": 0, "l_shoulder": 11, "r_shoulder": 12, "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24}


def head_pose(m) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees from MediaPipe's facial transformation matrix. yaw > 0: the face turns toward
    the right of the image; pitch > 0: looking up; roll > 0: head tilts clockwise in the image."""
    r = np.asarray(m)[:3, :3]
    yaw = math.degrees(math.atan2(r[0, 2], r[2, 2]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r[1, 2]))))
    roll = math.degrees(math.atan2(r[1, 0], r[1, 1]))
    return round(yaw, 1), round(-pitch, 1), round(-roll, 1)


# overlapping half-size tiles, scanned when no face is found in the whole frame
TILES = [(x, y, 0.5, 0.6) for y in (0.0, 0.4) for x in (0.0, 0.25, 0.5)]


def detect_faces(model, rgb: np.ndarray, region: tuple) -> list[dict]:
    """Faces in a region (x, y, w, h as fractions of the frame), with boxes in whole-frame fractions."""
    h, w = rgb.shape[:2]
    x0, y0 = int(region[0] * w), int(region[1] * h)
    x1, y1 = min(w, x0 + max(8, int(region[2] * w))), min(h, y0 + max(8, int(region[3] * h)))
    crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    r = model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
    cw, ch = (x1 - x0) / w, (y1 - y0) / h
    out = []
    for i, lm in enumerate(r.face_landmarks):
        xs, ys = [p.x for p in lm], [p.y for p in lm]
        f = {"box": [round(region[0] + min(xs) * cw, 3), round(region[1] + min(ys) * ch, 3),
                     round((max(xs) - min(xs)) * cw, 3), round((max(ys) - min(ys)) * ch, 3)]}
        if r.facial_transformation_matrixes:
            f["yaw"], f["pitch"], f["roll"] = head_pose(r.facial_transformation_matrixes[i])
        if r.face_blendshapes:
            f["blend"] = {c.category_name: round(c.score, 3) for c in r.face_blendshapes[i] if c.category_name != "_neutral"}
        out.append(f)
    return out


def main() -> None:
    inp, out = sys.stdin.buffer, sys.stdout
    cfg = json.loads(inp.readline())
    d, mode = cfg["models"], vision.RunningMode.VIDEO
    face = hands = pose = None
    if cfg.get("face", True):
        face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=f"{d}/face_landmarker.task"), running_mode=vision.RunningMode.IMAGE,
            num_faces=cfg.get("num_faces", 2), output_face_blendshapes=True, output_facial_transformation_matrixes=True))
    roi, misses, k = None, 0, 0
    if cfg.get("hands"):
        hands = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=f"{d}/gesture_recognizer.task"), running_mode=mode, num_hands=2))
    if cfg.get("pose"):
        pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=f"{d}/pose_landmarker_lite.task"), running_mode=mode))
    mirrored = bool(cfg.get("mirrored"))
    out.write(json.dumps({"ready": True}) + "\n")
    out.flush()
    last_ts = -1
    while True:
        head = inp.read(12)
        if len(head) < 12:
            return
        n, w, h, ts = struct.unpack(">IHHI", head)
        data = inp.read(n)
        ts = max(ts, last_ts + 1)
        last_ts = ts
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.frombuffer(data, np.uint8).reshape(h, w, 3))
        res: dict = {"ts": ts}
        if face:
            k += 1
            rgb = np.frombuffer(data, np.uint8).reshape(h, w, 3)
            faces = detect_faces(face, rgb, (0.0, 0.0, 1.0, 1.0))
            # MediaPipe's face detector is short-range: a face far from the camera is too small for it. Follow the
            # last face with a crop around it, and when nothing is known scan overlapping tiles every few frames.
            if not faces and roi is not None:
                faces = detect_faces(face, rgb, roi)
            if not faces and roi is None and k % 5 == 0:
                for tile in TILES:
                    faces = detect_faces(face, rgb, tile)
                    if faces:
                        break
            if faces:
                bx, by, bw, bh = faces[0]["box"]
                side = max(bw * w, bh * h) * 3.0
                cx, cy = (bx + bw / 2) * w, (by + bh / 2) * h
                if side < 0.6 * min(w, h):          # only worth a crop when the face is small in the frame
                    x0, y0 = max(0.0, cx - side / 2), max(0.0, cy - side / 2)
                    roi = (x0 / w, y0 / h, min(side, w - x0) / w, min(side, h - y0) / h)
                else:
                    roi = None
                misses = 0
            else:
                misses += 1
                if misses > 15:
                    roi = None
            res["faces"] = sorted(faces, key=lambda f: -f["box"][2] * f["box"][3])
        if hands:
            r = hands.recognize_for_video(img, ts)
            hs = []
            for i, lm in enumerate(r.hand_landmarks):
                side = r.handedness[i][0].category_name if r.handedness else "?"
                if not mirrored:     # MediaPipe labels hands as if the image were a mirrored selfie
                    side = {"Left": "Right", "Right": "Left"}.get(side, side)
                g = r.gestures[i][0] if r.gestures and r.gestures[i] else None
                size = math.dist((lm[0].x, lm[0].y), (lm[9].x, lm[9].y)) + 1e-6
                hs.append({"hand": side, "gesture": g.category_name if g else None, "score": round(g.score, 2) if g else 0.0,
                           "x": round(sum(lm[k].x for k in PALM) / 5, 3), "y": round(sum(lm[k].y for k in PALM) / 5, 3),
                           "pinch": round(min(1.0, math.dist((lm[4].x, lm[4].y), (lm[8].x, lm[8].y)) / size), 3)})
            res["hands"] = hs
        if pose:
            r = pose.detect_for_video(img, ts)
            if r.pose_landmarks:
                lm = r.pose_landmarks[0]
                res["pose"] = {k: [round(lm[i].x, 3), round(lm[i].y, 3), round(lm[i].visibility, 2)] for k, i in POSE_KEYS.items()}
        out.write(json.dumps(res) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
