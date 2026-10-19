"""Child process for Apple Vision OCR (see extract._OcrWorker). Reads length-prefixed PNGs, writes JSON lines."""
import json
import sys

import cv2
import numpy as np

from .extract import _ocr_direct


def main() -> None:
    inp, out = sys.stdin.buffer, sys.stdout
    while True:
        head = inp.read(4)
        if len(head) < 4:
            return
        data = inp.read(int.from_bytes(head, "big"))
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        try:
            lines = _ocr_direct(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "apple-vision")
        except Exception:
            lines = []
        out.write(json.dumps(lines) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
