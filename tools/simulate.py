"""Desktop HUD simulator: run the full pipeline on a video file / webcam (+ optional WAV) and
render the monochrome-green HUD exactly as the glasses would, without any Rokid hardware.

    python tools/simulate.py --video dog.mp4 [--wav dog.wav] [--skeleton] [--out hud.mp4]
    python tools/simulate.py --video 0          # webcam
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dogsense.keypoints import EDGES  # noqa: E402
from dogsense.pipeline import Models, Session  # noqa: E402

G = (60, 255, 60)  # the only "colour" the waveguide has; intensity encodes emphasis


def bar(v, n=10):
    k = int(round(np.clip(v, 0, 1) * n))
    return "[" + "|" * k + "." * (n - k) + "]"


def valence_bar(v, n=3):
    k = int(round(abs(np.clip(v, -1, 1)) * n))
    neg = "- " * k if v < 0 else ""
    pos = " +" * k if v > 0 else ""
    return "[" + neg.rjust(2 * n) + "|" + pos.ljust(2 * n) + "]"


def draw_hud(tel: dict, w: int, h: int, skeleton: bool) -> np.ndarray:
    hud = np.zeros((h, w, 3), np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX
    if tel.get("box"):
        x1, y1, x2, y2 = (np.array(tel["box"]) * [w, h, w, h]).astype(int)
        L = max(12, (x2 - x1) // 6)
        thick = 4 if tel["alert"] else 2  # geometry, not colour, carries urgency
        for (cx, cy, sx, sy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(hud, (cx, cy), (cx + sx * L, cy), G, thick)
            cv2.line(hud, (cx, cy), (cx, cy + sy * L), G, thick)
        if skeleton and tel.get("kpts"):
            k = np.array(tel["kpts"])
            for a, b in EDGES:
                if k[a, 2] > 0.3 and k[b, 2] > 0.3:
                    cv2.line(hud, (int(k[a, 0] * w), int(k[a, 1] * h)), (int(k[b, 0] * w), int(k[b, 1] * h)), (30, 140, 30), 1)
    s = h / 720
    cv2.putText(hud, f"AROUSAL {bar(tel['arousal'])} {tel['arousal']*100:3.0f}%", (20, int(40 * s)), cv2.FONT_HERSHEY_PLAIN, 1.6 * s, G, 2)
    cv2.putText(hud, f"VALENCE {valence_bar(tel['valence'])}", (20, int(75 * s)), cv2.FONT_HERSHEY_PLAIN, 1.6 * s, G, 2)
    if tel.get("trigger"):
        cv2.putText(hud, f"TRIGGER: {tel['trigger']}", (20, int(110 * s)), cv2.FONT_HERSHEY_PLAIN, 1.4 * s, G, 1)
    y0 = h - int(90 * s)
    cv2.rectangle(hud, (10, y0 - int(30 * s)), (w - 10, h - 10), (0, 40, 0), -1)
    if tel["alert"]:
        cv2.rectangle(hud, (10, y0 - int(30 * s)), (w - 10, h - 10), G, 3)
        cv2.putText(hud, f"ALERT: {tel['state']} - {tel['alert']}", (25, y0), f, 0.8 * s, G, 2)
    else:
        cv2.putText(hud, f"STATE: {tel['state']}" + (f" ({tel['detail']})" if tel['detail'] else ""), (25, y0), f, 0.8 * s, G, 2)
    cv2.putText(hud, " | ".join(tel["cues"]), (25, y0 + int(35 * s)), f, 0.55 * s, (40, 200, 40), 1)
    return hud


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--wav", default=None)
    ap.add_argument("--skeleton", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--log", default=None, help="write per-frame telemetry as JSON lines")
    args = ap.parse_args()

    cap = cv2.VideoCapture(int(args.video) if args.video.isdigit() else args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sess = Session(Models())
    audio = None
    if args.wav:
        import soundfile as sf
        from scipy.signal import resample_poly
        y, sr = sf.read(args.wav, dtype="float32", always_2d=True)
        y = resample_poly(y.mean(1), 16000, sr) if sr != 16000 else y.mean(1)
        audio = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    log = open(args.log, "w") if args.log else None
    writer, i, t0 = None, 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and i >= args.max_frames):
            break
        t = i / fps
        if audio is not None:
            a, b = int(t * 16000), int((t + 1 / fps) * 16000)
            if a < len(audio):
                sess.on_audio(audio[a:b])
        tel = sess.on_frame(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(), t)
        if log:
            log.write(json.dumps({"video_t": round(t, 2), **{k: v for k, v in tel.items() if k != "kpts"}}) + "\n")
        h, w = frame.shape[:2]
        hud = draw_hud(tel, w, h, args.skeleton)
        view = cv2.addWeighted((frame * 0.55).astype(np.uint8), 1.0, hud, 1.0, 0)  # see-through optics
        if args.out:
            writer = writer or cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(view)
        if not args.headless:
            cv2.imshow("DogSense HUD", view)
            if cv2.waitKey(1) == 27:
                break
        i += 1
    print(f"{i} frames in {time.time() - t0:.1f}s; last state: {tel['state'] if i else '-'}")
    if writer:
        writer.release()


if __name__ == "__main__":
    main()
