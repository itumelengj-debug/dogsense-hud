"""Turn a recorded live session into a reviewable HUD video (with sound).

    python tools/export_session.py recordings/20261023-101500 [--out walk.mp4] [--skeleton]

Frames are placed at their real capture times, so dropped frames show as held frames
and the audio stays in sync.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dogsense.pipeline import ROTATE  # noqa: E402
from tools.simulate import draw_hud  # noqa: E402

FPS = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skeleton", action="store_true")
    args = ap.parse_args()
    d = args.session
    rows = list(csv.DictReader(open(d / "frames.csv")))
    tels = [json.loads(l) for l in open(d / "telemetry.jsonl")]
    if not rows:
        sys.exit("no frames recorded")
    # telemetry lines are written once per recorded frame, in order
    t0 = float(rows[0]["device_t"])
    out = args.out or d / "session_hud.mp4"
    silent = d / "_video.mp4"
    writer, j = None, 0
    n_out = int((float(rows[-1]["device_t"]) - t0) * FPS) + 1
    frame = None
    for k in range(n_out):
        t = t0 + k / FPS
        while j + 1 < len(rows) and float(rows[j + 1]["device_t"]) <= t:
            j += 1
        r = rows[j]
        img = cv2.imread(str(d / "frames" / f"{int(r['frame']):06d}.jpg"))
        rot = int(r.get("rotation") or 0) % 360
        if rot:
            img = cv2.rotate(img, ROTATE[rot])
        h, w = img.shape[:2]
        hud = draw_hud(tels[min(j, len(tels) - 1)], w, h, args.skeleton)
        frame = cv2.addWeighted((img * 0.55).astype(np.uint8), 1.0, hud, 1.0, 0)
        if writer is None:
            size = (w, h)
            writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
        writer.write(frame if (w, h) == size else cv2.resize(frame, size))  # camera resolution changed
    writer.release()
    wav = d / "audio.wav"
    if shutil.which("ffmpeg") and wav.exists() and wav.stat().st_size > 44:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", str(wav), "-c:v", "libx264",
                        "-crf", "22", "-c:a", "aac", "-shortest", str(out)], check=True)
        silent.unlink()
    else:
        silent.replace(out)
    print(f"{len(rows)} frames, {n_out / FPS:.0f} s -> {out}")


if __name__ == "__main__":
    main()
