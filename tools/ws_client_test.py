"""Emulates the glasses: streams images + synthetic audio to the server over the real wire protocol.

    python tools/ws_client_test.py --url ws://127.0.0.1:8765/ws --images datasets/dog-pose/images/val
"""
import argparse
import asyncio
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import websockets

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "training"), str(Path(__file__).resolve().parents[1])]
from synth import synth_vocal  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--sound", default="growl")
    ap.add_argument("--rotate", type=int, default=0, help="send sideways frames (0x03) that the server must rotate by this many degrees")
    args = ap.parse_args()
    imgs = sorted(args.images.glob("*.jpg"))[: args.n]
    pcm = (synth_vocal(args.sound, np.random.RandomState(0)) * 32767).astype("<i2")
    lat = []
    async with websockets.connect(args.url, max_size=None) as ws:
        for i, p in enumerate(imgs):
            if i % 5 == 0:
                for c in range(0, len(pcm), 1600):
                    await ws.send(b"\x02" + struct.pack("<f", 0.0) + pcm[c:c + 1600].tobytes())
            t = time.time()
            if args.rotate:
                import cv2
                img = cv2.imread(str(p))
                # Undo the rotation here so the server's rotation restores the upright image.
                undo = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}[args.rotate]
                jpg = cv2.imencode(".jpg", cv2.rotate(img, undo))[1].tobytes()
                await ws.send(b"\x03" + struct.pack("<dfH", t, 0.0, args.rotate) + jpg)
            else:
                await ws.send(b"\x01" + struct.pack("<df", t, 0.0) + p.read_bytes())
            while True:  # skip audio-triggered replies until the frame reply arrives
                tel = json.loads(await ws.recv())
                if "latency_ms" in tel:
                    break
            lat.append((time.time() - t) * 1000)
            print(f"{p.name:28s} box={'Y' if tel['box'] else '-'} {tel['state']:24s} A={tel['arousal']:.2f} V={tel['valence']:+.2f} cues={tel['cues']}")
    print(f"round-trip ms: median {np.median(lat):.1f}, p90 {np.percentile(lat, 90):.1f}")


asyncio.run(main())
