"""Records a live session to disk for later review, labelling and fine-tuning.

    recordings/<YYYYmmdd-HHMMSS>/
        frames/000001.jpg ...   JPEGs exactly as received (rotate by the csv's rotation)
        frames.csv              frame index, device timestamp, server receive time, rotation
        audio.wav               16 kHz mono PCM16 (speech-flagged chunks included)
        telemetry.jsonl         every telemetry message sent to the glasses

Export a reviewable video with tools/export_session.py.
"""
from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import numpy as np


class SessionRecorder:
    def __init__(self, root: Path, every: int = 1):
        self.dir = Path(root) / time.strftime("%Y%m%d-%H%M%S")
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self.every = max(1, every)
        self.n = 0
        self.saved = 0
        self.csv = open(self.dir / "frames.csv", "w")
        self.csv.write("frame,device_t,server_t,rotation\n")
        self.tel = open(self.dir / "telemetry.jsonl", "w")
        self.wav = wave.open(str(self.dir / "audio.wav"), "wb")
        self.wav.setnchannels(1)
        self.wav.setsampwidth(2)
        self.wav.setframerate(16000)

    def frame(self, jpeg: bytes, device_t: float, rotation: int = 0):
        self.n += 1
        if self.n % self.every:
            return
        self.saved += 1
        (self.dir / "frames" / f"{self.saved:06d}.jpg").write_bytes(jpeg)
        self.csv.write(f"{self.saved},{device_t:.3f},{time.time():.3f},{rotation}\n")

    def audio(self, pcm: np.ndarray):
        self.wav.writeframes(pcm.astype("<i2").tobytes())

    def telemetry(self, tel: dict):
        self.tel.write(json.dumps(tel) + "\n")

    def close(self):
        for f in (self.csv, self.tel):
            f.close()
        self.wav.close()
        print(f"recorded {self.saved} frames to {self.dir}")
