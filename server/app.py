"""DogSense inference server (runs on the local GPU box; phone/glasses connect over Wi-Fi).

WebSocket  ws://<host>:8765/ws   binary frames, little-endian:
  0x01 | f64 timestamp | f32 head_roll_rad | JPEG bytes          -> video frame (already upright)
  0x03 | f64 timestamp | f32 head_roll_rad | u16 rotation_deg | JPEG  -> frame the server rotates
       clockwise by 0/90/180/270 to make upright (CameraX rotationDegrees)
  0x02 | f32 speech_prob (device VAD, 0..1) | PCM16 mono 16 kHz  -> audio chunk
Server replies with a JSON telemetry text message after each frame (and after audio at <=10 Hz).

    python -m server.app --host 0.0.0.0 --port 8765 [--vlm qwen2.5vl:7b] [--record recordings]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import struct
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from dogsense.pipeline import Models, Session
from dogsense.vlm import SceneContext
from server.recorder import SessionRecorder

app = FastAPI(title="DogSense")
STATE: dict = {}


STATIC = Path(__file__).resolve().parent / "static"


@app.get("/")
def index():
    """Browser HUD: webcam/mic or a video file, rendered like the glasses display."""
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    m: Models = STATE["models"]
    return {"ok": True, "device": m.device, "pose": m.pose_weights,
            "temporal": m.temporal is not None, "audio": m.audio is not None}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    args = STATE["args"]
    vlm = SceneContext(model=args.vlm, url=args.vlm_url) if args.vlm else None
    sess = Session(STATE["models"], vlm)
    rec = SessionRecorder(args.record, args.record_every) if args.record else None
    lock = asyncio.Lock()  # one inference at a time per session; stale frames are dropped
    last_audio_reply = 0.0
    loop = asyncio.get_running_loop()
    try:
        while True:
            msg = await sock.receive_bytes()
            kind = msg[0]
            if kind in (0x01, 0x03):
                if lock.locked():
                    continue  # GPU busy with the previous frame: drop to keep latency low
                t, roll = struct.unpack_from("<df", msg, 1)
                rot, jpeg = (struct.unpack_from("<H", msg, 13)[0], msg[15:]) if kind == 0x03 else (0, msg[13:])
                async with lock:
                    tel = await loop.run_in_executor(None, sess.on_frame, jpeg, t, roll, rot)
                # Device and server clocks may differ: the glasses measure true round trip
                # themselves; this is only the server-side processing time.
                tel["latency_ms"] = round((time.time() - t) * 1000, 1)
                tel["server_ms"] = tel.pop("_proc_ms", None)
                if rec:
                    rec.frame(jpeg, t, rot)
                    rec.telemetry(tel)
                await sock.send_text(json.dumps(tel))
            elif kind == 0x02:
                (sp,) = struct.unpack_from("<f", msg, 1)
                pcm = np.frombuffer(msg[5:], dtype="<i2")
                if rec:
                    rec.audio(pcm)
                tel = await loop.run_in_executor(None, sess.on_audio, pcm, sp)
                if time.time() - last_audio_reply > 0.1:
                    last_audio_reply = time.time()
                    await sock.send_text(json.dumps(tel))
    except WebSocketDisconnect:
        pass
    finally:
        if rec:
            rec.close()


def lan_ips() -> list[str]:
    """IPv4 addresses of this PC, the default-route (Wi-Fi/Ethernet) one first. Others are
    often virtual adapters (WSL, Hyper-V, VPN) that the glasses cannot reach."""
    import socket

    primary = None
    try:  # the interface used for the default route (no packets are actually sent)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        primary = s.getsockname()[0]
        s.close()
    except OSError:
        pass
    others = {info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
    others = sorted(ip for ip in others if ip != primary and not ip.startswith(("127.", "169.254.")))
    return ([primary] if primary else []) + others


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--weights", default=None, help="pose weights (default models/dog_pose.pt)")
    ap.add_argument("--vlm", default="", help="Ollama VLM model name, e.g. qwen2.5vl:7b (empty = off)")
    ap.add_argument("--vlm-url", default="http://127.0.0.1:11434/api/generate")
    ap.add_argument("--record", type=Path, default=None, help="save each session (frames, audio, telemetry) under this folder")
    ap.add_argument("--record-every", type=int, default=1, help="keep every Nth frame when recording")
    args = ap.parse_args()
    STATE["args"] = args
    STATE["models"] = Models(pose_weights=args.weights)
    print("models:", health())
    print(f"Browser HUD:  http://localhost:{args.port}/")
    for n, ip in enumerate(lan_ips()):
        note = "  <- use this one (Wi-Fi/LAN)" if n == 0 else "  (other adapter, usually not reachable)"
        print(f"Glasses app:  -PserverUrl=ws://{ip}:{args.port}/ws{note}")
    if args.record:
        print(f"Recording sessions to {args.record.resolve()}")
    uvicorn.run(app, host=args.host, port=args.port, ws_max_size=4 * 1024 * 1024)


if __name__ == "__main__":
    main()
