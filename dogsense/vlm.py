"""Async scene-context pass with a local Vision-Language Model (Ollama API, e.g. qwen2.5vl:7b).

Runs every few seconds off the hot path; the latest trigger tag is attached to the fused
decision so the HUD can say *why* the dog reacts. Backs off quietly if no server is reachable.
"""
from __future__ import annotations

import asyncio
import base64
import json
import time
import urllib.request

PROMPT = (
    "You see a dog through smart glasses. In at most 4 words, name the most likely environmental "
    "trigger for the dog's behaviour. Choose from: Approaching Stranger, Other Dog, Resource/Toy, "
    "Food, Leash Tension, Vehicle, Child, Owner, None. Reply with only the label."
)


class SceneContext:
    def __init__(self, url="http://127.0.0.1:11434/api/generate", model="qwen2.5vl:7b", period=3.0, enabled=True):
        self.url, self.model, self.period, self.enabled = url, model, period, enabled
        self.trigger, self._last, self._busy = "", 0.0, False

    def maybe_submit(self, jpeg: bytes):
        if not self.enabled or self._busy or time.time() - self._last < self.period:
            return
        self._busy, self._last = True, time.time()
        asyncio.get_running_loop().run_in_executor(None, self._query, jpeg)

    def _query(self, jpeg: bytes):
        try:
            body = json.dumps({"model": self.model, "prompt": PROMPT, "stream": False,
                               "images": [base64.b64encode(jpeg).decode()],
                               "options": {"num_predict": 8, "temperature": 0}}).encode()
            req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=8) as r:
                label = json.loads(r.read())["response"].strip().strip(".").split("\n")[0][:32]
            self.trigger = "" if label.lower() == "none" else label
        except Exception:
            self._last = time.time() + 60  # VLM server unreachable: back off for a minute
        finally:
            self._busy = False
