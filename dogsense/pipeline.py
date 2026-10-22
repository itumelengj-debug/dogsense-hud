"""End-to-end per-session pipeline: frames + audio in, HUD telemetry out."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .acoustics import AudioAnalyzer
from .fusion import fuse
from .kinematics import KinematicState, KinematicTracker
from .morphology import MorphologyCalibrator

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
COCO_DOG = 16
ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class Models:
    """Shared, loaded-once models (GPU if available)."""

    def __init__(self, device: str | None = None, pose_weights: str | None = None, source_gate: bool = True):
        import torch
        from ultralytics import YOLO

        from .models import AudioClassifier, PoseClassifier, SoundSourceGate

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        w = pose_weights or next((str(p) for p in [MODELS / "dog_pose.pt", MODELS / "yolo26n-pose.pt"] if p.exists()), "yolo26n-pose.pt")
        self.pose_weights = w
        self.yolo = YOLO(w)
        # COCO detector, second stage for occluded / head-on dogs (Ultralytics downloads it on first use)
        det = MODELS / "yolo26s.pt"
        try:
            self.detector = YOLO(str(det) if det.exists() else "yolo26s.pt")
        except Exception as e:
            print("second-stage detector unavailable:", e)
            self.detector = None
        self.temporal = PoseClassifier(MODELS / "temporal_pose.pt", self.device) if (MODELS / "temporal_pose.pt").exists() else None
        self.audio = AudioClassifier(MODELS / "vocal_cnn.pt", self.device) if (MODELS / "vocal_cnn.pt").exists() else None
        try:  # rejects the wearer's own voice / whistles (downloads AST once, ~350 MB)
            self.source_gate = SoundSourceGate(self.device) if source_gate else None
        except Exception as e:  # offline and not cached: fall back to the heuristic gate only
            print("sound-source gate unavailable:", e)
            self.source_gate = None

    def detect(self, img: np.ndarray):
        """Dog boxes (N,4) and keypoints (N,24,3) in image pixels.

        Two stages: the pose model on the full frame, plus a general COCO detector that finds
        dogs the pose model misses (head-on, behind gate bars or fences). Each missed dog is
        re-read by the pose model on a padded, zoomed crop; if that still yields no skeleton the
        box is kept with zero-confidence keypoints, so the dog counts as seen.
        """
        r = self.yolo.predict(img, imgsz=640, conf=0.35, device=self.device, verbose=False)[0]
        boxes, kps = np.zeros((0, 4)), np.zeros((0, 24, 3))
        if r.keypoints is not None and len(r.boxes) and r.keypoints.data.shape[1] == 24:
            boxes, kps = r.boxes.xyxy.cpu().numpy(), r.keypoints.data.cpu().numpy()
        if self.detector is None:
            return boxes, kps
        d = self.detector.predict(img, imgsz=640, conf=0.3, classes=[COCO_DOG], device=self.device, verbose=False)[0]
        H, W = img.shape[:2]
        extra_b, extra_k = [], []
        for db in d.boxes.xyxy.cpu().numpy():
            if len(boxes) and _iou(db, boxes).max() > 0.3:
                continue  # already found by the pose model
            x1, y1, x2, y2 = db
            pw, ph = (x2 - x1) * 0.2, (y2 - y1) * 0.2
            cx1, cy1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
            cx2, cy2 = int(min(W, x2 + pw)), int(min(H, y2 + ph))
            p = self.yolo.predict(img[cy1:cy2, cx1:cx2], imgsz=640, conf=0.15, device=self.device, verbose=False)[0]
            k = np.zeros((24, 3), np.float32)
            if p.keypoints is not None and len(p.boxes) and p.keypoints.data.shape[1] == 24:
                k = p.keypoints.data[int(p.boxes.conf.argmax())].cpu().numpy().copy()
                k[:, 0] += cx1
                k[:, 1] += cy1
            extra_b.append(db); extra_k.append(k)
        if extra_b:
            boxes = np.concatenate([boxes, np.array(extra_b)]) if len(boxes) else np.array(extra_b)
            kps = np.concatenate([kps, np.array(extra_k)]) if len(kps) else np.array(extra_k)
        return boxes, kps


def _iou(b, boxes):
    x1 = np.maximum(boxes[:, 0], b[0]); y1 = np.maximum(boxes[:, 1], b[1])
    x2 = np.minimum(boxes[:, 2], b[2]); y2 = np.minimum(boxes[:, 3], b[3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6)


class Session:
    def __init__(self, models: Models, vlm=None, ema: float = 0.35):
        self.m = models
        self.kin = KinematicTracker(window=30)
        self.morph = MorphologyCalibrator()
        self.audio = AudioAnalyzer(classifier=models.audio, source_gate=models.source_gate)
        self.vlm = vlm
        self.ema = ema
        self.box = None
        self.kpts = None
        self.lost = 0
        self.arousal = 0.0
        self.valence = 0.0
        self.img_wh = (1, 1)
        self.last_audio = self.audio._last
        self.pose_probs = None

    # --- inputs ---------------------------------------------------------------------------
    def on_frame(self, jpeg: bytes, t: float | None = None, roll: float = 0.0, rotation: int = 0) -> dict:
        """rotation: clockwise degrees (0/90/180/270) needed to make the image upright."""
        t0 = time.perf_counter()
        t = time.time() if t is None else t
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return self.telemetry()
        if rotation % 360:
            img = cv2.rotate(img, ROTATE[rotation % 360])
        self.img_wh = (img.shape[1], img.shape[0])
        boxes, kps = self.m.detect(img)
        i = self._pick(boxes)
        if i is None:
            self.lost += 1
            if self.lost > 15:  # ~1 s without the dog: drop the track and its baseline
                self.kin.reset(); self.morph.reset(); self.box = self.kpts = None
        else:
            self.lost = 0
            self.box, self.kpts = boxes[i], kps[i]
            if self.kin.push(kps[i], t, roll):
                st = self.kin.state()
                if st.valid:
                    mo = self.morph.update(self.kin.feats[-1], st.stiffness)
                    if mo.calibrated:
                        self.kin.baseline = mo.baseline
                if self.m.temporal is not None and len(self.kin.xy) >= 10:
                    self.pose_probs = self.m.temporal(self.kin.sequence())
        if self.vlm is not None and self.box is not None:
            self.vlm.maybe_submit(jpeg)
        tel = self.telemetry()
        tel["_proc_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return tel

    def on_audio(self, pcm: np.ndarray, speech_prob: float = 0.0) -> dict:
        self.last_audio = self.audio.feed(pcm, speech_prob)
        return self.telemetry()

    # --- helpers --------------------------------------------------------------------------
    def _pick(self, boxes):
        if len(boxes) == 0:
            return None
        if self.box is None:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            return int(np.argmax(areas))
        # Stick with the same dog: highest IoU with the previous box.
        return int(np.argmax(_iou(self.box, boxes)))

    def telemetry(self) -> dict:
        # Only report posture while the dog is actually being seen; a short grace period rides
        # over single-frame detector misses, longer gaps must not show stale readings.
        visible = self.box is not None and self.lost <= 3
        k = self.kin.state() if visible else KinematicState()
        trig = self.vlm.trigger if self.vlm else ""
        d = fuse(k, self.last_audio, self.pose_probs if k.valid else None, trig, dog_visible=visible)
        self.arousal += self.ema * (d.arousal - self.arousal)
        self.valence += self.ema * (d.valence - self.valence)
        W, H = self.img_wh
        out = {
            "t": time.time(),
            "state": d.state, "detail": d.detail, "alert": d.alert, "trigger": d.trigger,
            "cues": d.cues, "confidence": round(d.confidence, 2),
            "arousal": round(self.arousal, 3), "valence": round(self.valence, 3),
            "box": None, "kpts": None,
            "morph": {"ears": self.morph.result.ears, "tail": self.morph.result.tail, "calibrated": self.morph.result.calibrated},
            "kin": {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in k.dict().items()},
            "audio": {kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in self.last_audio.dict().items() if kk != "probs"},
            "pose_probs": {c: round(p, 2) for c, p in (self.pose_probs or {}).items()},
        }
        if visible:
            out["box"] = [round(float(self.box[0]) / W, 4), round(float(self.box[1]) / H, 4),
                          round(float(self.box[2]) / W, 4), round(float(self.box[3]) / H, 4)]
            out["kpts"] = [[round(float(x) / W, 4), round(float(y) / H, 4), round(float(c), 2)] for x, y, c in self.kpts]
        return out
