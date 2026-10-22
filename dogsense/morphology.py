"""Breed-morphology calibration.

Instead of a brittle breed classifier, the first few seconds of observation are used to
learn this dog's neutral baseline (ear carriage, tail set, front/rear height, tail visibility)
and categorise its morphology. Rules then score *deviations* from that baseline, so a Frenchie's
bat ears or a Greyhound's low tail set do not read as "alert" or "fearful".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Morphology:
    calibrated: bool = False
    ears: str = "unknown"   # prick / semi / drop
    tail: str = "unknown"   # full / short_or_docked / curled
    baseline: dict = field(default_factory=dict)


class MorphologyCalibrator:
    def __init__(self, frames_needed: int = 45):
        self.frames_needed = frames_needed
        self.samples: list[np.ndarray] = []
        self.result = Morphology()

    def reset(self):
        self.samples.clear()
        self.result = Morphology()

    def update(self, frame_feats: np.ndarray, stiffness: float) -> Morphology:
        if self.result.calibrated:
            return self.result
        # Only learn from relatively neutral frames (not mid-zoomie or frozen).
        if 0.2 < stiffness < 0.8 or not self.samples:
            self.samples.append(frame_feats)
        if len(self.samples) >= self.frames_needed:
            F = np.stack(self.samples)
            med = lambda i: float(np.nanmedian(F[:, i])) if np.any(~np.isnan(F[:, i])) else 0.0
            ear = np.nanmedian(F[:, 4:6]) if np.any(~np.isnan(F[:, 4:6])) else 0.0
            tail_vis = med(8)
            tail_h = med(1)
            m = Morphology(calibrated=True)
            m.baseline = {"ear_fwd": float(ear), "front_drop": med(2), "tail_height": tail_h}
            m.ears = "prick" if ear > 0.25 else "drop" if ear < -0.1 else "semi"
            m.tail = "short_or_docked" if tail_vis < 0.4 else "curled" if tail_h > 0.35 else "full"
            self.result = m
        return self.result
