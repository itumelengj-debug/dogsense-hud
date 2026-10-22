"""Temporal kinematic features over a sliding window of body-frame skeletons.

All measurements are in the dog's body frame (see keypoints.body_frame), so they are
invariant to wearer head motion and distance. Signals:
  * tail wag frequency / amplitude / left-right asymmetry (Quaranta et al. 2007)
  * centre-of-mass shift over the support polygon (forward-weighted vs. withdrawn)
  * play bow (front lowered relative to rear)
  * ear carriage (forward / neutral / pinned) and fast ear flicks (~200 ms)
  * global stiffness (low skeletal motion energy + still tail)
"""
from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, asdict

import numpy as np

from .keypoints import K, MIN_CONF, NUM_KPTS, body_frame, facing

FRAME_FEATS = 10  # per-frame scalar features (see frame_features)


def _ok(conf, *names):
    return all(conf[K[n]] >= MIN_CONF for n in names)


def _mean_pts(xy, conf, names):
    idx = [K[n] for n in names if conf[K[n]] >= MIN_CONF]
    return xy[idx].mean(0) if idx else None


def frame_features(xy: np.ndarray, conf: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Per-frame scalar features from a body-frame skeleton (NaN where unobservable).
    `up` is the gravity-aligned normalised skeleton (for pitch-dependent cues).

    [tail_angle, tail_height, front_drop, com_shift, ear_fwd_L, ear_fwd_R,
     head_height, head_angle, tail_visible, leg_spread]
    """
    f = np.full(FRAME_FEATS, np.nan, np.float32)
    # Tail angle relative to the backward spine direction (-x). +ve = toward +y (image down).
    if _ok(conf, "tail_start", "tail_end"):
        v = xy[K["tail_end"]] - xy[K["tail_start"]]
        f[0] = np.arctan2(v[1], -v[0])
        f[1] = -v[1]  # tail carriage: up is positive
    # Play bow: chest/elbows dropped below the hindquarters relative to gravity.
    front = _mean_pts(up, conf, ["front_left_elbow", "front_right_elbow", "withers"])
    rear = _mean_pts(up, conf, ["rear_left_elbow", "rear_right_elbow", "tail_start"])
    if front is not None and rear is not None:
        f[2] = front[1] - rear[1]  # >0: front lower than rear
        # A bow keeps the hind legs standing; a dog lying down also has a low front but
        # its hips sit near the rear paws, so it is not a bow.
        rpaw_up = _mean_pts(up, conf, ["rear_left_paw", "rear_right_paw"])
        hips = _mean_pts(up, conf, ["rear_left_elbow", "rear_right_elbow"])
        # Head-on views foreshorten the body so the front always looks low: side views only.
        lateral = _ok(conf, "withers", "tail_start")
        if not lateral or rpaw_up is None or hips is None or rpaw_up[1] - hips[1] < 0.3:
            f[2] = min(f[2], 0.0)  # standing hind legs not confirmed: no bow
    fpaw = _mean_pts(xy, conf, ["front_left_paw", "front_right_paw"])
    rpaw = _mean_pts(xy, conf, ["rear_left_paw", "rear_right_paw"])
    # Weight shift needs a lateral view where the spine defines "forward".
    if fpaw is not None and rpaw is not None and _ok(conf, "withers", "tail_start") and abs(fpaw[0] - rpaw[0]) > 0.3:
        # CoM proxy = spine midpoint (origin). 0 = over rear paws, 1 = over front paws.
        f[3] = (0.0 - rpaw[0]) / (fpaw[0] - rpaw[0])
        f[9] = abs(fpaw[0] - rpaw[0])
    nose = xy[K["nose"]] if _ok(conf, "nose") else None
    for j, side in enumerate(("left", "right")):
        b, t = f"{side}_ear_base", f"{side}_ear_tip"
        if nose is not None and _ok(conf, b, t):
            head_axis = nose - xy[K[b]]
            n = np.linalg.norm(head_axis)
            if n > 1e-3:
                # +1: ear pointing toward the nose (forward/alert), -1: pinned back.
                f[4 + j] = float(np.dot(xy[K[t]] - xy[K[b]], head_axis / n) / n)
    if nose is not None:
        f[6] = -nose[1]
        if _ok(conf, "withers"):
            v = nose - xy[K["withers"]]
            f[7] = np.arctan2(-v[1], v[0])
    f[8] = float(conf[K["tail_end"]] >= MIN_CONF)  # tail visible (0 for docked / hidden tails)
    return f


def _nanmean(a):
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else float("nan")


@dataclass
class KinematicState:
    valid: bool = False
    view: str = "side"
    wag_hz: float = 0.0
    wag_amp: float = 0.0
    wag_bias: float = 0.0       # +ve = dog's right (approach), -ve = dog's left (withdrawal)
    wag_bias_reliable: bool = False
    tail_height: float = 0.0
    play_bow: float = 0.0       # 0..1
    forward_weight: float = 0.0  # -1 (back) .. +1 (forward)
    ear_carriage: float = 0.0    # -1 pinned .. +1 forward
    ear_flick: float = 0.0       # 200 ms ear jump in units of this track's jitter (>1 = flick)
    stiffness: float = 0.0       # 0 loose .. 1 frozen
    motion: float = 0.0
    track_conf: float = 0.0

    def dict(self):
        return asdict(self)


class KinematicTracker:
    """Keeps a sliding window of skeletons for one dog and computes KinematicState."""

    def __init__(self, window: int = 30, fps: float = 15.0):
        self.window, self.fps = window, fps
        self.xy = deque(maxlen=window)
        self.feats = deque(maxlen=window)
        self.views = deque(maxlen=window)
        self.ts = deque(maxlen=window)
        self.baseline = None  # set by morphology calibration

    def reset(self):
        for d in (self.xy, self.feats, self.views, self.ts):
            d.clear()

    def push(self, kpts: np.ndarray, t: float, roll: float = 0.0) -> bool:
        xy, conf, _, ok, up = body_frame(kpts, roll)
        if not ok:
            return False
        self.xy.append(xy)
        self.feats.append(frame_features(xy, conf, up))
        self.views.append(facing(kpts))
        self.ts.append(t)
        return True

    def sequence(self) -> np.ndarray:
        """(T, 24*2 + FRAME_FEATS) tensor for the temporal network, NaN -> 0."""
        if not self.xy:
            return np.zeros((0, NUM_KPTS * 2 + FRAME_FEATS), np.float32)
        seq = np.concatenate([np.stack(self.xy).reshape(len(self.xy), -1), np.stack(self.feats)], 1)
        return np.nan_to_num(seq).astype(np.float32)

    def state(self) -> KinematicState:
        s = KinematicState()
        n = len(self.feats)
        if n < max(6, self.window // 3):
            return s
        F = np.stack(self.feats)
        ts = np.array(self.ts)
        dur = max(ts[-1] - ts[0], 1e-3)
        fps = (n - 1) / dur
        s.valid, s.track_conf = True, n / self.window
        s.view = max(set(self.views), key=list(self.views).count)

        tail = F[:, 0]
        good = ~np.isnan(tail)
        if good.sum() >= 6:
            a = np.unwrap(tail[good])
            centre = np.median(a)
            d = a - centre
            ac = d - d.mean()
            crossings = np.sum(np.diff(np.sign(ac)) != 0)
            s.wag_hz = float(crossings / 2 / (good.sum() / fps))
            s.wag_amp = float(np.std(a))
            # Lateral bias of the wag midpoint. Only a rear/front view shows lateral sweep;
            # in a rear view image-right == dog's right, in a front view it's mirrored.
            peak_r, peak_l = np.percentile(d, 90), -np.percentile(d, 10)
            bias = (peak_r - peak_l) / (peak_r + peak_l + 1e-6)
            if s.view == "front":
                bias = -bias
            s.wag_bias = float(np.clip(bias, -1, 1))
            s.wag_bias_reliable = s.view in ("front", "rear") and s.wag_amp > 0.15
            s.tail_height = _nanmean(F[:, 1])

        bow = _nanmean(F[:, 2])
        base_bow = self.baseline.get("front_drop", 0.0) if self.baseline else 0.0
        s.play_bow = float(np.clip((bow - base_bow - 0.15) / 0.35, 0, 1)) if not np.isnan(bow) else 0.0

        com = _nanmean(F[:, 3])
        s.forward_weight = float(np.clip((com - 0.5) * 4, -1, 1)) if not np.isnan(com) else 0.0

        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # frames with neither ear visible
            ears = np.nanmean(F[:, 4:6], axis=1)
        ear = _nanmean(ears)
        base_ear = self.baseline.get("ear_fwd", 0.0) if self.baseline else 0.0
        s.ear_carriage = float(np.clip((ear - base_ear) * 2.5, -1, 1)) if not np.isnan(ear) else 0.0
        k = max(2, int(round(0.2 * fps)))
        e = ears[~np.isnan(ears)]
        if e.size > k + 2:
            # Median-of-3 removes single-frame keypoint glitches; a flick must then stand out
            # from this track's own frame-to-frame jitter (robust MAD), not a fixed threshold.
            sm = np.array([np.median(e[max(0, i - 1):i + 2]) for i in range(e.size)])
            jitter = 1.4826 * np.median(np.abs(np.diff(e))) + 0.02
            s.ear_flick = float(np.max(np.abs(sm[k:] - sm[:-k])) / (4 * jitter))

        X = np.stack(self.xy)
        motion = float(np.mean(np.linalg.norm(np.diff(X, axis=0), axis=2)))
        s.motion = motion
        still_tail = 1.0 - min(s.wag_amp / 0.3, 1.0)
        still_body = 1.0 - min(motion / 0.04, 1.0)
        s.stiffness = float(0.5 * still_tail + 0.5 * still_body)
        return s
