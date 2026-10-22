"""Dog-Pose keypoint layout (Ultralytics / StanfordExtra, 24 points) and body-frame normalisation.

Ego-motion compensation: the glasses camera moves with the wearer's head, so raw screen
coordinates are meaningless for micro-movements. Every skeleton is re-expressed in a frame
anchored on the dog's own spine (tail_start -> withers), which cancels camera translation,
in-plane rotation (head roll) and distance (scale).
"""
from __future__ import annotations

import numpy as np

NAMES = [
    "front_left_paw", "front_left_knee", "front_left_elbow",
    "rear_left_paw", "rear_left_knee", "rear_left_elbow",
    "front_right_paw", "front_right_knee", "front_right_elbow",
    "rear_right_paw", "rear_right_knee", "rear_right_elbow",
    "tail_start", "tail_end", "left_ear_base", "right_ear_base",
    "nose", "chin", "left_ear_tip", "right_ear_tip",
    "left_eye", "right_eye", "withers", "throat",
]
K = {n: i for i, n in enumerate(NAMES)}
NUM_KPTS = len(NAMES)
FLIP_IDX = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 13, 15, 14, 16, 17, 19, 18, 21, 20, 22, 23]

# Skeleton edges for the optional HUD debug overlay.
EDGES = [
    (0, 1), (1, 2), (6, 7), (7, 8), (3, 4), (4, 5), (9, 10), (10, 11),
    (2, 22), (8, 22), (5, 12), (11, 12), (22, 12), (12, 13),
    (22, 23), (23, 17), (17, 16), (16, 20), (16, 21), (20, 14), (21, 15),
    (14, 18), (15, 19),
]

MIN_CONF = 0.3


def body_frame(kpts: np.ndarray, roll: float = 0.0):
    """Transform one skeleton (24,3: x, y, conf) into the dog's body frame.

    Returns (xy_body (24,2), conf (24,), spine_len_px, ok, xy_upright (24,2)). In the body
    frame the origin is the spine midpoint, +x points tail->withers and the unit is one spine
    length. xy_upright is the same translation/scale normalisation but kept aligned with
    gravity (after undoing head roll, radians, from the glasses IMU) so spine pitch - e.g. a
    play bow - stays measurable.
    """
    xy, conf = kpts[:, :2].astype(np.float32), kpts[:, 2].astype(np.float32)
    if roll:
        c, s = np.cos(-roll), np.sin(-roll)
        xy = xy @ np.array([[c, -s], [s, c]], np.float32).T
    w, t = K["withers"], K["tail_start"]
    vis = xy[conf >= MIN_CONF]
    if len(vis) < 5:
        return np.zeros((NUM_KPTS, 2), np.float32), conf, 0.0, False, None
    spread = float(np.ptp(vis, axis=0).max())
    spine_ok = conf[w] >= MIN_CONF and conf[t] >= MIN_CONF
    axis = xy[w] - xy[t]
    L = float(np.linalg.norm(axis)) if spine_ok else 0.0
    if spread < 1e-3:
        return np.zeros((NUM_KPTS, 2), np.float32), conf, 0.0, False, None
    if L < 0.25 * spread:
        # Spine hidden or foreshortened (head-on / rear view): anchor on the visible skeleton's
        # centroid and extent, with the (roll-corrected) image horizontal as the x-axis.
        ex, L = np.array([1.0, 0.0], np.float32), spread * 0.5
        origin = vis.mean(0)
    else:
        ex = axis / L
        origin = (xy[w] + xy[t]) / 2
    # Keep the gravity sense of y: the perpendicular that points "down" in the image.
    ey = np.array([-ex[1], ex[0]], np.float32)
    if ey[1] < 0:
        ey = -ey
    rel = (xy - origin) / L
    out = np.stack([rel @ ex, rel @ ey], axis=1)
    out[conf < MIN_CONF] = 0.0
    rel[conf < MIN_CONF] = 0.0
    return out.astype(np.float32), conf, L, True, rel.astype(np.float32)


def facing(kpts: np.ndarray) -> str:
    """Rough viewpoint: 'front', 'rear' or 'side' (needed to map image left/right to the dog's)."""
    c = kpts[:, 2]
    nose_vis = c[K["nose"]] > MIN_CONF
    eyes = (c[K["left_eye"]] > MIN_CONF) + (c[K["right_eye"]] > MIN_CONF)
    tail_vis = c[K["tail_start"]] > MIN_CONF
    lx, rx = kpts[K["left_ear_base"], 0], kpts[K["right_ear_base"], 0]
    both_ears = c[K["left_ear_base"]] > MIN_CONF and c[K["right_ear_base"]] > MIN_CONF
    if nose_vis and eyes == 2 and both_ears:
        return "front"
    if not nose_vis and tail_vis and both_ears:
        return "rear"
    return "side"
