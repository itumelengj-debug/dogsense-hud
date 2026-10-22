"""Procedural data generators used to bootstrap the temporal and vocal heads.

There is no public dataset of *labelled behaviour sequences* aligned with Dog-Pose keypoints,
so the temporal net is pre-trained on kinematically-plausible synthetic trajectories (rendered
through random viewpoints, head-motion, scale, jitter and keypoint dropout) and should then be
fine-tuned on real clips via `train_temporal.py --videos`. Likewise the vocal CNN can be
pre-trained on synthetic vocalisations and fine-tuned on DMAD / CREMD / your own recordings.
"""
from __future__ import annotations

import numpy as np

from dogsense.acoustics import SR
from dogsense.keypoints import K, NUM_KPTS

# Side view, dog facing +x, y down, spine length = 1.
BASE = {
    "withers": (0.5, 0.0), "tail_start": (-0.5, 0.0), "throat": (0.72, -0.08),
    "chin": (0.95, -0.12), "nose": (1.08, -0.22), "left_eye": (0.92, -0.33), "right_eye": (0.95, -0.32),
    "left_ear_base": (0.76, -0.42), "right_ear_base": (0.8, -0.41),
    "left_ear_tip": (0.80, -0.68), "right_ear_tip": (0.84, -0.67), "tail_end": (-0.95, -0.28),
    "front_left_elbow": (0.42, 0.35), "front_left_knee": (0.44, 0.68), "front_left_paw": (0.46, 1.0),
    "front_right_elbow": (0.48, 0.35), "front_right_knee": (0.50, 0.68), "front_right_paw": (0.52, 1.0),
    "rear_left_elbow": (-0.42, 0.35), "rear_left_knee": (-0.50, 0.68), "rear_left_paw": (-0.44, 1.0),
    "rear_right_elbow": (-0.38, 0.35), "rear_right_knee": (-0.46, 0.68), "rear_right_paw": (-0.40, 1.0),
}
BASE_ARR = np.array([BASE[n] for n in sorted(BASE, key=lambda n: K[n])], np.float32)
FRONT = [K[n] for n in BASE if n.startswith("front")] + [K[n] for n in ("withers", "throat", "chin", "nose", "left_eye", "right_eye", "left_ear_base", "right_ear_base", "left_ear_tip", "right_ear_tip")]
HEAD = [K[n] for n in ("throat", "chin", "nose", "left_eye", "right_eye", "left_ear_base", "right_ear_base", "left_ear_tip", "right_ear_tip")]
EARTIPS = [(K["left_ear_base"], K["left_ear_tip"]), (K["right_ear_base"], K["right_ear_tip"])]


def _rot(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


BEHAVIOURS = {
    # wag_hz, wag_amp(rad), wag_bias(rad), tail_lift, bow, lean(+fwd), ear(+fwd/-pinned), jitter, bounce
    "relaxed":         dict(hz=(0, 1.2), amp=(0, .25), bias=(-.05, .05), lift=(-.15, .1), bow=(0, .05), lean=(-.05, .05), ear=(-.1, .1), jit=(.006, .015), bounce=(0, .01)),
    "play_bow":        dict(hz=(1.5, 4), amp=(.3, .7), bias=(-.05, .2), lift=(.05, .4), bow=(.35, .6), lean=(-.15, 0), ear=(-.1, .25), jit=(.01, .025), bounce=(.01, .04)),
    "alert":           dict(hz=(0, .6), amp=(0, .1), bias=(-.05, .05), lift=(.2, .5), bow=(-.05, .05), lean=(.05, .15), ear=(.25, .5), jit=(.003, .008), bounce=(0, .005)),
    "defensive_stiff": dict(hz=(0, .8), amp=(0, .08), bias=(-.05, .05), lift=(.3, .7), bow=(-.12, -.02), lean=(.15, .35), ear=(.15, .5), jit=(.001, .004), bounce=(0, .002)),
    "anxious":         dict(hz=(0, 2.5), amp=(0, .2), bias=(-.35, -.1), lift=(-.6, -.3), bow=(-.05, .1), lean=(-.35, -.15), ear=(-.6, -.3), jit=(.008, .02), bounce=(0, .01)),
    "excited":         dict(hz=(2.5, 6), amp=(.4, .9), bias=(.05, .35), lift=(.0, .4), bow=(0, .1), lean=(-.1, .15), ear=(-.1, .2), jit=(.02, .05), bounce=(.03, .08)),
}


def synth_sequence(label: str, T: int = 30, fps: float = 15.0, rng=np.random) -> np.ndarray:
    """Returns (T, 24, 3) image-space keypoints (x, y, conf) for one behaviour clip."""
    p = {k: rng.uniform(*v) for k, v in BEHAVIOURS[label].items()}
    morph_ear = rng.uniform(-0.6, 0.3)   # drop vs prick ears (breed baseline)
    tail_len = rng.choice([1.0, 1.0, 0.6, 0.15])  # 0.15 ~ docked
    leg = rng.uniform(0.7, 1.2)            # leg length (Corgi .. Greyhound)
    facing = rng.choice([1, -1])
    yaw = rng.uniform(0, 1)                # 0 side view .. 1 strongly foreshortened
    scale = rng.uniform(80, 400)
    centre = rng.uniform([200, 150], [1080, 570])
    phase = rng.uniform(0, 2 * np.pi)
    # Wearer head motion: slow drift + sway, plus roll (removed by IMU in real use; keep a residual).
    drift = rng.normal(0, 4, (T, 2)).cumsum(0) + rng.normal(0, 30, 2) * np.linspace(0, 1, T)[:, None]
    zoom = 1 + np.linspace(0, rng.uniform(-.15, .15), T)
    roll = rng.normal(0, .04) + rng.normal(0, .01, T).cumsum()
    out = np.zeros((T, NUM_KPTS, 3), np.float32)
    for t in range(T):
        s = BASE_ARR.copy()
        s[:, 1] = np.where(s[:, 1] > 0, s[:, 1] * leg, s[:, 1])
        # ears: rotate tip around base, toward nose (+) or back (-)
        for b, tip in EARTIPS:
            s[tip] = s[b] + _rot(s[tip] - s[b], (p["ear"] + morph_ear * 0.3) * 1.2 + rng.normal(0, .03))
        # tail: lift + wag (lateral wag appears as vertical/horizontal sweep depending on yaw)
        wag = p["amp"] * np.sin(2 * np.pi * p["hz"] * t / fps + phase) + p["bias"]
        ang = -p["lift"] * 1.5
        tv = np.array([-0.45, -0.28]) * tail_len
        tv = _rot(tv, ang + wag * (1 - yaw) * 0.6)
        tv[0] += wag * yaw * 0.4 * tail_len  # lateral component visible when viewed from behind
        s[K["tail_end"]] = s[K["tail_start"]] + tv
        # bow: drop the front half (pitch around hindquarters)
        s[FRONT, 1] += p["bow"] * (s[FRONT, 0] + 0.5)
        s[FRONT, 0] -= p["bow"] * 0.2
        # lean: shift torso and head over the paws
        body = [i for i in range(NUM_KPTS) if not BASE_ARR[i, 1] > 0.9]
        s[body, 0] += p["lean"] * 0.4
        s[HEAD, 1] += rng.normal(0, .01) - p["lean"] * 0.1
        s[:, 1] += p["bounce"] * np.sin(2 * np.pi * 2.5 * t / fps + phase)
        s += rng.normal(0, p["jit"], s.shape)
        s[:, 0] *= facing * (1 - 0.75 * yaw)
        s = _rot(s, roll[t]) * scale * zoom[t] + centre + drift[t]
        out[t, :, :2] = s + rng.normal(0, 1.5, s.shape)  # detector noise (px)
        conf = rng.uniform(0.5, 1.0, NUM_KPTS)
        conf[rng.random(NUM_KPTS) < 0.08] = 0.05  # occlusions / misses
        if tail_len < 0.3:
            conf[K["tail_end"]] = 0.05
        conf[K["withers"]] = max(conf[K["withers"]], 0.5)
        conf[K["tail_start"]] = max(conf[K["tail_start"]], 0.5)
        out[t, :, 2] = conf
    return out


# --------------------------------------------------------------------------- audio
def _tone(dur, f0, harm, hnr_db, rng, glide=0.0, am=0.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f0 * (1 + glide * t / max(dur, 1e-3) + 0.02 * np.sin(2 * np.pi * 5 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = sum((0.7 ** k) * np.sin((k + 1) * ph) for k in range(harm))
    noise = rng.normal(0, 1, n)
    g = 10 ** (-hnr_db / 20)
    y = y / (np.std(y) + 1e-9) + g * noise
    if am:
        y *= 1 + am * np.sin(2 * np.pi * rng.uniform(20, 40) * t)  # growl rattle
    env = np.minimum(1, np.minimum(t / 0.01, (dur - t) / 0.03).clip(0))
    return (y * env).astype(np.float32)


def synth_vocal(label: str, rng=np.random, sec: float = 1.0) -> np.ndarray:
    y = np.zeros(int(sec * SR), np.float32)
    def put(seg, at):
        a = int(at * SR); b = min(len(y), a + len(seg)); y[a:b] += seg[: b - a]
    if label == "growl":
        put(_tone(rng.uniform(.6, 1.0), rng.uniform(80, 220), 8, rng.uniform(-6, 2), rng, am=.6), rng.uniform(0, .1))
    elif label == "bark_alert":
        for at in np.arange(rng.uniform(0, .2), sec, rng.uniform(.35, .7)):
            put(_tone(rng.uniform(.08, .2), rng.uniform(250, 600), 6, rng.uniform(0, 5), rng, glide=-.2), at)
    elif label == "bark_play":
        for at in np.arange(rng.uniform(0, .1), sec, rng.uniform(.15, .3)):
            put(_tone(rng.uniform(.05, .12), rng.uniform(450, 900), 5, rng.uniform(3, 9), rng, glide=.15), at)
    elif label == "whine":
        put(_tone(rng.uniform(.6, 1.0), rng.uniform(600, 1500), 3, rng.uniform(12, 25), rng, glide=rng.uniform(-.3, .3)), rng.uniform(0, .2))
    elif label == "howl":
        put(_tone(sec, rng.uniform(350, 650), 4, rng.uniform(10, 20), rng, glide=rng.uniform(-.2, .3)), 0)
    y = y / (np.abs(y).max() + 1e-9) * rng.uniform(.05, .8) if label != "none" else y
    # Background: wind (low-frequency rumble), traffic hiss, occasional human speech-like hum.
    wind = np.cumsum(rng.normal(0, 1, len(y))); wind -= np.convolve(wind, np.ones(400) / 400, "same")
    y += rng.uniform(0, .05) * wind / (np.std(wind) + 1e-9)
    y += rng.uniform(.001, .02) * rng.normal(0, 1, len(y))
    if rng.random() < 0.3:
        y += _tone(sec, rng.uniform(90, 220), 10, 5, rng) * rng.uniform(.01, .08)
    return y.astype(np.float32)
