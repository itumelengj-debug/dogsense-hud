"""Neuro-symbolic fusion: continuous kinematics + acoustics + NN priors + scene context -> HUD state.

Neural heads give soft evidence; a deterministic, ordered rule base makes the final call so that
conflicting cues are resolved explicitly (a play bow overrides a growl; a stiff, forward-weighted
dog with a low harsh growl is a warning regardless of tail wag). Every decision carries the cues
that fired it, which become the HUD caption.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .acoustics import AcousticState
from .kinematics import KinematicState

# (valence, arousal) priors for the neural pose classes.
POSE_PRIOR = {
    "relaxed": (0.5, 0.2), "play_bow": (0.8, 0.7), "alert": (0.0, 0.6),
    "defensive_stiff": (-0.8, 0.9), "anxious": (-0.6, 0.6), "excited": (0.6, 0.85),
}
AUDIO_PRIOR = {
    "none": (0.0, 0.0), "growl": (-0.6, 0.7), "bark_alert": (-0.2, 0.7),
    "bark_play": (0.6, 0.7), "whine": (-0.5, 0.5), "howl": (-0.2, 0.6),
}


@dataclass
class Decision:
    state: str = "No dog"
    detail: str = ""
    alert: str = ""
    arousal: float = 0.0
    valence: float = 0.0
    confidence: float = 0.0
    cues: list = field(default_factory=list)
    trigger: str = ""

    def dict(self):
        return asdict(self)


def _audio_label(a: AcousticState) -> str:
    if not a.active:
        return "none"
    # Trust the network when it is confident, otherwise the acoustic rules.
    return a.nn_class if a.nn_conf > 0.6 and a.nn_class != "none" else a.rule_class


def fuse(k: KinematicState, a: AcousticState, pose_probs: dict | None = None, trigger: str = "",
         dog_visible: bool = False) -> Decision:
    d = Decision(trigger=trigger)
    if not k.valid and not a.active:
        if dog_visible:
            d.state = "Reading posture..."
        return d
    snd = _audio_label(a)
    growl = snd == "growl" or (a.active and a.pitch_band == "low" and a.tonality == "harsh")
    whine = snd == "whine"
    bark = snd in ("bark_alert", "bark_play")
    cues = []
    if k.valid:
        if k.play_bow > 0.5: cues.append("bow")
        if k.stiffness > 0.7: cues.append("stiff")
        if k.forward_weight > 0.3: cues.append("weight fwd")
        if k.forward_weight < -0.3: cues.append("weight back")
        if k.ear_carriage < -0.4: cues.append("ears pinned")
        if k.ear_carriage > 0.4: cues.append("ears fwd")
        if k.ear_flick > 1.0: cues.append("ear flick")  # jump > 4x this track's jitter
        if k.wag_hz > 1.5 and k.wag_amp > 0.15:
            side = ""
            if k.wag_bias_reliable and abs(k.wag_bias) > 0.25:
                side = " R-bias" if k.wag_bias > 0 else " L-bias"
            cues.append(f"wag {k.wag_hz:.1f}Hz{side}")
        if k.tail_height < -0.3: cues.append("tail low")
    if a.active and snd != "none":
        cues.append(f"{snd.replace('_', ' ')} {a.f0:.0f}Hz")

    resource = "toy" in trigger.lower() or "food" in trigger.lower() or "resource" in trigger.lower()
    loose = k.stiffness < 0.45
    threat_posture = k.stiffness > 0.7 and k.forward_weight > 0.3
    withdrawal = (k.ear_carriage < -0.4 and k.forward_weight < -0.2) or (k.wag_bias_reliable and k.wag_bias < -0.3)

    # ---- ordered symbolic rules -------------------------------------------------------------
    if not k.valid:
        # Sound without readable posture: report what was heard, never a behavioural state
        # (the same bark means play or frustration depending on the body).
        heard = {"growl": "Growling", "bark_alert": "Barking", "bark_play": "Barking",
                 "whine": "Whining", "howl": "Howling"}.get(snd)
        where = "posture unclear" if dog_visible else "dog not in view"
        if heard:
            d.state, d.detail = heard, where
        else:
            d.state = "Reading posture..." if dog_visible else "Vocalising (dog not in view)"
        rv, ra = AUDIO_PRIOR.get(snd, (0, 0.3))
    elif k.play_bow > 0.5 and not threat_posture:
        d.state, d.detail = "Play Solicitation", "Bow" + (" + play growl" if growl else " + bark" if bark else "")
        rv, ra = 0.8, 0.7
    elif threat_posture and resource:
        d.state, d.alert = "Resource Guarding", "Stiff posture over resource - give space"
        rv, ra = -0.9, 0.9
    elif threat_posture and (growl or (bark and a.pitch_band == "low")):
        d.state, d.alert = "Defensive Warning", "Stiff + forward + low growl - do not approach"
        rv, ra = -0.9, 0.95
    elif threat_posture:
        d.state, d.alert = "High Arousal / Fixated", "Frozen, weight forward"
        rv, ra = -0.5, 0.8
    elif withdrawal or (whine and not loose):
        d.state, d.detail = "Anxious / Appeasing", "Withdrawal cues" + (" + whine" if whine else "")
        rv, ra = -0.6, 0.6
    elif whine:
        d.state, d.detail = "Seeking / Distress Call", "Tonal high-pitch whine"
        rv, ra = -0.3, 0.5
    elif k.valid and k.wag_hz > 1.5 and loose and k.wag_bias >= -0.1:
        d.state, d.detail = "Friendly / Excited", "Loose body, broad wag"
        rv, ra = 0.7, 0.6 + min(k.motion * 5, 0.3)
    elif bark or (k.valid and k.ear_carriage > 0.4 and k.stiffness > 0.5):
        d.state, d.detail = "Alert / Vigilant", "Ears forward" + (", alert barking" if bark else "")
        rv, ra = -0.1, 0.65
    else:
        d.state, d.detail = "Relaxed", "Neutral carriage"
        rv, ra = 0.4, 0.2

    # ---- continuous blend with neural evidence ---------------------------------------------
    vals, wts = [(rv, ra)], [0.6]
    if pose_probs:
        pv = sum(p * POSE_PRIOR[c][0] for c, p in pose_probs.items())
        pa = sum(p * POSE_PRIOR[c][1] for c, p in pose_probs.items())
        vals.append((pv, pa)); wts.append(0.25)
    if a.active and not (d.state == "Play Solicitation" and growl):  # bow overrides growl valence
        vals.append(AUDIO_PRIOR.get(snd, (0, 0))); wts.append(0.15)
    v = np.average([x[0] for x in vals], weights=wts)
    ar = np.average([x[1] for x in vals], weights=wts)
    ar = ar + 0.1 * min(a.burst_rate, 2) + 0.1 * min(max(k.ear_flick - 1, 0), 1)
    d.valence, d.arousal = float(np.clip(v, -1, 1)), float(np.clip(ar, 0, 1))
    d.confidence = float(np.clip(0.4 * k.track_conf + 0.3 * a.active + 0.3 * (max(pose_probs.values()) if pose_probs else 0.3), 0, 1))
    d.cues = cues[:5]
    return d
