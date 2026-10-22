import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]

from dogsense.acoustics import AcousticState, AudioAnalyzer, f0_hz, hnr_db, SR  # noqa: E402
from dogsense.fusion import fuse  # noqa: E402
from dogsense.keypoints import body_frame  # noqa: E402
from dogsense.kinematics import KinematicTracker  # noqa: E402
from synth import synth_sequence, synth_vocal, _tone  # noqa: E402


def track(label, seed=0, T=30):
    rng = np.random.RandomState(seed)
    tr = KinematicTracker(window=T)
    for t, k in enumerate(synth_sequence(label, T, rng=rng)):
        tr.push(k, t / 15)
    return tr.state()


def test_body_frame_invariant_to_camera_motion():
    k = synth_sequence("relaxed", 1, rng=np.random.RandomState(1))[0]
    a = body_frame(k)[0]
    moved = k.copy()
    c, s = np.cos(0.3), np.sin(0.3)
    moved[:, :2] = (k[:, :2] @ np.array([[c, -s], [s, c]]).T) * 1.7 + [120, -40]
    b = body_frame(moved)[0]
    assert np.allclose(a, b, atol=1e-3)


def test_kinematic_signatures():
    bow = np.mean([track("play_bow", s).play_bow for s in range(8)])
    rel = np.mean([track("relaxed", s).play_bow for s in range(8)])
    assert bow > rel + 0.3
    stiff = np.mean([track("defensive_stiff", s).stiffness for s in range(8)])
    exc = np.mean([track("excited", s).stiffness for s in range(8)])
    assert stiff > exc
    assert np.mean([track("excited", s).wag_hz for s in range(8)]) > np.mean([track("alert", s).wag_hz for s in range(8)])


def growl_state():
    return AcousticState(active=True, f0=150, hnr=1, pitch_band="low", tonality="harsh", rule_class="growl", burst_rate=0.5)


def test_play_bow_overrides_growl():
    k = track("play_bow", 3)
    k.play_bow, k.stiffness, k.forward_weight = 0.9, 0.3, 0.0
    d = fuse(k, growl_state())
    assert d.state == "Play Solicitation" and not d.alert and d.valence > 0


def test_stiff_forward_growl_is_warning():
    k = track("defensive_stiff", 3)
    k.play_bow, k.stiffness, k.forward_weight = 0.0, 0.85, 0.6
    d = fuse(k, growl_state())
    assert d.state == "Defensive Warning" and d.alert and d.valence < 0
    assert fuse(k, AcousticState(), trigger="Resource/Toy").state == "Resource Guarding"


def test_pitch_and_hnr():
    rng = np.random.RandomState(0)
    tone = _tone(0.5, 440, 3, 30, rng)
    assert abs(f0_hz(tone) - 440) < 15
    assert hnr_db(tone) > 10
    assert hnr_db(rng.normal(0, 1, SR // 2).astype(np.float32)) < 3


@pytest.mark.parametrize("label,band", [("growl", "low"), ("whine", "high")])
def test_audio_analyzer_bands(label, band):
    rng = np.random.RandomState(2)
    an = AudioAnalyzer()
    voc = synth_vocal(label, rng)
    hiss = lambda n: rng.normal(0, 0.01, n).astype(np.float32)  # real mics are never digitally silent
    clip = np.concatenate([hiss(SR // 2), voc + hiss(len(voc)), hiss(SR // 4)])
    for i in range(0, len(clip), 1600):
        st = an.feed((clip[i:i + 1600] * 32767).astype(np.int16))
    assert st.pitch_band == band, st


def test_sound_without_posture_reports_only_what_was_heard():
    from dogsense.kinematics import KinematicState

    bark = AcousticState(active=True, f0=450, hnr=5, pitch_band="mid", tonality="mixed", rule_class="bark_alert")
    d = fuse(KinematicState(), bark, dog_visible=False)
    assert (d.state, d.detail, d.alert) == ("Barking", "dog not in view", "")
    assert fuse(KinematicState(), bark, dog_visible=True).detail == "posture unclear"


def test_source_gate_rejects_human_runs_and_keeps_dog_calls():
    # Stub gate: the 1st and 3rd bursts are classified human, the 2nd is ambiguous, the 4th dog.
    labels = iter(["human", "other", "human", "dog"])
    an = AudioAnalyzer(source_gate=lambda audio: next(labels))
    rng = np.random.RandomState(0)
    burst = lambda: _tone(0.15, 300, 6, 3, rng)
    gap = np.zeros(int(0.3 * SR), np.float32)
    clip = np.concatenate([gap, burst(), gap, burst(), gap, burst(), np.zeros(int(1.5 * SR), np.float32), burst(), np.zeros(SR, np.float32)])
    for i in range(0, len(clip), 1600):
        an.feed((clip[i:i + 1600] * 32767 * 0.5).astype(np.int16))
    assert an.rejected_by["human_source"] == 3  # the ambiguous middle "hey" goes with its run
    assert len(an.events) == 1
