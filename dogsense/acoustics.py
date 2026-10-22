"""Bioacoustic front-end: denoise -> vocal event detection -> F0 / HNR / inter-bark interval.

Input is 16 kHz mono PCM16 from the glasses mic array. Wind is suppressed with a high-pass
filter + spectral floor; human speech is rejected with the device VAD flag (if provided) and
a syllable-rate / duration heuristic, keeping short transient bursts typical of dog vocalisations.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np
import scipy.signal as sps
from scipy.fft import dct as sp_dct

SR = 16000
N_MELS = 64
CLIP_SEC = 1.0
GATE_WIN_SEC = 1.5  # window (centred on the event) the sound-source gate classifies; shorter
                    # snippets were unreliable in field tests (AST is trained on 10 s clips)
AUDIO_CLASSES = ["none", "growl", "bark_alert", "bark_play", "whine", "howl"]

_HP = sps.butter(4, 150, "highpass", fs=SR, output="sos")


def denoise(x: np.ndarray) -> np.ndarray:
    """Offline clip cleaning (training / classifier input): high-pass (wind buffeting is mostly < 150 Hz) + crude spectral-floor subtraction."""
    y = sps.sosfilt(_HP, x.astype(np.float32))
    f, t, Z = sps.stft(y, SR, nperseg=512, noverlap=384)
    mag = np.abs(Z)
    floor = np.percentile(mag, 20, axis=1, keepdims=True)
    Z = Z * np.maximum(mag - 1.5 * floor, 0) / (mag + 1e-9)
    _, y = sps.istft(Z, SR, nperseg=512, noverlap=384)
    return y[: len(x)].astype(np.float32)


_MEL = None


def _mel():
    global _MEL
    if _MEL is None:
        import torchaudio

        _MEL = torchaudio.transforms.MelSpectrogram(SR, n_fft=512, hop_length=160, n_mels=N_MELS, f_min=50, f_max=8000)
    return _MEL


def log_mel(x: np.ndarray) -> np.ndarray:
    """(64, T) log-mel power in dB, floor at -80 dB."""
    import torch

    m = _mel()(torch.from_numpy(np.ascontiguousarray(x, np.float32))).numpy()
    db = 10 * np.log10(np.maximum(m, 1e-10))
    return np.maximum(db, db.max() - 80).astype(np.float32)


def mfcc(x: np.ndarray, n: int = 40) -> np.ndarray:
    """(n, T) MFCCs (DCT-II of the log-mel spectrogram)."""
    return sp_dct(log_mel(x), axis=0, norm="ortho")[:n].astype(np.float32)


def _autocorr(seg: np.ndarray) -> np.ndarray:
    seg = seg - seg.mean()
    ac = sps.correlate(seg, seg, "full", method="fft")[len(seg) - 1:]
    return ac / (ac[0] + 1e-12)


def hnr_db(seg: np.ndarray, fmin=80, fmax=2000, frame=640) -> float:
    """Harmonics-to-noise ratio (Boersma 1993): median over 40 ms frames, so pitch glides
    within a call don't smear the autocorrelation peak."""
    if len(seg) >= 2 * frame:
        v = [_hnr_frame(seg[s:s + frame], fmin, fmax) for s in range(0, len(seg) - frame + 1, frame // 2)]
        return float(np.median(v))
    return _hnr_frame(seg, fmin, fmax)


def _hnr_frame(seg, fmin, fmax) -> float:
    if np.sum((seg - seg.mean()) ** 2) < 1e-8:
        return -10.0
    ac = _autocorr(seg)
    lo, hi = SR // fmax, min(SR // fmin, len(ac) - 1)
    if hi <= lo:
        return -10.0
    r = float(np.clip(ac[lo:hi].max(), 1e-4, 0.9999))
    return float(10 * np.log10(r / (1 - r)))


def dominant_hz(seg: np.ndarray, fmin=50, fmax=2000) -> float:
    """Spectral-peak frequency; pitch proxy for atonal sounds (growls) where F0 is undefined."""
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    band = (f >= fmin) & (f <= fmax)
    # smooth so the broad formant hump wins over isolated noise spikes
    sm = np.convolve(spec[band], np.ones(9) / 9, "same")
    return float(f[band][np.argmax(sm)])


def f0_hz(seg: np.ndarray, fmin=80, fmax=2000, frame=1024, hop=512) -> float:
    """Median F0 over voiced frames, via normalised autocorrelation peak picking."""
    if len(seg) < frame:
        seg = np.pad(seg, (0, frame - len(seg)))
    lo, hi = SR // fmax, SR // fmin
    est = []
    for s in range(0, len(seg) - frame + 1, hop):
        ac = _autocorr(seg[s:s + frame] * np.hanning(frame))
        k = lo + int(np.argmax(ac[lo:hi]))
        # Octave guard: multiples of the true period correlate almost as well, so take the
        # shortest lag that is a local peak within 10 % of the best one.
        for c in range(lo + 1, k):
            if ac[c] >= 0.9 * ac[k] and ac[c] >= ac[c - 1] and ac[c] >= ac[c + 1]:
                k = c
                break
        if ac[k] > 0.3:
            # parabolic interpolation around the peak
            a, b, c = ac[k - 1], ac[k], ac[min(k + 1, len(ac) - 1)]
            den = a - 2 * b + c
            k = k + (float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den < 0 else 0.0)
            est.append(SR / k)
    return float(np.median(est)) if est else 0.0


@dataclass
class VocalEvent:
    t: float
    dur: float
    f0: float
    hnr: float
    rms: float
    speech_prob: float = 0.0  # device VAD score for the chunk the event arrived in
    src: str = ""             # sound-source gate verdict: human / dog / other


@dataclass
class AcousticState:
    active: bool = False
    f0: float = 0.0
    hnr: float = 0.0
    ibi: float = 0.0            # mean inter-bark interval (s)
    burst_rate: float = 0.0     # events / s over the window
    pitch_band: str = "-"       # low / mid / high
    tonality: str = "-"         # harsh / mixed / tonal
    rule_class: str = "none"
    nn_class: str = "none"
    nn_conf: float = 0.0
    loudness: float = 0.0
    speech_rejected: int = 0
    probs: dict = field(default_factory=dict)

    def dict(self):
        return asdict(self)


def rule_class(f0: float, hnr: float, dur: float, rate: float) -> str:
    """Acoustic heuristics (Morton's motivation-structural rules, Yin & McCowan 2004)."""
    if f0 <= 0:
        return "none"
    if dur > 1.2 and hnr > 8:
        return "howl" if f0 < 700 else "whine"
    if f0 > 600 and hnr > 6:
        return "whine"
    if f0 < 250 and hnr < 4 and dur > 0.35:
        return "growl"
    if rate > 1.2 and 250 <= f0 <= 900:
        return "bark_play"
    return "bark_alert"


class AudioAnalyzer:
    """Streaming analyser. feed() PCM chunks, read state()."""

    def __init__(self, window_sec: float = 4.0, classifier=None, source_gate=None):
        self.buf = deque(maxlen=int(SR * window_sec))
        self.events: deque[VocalEvent] = deque(maxlen=32)
        self.t = 0.0
        self.window_sec = window_sec
        self.classifier = classifier  # callable(np.ndarray 1s @16k) -> probs dict
        self.source_gate = source_gate  # callable(event audio) -> "human" | "dog" | "other"
        self.rejected_by = {"vad": 0, "speech_shape": 0, "human_source": 0}
        self._last_human_t = -1e9
        self._waiting: list[VocalEvent] = []  # detected, awaiting post-roll for the source gate
        self.speech_rejected = 0
        self._last = AcousticState()
        self._noise_rms = 1e-3
        self._pend = np.zeros(0, np.float32)  # audio holding a vocal event still in progress
        self._pend_t = 0.0
        self._zi = np.zeros((_HP.shape[0], 2))

    def feed(self, pcm: np.ndarray, speech_prob: float = 0.0) -> AcousticState:
        x = pcm.astype(np.float32)
        if pcm.dtype == np.int16:
            x = x / 32768.0
        if not len(self._pend):
            self._pend_t = self.t
        self.t += len(x) / SR
        y, self._zi = sps.sosfilt(_HP, x, zi=self._zi)
        y = y.astype(np.float32)
        self.buf.extend(y)
        self._pend = np.concatenate([self._pend, y])
        events, open_from = self._detect(self._pend, self._pend_t)
        if open_from is not None and len(self._pend) - open_from < 3 * SR:
            # Last event runs into the chunk edge: hold it until it finishes.
            self._pend_t += open_from / SR
            self._pend = self._pend[open_from:]
        else:
            self._pend = np.zeros(0, np.float32)
        for ev in events:
            ev.speech_prob = speech_prob
            self._waiting.append(ev)
        # With a source gate, decide each event only once the rest of its classification window
        # has arrived, so the classifier hears it in context (a bare 100 ms burst is ambiguous).
        while self._waiting and self._ready_at(self._waiting[0]) <= self.t:
            self._decide(self._waiting.pop(0))
        self._last = self._summarise()
        return self._last

    def _ready_at(self, ev: VocalEvent) -> float:
        if self.source_gate is None:
            return ev.t + ev.dur
        return max(ev.t + ev.dur, ev.t + ev.dur / 2 + GATE_WIN_SEC / 2)

    def _context(self, ev: VocalEvent) -> np.ndarray:
        """Filtered audio in a GATE_WIN_SEC window centred on the event."""
        buf = np.asarray(self.buf, np.float32)
        mid = ev.t + ev.dur / 2
        a = len(buf) - int((self.t - (mid - GATE_WIN_SEC / 2)) * SR)
        b = len(buf) - int((self.t - (mid + GATE_WIN_SEC / 2)) * SR)
        return buf[max(0, a):max(0, min(len(buf), b))]

    def _decide(self, ev: VocalEvent):
        why = "vad" if ev.speech_prob > 0.6 else "speech_shape" if self._speech_like(ev) else None
        if why is None and self.source_gate is not None:
            ev.src = self.source_gate(self._context(ev))
            # Speech comes in runs ("hey, hey, hey"): an ambiguous event within 0.8 s of
            # confirmed human sound - before or after it - belongs to the same run.
            if ev.src == "human" or (ev.src == "other" and ev.t - self._last_human_t < 0.8):
                why = "human_source"
        if why:
            self._last_human_t = ev.t
            self.speech_rejected += 1
            self.rejected_by[why] += 1
            for old in [e for e in self.events if e.src == "other" and ev.t - e.t < 0.8]:
                self.events.remove(old)  # retroactively part of this human run
                self.speech_rejected += 1
                self.rejected_by["human_source"] += 1
        else:
            self.events.append(ev)

    def _detect(self, y: np.ndarray, t0: float):
        hop = 160  # 10 ms
        frames = len(y) // hop
        if frames == 0:
            return [], None
        rms = np.sqrt(np.mean(y[: frames * hop].reshape(frames, hop) ** 2, axis=1) + 1e-12)
        # Noise floor from the last 2 s of audio (not just this chunk, which for small live
        # chunks is mostly the call itself): 20th percentile of 10 ms frame energy.
        hist = np.asarray(self.buf, np.float32)[-2 * SR:]
        nf = len(hist) // hop
        if nf >= 10:
            hr = np.sqrt(np.mean(hist[: nf * hop].reshape(nf, hop) ** 2, axis=1) + 1e-12)
            self._noise_rms = float(np.percentile(hr, 20))
        on = rms > max(4 * self._noise_rms, 0.01)
        out, i, open_from = [], 0, None
        while i < frames:
            if on[i]:
                j = i
                while j < frames and (on[j] or (j + 3 < frames and on[j:j + 3].any())):
                    j += 1
                dur = (j - i) * hop / SR
                if j >= frames - 2:  # touches the end of the buffer: not finished yet
                    open_from = i * hop
                    break
                if dur >= 0.04:
                    seg = y[i * hop: j * hop]
                    h = hnr_db(seg)
                    pitch = f0_hz(seg) if h >= 3 else dominant_hz(seg)
                    out.append(VocalEvent(t0 + i * hop / SR, dur, pitch, h, float(rms[i:j].mean())))
                i = j
            i += 1
        return out, open_from

    @staticmethod
    def _speech_like(ev: VocalEvent) -> bool:
        # Long, moderately harmonic, in the 85-255 Hz adult speech F0 range -> likely the wearer talking.
        return ev.dur > 0.6 and 85 <= ev.f0 <= 255 and 3 < ev.hnr < 12

    def _summarise(self) -> AcousticState:
        s = AcousticState(speech_rejected=self.speech_rejected)
        recent = [e for e in self.events if e.t > self.t - self.window_sec]
        if self.classifier is not None and len(self.buf) >= SR * CLIP_SEC:
            clip = np.array(list(self.buf)[-int(SR * CLIP_SEC):], np.float32)
            probs = self.classifier(clip)
            s.probs = probs
            s.nn_class = max(probs, key=probs.get)
            s.nn_conf = float(probs[s.nn_class])
        if not recent:
            return s
        s.active = recent[-1].t > self.t - 1.5
        w = np.array([e.rms * e.dur for e in recent])
        s.f0 = float(np.average([e.f0 for e in recent], weights=w))
        s.hnr = float(np.average([e.hnr for e in recent], weights=w))
        s.loudness = float(max(e.rms for e in recent))
        dur = float(np.average([e.dur for e in recent], weights=w))
        if len(recent) > 1:
            s.ibi = float(np.mean(np.diff([e.t for e in recent])))
        s.burst_rate = len(recent) / self.window_sec
        s.pitch_band = "low" if s.f0 < 300 else "high" if s.f0 > 600 else "mid"
        s.tonality = "harsh" if s.hnr < 3 else "tonal" if s.hnr > 9 else "mixed"
        s.rule_class = rule_class(s.f0, s.hnr, dur, s.burst_rate) if s.active else "none"
        return s
