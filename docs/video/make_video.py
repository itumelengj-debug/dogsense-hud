"""Renders the 90 s Symbolic Synapse DogSense HUD video (1920x1080, 25 fps).

    python docs/video/make_video.py --fonts <dir with InstrumentSerif/Inter/JetBrainsMono> --out media/DogSense_HUD_90s.mp4

Fonts (Google Fonts, OFL): Instrument Serif (Regular + Italic), Inter, JetBrains Mono.
Needs the field-test clips in data/test/, which are not in the public repository.

Visual language follows Symbolic_Synapse_Augmented_Psychiatry_90s: deep navy field, serif titles
with a cyan italic accent word, numbered concept slides with a green console panel, ambient score.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "training")]

W, H, FPS, DUR = 1920, 1080, 25, 90.0
NAVY = (9, 18, 32)
WHITE = (232, 238, 246)
BODY = (178, 192, 210)
CYAN = (79, 195, 220)
DIMC = (70, 120, 150)
GREEN = (60, 255, 110)
GDIM = (36, 140, 70)
GRID = (16, 30, 48)

# ------------------------------------------------------------------------------- helpers
def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def fade(t, start, dur=0.6):
    return ease((t - start) / dur)


class Fonts:
    def __init__(self, d: Path):
        self.d = d
        self.cache = {}

    def get(self, kind, size, weight=None):
        key = (kind, size, weight)
        if key not in self.cache:
            f = {"serif": "InstrumentSerif-Regular.ttf.ttf", "serif_i": "InstrumentSerif-Italic.ttf.ttf",
                 "sans": "Inter.ttf", "mono": "JetBrainsMono.ttf"}[kind]
            font = ImageFont.truetype(str(self.d / f), size)
            if weight:
                try:
                    font.set_variation_by_axes([weight] if kind == "mono" else [14, weight])
                except Exception:
                    pass
            self.cache[key] = font
        return self.cache[key]


def blend(base: Image.Image, layer: Image.Image, alpha: float, xy=(0, 0)):
    """Paste RGBA layer onto RGB base with extra opacity."""
    if alpha <= 0:
        return
    if alpha < 1:
        a = layer.getchannel("A").point(lambda v: int(v * alpha))
        layer = layer.copy()
        layer.putalpha(a)
    base.paste(layer, xy, layer)


def text_layer(parts, size_box=(W, 200)):
    """parts: list of (text, font, color) drawn on one line. Returns RGBA image cropped to box."""
    img = Image.new("RGBA", size_box, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = 0
    for txt, font, col in parts:
        d.text((x, 0), txt, font=font, fill=col)
        x += d.textlength(txt, font=font)
    return img


def draw_text_fade(img, xy, parts, alpha, dy=18):
    lay = text_layer(parts)
    blend(img, lay, alpha, (int(xy[0]), int(xy[1] + (1 - alpha) * dy)))


def wrap(d, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) > width and cur:
            lines.append(cur); cur = w
        else:
            cur = t
    lines.append(cur)
    return lines


# ------------------------------------------------------------------------------- background
def make_background():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    g1 = np.exp(-(((x - 1450) / 900) ** 2 + ((y - 120) / 520) ** 2))
    g2 = np.exp(-(((x - 300) / 800) ** 2 + ((y - 1000) / 500) ** 2))
    base = np.array(NAVY, np.float32)
    img = base + g1[..., None] * np.array([14, 40, 52]) + g2[..., None] * np.array([10, 22, 48])
    img = img.clip(0, 255).astype(np.uint8)
    for gx in range(0, W, 160):  # faint vertical guides
        img[:, gx] = np.maximum(img[:, gx], np.array(GRID, np.uint8))
    return Image.fromarray(img)


class Renderer:
    def __init__(self, fonts: Fonts):
        self.F = fonts
        self.bg = make_background()
        logo = Image.open(ROOT / "docs" / "build" / "logo.png").convert("RGBA")
        self.logo_small = logo.resize((44, 50), Image.LANCZOS)
        self.logo_big = logo.resize((300, 344), Image.LANCZOS)
        glow = Image.new("RGBA", (360, 404), (0, 0, 0, 0))
        glow.paste(self.logo_big, (30, 30), self.logo_big)
        self.logo_glow = glow.filter(ImageFilter.GaussianBlur(28))
        self.photo = Image.open(ROOT / "docs" / "hud_preview.png").convert("RGB")
        self.panel_glow = self._panel_glow()
        self.vid1 = cv2.VideoCapture(str(ROOT / "media" / "DogSense_HUD_test.mp4"))
        self.vid2 = cv2.VideoCapture(str(ROOT / "data" / "test" / "hud2390.mp4"))
        self.spec = self._spectrogram()
        self.skel = self._skeleton_seq()

    # ---- shared chrome
    def chrome(self, img, t, label, idx, total=5):
        d = ImageDraw.Draw(img)
        img.paste(self.logo_small, (120, 58), self.logo_small)
        d.text((180, 70), "S Y M B O L I C   S Y N A P S E", font=self.F.get("sans", 19, 500), fill=WHITE)
        right = f"D O G S E N S E   H U D"
        f = self.F.get("sans", 19, 400)
        num = f"{idx:02d} / {total:02d}" if idx else ""
        wn = d.textlength(num, font=f)
        d.text((W - 120 - wn, 70), num, font=f, fill=CYAN)
        d.text((W - 120 - wn - 40 - d.textlength(right, font=f), 70), right, font=f, fill=DIMC)
        # progress rail
        d.line([(120, 1015), (W - 120, 1015)], fill=(30, 52, 76), width=2)
        p = min(t / DUR, 1)
        d.line([(120, 1015), (120 + (W - 240) * p, 1015)], fill=CYAN, width=2)
        for k in range(1, 6):
            x = 120 + (W - 240) * (k * 13 - 0.5) / 90
            d.line([(x, 1009), (x, 1021)], fill=(50, 90, 120), width=2)

    def _panel_glow(self):
        g = Image.new("RGBA", (1000, 760), (0, 0, 0, 0))
        ImageDraw.Draw(g).rounded_rectangle((40, 40, 960, 720), 10, outline=(60, 255, 110, 170), width=10)
        return g.filter(ImageFilter.GaussianBlur(18))

    def panel(self, img, title, tag, alpha, box=(980, 250, 1800, 870)):
        x1, y1, x2, y2 = box
        lay = Image.new("RGBA", (x2 - x1 + 80, y2 - y1 + 80), (0, 0, 0, 0))
        glow = self.panel_glow.resize(lay.size)
        lay.paste(glow, (0, 0), glow)
        d = ImageDraw.Draw(lay)
        d.rounded_rectangle((40, 40, 40 + x2 - x1, 40 + y2 - y1), 8, fill=(4, 10, 8, 245), outline=GREEN, width=2)
        d.text((70, 72), "[SYS]", font=self.F.get("mono", 20), fill=GDIM)
        d.text((140, 64), title, font=self.F.get("mono", 30, 500), fill=GREEN)
        f = self.F.get("mono", 17, 500)
        tw = d.textlength(tag, font=f)
        d.rectangle((40 + x2 - x1 - 40 - tw - 24, 66, 40 + x2 - x1 - 40, 100), outline=GREEN, width=2)
        d.text((40 + x2 - x1 - 52 - tw, 73), tag, font=f, fill=GREEN)
        d.line((70, 124, 10 + x2 - x1, 124), fill=(30, 90, 50), width=1)
        blend(img, lay, alpha, (x1 - 40, y1 - 40))
        return (x1 + 30, y1 + 100, x2 - 30, y2 - 30)  # inner content box

    def concept_text(self, img, t, t0, num, eyebrow, title, accent, body, foot):
        d = ImageDraw.Draw(img)
        a = fade(t, t0 + 0.1)
        big = text_layer([(f"{num:02d}", self.F.get("serif", 300), (22, 52, 70))], (500, 340))
        blend(img, big, a * 0.9, (120, 160))
        draw_text_fade(img, (128, 330), [(eyebrow, self.F.get("sans", 21, 500), CYAN)], fade(t, t0 + 0.3))
        draw_text_fade(img, (124, 360), [(title + " ", self.F.get("serif", 104), WHITE),
                                        (accent, self.F.get("serif_i", 104), CYAN)], fade(t, t0 + 0.45))
        fb = self.F.get("sans", 33, 300)
        for i, line in enumerate(wrap(d, body, fb, 720)):
            draw_text_fade(img, (128, 520 + i * 50), [(line, fb, BODY)], fade(t, t0 + 0.8 + 0.12 * i))
        af = fade(t, t0 + 1.6)
        lay = Image.new("RGBA", (900, 60), (0, 0, 0, 0))
        dd = ImageDraw.Draw(lay)
        dd.line((0, 0, 70, 0), fill=(40, 90, 110), width=2)
        dd.text((0, 26), foot, font=self.F.get("mono", 20), fill=(120, 140, 160))
        blend(img, lay, af, (128, 800))

    # ---- data generators
    def _skeleton_seq(self):
        from synth import synth_sequence

        rng = np.random.RandomState(55)  # a side-view sample
        seq = synth_sequence("play_bow", 60, fps=15, rng=rng)
        return seq

    def _source_timeline(self, x):
        """Sound-source gate verdicts every 0.25 s over the stretch that plays, from the same
        AST model the server uses - so the on-screen DOG / HUMAN calls are real."""
        from dogsense.models import SoundSourceGate

        g = SoundSourceGate("cuda" if __import__("torch").cuda.is_available() else "cpu")
        out = {}
        for k in range(int(10.5 / 0.25)):
            c = AUDIO_BARK_FROM + k * 0.25
            seg = x[int((c - 0.75) * 16000):int((c + 0.75) * 16000)]
            out[k] = g(seg)
        return out

    def _spectrogram(self):
        with wave.open(str(ROOT / "data" / "test" / BARK_WAV)) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
        import scipy.signal as sps

        from dogsense.acoustics import _HP

        self.bark_audio = sps.sosfilt(_HP, x).astype(np.float32)  # same 150 Hz wind filter as the server
        self.sources = self._source_timeline(x)
        f, t, Z = cv2_spec(x)
        return f, t, Z

    # ---- scenes ------------------------------------------------------------------------------
    def frame(self, t):
        img = self.bg.copy()
        for start, end, fn in SCENES:
            if start <= t < end:
                fn(self, img, t, start, end)
                # scene fade in / out through the dark field
                k = min(fade(t, start, 0.45), 1 - ease((t - (end - 0.45)) / 0.45))
                if k < 1:
                    img = Image.blend(self.bg, img, max(k, 0))
                break
        return img

    def s_intro(self, img, t, s, e):
        d = ImageDraw.Draw(img)
        f = self.F.get("serif", 64)
        fi = self.F.get("serif_i", 64)
        draw_text_fade(img, (160, 300), [("On every walk,", f, WHITE)], fade(t, s + 0.4, 0.9))
        draw_text_fade(img, (160, 384), [("a dog is already telling you how it feels —", f, WHITE)], fade(t, s + 1.8, 0.9))
        draw_text_fade(img, (160, 468), [("in ", f, BODY), ("posture and sound.", fi, CYAN)], fade(t, s + 3.2, 0.9))
        # tail-wag waveform drawing in (the canine heartbeat line)
        n = int(min((t - s) / (e - s - 0.5), 1) * W)
        pts = []
        for x in range(0, n, 4):
            env = math.exp(-((x % 480 - 240) / 60) ** 2)
            y = 760 + math.sin(x / 7.0) * 26 * env
            pts.append((x, y))
        if len(pts) > 1:
            d.line(pts, fill=(40, 120, 150), width=2)

    def s_title(self, img, t, s, e):
        d = ImageDraw.Draw(img)
        draw_text_fade(img, (200, 330), [("CANINE BEHAVIOUR TELEMETRY  ·  ROKID RV101 GLASSES", self.F.get("sans", 21, 500), CYAN)], fade(t, s + 0.2))
        draw_text_fade(img, (192, 370), [("DogSense", self.F.get("serif", 170), WHITE)], fade(t, s + 0.4))
        draw_text_fade(img, (196, 540), [("HUD", self.F.get("serif_i", 170), CYAN)], fade(t, s + 0.7))
        draw_text_fade(img, (200, 760), [("A transparent display.", self.F.get("sans", 30, 300), BODY)], fade(t, s + 1.4))
        draw_text_fade(img, (200, 802), [("Your eyes stay on the dog.", self.F.get("sans", 30, 300), CYAN)], fade(t, s + 1.9))
        # glasses outline with live telemetry in the right lens
        a = fade(t, s + 0.5, 1.0)
        lay = Image.new("RGBA", (900, 300), (0, 0, 0, 0))
        g = ImageDraw.Draw(lay)
        c = (150, 175, 200, 255)
        g.rounded_rectangle((120, 60, 400, 230), 60, outline=c, width=4)
        g.rounded_rectangle((480, 60, 760, 230), 60, outline=c, width=4)
        g.arc((395, 90, 485, 170), 200, 340, fill=c, width=4)
        g.line((0, 110, 120, 100), fill=c, width=4)
        g.line((760, 100, 880, 110), fill=c, width=4)
        if t > s + 1.6:
            ar = 0.35 + 0.3 * math.sin((t - s) * 2.2) ** 2
            g.text((520, 105), f"AROUSAL [{'|' * int(ar * 10)}{'.' * (10 - int(ar * 10))}]", font=self.F.get("mono", 16), fill=GREEN)
            g.text((520, 132), "VALENCE [   |+ + ]", font=self.F.get("mono", 16), fill=GREEN)
            g.text((520, 166), "STATE: Play", font=self.F.get("mono", 16, 700), fill=GREEN)
        blend(img, lay, a, (1000, 480))

    def s_kinematics(self, img, t, s, e):
        self.concept_text(img, t, s, 1, "CONCEPT 01  —  VISION", "Body", "language",
                          "Twenty-four keypoints per dog, re-expressed in the dog's own body frame — so your head can move and the reading still holds.",
                          "YOLO26N-POSE  ·  DOG-POSE DATASET  ·  POSE mAP50 0.936")
        box = self.panel(img, "POSE_KINEMATICS.exe", "LIVE", fade(t, s + 0.6))
        if t < s + 0.9:
            return
        d = ImageDraw.Draw(img)
        x1, y1, x2, y2 = box
        fm = self.F.get("mono", 21, 500)
        k = int((t - s) * 15) % len(self.skel)
        kp = self.skel[k].copy()
        # fit skeleton into the panel
        # centre on this frame's body and use a fixed scale, so camera drift doesn't shrink it
        vis = kp[kp[:, 2] > .3, :2]
        c = (vis.min(0) + vis.max(0)) / 2
        ext = max(np.ptp(f[f[:, 2] > .3, :2], axis=0).max() for f in self.skel)
        sc = min(x2 - x1 - 140, y2 - y1 - 190) / ext
        P = (kp[:, :2] - c) * sc + [(x1 + x2) / 2, (y1 + 100 + y2 - 50) / 2]
        from dogsense.keypoints import EDGES

        for a_, b_ in EDGES:
            if kp[a_, 2] > .3 and kp[b_, 2] > .3:
                d.line([tuple(P[a_]), tuple(P[b_])], fill=GDIM, width=3)
        for i, (x, y) in enumerate(P):
            if kp[i, 2] > .3:
                d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=GREEN)
        wag = 2.6 + 0.4 * math.sin(t)
        d.text((x1, y1), f"> WAG: {wag:.1f} Hz", font=fm, fill=GREEN)
        d.text((x1 + 270, y1), f"> BOW: {0.78 + 0.05 * math.sin(t * 3):.2f}", font=fm, fill=GREEN)
        d.text((x1 + 500, y1), "> EARS: FWD", font=fm, fill=GREEN)
        d.text((x1, y1 + 40), "[TRAINING SEQ 1 / 30,000] body frame · ego-motion cancelled", font=self.F.get("mono", 17), fill=GDIM)
        st = "STATE: PLAY SOLICITATION" if t > s + 5 else "STATE: READING POSTURE..."
        d.text((x1, y2 - 36), st, font=self.F.get("mono", 24, 700), fill=GREEN)

    def s_acoustics(self, img, t, s, e):
        self.concept_text(img, t, s, 2, "CONCEPT 02  —  SOUND", "Bark", "acoustics",
                          "Pitch, harmonicity and the gap between barks — measured, not guessed. A sound classifier checks every call: your own voice is ignored.",
                          "F0  ·  HNR  ·  AST SOURCE GATE  ·  REAL AUDIO, IMG_2389")
        box = self.panel(img, "BIOACOUSTIC_NODE.exe", "LISTENING", fade(t, s + 0.6))
        if t < s + 0.9:
            return
        x1, y1, x2, y2 = box
        f, tt, Z = self.spec
        # scrolling 4 s window of the real recording (same clip that plays in the score)
        cur = AUDIO_BARK_FROM + (t - s - 0.9)
        j1 = np.searchsorted(tt, cur)
        j0 = max(0, j1 - int(4.0 / (tt[1] - tt[0])))
        seg = Z[:, j0:j1]
        if seg.shape[1] > 2:
            hh, ww = 330, x2 - x1
            im = (np.clip((seg - (-80)) / 70, 0, 1) * 255).astype(np.uint8)[::-1]
            im = cv2.resize(im, (ww, hh), interpolation=cv2.INTER_LINEAR)
            rgb = np.zeros((hh, ww, 3), np.uint8)
            rgb[..., 1] = im
            rgb[..., 0] = (im * 0.25).astype(np.uint8)
            rgb[..., 2] = (im * 0.45).astype(np.uint8)
            img.paste(Image.fromarray(rgb), (x1, y1 + 70))
        d = ImageDraw.Draw(img)
        d.rectangle((x1, y1 + 70, x2, y1 + 400), outline=GDIM, width=1)
        d.text((x1 + 10, y1 + 78), "[LIVE] SPECTROGRAM 0–4 kHz", font=self.F.get("mono", 16), fill=GDIM)
        fm = self.F.get("mono", 21, 500)
        lvl = float(np.abs(self.bark_audio[int(cur * 16000):int(cur * 16000) + 1600]).mean()) if cur * 16000 < len(self.bark_audio) else 0
        barking = lvl > 0.01  # median level of this recording is ~0.013
        if barking:  # measured from the real recording with the pipeline's own estimators
            from dogsense.acoustics import dominant_hz, f0_hz, hnr_db

            i0 = int(cur * 16000)
            seg = self.bark_audio[i0:i0 + 3200]
            self._hnr = hnr_db(seg)
            p = f0_hz(seg) if self._hnr >= 3 else dominant_hz(seg)  # same rule as the pipeline
            self._f0 = 0.6 * getattr(self, "_f0", p) + 0.4 * p
        d.text((x1, y1), f"> F0: {self._f0:.0f} Hz" if barking else "> F0: ---", font=fm, fill=GREEN)
        d.text((x1 + 250, y1), f"> HNR: {self._hnr:.1f} dB" if barking else "> HNR: ---", font=fm, fill=GREEN)
        k = int((cur - AUDIO_BARK_FROM) / 0.25)
        near = [self.sources.get(k + d, "other") for d in (-2, -1, 0, 1, 2)]
        src = "human" if "human" in near[1:4] else self.sources.get(k, "other")  # hold a human verdict ~0.5 s
        if src == "other" and not barking:
            src = ""
        human = src == "human"
        d.text((x1 + 500, y1), "> SRC: " + ({"dog": "DOG", "human": "HUMAN"}.get(src, "—")), font=fm, fill=GREEN)
        lvl_w = int(min(lvl * 3000, x2 - x1))
        d.rectangle((x1, y2 - 70, x1 + lvl_w, y2 - 58), fill=GDIM if human else GREEN)
        if human:  # the wearer talking: shown, then ignored
            d.rectangle((x1 - 6, y2 - 46, x2, y2 - 8), fill=GREEN)
            d.text((x1, y2 - 40), "HUMAN VOICE  ->  IGNORED  (AST source gate)", font=self.F.get("mono", 22, 700), fill=(4, 10, 8))
        else:
            d.text((x1, y2 - 40), "STATE: BARKING  (posture decides the meaning)", font=self.F.get("mono", 22, 700), fill=GREEN)

    def s_fusion(self, img, t, s, e):
        self.concept_text(img, t, s, 3, "CONCEPT 03  —  REASONING", "Symbolic", "fusion",
                          "A bark alone means nothing. Posture decides. Explicit rules resolve the conflicts — and every state traces back to the cues that fired it.",
                          "NEURO-SYMBOLIC  ·  ORDERED RULES  ·  NO BLACK-BOX VERDICTS")
        box = self.panel(img, "FUSION_RULES.engine", "ENGINE", fade(t, s + 0.6))
        if t < s + 0.9:
            return
        x1, y1, x2, y2 = box
        d = ImageDraw.Draw(img)
        fm = self.F.get("mono", 22, 500)
        rules = [
            ("IF   bow > 0.5  ∧  growl(low F0)", "=> PLAY SOLICITATION  (safe)"),
            ("IF   stiff ∧ weight fwd ∧ growl", "=> DEFENSIVE WARNING  [ALERT]"),
            ("IF   ears pinned ∧ weight back", "=> ANXIOUS / APPEASING"),
            ("IF   no posture ∧ bark heard", "=> BARKING  (dog not in view)"),
        ]
        chars = int((t - s - 0.9) * 38)
        y = y1 + 10
        for i, (a_, b_) in enumerate(rules):
            line = a_
            shown = line[:max(0, min(len(line), chars - i * 40))]
            d.text((x1, y), shown, font=fm, fill=GREEN)
            if chars - i * 40 > len(line) + 4:
                hl = (t - s) > 6.2 and i == 0
                col = (4, 10, 8) if hl else GREEN
                if hl:
                    d.rectangle((x1 - 6, y + 34, x2, y + 70), fill=GREEN)
                d.text((x1 + 30, y + 38), b_, font=fm, fill=col)
            y += 112
        if (t - s) > 6.2:
            d.text((x1, y2 - 36), "FIRED: rule 1 — cues: bow · play growl 180 Hz", font=self.F.get("mono", 19), fill=GREEN)

    def s_photo(self, img, t, s, e):
        a = fade(t, s + 0.2, 0.9)
        p = (t - s) / (e - s)
        ph = self.photo
        z = 1.0 + 0.08 * p
        w0, h0 = 1040, 780
        crop_w, crop_h = int(ph.width / z), int(ph.height / z)
        cx, cy = ph.width // 2, int(ph.height * 0.48)
        c = ph.crop((cx - crop_w // 2, cy - crop_h // 2, cx + crop_w // 2, cy + crop_h // 2)).resize((w0, h0), Image.LANCZOS)
        frame = Image.new("RGBA", (w0 + 40, h0 + 40), (230, 236, 244, 255))
        frame.paste(c, (20, 20))
        frame = frame.rotate(-2.2, expand=True, resample=Image.BICUBIC)
        sh = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        sh.putalpha(frame.getchannel("A").point(lambda v: int(v * 0.6)))
        sh = sh.filter(ImageFilter.GaussianBlur(24))
        blend(img, sh, a, (800, 170))
        blend(img, frame, a, (780, 140))
        draw_text_fade(img, (130, 360), [("It started with", self.F.get("serif", 92), WHITE)], fade(t, s + 0.5))
        draw_text_fade(img, (130, 460), [("one photo.", self.F.get("serif_i", 92), CYAN)], fade(t, s + 0.9))
        draw_text_fade(img, (134, 600), [("The first HUD the system ever drew —", self.F.get("sans", 28, 300), BODY)], fade(t, s + 1.6))
        draw_text_fade(img, (134, 640), [("Figure 2 in the documentation.", self.F.get("sans", 28, 300), BODY)], fade(t, s + 1.9))
        draw_text_fade(img, (134, 720), [("DOCS · FIG. 2 · FIRST RENDER", self.F.get("mono", 20), (120, 140, 160))], fade(t, s + 2.3))

    def _video_scene(self, img, t, s, cap, src_from, box, label):
        x1, y1, x2, y2 = box
        a = fade(t, s + 0.4)
        src_t = src_from + (t - s)
        cap.set(cv2.CAP_PROP_POS_MSEC, src_t * 1000)
        ok, fr = cap.read()
        if not ok:
            return
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        h, w = fr.shape[:2]
        sc = min((x2 - x1) / w, (y2 - y1) / h)
        fr = cv2.resize(fr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        ox = x1 + ((x2 - x1) - fr.shape[1]) // 2
        lay = Image.fromarray(fr).convert("RGBA")
        g = self.panel_glow.resize((fr.shape[1] + 80, fr.shape[0] + 80))
        blend(img, g, a, (ox - 40, y1 - 40))
        blend(img, lay, a, (ox, y1))
        d = ImageDraw.Draw(img)
        d.rectangle((ox, y1, ox + fr.shape[1], y1 + fr.shape[0]), outline=GREEN, width=2)
        rec = int(t * 2) % 2 == 0
        d.ellipse((ox + 18, y1 + 18, ox + 32, y1 + 32), fill=(255, 80, 80) if rec else (90, 30, 30))
        d.text((ox + 42, y1 + 12), label, font=self.F.get("mono", 18, 700), fill=WHITE)

    def s_field(self, img, t, s, e):
        self.concept_text(img, t, s, 4, "FIELD TEST  —  ROKID GLASSES", "Field", "tested",
                          "Filmed through the glasses: two dogs, one garden, 61 seconds, every frame processed. Four bugs surfaced in the field — all four fixed.",
                          "1,830 FRAMES  ·  DOGS TRACKED 56%  ·  WEARER'S VOICE FILTERED")
        self._video_scene(img, t, s, self.vid1, VID1_FROM, (1000, 150, 1800, 960), "ROKID RV101  ·  REAL FOOTAGE")

    def s_gate(self, img, t, s, e):
        self.concept_text(img, t, s, 5, "STRESS TEST  —  HEAD-ON, BEHIND A GATE", "Through the", "bars",
                          "Dogs facing you, broken up by gate bars. A second detection stage finds them — and re-reads each one on a zoomed crop.",
                          "TWO-STAGE DETECTION  ·  YOLO26S + DOG-POSE")
        self._video_scene(img, t, s, self.vid2, VID2_FROM, (1000, 150, 1800, 960), "GATE TEST  ·  IMG_2390")
        # tracking-rate counter
        a = fade(t, s + 2.5)
        p = ease((t - s - 3.0) / 2.5)
        val = 35 + (79 - 35) * p
        lay = Image.new("RGBA", (760, 110), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        d.text((0, 0), "DOGS TRACKED", font=self.F.get("mono", 18), fill=(120, 140, 160))
        d.rectangle((0, 40, 600, 58), outline=(40, 90, 110), width=2)
        d.rectangle((0, 40, int(600 * val / 100), 58), fill=CYAN)
        d.text((620, 24), f"{val:.0f}%", font=self.F.get("serif", 56), fill=WHITE)
        d.text((0, 72), "before: 35%", font=self.F.get("mono", 16), fill=(120, 140, 160))
        blend(img, lay, a, (128, 880))

    def s_numbers(self, img, t, s, e):
        stats = [("37.8", "ms", "median round trip on the GPU server"),
                 ("24", "pts", "keypoints read per dog, every frame"),
                 ("9/9", "", "unit tests passing"),
                 ("3", "", "real Rokid clips tested · 10 field issues fixed")]
        draw_text_fade(img, (128, 200), [("Built, trained and tested ", self.F.get("serif", 80), WHITE),
                                        ("locally.", self.F.get("serif_i", 80), CYAN)], fade(t, s + 0.2))
        draw_text_fade(img, (132, 310), [("RTX 3090  ·  Python server  ·  Kotlin glasses app  ·  open repository", self.F.get("mono", 22), (120, 140, 160))], fade(t, s + 0.6))
        for i, (n, u, lab) in enumerate(stats):
            x = 128 + i * 420
            a = fade(t, s + 0.9 + i * 0.35)
            lay = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
            d = ImageDraw.Draw(lay)
            d.line((0, 0, 360, 0), fill=(40, 90, 110), width=2)
            d.text((0, 30), n, font=self.F.get("serif", 130), fill=WHITE)
            d.text((d.textlength(n, font=self.F.get("serif", 130)) + 12, 110), u, font=self.F.get("serif_i", 50), fill=CYAN)
            for j, line in enumerate(wrap(d, lab, self.F.get("sans", 24, 300), 340)):
                d.text((0, 200 + j * 34), line, font=self.F.get("sans", 24, 300), fill=BODY)
            blend(img, lay, a, (x, 470 + int((1 - a) * 20)))

    def s_tagline(self, img, t, s, e):
        f, fi = self.F.get("serif", 96), self.F.get("serif_i", 96)
        d = ImageDraw.Draw(img)
        w = d.textlength("Measured, ", font=f) + d.textlength("never imagined.", font=fi)
        draw_text_fade(img, ((W - w) / 2, 430), [("Measured, ", f, WHITE), ("never imagined.", fi, CYAN)], fade(t, s + 0.3, 1.0))
        sub = "It reports what it saw and heard — not what a dog is thinking."
        fs = self.F.get("sans", 28, 300)
        draw_text_fade(img, ((W - d.textlength(sub, font=fs)) / 2, 580), [(sub, fs, BODY)], fade(t, s + 1.4, 1.0))

    def s_end(self, img, t, s, e):
        a = fade(t, s + 0.2, 1.2)
        pulse = 0.75 + 0.25 * math.sin((t - s) * 2.0)
        blend(img, self.logo_glow, a * pulse, (W // 2 - 180, 150))
        blend(img, self.logo_big, a, (W // 2 - 150, 180))
        d = ImageDraw.Draw(img)
        f = self.F.get("sans", 58, 700)
        for i, (txt, col) in enumerate([("SYMBOLIC", (110, 160, 230)), ("SYNAPSE", (90, 200, 235))]):
            w = d.textlength(txt, font=f) + 12 * (len(txt) - 1)
            lay = Image.new("RGBA", (int(w) + 10, 80), (0, 0, 0, 0))
            dd = ImageDraw.Draw(lay)
            x = 0
            for ch in txt:
                dd.text((x, 0), ch, font=f, fill=col)
                x += dd.textlength(ch, font=f) + 12
            blend(img, lay, fade(t, s + 0.6 + i * 0.2), (int((W - w) / 2), 560 + i * 72))
        tag = "INTELLIGENCE THROUGH LOGIC & LEARNING"
        ft = self.F.get("sans", 20, 500)
        draw_text_fade(img, ((W - d.textlength(tag, font=ft)) / 2, 718), [(tag, ft, CYAN)], fade(t, s + 1.1))
        cta = "DogSense HUD  ·  open-source repository  ·  link in the post"
        fc = self.F.get("mono", 24, 500)
        draw_text_fade(img, ((W - d.textlength(cta, font=fc)) / 2, 830), [(cta, fc, WHITE)], fade(t, s + 1.6))


def cv2_spec(x):
    import scipy.signal as sps

    f, t, Z = sps.stft(x, 16000, nperseg=512, noverlap=384)
    m = 20 * np.log10(np.abs(Z) + 1e-6)
    keep = f <= 4000
    return f[keep], t, m[keep]


# timeline (seconds)
BARK_WAV = "v2389.wav"   # real off-screen barking, with the wearer speaking at ~12.5 s
AUDIO_BARK_FROM = 6.2    # offset that plays under the acoustics scene
VID1_FROM = 5.0          # DogSense_HUD_test.mp4 offset (two dogs at ~7 s)
VID2_FROM = 6.0          # hud2390.mp4 offset (dogs at the gate)
SCENES = [
    (0.0, 7.0, Renderer.s_intro),
    (7.0, 13.0, Renderer.s_title),
    (13.0, 24.0, Renderer.s_kinematics),
    (24.0, 35.0, Renderer.s_acoustics),
    (35.0, 45.0, Renderer.s_fusion),
    (45.0, 51.5, Renderer.s_photo),
    (51.5, 63.5, Renderer.s_field),
    (63.5, 74.0, Renderer.s_gate),
    (74.0, 80.5, Renderer.s_numbers),
    (80.5, 85.0, Renderer.s_tagline),
    (85.0, 90.0, Renderer.s_end),
]
SCENE_INDEX = {Renderer.s_kinematics: 1, Renderer.s_acoustics: 2, Renderer.s_fusion: 3, Renderer.s_field: 4, Renderer.s_gate: 5}


# ------------------------------------------------------------------------------- score
def make_audio(path: Path, sr=48000):
    n = int(DUR * sr)
    t = np.arange(n) / sr
    rng = np.random.RandomState(3)
    # ambient pad: A minor add9, slow swells, detuned pairs
    pad = np.zeros(n)
    for f0, amp in [(110, .5), (164.81, .35), (220, .3), (261.63, .22), (329.63, .16), (493.88, .08)]:
        for det in (-0.4, 0.4):
            pad += amp * np.sin(2 * np.pi * (f0 + det) * t + rng.uniform(0, 6))
    pad *= 0.55 + 0.45 * np.sin(2 * np.pi * t / 11.0) ** 2
    # soft low pulse that quickens in the field-test section
    pulse = np.zeros(n)
    bpm_times = np.arange(0.5, DUR - 4, 1.2)
    for bt in bpm_times:
        i = int(bt * sr); L = int(0.5 * sr)
        if i + L < n:
            env = np.exp(-np.arange(L) / (0.08 * sr))
            pulse[i:i + L] += np.sin(2 * np.pi * 55 * np.arange(L) / sr) * env
    # whooshes on scene changes
    wh = np.zeros(n)
    for s, _, _ in SCENES[1:]:
        i = int((s - 0.5) * sr); L = int(1.0 * sr)
        if i + L < n:
            noise = rng.normal(0, 1, L)
            b = np.convolve(noise, np.ones(40) / 40, "same")
            env = np.sin(np.linspace(0, np.pi, L)) ** 2
            wh[i:i + L] += b * env
    mix = 0.10 * pad / np.abs(pad).max() + 0.10 * pulse + 0.25 * wh / (np.abs(wh).max() + 1e-9)
    # fade in/out
    mix *= np.minimum(1, t / 2.0) * np.minimum(1, (DUR - t) / 3.0)

    def put(clip_sr16k, at, gain):
        c = np.interp(np.arange(int(len(clip_sr16k) * sr / 16000)) * 16000 / sr, np.arange(len(clip_sr16k)), clip_sr16k)
        i = int(at * sr)
        L = min(len(c), n - i)
        ramp = np.minimum(1, np.minimum(np.arange(L) / (0.3 * sr), (L - np.arange(L)) / (0.5 * sr)))
        mix[i:i + L] += gain * c[:L] * ramp

    with wave.open(str(ROOT / "data" / "test" / "dog.wav")) as w:
        dog = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    # real barks under the acoustics scene, then the field-test clip's own sound
    a0 = 24.9
    with wave.open(str(ROOT / "data" / "test" / BARK_WAV)) as w:
        barks = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    put(barks[int(AUDIO_BARK_FROM * 16000): int((AUDIO_BARK_FROM + 9.6) * 16000)], a0, 0.9)
    put(dog[int(VID1_FROM * 16000): int((VID1_FROM + 11.5) * 16000)], 51.9, 0.55)
    with wave.open(str(ROOT / "data" / "test" / "v2390.wav")) as w:
        gate = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    put(gate[int(VID2_FROM * 16000): int((VID2_FROM + 10) * 16000)], 63.9, 0.5)
    mix /= max(1.0, np.abs(mix).max() / 0.9)
    st = np.stack([mix, mix], 1)
    st[:, 1] = np.roll(st[:, 1], int(0.012 * sr)) * 0.98 + 0.02 * st[:, 0]  # a little width
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((st * 32767).astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "media" / "DogSense_HUD_90s.mp4")
    ap.add_argument("--preview", type=str, default="", help="comma-separated times: write PNG stills only")
    args = ap.parse_args()
    r = Renderer(Fonts(args.fonts))

    def chrome_for(img, t):
        for s, e, fn in SCENES:
            if s <= t < e and fn in SCENE_INDEX:
                r.chrome(img, t, "", SCENE_INDEX[fn])
            elif s <= t < e and fn not in (Renderer.s_intro, Renderer.s_end):
                r.chrome(img, t, "", 0)

    if args.preview:
        for ts in args.preview.split(","):
            t = float(ts)
            img = r.frame(t); chrome_for(img, t)
            img.save(args.out.with_name(f"preview_{t:05.1f}.png"))
        return
    wav = args.out.with_suffix(".wav")
    make_audio(wav)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-i", str(wav), "-c:v", "libx264", "-preset", "medium",
                           "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
                           "-movflags", "+faststart", str(args.out)], stdin=subprocess.PIPE)
    n = int(DUR * FPS)
    for i in range(n):
        t = i / FPS
        img = r.frame(t)
        chrome_for(img, t)
        ff.stdin.write(img.tobytes())
        if i % 250 == 0:
            print(f"{t:5.1f}s", flush=True)
    ff.stdin.close(); ff.wait()
    wav.unlink()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
