"""Small neural heads: temporal pose classifier (1D-CNN) and log-mel vocalisation CNN."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .acoustics import AUDIO_CLASSES, denoise, log_mel
from .keypoints import NUM_KPTS
from .kinematics import FRAME_FEATS

POSE_CLASSES = ["relaxed", "play_bow", "alert", "defensive_stiff", "anxious", "excited"]
SEQ_DIM = NUM_KPTS * 2 + FRAME_FEATS
SEQ_LEN = 30


class TemporalPoseNet(nn.Module):
    """Dilated 1D-CNN over (B, T, SEQ_DIM) keypoint trajectories."""

    def __init__(self, in_dim=SEQ_DIM, n_cls=len(POSE_CLASSES), h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, h, 3, padding=1), nn.BatchNorm1d(h), nn.GELU(),
            nn.Conv1d(h, h, 3, padding=2, dilation=2), nn.BatchNorm1d(h), nn.GELU(),
            nn.Conv1d(h, h, 3, padding=4, dilation=4), nn.BatchNorm1d(h), nn.GELU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(2 * h, n_cls)

    def forward(self, x):
        z = self.net(x.transpose(1, 2))
        return self.head(torch.cat([z.mean(-1), z.amax(-1)], 1))


class VocalCNN(nn.Module):
    """Compact CNN over (B, 1, 64 mels, ~101 frames)."""

    def __init__(self, n_cls=len(AUDIO_CLASSES)):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.f = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 128), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_cls))

    def forward(self, x):
        return self.head(self.f(x))


def norm_mel(m: np.ndarray) -> np.ndarray:
    return (m + 40.0) / 20.0


class PoseClassifier:
    def __init__(self, path: Path, device="cpu"):
        self.device = device
        self.model = TemporalPoseNet().to(device).eval()
        self.model.load_state_dict(torch.load(path, map_location=device, weights_only=True))

    @torch.no_grad()
    def __call__(self, seq: np.ndarray) -> dict:
        if len(seq) < SEQ_LEN:
            seq = np.concatenate([np.repeat(seq[:1], SEQ_LEN - len(seq), 0), seq]) if len(seq) else np.zeros((SEQ_LEN, SEQ_DIM), np.float32)
        x = torch.from_numpy(seq[-SEQ_LEN:]).float()[None].to(self.device)
        p = torch.softmax(self.model(x), -1)[0].cpu().numpy()
        return {c: float(v) for c, v in zip(POSE_CLASSES, p)}


class AudioClassifier:
    def __init__(self, path: Path, device="cpu"):
        self.device = device
        self.model = VocalCNN().to(device).eval()
        self.model.load_state_dict(torch.load(path, map_location=device, weights_only=True))

    @torch.no_grad()
    def __call__(self, clip: np.ndarray) -> dict:
        m = norm_mel(log_mel(denoise(clip)))  # same cleaning as training
        x = torch.from_numpy(m)[None, None].to(self.device)
        p = torch.softmax(self.model(x), -1)[0].cpu().numpy()
        return {c: float(v) for c, v in zip(AUDIO_CLASSES, p)}


class SoundSourceGate:
    """Decides whether a vocal event came from a person or a dog, using the Audio Spectrogram
    Transformer fine-tuned on AudioSet (527 classes, incl. Speech/Shout/Whistling vs Dog/Bark).

    The wearer's own voice reaches the glasses' mic as loud as a nearby dog, and short shouts
    ("hey!") or whistles look like barks and whines to pitch/HNR rules. Field testing showed
    exactly that, so every event is checked here before the dog rules see it.
    """

    NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
    HUMAN = ("Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
             "Conversation", "Narration, monologue", "Shout", "Yell", "Bellow", "Whoop", "Screaming",
             "Children shouting", "Battle cry", "Whistling", "Laughter", "Throat clearing", "Cough", "Sneeze")
    DOG = ("Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling", "Whimper (dog)", "Canidae, dogs, wolves")

    def __init__(self, device="cpu", margin: float = 0.1):
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self.device, self.margin = device, margin
        try:  # cached copy first: no network needed once downloaded
            self.fe = ASTFeatureExtractor.from_pretrained(self.NAME, local_files_only=True)
            model = ASTForAudioClassification.from_pretrained(self.NAME, local_files_only=True)
        except OSError:
            try:  # first run: download, trusting the OS certificate store (TLS-inspecting proxies)
                import truststore

                truststore.inject_into_ssl()
            except ImportError:
                pass
            self.fe = ASTFeatureExtractor.from_pretrained(self.NAME)
            model = ASTForAudioClassification.from_pretrained(self.NAME)
        self.model = model.to(device).eval()
        lab = self.model.config.id2label
        self.h_idx = [i for i, l in lab.items() if l in self.HUMAN]
        self.d_idx = [i for i, l in lab.items() if l in self.DOG]

    @torch.no_grad()
    def scores(self, audio: np.ndarray) -> tuple[float, float]:
        if len(audio) < 8000:  # AST needs some context; pad short events to 0.5 s
            audio = np.pad(audio, (0, 8000 - len(audio)))
        x = self.fe(audio, sampling_rate=16000, return_tensors="pt").to(self.device)
        p = torch.sigmoid(self.model(**x).logits)[0]
        return float(p[self.h_idx].max()), float(p[self.d_idx].max())

    def __call__(self, audio: np.ndarray) -> str:
        if audio is None or len(audio) == 0:
            return "other"
        h, d = self.scores(audio)
        if h > d + self.margin and h > 0.15:
            return "human"
        return "dog" if d >= 0.1 else "other"
