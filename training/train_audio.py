"""Train the vocalisation CNN on 1 s log-mel clips (16 kHz).

Synthetic pre-training:
    python training/train_audio.py --synthetic 12000
Real data (DMAD, CREMD audio, or your own Rokid recordings), one folder per class in
dogsense.acoustics.AUDIO_CLASSES (none, growl, bark_alert, bark_play, whine, howl):
    python training/train_audio.py --data data/audio --init models/vocal_cnn.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
from dogsense.acoustics import AUDIO_CLASSES, SR, denoise, log_mel  # noqa: E402
from dogsense.models import VocalCNN, norm_mel  # noqa: E402

N = SR  # 1 s


def _clip_feats(y):
    y = y[:N] if len(y) >= N else np.pad(y, (0, N - len(y)))
    return norm_mel(log_mel(denoise(y)))


def _synth_worker(args):
    from synth import synth_vocal

    ci, seed = args
    return _clip_feats(synth_vocal(AUDIO_CLASSES[ci], np.random.RandomState(seed))), ci


def build_synthetic(n):
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=4) as ex:  # bounded: each worker imports torch
        res = list(ex.map(_synth_worker, [(i % len(AUDIO_CLASSES), i) for i in range(n)], chunksize=128))
    return np.stack([r[0] for r in res]), np.array([r[1] for r in res])


def build_folder(root: Path):
    import soundfile as sf
    from scipy.signal import resample_poly

    X, Y = [], []
    for ci, c in enumerate(AUDIO_CLASSES):
        for f in sorted((root / c).glob("*")):
            if f.suffix.lower() not in (".wav", ".flac", ".ogg", ".mp3"):
                continue
            y, sr = sf.read(f, dtype="float32", always_2d=True)
            y = y.mean(1)
            if sr != SR:
                y = resample_poly(y, SR, sr).astype(np.float32)
            for s in range(0, max(1, len(y) - N + 1), N // 2):  # 50% overlapping 1 s windows
                X.append(_clip_feats(y[s:s + N])); Y.append(ci)
    return np.stack(X), np.array(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=12000)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--init", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "vocal_cnn.pt")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X, Y = build_folder(args.data) if args.data else build_synthetic(args.synthetic)
    print("dataset", X.shape, np.bincount(Y, minlength=len(AUDIO_CLASSES)))
    perm = np.random.RandomState(0).permutation(len(X))
    nv = max(1, len(X) // 10)
    va, tr = perm[:nv], perm[nv:]
    Xt, Yt = torch.tensor(X)[:, None].to(dev), torch.tensor(Y).to(dev)
    counts = np.bincount(Y[tr], minlength=len(AUDIO_CLASSES)).clip(1)
    weight = torch.tensor(counts.sum() / counts / len(counts), dtype=torch.float32, device=dev)

    model = VocalCNN().to(dev)
    if args.init and args.init.exists():
        model.load_state_dict(torch.load(args.init, map_location=dev, weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        idx = torch.tensor(np.random.permutation(tr), device=dev)
        for b in range(0, len(idx), 128):
            j = idx[b:b + 128]
            x = Xt[j].clone()
            # SpecAugment: random time + frequency masks
            for k in range(len(j)):
                f0 = np.random.randint(0, 56); x[k, :, f0:f0 + np.random.randint(0, 8)] = 0
                t0 = np.random.randint(0, 90); x[k, :, :, t0:t0 + np.random.randint(0, 10)] = 0
            loss = lossf(model(x), Yt[j])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xt[va]).argmax(1) == Yt[va]).float().mean().item()
        if acc >= best:
            best = acc
            args.out.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), args.out)
        print(f"epoch {ep + 1:02d} loss {loss.item():.3f} val_acc {acc:.3f}")
    print("best val acc", round(best, 3))


if __name__ == "__main__":
    main()
