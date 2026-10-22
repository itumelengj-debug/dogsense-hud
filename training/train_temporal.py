"""Train the temporal behaviour head (1D-CNN over 30-frame keypoint windows).

Synthetic pre-training (default):
    python training/train_temporal.py --synthetic 30000
Fine-tune on real labelled clips (folder per class, names from dogsense.models.POSE_CLASSES):
    python training/train_temporal.py --videos data/behaviour_clips --init models/temporal_pose.pt
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
from dogsense.kinematics import KinematicTracker  # noqa: E402
from dogsense.models import POSE_CLASSES, SEQ_DIM, SEQ_LEN, TemporalPoseNet  # noqa: E402


def featurise(kp_seq: np.ndarray, fps=15.0) -> np.ndarray | None:
    tr = KinematicTracker(window=SEQ_LEN, fps=fps)
    for t, k in enumerate(kp_seq):
        tr.push(k, t / fps)
    s = tr.sequence()
    if len(s) < SEQ_LEN // 2:
        return None
    if len(s) < SEQ_LEN:
        s = np.concatenate([np.repeat(s[:1], SEQ_LEN - len(s), 0), s])
    return s


def _synth_worker(args):
    from synth import synth_sequence

    label_idx, seed = args
    rng = np.random.RandomState(seed)
    return featurise(synth_sequence(POSE_CLASSES[label_idx], SEQ_LEN, rng=rng)), label_idx


def build_synthetic(n: int):
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(i % len(POSE_CLASSES), i) for i in range(n)]
    X, Y = [], []
    with ProcessPoolExecutor(max_workers=4) as ex:  # bounded: each worker imports torch
        for x, y in ex.map(_synth_worker, jobs, chunksize=256):
            if x is not None:
                X.append(x); Y.append(y)
    return np.stack(X), np.array(Y)


def build_from_videos(root: Path, stride=5, fps=15.0):
    import cv2
    from ultralytics import YOLO

    w = ROOT / "models" / "dog_pose.pt"
    yolo = YOLO(str(w))
    X, Y = [], []
    for ci, c in enumerate(POSE_CLASSES):
        for vid in sorted((root / c).glob("*")):
            cap = cv2.VideoCapture(str(vid))
            step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30) / fps))
            kps, i = [], 0
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                if i % step == 0:
                    r = yolo.predict(fr, verbose=False, conf=0.35)[0]
                    if r.keypoints is not None and len(r.boxes):
                        j = int(r.boxes.conf.argmax())
                        kps.append(r.keypoints.data[j].cpu().numpy())
                i += 1
            for s in range(0, max(1, len(kps) - SEQ_LEN + 1), stride):
                x = featurise(np.array(kps[s:s + SEQ_LEN]))
                if x is not None:
                    X.append(x); Y.append(ci)
            print(f"{c}/{vid.name}: {len(kps)} frames")
    return np.stack(X), np.array(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=30000)
    ap.add_argument("--videos", type=Path, default=None)
    ap.add_argument("--init", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "temporal_pose.pt")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X, Y = build_from_videos(args.videos) if args.videos else build_synthetic(args.synthetic)
    print("dataset", X.shape, np.bincount(Y, minlength=len(POSE_CLASSES)))
    assert X.shape[1:] == (SEQ_LEN, SEQ_DIM)
    perm = np.random.RandomState(0).permutation(len(X))
    nv = max(1, len(X) // 10)
    va, tr = perm[:nv], perm[nv:]
    Xt, Yt = torch.tensor(X).to(dev), torch.tensor(Y).to(dev)

    model = TemporalPoseNet().to(dev)
    if args.init and args.init.exists():
        model.load_state_dict(torch.load(args.init, map_location=dev, weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=args.epochs * (len(tr) // 256 + 1))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        idx = torch.tensor(np.random.permutation(tr), device=dev)
        for b in range(0, len(idx), 256):
            j = idx[b:b + 256]
            x = Xt[j] * (1 + 0.05 * torch.randn(len(j), 1, 1, device=dev))  # scale jitter
            loss = lossf(model(x), Yt[j])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xt[va]).argmax(1)
            acc = (pred == Yt[va]).float().mean().item()
        if acc >= best:
            best = acc
            args.out.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), args.out)
        print(f"epoch {ep + 1:02d} loss {loss.item():.3f} val_acc {acc:.3f}")
    cm = np.zeros((len(POSE_CLASSES),) * 2, int)
    model.load_state_dict(torch.load(args.out, weights_only=True))
    with torch.no_grad():
        for t, p in zip(Yt[va].cpu().numpy(), model(Xt[va]).argmax(1).cpu().numpy()):
            cm[t, p] += 1
    print("best val acc", round(best, 3), "\nconfusion (rows=true):", POSE_CLASSES, "\n", cm)


if __name__ == "__main__":
    main()
