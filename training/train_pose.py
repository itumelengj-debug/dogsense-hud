"""Fine-tune YOLO26n-pose on the Ultralytics Dog-Pose dataset (24 keypoints).

The dataset (~8.5k images) downloads automatically on first run.
Output: models/dog_pose.pt (+ optional ONNX / TFLite export for on-phone use).

    python training/train_pose.py --epochs 100 --batch 32
"""
import argparse
import shutil
from pathlib import Path

try:  # use the OS certificate store (needed behind TLS-inspecting proxies / AV)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n-pose.pt", help="base weights (yolo26n-pose.pt / yolov8n-pose.pt)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--export", nargs="*", default=["onnx"], help="extra export formats, e.g. onnx tflite")
    ap.add_argument("--resume", action="store_true", help="continue runs/dog_pose/weights/last.pt")
    args = ap.parse_args()

    if args.resume:
        YOLO(str(ROOT / "runs" / "dog_pose" / "weights" / "last.pt")).train(resume=True)
    else:
        _train(args)
    best = ROOT / "runs" / "dog_pose" / "weights" / "best.pt"
    out = ROOT / "models" / "dog_pose.pt"
    out.parent.mkdir(exist_ok=True)
    shutil.copy(best, out)
    print(f"saved {out}")

    trained = YOLO(str(out))
    for fmt in args.export:
        print("exported", trained.export(format=fmt, imgsz=args.imgsz))


def _train(args):
    model = YOLO(args.model)
    model.train(
        data=str(ROOT / "training" / "dog-pose-flip.yaml"),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=4,  # Windows dataloader workers are expensive to spawn
        project=str(ROOT / "runs"),
        name="dog_pose",
        exist_ok=True,
        patience=20,
        # Head-mounted camera: tolerate tilt/blur, but never flip vertically.
        degrees=10.0,
        fliplr=0.5,  # flip_idx in dog-pose-flip.yaml swaps left/right keypoints correctly
        mosaic=1.0,
    )


if __name__ == "__main__":
    main()
