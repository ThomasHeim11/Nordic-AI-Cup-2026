"""Fine-tune a YOLO detector on the multi-scale Helsinki crops.

    python train_yolo.py --model yolo11s.pt --epochs 80 --imgsz 960 --batch 8
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--name", default="drone")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--amp", type=int, default=1)
    a = ap.parse_args()

    model = YOLO(a.model)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
        workers=a.workers, name=a.name, project="runs", exist_ok=True,
        # Top-down imagery: vertical flips are as valid as horizontal ones.
        fliplr=0.5, flipud=0.5, degrees=0.0, scale=0.3, mosaic=1.0, close_mosaic=10,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, translate=0.1,
        patience=30, cos_lr=True, plots=False, verbose=True,
        lr0=a.lr0, amp=bool(a.amp), warmup_epochs=1.0,
    )


if __name__ == "__main__":
    main()
