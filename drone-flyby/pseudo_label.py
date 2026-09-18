"""Pseudo-label recorded validation views with a trained model (self-training round 2).

Every recorded view is already rendered exactly like the evaluator renders it,
so it is a training image as-is.  Confident detections become labels; views
with nothing above the low threshold become background images; views with only
uncertain detections are skipped (ambiguous supervision).

    python pseudo_label.py --weights runs/detect/runs/synth_yolo11m/weights/best.pt \
        --recordings recordings --out data/pseudo --conf 0.55 --low 0.15
"""
import argparse
import glob
import os
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

from dtos import OBJECT_CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--recordings", default="recordings")
    ap.add_argument("--out", default="data/pseudo")
    ap.add_argument("--conf", type=float, default=0.55, help="min confidence to become a label")
    ap.add_argument("--low", type=float, default=0.15, help="views with any detection between low and conf are skipped")
    ap.add_argument("--max-per-class", type=int, default=6, help="more than this of one class in a view = suspicious, skip")
    a = ap.parse_args()
    model = YOLO(a.weights)
    out = Path(a.out)
    for d in ("images/train", "labels/train"):
        (out / d).mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.recordings, "*", "*.png")))
    n_lab, n_bg, n_skip, n_boxes = 0, 0, 0, 0
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        r = model.predict(img, imgsz=960, conf=a.low, verbose=False)[0]
        rows, uncertain, per_class = [], False, {}
        h, w = img.shape[:2]
        for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
            if c >= a.conf:
                per_class[k] = per_class.get(k, 0) + 1
                rows.append(f"{k} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            else:
                uncertain = True
        if uncertain or any(v > a.max_per_class for v in per_class.values()):
            n_skip += 1
            continue
        stem = Path(f).parent.name[:8] + "_" + Path(f).stem
        shutil.copy(f, out / "images/train" / f"{stem}.png")
        with open(out / "labels/train" / f"{stem}.txt", "w") as fh:
            fh.write("\n".join(rows) + ("\n" if rows else ""))
        if rows:
            n_lab += 1
            n_boxes += len(rows)
        else:
            n_bg += 1
    with open(out / "data.yaml", "w") as fh:
        fh.write(f"path: {out.resolve()}\ntrain: images/train\nval: images/train\n")
        fh.write("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(OBJECT_CLASSES)))
    print(f"pseudo-labelled {n_lab} views ({n_boxes} boxes), {n_bg} background views, {n_skip} skipped as ambiguous -> {out}")


if __name__ == "__main__":
    main()
