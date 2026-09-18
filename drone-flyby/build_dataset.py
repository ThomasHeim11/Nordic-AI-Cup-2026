"""Build a YOLO dataset from the 25 Helsinki frames.

Every training image is rendered exactly the way the evaluator renders a view:
crop a source region of 3840x2160 (L0), 1920x1080 (L1) or 960x540 (L2) and
INTER_AREA-resize it to 960x540.  Boxes are clipped to the crop and dropped
when less than half of them survives or they end up smaller than MIN_SIDE px.

    python build_dataset.py --out data/yolo --l1-per-frame 8 --l2-per-frame 16
"""
import argparse, json, random
from pathlib import Path

import cv2
import numpy as np

from dtos import OBJECT_CLASSES, SOURCE_REGION_SIZES, TRANSMITTED_VIEW_SIZE
from utils import frame_numbers, load_annotations, load_frame

CLASS_INDEX = {name: i for i, name in enumerate(OBJECT_CLASSES)}
MIN_SIDE = 5          # pixels in the rendered 960x540 image
MIN_VISIBLE = 0.45    # fraction of the source box that must survive the crop


def render(frame_image, region):
    x1, y1, x2, y2 = region
    view = frame_image[y1:y2, x1:x2]
    if (view.shape[1], view.shape[0]) != TRANSMITTED_VIEW_SIZE:
        view = cv2.resize(view, TRANSMITTED_VIEW_SIZE, interpolation=cv2.INTER_AREA)
    return view


def labels_for(annotations, region):
    x1, y1, x2, y2 = region
    rw, rh = x2 - x1, y2 - y1
    vw, vh = TRANSMITTED_VIEW_SIZE
    sx, sy = vw / rw, vh / rh
    out = []
    for a in annotations:
        bx1, by1, bx2, by2 = a["bbox"]
        area = max(1.0, (bx2 - bx1) * (by2 - by1))
        cx1, cy1 = max(bx1, x1), max(by1, y1)
        cx2, cy2 = min(bx2, x2), min(by2, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        if (cx2 - cx1) * (cy2 - cy1) / area < MIN_VISIBLE:
            continue
        w, h = (cx2 - cx1) * sx, (cy2 - cy1) * sy
        if max(w, h) < MIN_SIDE:
            continue
        ccx, ccy = ((cx1 + cx2) / 2 - x1) * sx, ((cy1 + cy2) / 2 - y1) * sy
        out.append((CLASS_INDEX[a["object_id"]], ccx / vw, ccy / vh, w / vw, h / vh))
    return out


def random_region(level, rng, focus=None):
    W, H = SOURCE_REGION_SIZES[level]
    if level == 0:
        return (0, 0, 3840, 2160)
    if focus is not None:
        fx, fy = focus
        cx = int(fx + rng.uniform(-W * 0.4, W * 0.4))
        cy = int(fy + rng.uniform(-H * 0.4, H * 0.4))
    else:
        cx = rng.randint(W // 2, 3840 - W // 2)
        cy = rng.randint(H // 2, 2160 - H // 2)
    cx = min(max(cx, W // 2), 3840 - W // 2)
    cy = min(max(cy, H // 2), 2160 - H // 2)
    return (cx - W // 2, cy - H // 2, cx + W // 2, cy + H // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/yolo")
    ap.add_argument("--l1-per-frame", type=int, default=8)
    ap.add_argument("--l2-per-frame", type=int, default=16)
    ap.add_argument("--val-frames", default="22,23,24")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    val_frames = {int(s) for s in a.val_frames.split(",") if s}
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for frame in frame_numbers():
        image = load_frame(frame)
        ann = load_annotations(frame)
        split = "val" if frame in val_frames else "train"
        centres = [((b["bbox"][0] + b["bbox"][2]) / 2, (b["bbox"][1] + b["bbox"][3]) / 2) for b in ann]

        regions = [(0, (0, 0, 3840, 2160))]
        for i in range(a.l1_per_frame):
            focus = rng.choice(centres) if centres and i % 2 == 0 else None
            regions.append((1, random_region(1, rng, focus)))
        for i in range(a.l2_per_frame):
            focus = rng.choice(centres) if centres and i % 3 != 2 else None
            regions.append((2, random_region(2, rng, focus)))

        for k, (level, region) in enumerate(regions):
            labels = labels_for(ann, region)
            if level > 0 and not labels and rng.random() < 0.6:
                continue  # keep a few pure-background crops, not most of them
            name = f"f{frame:03d}_L{level}_{k:02d}"
            cv2.imwrite(str(out / "images" / split / f"{name}.png"), render(image, region))
            with open(out / "labels" / split / f"{name}.txt", "w") as f:
                for c, x, y, w, h in labels:
                    f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            counts[split] += 1

    with open(out / "data.yaml", "w") as f:
        f.write(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n")
        f.write("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(OBJECT_CLASSES)))
    print("images written:", counts)


if __name__ == "__main__":
    main()
