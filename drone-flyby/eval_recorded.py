"""Score a detector on the human-verified objects of the recorded validation views.

cache/mined_objects.json (from mine_recordings.py accept ...) holds the boxes a
human confirmed on recorded frames.  It is not a complete labelling -- objects
nobody verified may exist -- so this reports recall per class at several
confidence thresholds (exact) and the number of unmatched detections
(an upper bound on false positives), which is what the per-class thresholds
and the checkpoint choice need.

    python eval_recorded.py --weights weights/best.pt [--mined cache/mined_objects.json]
"""
import argparse, collections, json, os

import cv2
import numpy as np
from ultralytics import YOLO

from dtos import OBJECT_CLASSES

HERE = os.path.dirname(os.path.abspath(__file__))
THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--mined", default=os.path.join(HERE, "cache", "mined_objects.json"))
    ap.add_argument("--root", default=HERE, help="directory the mined 'file' paths are relative to")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--tta", type=int, default=0)
    a = ap.parse_args()

    objs = [o for o in json.load(open(a.mined)) if o.get("cls") is not None]
    by_file = collections.defaultdict(list)
    for o in objs:
        by_file[o["file"]].append(o)
    model = YOLO(a.weights)
    # detections per file at conf 0.01, thresholded later
    dets = {}
    for f in sorted(by_file):
        path = os.path.join(a.root, f)
        img = cv2.imread(path)
        if img is None:
            print("missing", path)
            continue
        r = model.predict(img, imgsz=960, conf=0.01, iou=0.6, verbose=False, augment=bool(a.tta))[0]
        dets[f] = [(b.tolist(), float(c), int(k)) for b, c, k in
                   zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy())]

    n_gold = collections.Counter(o["cls"] for o in objs)
    print(f"{os.path.basename(a.weights)}: {len(objs)} verified objects on {len(by_file)} frames; "
          f"gold per class: " + ", ".join(f"{OBJECT_CLASSES[k]}={v}" for k, v in sorted(n_gold.items())))
    print(f"{'class':14s} gold " + " ".join(f"rec@{t:<4}" for t in THRESHOLDS) + "   unmatched dets @0.1/@0.3")
    total = {t: [0, 0] for t in THRESHOLDS}
    for k in sorted(n_gold):
        row = []
        for t in THRESHOLDS:
            hit = 0
            for f, gold in by_file.items():
                for g in gold:
                    if g["cls"] != k:
                        continue
                    if any(c >= t and kk == k and iou(b, g["bbox"]) >= a.iou for b, c, kk in dets.get(f, [])):
                        hit += 1
            row.append(hit)
            total[t][0] += hit
            total[t][1] += n_gold[k]
        unmatched = {}
        for t in (0.1, 0.3):
            u = 0
            for f, dd in dets.items():
                for b, c, kk in dd:
                    if kk == k and c >= t and not any(g["cls"] == k and iou(b, g["bbox"]) >= a.iou for g in by_file[f]):
                        u += 1
            unmatched[t] = u
        print(f"{OBJECT_CLASSES[k]:14s} {n_gold[k]:4d} " + " ".join(f"{h / n_gold[k]:8.2f}" for h in row)
              + f"   {unmatched[0.1]:5d}/{unmatched[0.3]}")
    print(f"{'ALL':14s} {sum(n_gold.values()):4d} " + " ".join(f"{h / n:8.2f}" for h, n in total.values()))
    # detections on classes with no verified object at all (pure FP candidates on these frames)
    extra = collections.Counter()
    for f, dd in dets.items():
        for b, c, kk in dd:
            if c >= 0.3 and kk not in n_gold:
                extra[OBJECT_CLASSES[kk]] += 1
    print("dets @0.3 for classes with no verified object on these frames:", dict(extra))


if __name__ == "__main__":
    main()
