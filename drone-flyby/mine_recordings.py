"""Mine object candidates from the recorded validation views.

The validation scene looks nothing like Helsinki (Copenhagen-style map, sparse,
small, dark-shaded objects), and no labels exist for it.  This runs several
detectors at a very low threshold with flip-TTA over every recorded frame,
merges their boxes, and dumps each candidate as a numbered crop on a contact
sheet plus cache/mined_candidates.json.  A human then keeps the true ones
(mine_recordings.py accept 3,7,12-15) -> cache/mined_objects.json, which
make_synth.py can paste as validation-domain cutouts.

    python mine_recordings.py scan --weights weights/best.pt weights/synth_mac_s.pt
    python mine_recordings.py accept 0,2,5-9 --classes 0:condor,2:hangar
"""
import argparse, glob, json, os, sys
import cv2, numpy as np
from ultralytics import YOLO
from dtos import OBJECT_CLASSES

CAND = "cache/mined_candidates.json"
OBJ = "cache/mined_objects.json"


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def scan(a):
    models = [YOLO(w) for w in a.weights]
    files = sorted(glob.glob(os.path.join(a.recordings, "*", "*.png")))
    cands = []
    for f in files:
        img = cv2.imread(f)
        boxes = []
        for mi, m in enumerate(models):
            r = m.predict(img, imgsz=960, conf=a.conf, augment=True, verbose=False)[0]
            for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                boxes.append([float(x1), float(y1), float(x2), float(y2), float(c), int(k), mi])
        boxes.sort(key=lambda b: -b[4])
        keep = []
        for b in boxes:                                   # class-agnostic merge across models
            if all(iou(b, k) < 0.5 for k in keep):
                keep.append(b)
        for b in keep[:a.per_frame]:
            cands.append({"file": f, "bbox": b[:4], "conf": b[4], "cls": b[5], "model": b[6]})
    json.dump(cands, open(CAND, "w"))
    # contact sheet: 128px crops with 30% context, index + class + conf
    tiles = []
    for i, c in enumerate(cands):
        img = cv2.imread(c["file"]); x1, y1, x2, y2 = c["bbox"]
        w, h = x2 - x1, y2 - y1; m = 0.3 * max(w, h) + 6
        crop = img[int(max(0, y1 - m)):int(min(img.shape[0], y2 + m)), int(max(0, x1 - m)):int(min(img.shape[1], x2 + m))]
        if crop.size == 0:
            crop = np.zeros((8, 8, 3), np.uint8)
        s = 128 / max(crop.shape[:2]); crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
        tile = np.zeros((150, 128, 3), np.uint8); tile[:crop.shape[0], :crop.shape[1]] = crop
        cv2.putText(tile, f"{i} {OBJECT_CLASSES[c['cls']][:6]} {c['conf']:.2f} L{os.path.basename(c['file']).split('_')[1][1]}", (2, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        tiles.append(tile)
    os.makedirs("debug", exist_ok=True)
    for s in range(0, len(tiles), 80):
        t = tiles[s:s + 80]
        while len(t) % 10: t.append(np.zeros((150, 128, 3), np.uint8))
        rows = [np.hstack(t[i:i + 10]) for i in range(0, len(t), 10)]
        cv2.imwrite(f"debug/mined_sheet{s // 80}.png", np.vstack(rows))
    print(f"{len(cands)} candidates from {len(files)} frames -> {CAND}, sheets debug/mined_sheet*.png")


def accept(a):
    cands = json.load(open(CAND))
    ids = set()
    for part in a.ids.split(","):
        if "-" in part:
            lo, hi = part.split("-"); ids.update(range(int(lo), int(hi) + 1))
        elif part:
            ids.add(int(part))
    relabel = {}
    for kv in (a.classes or "").split(","):
        if ":" in kv:
            k, v = kv.split(":"); relabel[int(k)] = v
    out = []
    for i in sorted(ids):
        c = cands[i]
        c = dict(c, cls=OBJECT_CLASSES.index(relabel[i]) if i in relabel else c["cls"], id=i)
        out.append(c)
    json.dump(out, open(OBJ, "w"), indent=1)
    print(f"kept {len(out)} objects -> {OBJ}; classes:", {OBJECT_CLASSES[c['cls']]: sum(1 for d in out if d['cls'] == c['cls']) for c in out})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("scan"); s.add_argument("--weights", nargs="+", default=["weights/best.pt"]); s.add_argument("--recordings", default="recordings")
    s.add_argument("--conf", type=float, default=0.05); s.add_argument("--per-frame", type=int, default=12)
    c = sub.add_parser("accept"); c.add_argument("ids"); c.add_argument("--classes", default="")
    a = ap.parse_args()
    scan(a) if a.cmd == "scan" else accept(a)
