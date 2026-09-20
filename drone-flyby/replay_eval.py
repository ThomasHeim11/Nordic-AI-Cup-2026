"""Replay a recorded validation sequence through the full pipeline and score the
reported full-frame boxes against the in-domain labels (data/real, view coords ->
4K via the recorded source region).  The camera path is the recorded one, so this
measures detector + tracker + report (persistence, class votes, thresholds), not the
camera policy.  Predictions are only judged inside the labelled view of each frame.

    DRONE_MISS_DECAY_L1=0.95 python replay_eval.py --recording <dir> --labels <data/real>
"""
import argparse, base64, collections, glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-12)


def average_precision(preds, n_gt, gt_by_frame, thr=0.5):
    """preds: [(conf, frame, box)] one class; gt_by_frame: {frame: [box]}.  VOC all-point AP."""
    if n_gt == 0:
        return None
    preds = sorted(preds, key=lambda p: -p[0])
    used = {f: [False] * len(b) for f, b in gt_by_frame.items()}
    tp = np.zeros(len(preds)); fp = np.zeros(len(preds))
    for i, (c, f, box) in enumerate(preds):
        best, bj = 0.0, -1
        for j, g in enumerate(gt_by_frame.get(f, [])):
            v = iou(box, g)
            if v > best:
                best, bj = v, j
        if best >= thr and not used[f][bj]:
            tp[i] = 1; used[f][bj] = True
        else:
            fp[i] = 1
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1e-12)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True)
    ap.add_argument("--labels", required=True, help="data/real (images/{train,val}, labels/{train,val})")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.environ.setdefault("DRONE_RECORD_DIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "replay_rec"))
    import example
    from dtos import DroneFlybyPredictRequestDto, OBJECT_CLASSES
    from utils import validate_response  # noqa: F401  (import check only)

    # labels: stem -> (split, [(cls, xn, yn, wn, hn)])
    labels = {}
    for split in ("train", "val"):
        for f in glob.glob(os.path.join(a.labels, "labels", split, "*.txt")):
            stem = os.path.basename(f)[:-4]
            key = "_".join(stem.split("_")[1:])          # 626281_0037_L1_2880_540 -> 0037_L1_2880_540
            rows = [tuple(float(v) for v in l.split()) for l in open(f) if l.strip()]
            labels[key] = (split, rows)

    files = sorted(glob.glob(os.path.join(a.recording, "*.json")))
    gt = collections.defaultdict(lambda: collections.defaultdict(list))   # cls -> frame -> [box4k]
    n_gt = collections.Counter(); preds = collections.defaultdict(list)   # cls -> [(conf, frame, box4k)]
    frames_scored = {"train": 0, "val": 0}; val_frames = set()
    for jf in files:
        d = json.load(open(jf))
        stem = os.path.basename(jf)[:-5]
        png = jf[:-5] + ".png"
        d["view"]["image"] = base64.b64encode(open(png, "rb").read()).decode()
        d.pop("response", None)
        req = DroneFlybyPredictRequestDto(**d)
        resp = example.predict(req)
        if stem not in labels:
            continue
        split, rows = labels[stem]
        frames_scored[split] += 1
        if split == "val":
            val_frames.add(d["frame"])
        rx1, ry1, rx2, ry2 = d["view"]["source_region_xyxy"]
        rw, rh = rx2 - rx1, ry2 - ry1
        fr = d["frame"]
        for k, xn, yn, wn, hn in rows:
            b = ((rx1 + (xn - wn / 2) * rw) / 3840, (ry1 + (yn - hn / 2) * rh) / 2160,
                 (rx1 + (xn + wn / 2) * rw) / 3840, (ry1 + (yn + hn / 2) * rh) / 2160)
            gt[OBJECT_CLASSES[int(k)]][fr].append(b); n_gt[OBJECT_CLASSES[int(k)]] += 1
        for ann in resp.annotations:
            x1, y1, x2, y2 = ann.bbox
            cx, cy = (x1 + x2) / 2 * 3840, (y1 + y2) / 2 * 2160
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:                   # only judge inside the labelled view
                preds[ann.object_id].append((ann.confidence, fr, (x1, y1, x2, y2)))

    def summarize(frame_filter):
        aps = {}
        for cls, n in n_gt.items():
            g = {f: b for f, b in gt[cls].items() if frame_filter(f)}
            ng = sum(len(b) for b in g.values())
            p = [x for x in preds[cls] if frame_filter(x[1])]
            apv = average_precision(p, ng, g)
            if apv is not None:
                aps[cls] = apv
        return aps
    all_aps = summarize(lambda f: True); val_aps = summarize(lambda f: f in val_frames)
    print(f"[{a.tag}] frames scored train {frames_scored['train']} val {frames_scored['val']} | GT {sum(n_gt.values())} | reported in-view {sum(len(v) for v in preds.values())}")
    print(f"[{a.tag}] in-view mAP@0.5 ALL {np.mean(list(all_aps.values())):.3f} ({len(all_aps)} classes) | VAL-split {np.mean(list(val_aps.values())) if val_aps else 0:.3f} ({len(val_aps)} classes)")
    print(f"[{a.tag}] per class: " + ", ".join(f"{c} {v:.2f}(n{n_gt[c]})" for c, v in sorted(all_aps.items(), key=lambda kv: -n_gt[kv[0]])))


if __name__ == "__main__":
    main()
