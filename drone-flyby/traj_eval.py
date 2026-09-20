"""Out-of-view persistence score on a recorded sequence.

Same physical object labelled in several recorded frames -> its 4K positions over
time.  Between two sightings the object is (linearly) interpolated, giving
pseudo ground truth for frames where it was OUTSIDE our view.  We then check
whether the reported (remembered) boxes hit those positions at IoU 0.5.
Identity: same class, next sighting within a plausible motion window.

    DRONE_MOTION_ONLINE_MIN=8 python traj_eval.py --recording <dir> --labels <data/real>
"""
import argparse, base64, collections, glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from replay_eval import iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--tag", default=""); ap.add_argument("--max-gap", type=int, default=40)
    a = ap.parse_args()
    os.environ.setdefault("DRONE_RECORD_DIR", "/tmp/replay_rec")
    import example
    from dtos import DroneFlybyPredictRequestDto, OBJECT_CLASSES
    labels = {}
    for split in ("train", "val"):
        for f in glob.glob(os.path.join(a.labels, "labels", split, "*.txt")):
            key = "_".join(os.path.basename(f)[:-4].split("_")[1:])
            labels[key] = [tuple(float(v) for v in l.split()) for l in open(f) if l.strip()]
    files = sorted(glob.glob(os.path.join(a.recording, "*.json")))
    sightings = collections.defaultdict(list)     # cls -> [(frame, box4k)]
    reports = {}                                   # frame -> [(cls, box4k, conf)]
    regions = {}
    for jf in files:
        d = json.load(open(jf)); stem = os.path.basename(jf)[:-5]
        d["view"]["image"] = base64.b64encode(open(jf[:-5] + ".png", "rb").read()).decode(); d.pop("response", None)
        resp = example.predict(DroneFlybyPredictRequestDto(**d))
        fr = d["frame"]; rx1, ry1, rx2, ry2 = d["view"]["source_region_xyxy"]; regions[fr] = (rx1, ry1, rx2, ry2)
        reports[fr] = [(an.object_id, an.bbox, an.confidence) for an in resp.annotations]
        for k, xn, yn, wn, hn in labels.get(stem, []):
            rw, rh = rx2 - rx1, ry2 - ry1
            sightings[OBJECT_CLASSES[int(k)]].append((fr, ((rx1 + (xn - wn / 2) * rw) / 3840, (ry1 + (yn - hn / 2) * rh) / 2160,
                                                          (rx1 + (xn + wn / 2) * rw) / 3840, (ry1 + (yn + hn / 2) * rh) / 2160)))
    # link sightings into trajectories: greedy by class, nearest predicted position (dy ~ +70 px/frame)
    tot = collections.Counter(); hit = collections.Counter(); n_traj = 0
    for cls, sl in sightings.items():
        sl.sort()
        used = [False] * len(sl)
        for i in range(len(sl)):
            if used[i]:
                continue
            traj = [sl[i]]; used[i] = True
            while True:
                f0, b0 = traj[-1]; cx0, cy0 = (b0[0] + b0[2]) / 2 * 3840, (b0[1] + b0[3]) / 2 * 2160
                best, bj = None, -1
                for j in range(len(sl)):
                    if used[j] or sl[j][0] <= f0 or sl[j][0] - f0 > a.max_gap:
                        continue
                    f1, b1 = sl[j]; cx1, cy1 = (b1[0] + b1[2]) / 2 * 3840, (b1[1] + b1[3]) / 2 * 2160
                    k = f1 - f0; pred = (cx0, cy0 + 70 * k)          # rough motion prior for identity only
                    dist = np.hypot(cx1 - pred[0], cy1 - pred[1])
                    if dist < 60 + 12 * k and (best is None or dist < best):
                        best, bj = dist, j
                if bj < 0:
                    break
                used[bj] = True; traj.append(sl[bj])
            if len(traj) < 2:
                continue
            n_traj += 1
            for (f0, b0), (f1, b1) in zip(traj, traj[1:]):
                for f in range(f0 + 1, f1):
                    if f not in reports:
                        continue
                    t = (f - f0) / (f1 - f0)
                    g = tuple(b0[i] + t * (b1[i] - b0[i]) for i in range(4))
                    rx1, ry1, rx2, ry2 = regions[f]; gcx, gcy = (g[0] + g[2]) / 2 * 3840, (g[1] + g[3]) / 2 * 2160
                    inview = rx1 <= gcx <= rx2 and ry1 <= gcy <= ry2
                    key = (cls, "inview" if inview else "outview")
                    tot[key] += 1
                    if any(c == cls and iou(b, g) >= 0.5 for c, b, _ in reports[f]):
                        hit[key] += 1
    def rate(where):
        t = sum(v for k, v in tot.items() if k[1] == where); h = sum(v for k, v in hit.items() if k[1] == where)
        return h / t if t else float("nan"), t
    ro, no = rate("outview"); ri, ni = rate("inview")
    print(f"[{a.tag}] trajectories {n_traj} | interpolated GT out-of-view {no}: recall@0.5 {ro:.3f} | in-view {ni}: recall {ri:.3f}")
    per = collections.defaultdict(lambda: [0, 0])
    for (c, w), v in tot.items():
        if w == "outview": per[c][1] += v; per[c][0] += hit[(c, w)]
    print(f"[{a.tag}] out-of-view per class: " + ", ".join(f"{c} {h}/{t}" for c, (h, t) in sorted(per.items(), key=lambda kv: -kv[1][1])))


if __name__ == "__main__":
    main()
