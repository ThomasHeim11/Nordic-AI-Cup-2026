"""Copy-paste synthetic dataset for the drone detector.

The 16 object classes are 3D models composited onto satellite imagery, so an
object patch cut from one scene is a valid object in any other.  We cut every
annotated instance out of the Helsinki 4K frames and paste it, at the correct
pixel scale for the target zoom level, onto backgrounds from (a) the recorded
validation-scene views and (b) random Helsinki crops.  Empty background crops
are kept as negatives so the detector learns that bushes and cars are not
targets.

    python make_synth.py --out data/synth --n 3000 --recordings recordings
    python make_synth.py --out data/synth2 --n 4000 --recordings recordings_site --mined cache/mined_objects.json

--mined adds human-verified cutouts from the recorded validation views
(mine_recordings.py): objects in the *validation* rendering, which differs
from Helsinki (darker shading, dry grass, Copenhagen-style terrain).  Any
recorded frame that contains a verified object is also labelled with that
object's real box whenever it is used as a background, so the model is never
taught that those objects are background.
"""
import argparse, glob, json, os, random
from pathlib import Path

import cv2
import numpy as np

from dtos import OBJECT_CLASSES, SOURCE_REGION_SIZES, TRANSMITTED_VIEW_SIZE
from utils import frame_numbers, load_annotations, load_frame

CLASS_INDEX = {n: i for i, n in enumerate(OBJECT_CLASSES)}
VW, VH = TRANSMITTED_VIEW_SIZE


def extract_patches(margin=3):
    """Every fully-visible annotated instance as a 4K BGR patch."""
    patches = []
    for f in frame_numbers():
        img = load_frame(f)
        for a in load_annotations(f):
            x1, y1, x2, y2 = a["bbox"]
            if x1 <= 2 or y1 <= 2 or x2 >= 3838 or y2 >= 2158:
                continue
            X1, Y1, X2, Y2 = max(0, x1 - margin), max(0, y1 - margin), min(3840, x2 + margin), min(2160, y2 + margin)
            patches.append((a["object_id"], img[Y1:Y2, X1:X2].copy(), (x1 - X1, y1 - Y1, x2 - X1, y2 - Y1)))
    return patches


def feather_mask(h, w, r):
    m = np.ones((h, w), np.float32)
    r = max(1, min(r, h // 3, w // 3))
    ramp = np.linspace(0, 1, r + 1)[1:]
    m[:r, :] *= ramp[:, None]; m[-r:, :] *= ramp[::-1][:, None]
    m[:, :r] *= ramp[None, :]; m[:, -r:] *= ramp[::-1][None, :]
    return cv2.GaussianBlur(m, (0, 0), 0.6)


def orient(patch, box, k, flip):
    x1, y1, x2, y2 = box
    h, w = patch.shape[:2]
    if flip:
        patch = patch[:, ::-1]; x1, x2 = w - x2, w - x1
    for _ in range(k):                      # rotate 90 deg CCW
        patch = np.ascontiguousarray(np.rot90(patch))
        h, w = patch.shape[:2]
        x1, y1, x2, y2 = y1, w - x2, y2, w - x1
    return patch, (x1, y1, x2, y2)


def paste(bg, patch, box, scale, rng):
    """Paste patch (scaled) at a random spot; return the pasted object's view box or None."""
    ph, pw = patch.shape[:2]
    nh, nw = max(2, int(round(ph * scale))), max(2, int(round(pw * scale)))
    p = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    # photometric jitter
    p = np.clip(p.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-15, 15), 0, 255).astype(np.uint8)
    if nh >= VH - 2 or nw >= VW - 2:
        return None
    x0, y0 = rng.randint(0, VW - nw - 1), rng.randint(0, VH - nh - 1)
    m = feather_mask(nh, nw, max(2, int(0.22 * min(nh, nw))))[..., None]
    roi = bg[y0:y0 + nh, x0:x0 + nw].astype(np.float32)
    bg[y0:y0 + nh, x0:x0 + nw] = (roi * (1 - m) + p.astype(np.float32) * m).astype(np.uint8)
    bx1, by1, bx2, by2 = box
    return (x0 + bx1 * scale, y0 + by1 * scale, x0 + bx2 * scale, y0 + by2 * scale)


LEVEL_SCALE = {0: 0.25, 1: 0.5, 2: 1.0}      # view pixels per 4K source pixel


def load_mined(path, margin=2):
    """Verified validation cutouts -> {class: [(patch, box, source_scale)]} and {frame: [(cls, box)]}."""
    by_class, by_frame = {}, {}
    for o in json.load(open(path)):
        img = cv2.imread(o["file"])
        x1, y1, x2, y2 = [int(round(v)) for v in o["bbox"]]
        X1, Y1 = max(0, x1 - margin), max(0, y1 - margin)
        X2, Y2 = min(img.shape[1], x2 + margin), min(img.shape[0], y2 + margin)
        lvl = int(os.path.basename(o["file"]).split("_L")[1][0])
        name = OBJECT_CLASSES[o["cls"]]
        by_class.setdefault(name, []).append((img[Y1:Y2, X1:X2].copy(), (x1 - X1, y1 - Y1, x2 - X1, y2 - Y1), LEVEL_SCALE[lvl]))
        by_frame.setdefault(os.path.abspath(o["file"]), []).append((o["cls"], (x1, y1, x2, y2)))
    return by_class, by_frame


def helsinki_background(level, rng, frame_cache):
    f = rng.choice(frame_numbers())
    img = frame_cache.setdefault(f, load_frame(f))
    W, H = SOURCE_REGION_SIZES[level]
    x = rng.randint(0, 3840 - W); y = rng.randint(0, 2160 - H)
    crop = img[y:y + H, x:x + W]
    return cv2.resize(crop, (VW, VH), interpolation=cv2.INTER_AREA) if level > 0 else cv2.resize(crop, (VW, VH), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--recordings", default="recordings")
    ap.add_argument("--neg-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jpeg", type=int, default=0, help="if >0, write JPEG at this quality instead of PNG")
    ap.add_argument("--mined", default="", help="verified validation cutouts from mine_recordings.py")
    ap.add_argument("--mined-frac", type=float, default=0.5, help="share of pasted objects drawn from the mined pool when the class has one")
    ap.add_argument("--rec-frac", type=float, default=0.7, help="share of backgrounds taken from recorded views")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    patches = extract_patches()
    by_class = {}
    for name, p, b in patches:
        by_class.setdefault(name, []).append((p, b))
    print("patches per class:", {k: len(v) for k, v in by_class.items()})
    mined, mined_frames = load_mined(a.mined) if a.mined else ({}, {})
    if mined:
        print("mined cutouts per class:", {k: len(v) for k, v in mined.items()}, "| labelled frames:", len(mined_frames))

    rec = []
    for png in glob.glob(os.path.join(a.recordings, "*", "*.png")):
        stem = os.path.basename(png)[:-4]
        try:
            lvl = int(stem.split("_L")[1][0])
        except Exception:
            continue
        rec.append((lvl, png))
    print("recorded backgrounds:", len(rec))
    frame_cache = {}
    ext = "jpg" if a.jpeg else "png"
    counts = {"train": 0, "val": 0}
    for i in range(a.n):
        split = "val" if rng.random() < a.val_frac else "train"
        # backgrounds: 70% recorded validation terrain, 30% Helsinki
        real_labels = []
        if rec and rng.random() < a.rec_frac:
            lvl, png = rng.choice(rec)
            bg = cv2.imread(png)
            real_labels = mined_frames.get(os.path.abspath(png), [])
        else:
            lvl = rng.choice([0, 1, 1, 2, 2])
            bg = helsinki_background(lvl, rng, frame_cache)
        bg = np.ascontiguousarray(bg)
        # global photometric augmentation of the whole image
        if rng.random() < 0.5:
            hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] *= rng.uniform(0.7, 1.3); hsv[..., 2] *= rng.uniform(0.75, 1.25)
            bg = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        labels = [(c, (x1 + x2) / 2 / VW, (y1 + y2) / 2 / VH, (x2 - x1) / VW, (y2 - y1) / VH) for c, (x1, y1, x2, y2) in real_labels]
        if rng.random() >= a.neg_frac:
            scale = {0: 0.25, 1: 0.5, 2: 1.0}[lvl]
            k_obj = rng.randint(1, 6)
            for _ in range(k_obj):
                name = rng.choice(sorted(set(by_class) | set(mined)))
                src_scale = 1.0
                if name in mined and (name not in by_class or rng.random() < a.mined_frac):
                    p, b, src_scale = rng.choice(mined[name])
                else:
                    p, b = rng.choice(by_class[name])
                p, b = orient(p, b, rng.randint(0, 3), rng.random() < 0.5)
                s = scale * rng.uniform(0.75, 1.3) / src_scale
                if s > 2.2:                      # do not blow a small L1 cutout up to mush
                    continue
                box = paste(bg, p, b, s, rng)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue
                labels.append((CLASS_INDEX[name], (x1 + x2) / 2 / VW, (y1 + y2) / 2 / VH, (x2 - x1) / VW, (y2 - y1) / VH))
        name = f"s{i:05d}_L{lvl}"
        if a.jpeg:
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, a.jpeg])
        else:
            cv2.imwrite(str(out / "images" / split / f"{name}.png"), bg, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        with open(out / "labels" / split / f"{name}.txt", "w") as f:
            for c, x, y, w, h in labels:
                f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        counts[split] += 1
        if i % 500 == 0:
            print("generated", i, flush=True)
    with open(out / "data.yaml", "w") as f:
        f.write(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n")
        f.write("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(OBJECT_CLASSES)))
    print("done:", counts)


if __name__ == "__main__":
    main()
