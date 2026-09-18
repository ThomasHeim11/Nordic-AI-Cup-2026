"""Run the detector over recorded validation views and summarise what it sees.

No labels exist for these frames, so this is a sanity check, not a score:
a healthy model reports a handful of confident objects per frame (the scene has
16 instances), not hundreds of one class.

    python check_recordings.py --weights weights/best.pt --recordings recordings
"""
import argparse, collections, glob, os
import cv2
from ultralytics import YOLO
from dtos import OBJECT_CLASSES
from utils import draw_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--recordings", default="recordings")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out", default="debug/check")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    model = YOLO(a.weights)
    files = sorted(glob.glob(os.path.join(a.recordings, "*", "*.png")))
    if a.limit:
        files = files[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    per_class = collections.Counter(); per_frame = []
    for i, f in enumerate(files):
        img = cv2.imread(f)
        r = model.predict(img, imgsz=960, conf=a.conf, verbose=False)[0]
        anns = []
        for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
            per_class[OBJECT_CLASSES[k]] += 1
            anns.append({"object_id": OBJECT_CLASSES[k], "bbox": [x1, y1, x2, y2], "confidence": float(c)})
        per_frame.append(len(anns))
        if i % 25 == 0:
            cv2.imwrite(os.path.join(a.out, os.path.basename(f)), draw_boxes(img, anns))
    print(f"frames {len(files)} | detections/frame mean {sum(per_frame)/max(1,len(per_frame)):.1f} max {max(per_frame) if per_frame else 0}")
    print("per class:", per_class.most_common())
    print("overlays written to", a.out)


if __name__ == "__main__":
    main()
