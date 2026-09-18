"""Replay a scene through example.predict() in-process and score it.

Same camera rules, rendering and scorer as local_evaluator.py, minus HTTP.
Detections are cached per (frame, level, cx, cy) in cache/dets_<scene>.json so
a tracker / camera-policy sweep does not re-run the detector.

    python eval_inprocess.py                       # score once
    python eval_inprocess.py --set CONF_THRESHOLD=0.2 --set REPORT_MIN_CONF=0.1
    python eval_inprocess.py --no-cache            # force the detector to run
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from utils import frame_numbers, global_bbox_to_source, load_frame
import local_evaluator as le


def run(scene="helsinki", overrides=None, use_cache=True, verbose=False):
    import example
    for k, v in (overrides or {}).items():
        setattr(example, k, v)
    example._SEQ.clear()

    cache_path = os.path.join(HERE, "cache", f"dets_{scene}.json")
    cache = {}
    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    real_detector = example.DETECTOR

    class CachedDetector:
        def __init__(self):
            self.key = None
            self.dirty = False

        def __call__(self, image):
            if self.key in cache:
                return [tuple(d) for d in cache[self.key]]
            dets = real_detector(image)
            cache[self.key] = [list(d) for d in dets]
            self.dirty = True
            return dets

    cd = CachedDetector()
    example.DETECTOR = cd

    frames = frame_numbers(scene)
    camera = le.Camera()
    feedback = None
    predictions = {}
    views = {}
    stats = le.Statistics(frames_total=len(frames))
    t_pred = []
    for frame_index, frame in enumerate(frames):
        image = load_frame(frame, scene)
        payload = le.build_request(frame, frame_index, camera, le.render_view(image, camera), feedback)
        cd.key = f"{frame}:{camera.resolution_level}:{camera.center_x}:{camera.center_y}"
        views[frame] = (camera.resolution_level, camera.source_region)
        req = DroneFlybyPredictRequestDto.model_validate(payload)
        t0 = time.perf_counter()
        resp = example.predict(req)
        t_pred.append(time.perf_counter() - t0)
        body = json.loads(resp.model_dump_json())
        try:
            response = DroneFlybyPredictResponseDto.model_validate(body)
            assert response.request_id == payload["request_id"] and response.frame == frame
        except Exception as exc:
            stats.invalid_responses += 1
            print(f"frame {frame}: invalid response: {exc}")
            continue
        stats.responses_accepted += 1
        predictions[frame] = [
            {"object_id": a.object_id, "bbox": global_bbox_to_source(a.bbox, 3840, 2160),
             "confidence": float(a.confidence)} for a in response.annotations]
        if verbose:
            print(f"frame {frame:3d} L{camera.resolution_level} ({camera.center_x},{camera.center_y}) -> {len(response.annotations)} dets")
        if response.requested_view is not None:
            rv = response.requested_view
            try:
                camera.apply(rv.resolution_level, rv.center_x, rv.center_y)
                stats.commands_applied += 1
                feedback = None
            except le.CameraRejection as exc:
                stats.invalid_commands += 1
                feedback = {"frame": frame, "requested_view": rv.model_dump(), "reason": str(exc)}
                print(f"frame {frame}: camera refused: {exc}")

    if cd.dirty:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f)
    example.DETECTOR = real_detector
    mAP, per_class = le.score(scene, predictions)
    if verbose or os.environ.get("COVERAGE"):
        coverage_report(scene, predictions, views)
    return mAP, per_class, stats, t_pred


def coverage_report(scene, predictions, views):
    """Per class: frames present / covered by a zoomed view / matched at IoU>=0.5."""
    from utils import load_annotations
    from example import iou_xyxy
    rows = {}
    for frame, (level, region) in views.items():
        for gt in load_annotations(frame, scene):
            r = rows.setdefault(gt["object_id"], {"present": 0, "zoomed": 0, "hit": 0, "miss_frames": []})
            r["present"] += 1
            b = gt["bbox"]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if level > 0 and region[0] <= cx <= region[2] and region[1] <= cy <= region[3]:
                r["zoomed"] += 1
            best = max((iou_xyxy(b, p["bbox"]) for p in predictions.get(frame, []) if p["object_id"] == gt["object_id"]), default=0.0)
            if best >= 0.5:
                r["hit"] += 1
            else:
                r["miss_frames"].append(frame)
    print("coverage  class            present zoomed hit  missed frames")
    for k, r in sorted(rows.items()):
        print(f"          {k:16s} {r['present']:5d} {r['zoomed']:6d} {r['hit']:4d}  {r['miss_frames']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="helsinki")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=PYEXPR")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    overrides = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        overrides[k] = eval(v)
    mAP, per_class, stats, t_pred = run(a.scene, overrides, not a.no_cache, a.verbose)
    for k, v in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16s} {v:.3f}")
    print(f"invalid {stats.invalid_responses} | moves applied {stats.commands_applied} refused {stats.invalid_commands} "
          f"| predict ms mean {1000*sum(t_pred)/len(t_pred):.0f} max {1000*max(t_pred):.0f}")
    print(f"mAP@0.50 = {mAP:.4f}   overrides={overrides}")


if __name__ == "__main__":
    main()
