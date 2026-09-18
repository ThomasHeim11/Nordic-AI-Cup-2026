"""Drone flyby: YOLO detector + dead-reckoning tracker + camera sweep.

The three parts and why they exist:

* **Detector** -- a YOLO model fine-tuned on multi-scale crops of the supplied
  frames, run on the 960x540 view.  Boxes are lifted into source pixels through
  ``source_region_xyxy``.
* **Tracker** -- the drone flies a straight line at constant speed with a
  forward-pitched camera, so every object scrolls down the frame with
  ``dy = 51.85 + 0.01333 * y`` px/frame (fitted on the Helsinki annotations,
  residual 0.8 px).  A detection made once at Level 2 can therefore be
  reported, at IoU > 0.5, for the rest of that object's life in the frame --
  which is what makes "answer for the whole frame from one crop" possible.
* **Camera policy** -- objects enter at the top edge, so after an initial coarse
  scan of the whole frame the camera sweeps the top band at Level 2, where even
  the 20-px classes are native-resolution.  Every command is derived from the
  request's ``camera_constraints`` so it is never rejected.
"""

import glob
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from dtos import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAXIMUM_CENTER_DELTA_PIXELS,
    OBJECT_CLASSES,
    SOURCE_REGION_SIZES,
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)
from utils import (center_bounds_for_level, clip_bbox_to_frame, decode_view,
                   describe_camera_rejection, source_bbox_to_global)

ALLOWED_LEVELS_FROM = {0: (0, 1), 1: (0, 1, 2), 2: (1, 2)}

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Motion model: the per-frame ground-plane homography, fitted on the Helsinki
# annotation tracks (residual 0.3 px in x, 0.7 px in y).  The camera is pitched
# forward, so objects flow radially: x drifts outwards by ~0.7% of its distance
# from the vanishing point per frame, y scrolls by 52 + 1.2% * y.
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
_H_DEFAULT = np.array([[1.006756, -0.001473, -12.614238],
                       [0.000337, 1.011899, 51.706644],
                       [0.0, -0.000001, 1.0]], dtype=np.float64)
try:
    MOTION_H = np.load(os.path.join(HERE, "cache", "homography.npy"))
except Exception:
    MOTION_H = _H_DEFAULT


def warp_point(x: float, y: float, H=None) -> Tuple[float, float]:
    H = MOTION_H if H is None else H
    v = H @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])

# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
WEIGHT_CANDIDATES = [
    os.path.join(HERE, "weights", "best.pt"),
    os.path.join(HERE, "runs", "detect", "runs", "drone_s", "weights", "best.pt"),
    os.path.join(HERE, "..", "runs", "detect", "runs", "drone_s", "weights", "best.pt"),
    os.path.join(HERE, "runs", "detect", "runs", "drone_s", "weights", "last.pt"),
    os.path.join(HERE, "..", "runs", "detect", "runs", "drone_s", "weights", "last.pt"),
]
CONF_THRESHOLD = float(os.environ.get("DRONE_CONF", "0.12"))
NMS_IOU = 0.5
IMGSZ = 960


class Detector:
    def __init__(self):
        self.model = None
        self.device = "cpu"
        path = next((p for p in WEIGHT_CANDIDATES if os.path.exists(p)), None)
        if path is None:
            logger.error("no YOLO weights found; detector disabled (candidates: %s)", WEIGHT_CANDIDATES)
            return
        try:
            import torch
            from ultralytics import YOLO
            self.device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu")
            self.model = YOLO(path)
            # Warm up: the first inference is by far the slowest.
            dummy = np.zeros((540, 960, 3), dtype=np.uint8)
            t0 = time.perf_counter()
            for _ in range(2):
                self.model.predict(dummy, imgsz=IMGSZ, device=self.device, verbose=False)
            logger.info("loaded %s on %s (warm-up %.0f ms)", path, self.device,
                        1000 * (time.perf_counter() - t0))
        except Exception:
            logger.exception("failed to load YOLO weights from %s", path)
            self.model = None

    def __call__(self, image: np.ndarray) -> List[Tuple[int, float, float, float, float, float]]:
        """Return (class_index, conf, x1, y1, x2, y2) in view pixels."""
        if self.model is None:
            return []
        out = self._run(image)
        if TTA_FLIP:
            # Top-down imagery: a mirrored view is an equally valid view.  Run
            # the flipped image too and merge (ultralytics' augment=True does
            # the same idea; done explicitly here so the merge is ours).
            w = image.shape[1]
            flipped = self._run(np.ascontiguousarray(image[:, ::-1]))
            out = _merge_tta(out + [(k, c, w - x2, y1, w - x1, y2) for k, c, x1, y1, x2, y2 in flipped])
        return out

    def _run(self, image: np.ndarray):
        results = self.model.predict(image, imgsz=IMGSZ, conf=CONF_THRESHOLD, iou=NMS_IOU,
                                     device=self.device, verbose=False, max_det=200)
        out = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                out.append((int(k), float(c), float(x1), float(y1), float(x2), float(y2)))
        return out


TTA_FLIP = os.environ.get("DRONE_TTA", "0") == "1"


def _merge_tta(dets, iou_thr: float = 0.55):
    """Greedy merge of detections from two passes: overlapping boxes of the
    same class become one box with the max confidence and averaged corners."""
    dets = sorted(dets, key=lambda d: d[1], reverse=True)
    merged = []
    for d in dets:
        for i, m in enumerate(merged):
            if m[0] == d[0] and iou_xyxy(m[2:], d[2:]) >= iou_thr:
                merged[i] = (m[0], max(m[1], d[1]), (m[2] + d[2]) / 2, (m[3] + d[3]) / 2, (m[4] + d[4]) / 2, (m[5] + d[5]) / 2)
                break
        else:
            merged.append(d)
    return merged


DETECTOR = Detector()

# --------------------------------------------------------------------------- #
# Tracker
# --------------------------------------------------------------------------- #

def advance(cx, cy, w, h, steps: int, H=None):
    """Dead-reckon a box through ``steps`` frames by warping its corners."""
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    for _ in range(steps):
        x1, y1 = warp_point(x1, y1, H)
        x2, y2 = warp_point(x2, y2, H)
    return (x1 + x2) / 2, (y1 + y2) / 2, max(1.0, x2 - x1), max(1.0, y2 - y1)


class MotionEstimator:
    """Online estimate of the per-frame ground motion, in source pixels.

    The fitted Helsinki matrix is only a prior: the evaluation drone may fly
    another heading.  Consecutive views at the same zoom level are registered
    with ORB features (mapped through their source regions, so the camera's
    own moves cancel out) and a RANSAC affine; plausible estimates are blended
    into the running matrix.  Thousands of inliers per pair on the recorded
    validation scene, so this converges within a few frames.
    """

    # Switch from the prior to the online estimate only when they disagree by
    # more than this at the frame centre (px/frame) or in scale; on a scene the
    # prior was fitted for, the prior is exact and the estimate is only noise.
    DISAGREE_PX = 10.0
    DISAGREE_SCALE = 0.006

    def __init__(self, prior):
        self.prior = np.array(prior, dtype=np.float64).copy()
        self.online = self.prior.copy()
        self.H = self.prior.copy()
        self.using_online = False
        self.last = None                      # (frame, level, region, gray)
        self.orb = cv2.ORB_create(1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.accepted = 0

    def _choose(self) -> None:
        c = np.array([1920.0, 1080.0, 1.0])
        dp, do = self.prior @ c, self.online @ c
        shift = float(np.hypot(dp[0] - do[0], dp[1] - do[1]))
        scale = float(max(abs(self.prior[0, 0] - self.online[0, 0]), abs(self.prior[1, 1] - self.online[1, 1])))
        want_online = self.accepted >= 2 and (shift > self.DISAGREE_PX or scale > self.DISAGREE_SCALE)
        if want_online != self.using_online:
            logger.info("motion model: %s (disagreement %.1f px, scale %.4f)",
                        "ONLINE estimate" if want_online else "fitted prior", shift, scale)
        self.using_online = want_online
        self.H = self.online if want_online else self.prior

    @staticmethod
    def _to_src(pts, region):
        x1, y1, x2, y2 = region
        sx, sy = (x2 - x1) / 960.0, (y2 - y1) / 540.0
        return np.column_stack([x1 + pts[:, 0] * sx, y1 + pts[:, 1] * sy]).astype(np.float32)

    def update(self, frame: int, level: int, region, image_bgr) -> None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        try:
            if self.last is not None:
                f0, l0, r0, g0 = self.last
                k = frame - f0
                if 1 <= k <= 3 and l0 == level:
                    self._register(g0, r0, gray, region, k)
        except Exception:
            logger.exception("motion estimation failed")
        self.last = (frame, level, region, gray)

    def _register(self, g0, r0, g1, r1, k: int) -> None:
        k0, d0 = self.orb.detectAndCompute(g0, None)
        k1, d1 = self.orb.detectAndCompute(g1, None)
        if d0 is None or d1 is None or len(k0) < 50 or len(k1) < 50:
            return
        m = self.bf.match(d0, d1)
        if len(m) < 40:
            return
        P0 = self._to_src(np.float32([k0[x.queryIdx].pt for x in m]), r0)
        P1 = self._to_src(np.float32([k1[x.trainIdx].pt for x in m]), r1)
        A, inl = cv2.estimateAffine2D(P0, P1, ransacReprojThreshold=6.0, method=cv2.RANSAC)
        if A is None or inl is None or int(inl.sum()) < 60:
            return
        # k-frame motion -> per-frame (first-order: the motion is near identity)
        M = np.eye(2) + (A[:, :2] - np.eye(2)) / k
        t = A[:, 2] / k
        shift = float(np.hypot(*(M @ np.array([1920.0, 1080.0]) + t - np.array([1920.0, 1080.0]))))
        if not (10.0 <= shift <= 250.0) or not (0.96 <= M[0, 0] <= 1.06 and 0.96 <= M[1, 1] <= 1.06) \
                or abs(M[0, 1]) > 0.05 or abs(M[1, 0]) > 0.05:
            return
        A1 = np.eye(3)
        A1[:2, :2] = M
        A1[:2, 2] = t
        alpha = 0.5 if self.accepted < 5 else 0.2        # converge fast, then smooth
        self.online = (1 - alpha) * self.online + alpha * A1
        self.accepted += 1
        self._choose()


def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Track:
    cx: float
    cy: float
    w: float
    h: float
    frame: int                      # frame the state refers to
    conf: float                     # best evidence so far, decayed on misses
    votes: Dict[int, float] = field(default_factory=dict)
    hits: int = 1
    misses_in_view: int = 0
    best_level: int = 0             # highest zoom this track has been seen at

    @property
    def cls(self) -> int:
        return max(self.votes.items(), key=lambda kv: kv[1])[0]

    def box(self):
        return (self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2)

    def predict_to(self, frame: int, H=None):
        steps = frame - self.frame
        if steps > 0:
            self.cx, self.cy, self.w, self.h = advance(self.cx, self.cy, self.w, self.h, steps, H)
            self.frame = frame


LEVEL_WEIGHT = {0: 1.0, 1: 2.0, 2: 3.0}
LEVEL_ALPHA = {0: 0.35, 1: 0.6, 2: 0.85}       # how much a detection moves the state
MATCH_IOU = 0.25
MATCH_DIST_FRAC = 0.9                         # or centre within this * max(w,h)
MISS_DECAY = {0: 0.97, 1: 0.85, 2: 0.6}       # conf decay when in view but not detected
OFFVIEW_DECAY = 0.995
REPORT_MIN_CONF = 0.08
DUPLICATE_IOU = 0.55


MAX_REPORT = 80
REPORT_NMS_IOU = 0.6


class Tracker:
    def __init__(self):
        self.tracks: List[Track] = []
        self.motion = MotionEstimator(MOTION_H)

    def step(self, frame: int, level: int, region, detections, image=None):
        """detections: list of (cls, conf, x1, y1, x2, y2) in SOURCE pixels."""
        if image is not None:
            self.motion.update(frame, level, region, image)
        H = self.motion.H
        for t in self.tracks:
            t.predict_to(frame, H)
        # Drop what has left the frame.
        self.tracks = [t for t in self.tracks if t.cy - t.h / 2 < IMAGE_HEIGHT + 20 and t.conf > 0.02]

        rx1, ry1, rx2, ry2 = region
        used = set()
        pairs = []
        for ti, t in enumerate(self.tracks):
            tb = t.box()
            for di, d in enumerate(detections):
                db = d[2:6]
                ov = iou_xyxy(tb, db)
                dcx, dcy = (db[0] + db[2]) / 2, (db[1] + db[3]) / 2
                dist = math.hypot(dcx - t.cx, dcy - t.cy)
                if ov >= MATCH_IOU or dist < MATCH_DIST_FRAC * max(t.w, t.h, 20):
                    pairs.append((ov + 1e-3 * d[1], ti, di))
        pairs.sort(reverse=True)
        matched_t = set()
        for _, ti, di in pairs:
            if ti in matched_t or di in used:
                continue
            matched_t.add(ti)
            used.add(di)
            t = self.tracks[ti]
            k, c, x1, y1, x2, y2 = detections[di]
            a = LEVEL_ALPHA[level]
            # A precise zoomed-in box should not be dragged around by a coarse one.
            if level < t.best_level:
                a *= 0.4
            t.cx = (1 - a) * t.cx + a * (x1 + x2) / 2
            t.cy = (1 - a) * t.cy + a * (y1 + y2) / 2
            t.w = (1 - a) * t.w + a * (x2 - x1)
            t.h = (1 - a) * t.h + a * (y2 - y1)
            t.votes[k] = t.votes.get(k, 0.0) + c * LEVEL_WEIGHT[level]
            t.conf = max(t.conf * 0.9 + c * LEVEL_WEIGHT[level] / 3 * 0.1, c * (0.7 + 0.1 * level))
            t.conf = min(1.0, t.conf)
            t.hits += 1
            t.misses_in_view = 0
            t.best_level = max(t.best_level, level)

        for ti, t in enumerate(self.tracks):
            if ti in matched_t:
                continue
            inside = rx1 <= t.cx <= rx2 and ry1 <= t.cy <= ry2
            if inside:
                # Was in view and not detected: weak evidence at L0 (tiny
                # objects vanish there), strong at L2.
                t.conf *= MISS_DECAY[level]
                t.misses_in_view += 1
            else:
                t.conf *= OFFVIEW_DECAY

        for di, d in enumerate(detections):
            if di in used:
                continue
            k, c, x1, y1, x2, y2 = d
            self.tracks.append(Track(
                cx=(x1 + x2) / 2, cy=(y1 + y2) / 2, w=x2 - x1, h=y2 - y1, frame=frame,
                conf=c * (0.7 + 0.1 * level), votes={k: c * LEVEL_WEIGHT[level]}, best_level=level))

        self._merge_duplicates()

    def _merge_duplicates(self):
        self.tracks.sort(key=lambda t: t.conf, reverse=True)
        kept: List[Track] = []
        for t in self.tracks:
            dup = None
            for k in kept:
                if iou_xyxy(t.box(), k.box()) > DUPLICATE_IOU:
                    dup = k
                    break
            if dup is None:
                kept.append(t)
            else:
                for c, v in t.votes.items():
                    dup.votes[c] = dup.votes.get(c, 0.0) + v
                dup.hits += t.hits
        self.tracks = kept

    def report(self) -> List[Tuple[str, Tuple[float, float, float, float], float]]:
        out = []
        for t in self.tracks:
            conf = t.conf
            if t.hits == 1:
                conf *= 0.75
            if t.misses_in_view >= 3 and t.hits <= 2:
                conf *= 0.5
            if conf < REPORT_MIN_CONF:
                continue
            out.append((OBJECT_CLASSES[t.cls], t.box(), conf))
        # The scorer applies no NMS: two boxes on one object are one hit and
        # one false positive.  Class-agnostic suppression, best first, then cap.
        out.sort(key=lambda r: r[2], reverse=True)
        kept: List[Tuple[str, Tuple[float, float, float, float], float]] = []
        for r in out:
            if all(iou_xyxy(r[1], k[1]) < REPORT_NMS_IOU for k in kept):
                kept.append(r)
            if len(kept) >= MAX_REPORT:
                break
        return kept


# --------------------------------------------------------------------------- #
# Camera policy
# --------------------------------------------------------------------------- #

@dataclass
class CameraPlan:
    phase: str = "init"            # init -> tiles -> sweep
    tile_index: int = 0
    sweep_dir: int = 1
    last_frame: int = -1
    last_dive_frame: int = -100
    dive_target: Optional[int] = None   # id(track) being dived on
    # True camera state.  The request's view can be STALE: frames are emitted
    # on a clock and a frame rendered before our previous command was applied
    # still carries the old view.  The evaluator applies our command when the
    # response arrives, so the true state is our last command unless
    # camera_command_feedback says it was refused.
    cam_level: int = 0
    cam_cx: int = 1920
    cam_cy: int = 1080
    pending: Optional[Tuple[int, int, int]] = None
    pending_frame: int = -1


# Level-1 tiles that cover the whole frame (centres), visited once at the start.
L1_TILES = [(960, 540), (2880, 540), (2880, 1620), (960, 1620)]
# Sweep band: objects enter at the top, so patrol y = SWEEP_Y at SWEEP_LEVEL.
SWEEP_LEVEL = 1
SWEEP_Y = 300
SWEEP_Y_BY_LEVEL = {1: 540, 2: 300}
# L2 dives: re-inspect a track at native resolution if it was only seen coarsely
# and is small or uncertain.  At most one dive every DIVE_EVERY frames so the
# sweep keeps covering the entry band.
DIVE_ENABLED = True
DIVE_MAX_SIZE = 60.0       # source px; larger objects are fine at L1
DIVE_MIN_CONF = 0.55
DIVE_EVERY = 3
DIVE_MARGIN = 80           # keep the predicted box this far inside the L2 crop


def choose_next_view(request: DroneFlybyPredictRequestDto, plan: CameraPlan,
                     tracker: Optional["Tracker"] = None) -> Optional[RequestedViewDto]:
    """Camera policy plus a last-line legality check.

    A refused command costs a frame of camera motion and, worse, leaves the
    camera wherever it was.  So every command is re-checked with the same rules
    the evaluator applies, and an illegal one is replaced by a hold.
    """
    _sync_camera_state(request, plan)
    req = _choose_next_view(request, plan, tracker)
    if req is None:
        return None
    reason = describe_camera_rejection(plan.cam_level, (plan.cam_cx, plan.cam_cy),
                                       req.resolution_level, (req.center_x, req.center_y))
    if reason is not None:
        logger.warning("suppressed illegal camera command %s from true state L%d (%d,%d): %s",
                       req, plan.cam_level, plan.cam_cx, plan.cam_cy, reason)
        return None
    plan.pending = (req.resolution_level, req.center_x, req.center_y)
    plan.pending_frame = request.frame
    return req


def _sync_camera_state(request: DroneFlybyPredictRequestDto, plan: CameraPlan) -> None:
    """Bring plan.cam_* up to date before deciding the next move."""
    v = request.view
    fb = request.camera_command_feedback
    if request.frame_index == 0 or plan.pending is None and plan.pending_frame < 0:
        plan.cam_level, plan.cam_cx, plan.cam_cy = v.resolution_level, v.center_x, v.center_y
    if plan.pending is not None:
        refused = fb is not None and fb.frame == plan.pending_frame
        if not refused:
            plan.cam_level, plan.cam_cx, plan.cam_cy = plan.pending
        plan.pending = None
    # If the view we were sent already reflects a newer state than we think
    # (should not happen, but the view is ground truth when it is fresher),
    # trust a view whose position equals a command we sent.
    if (v.resolution_level, v.center_x, v.center_y) == (plan.cam_level, plan.cam_cx, plan.cam_cy):
        return


def _choose_next_view(request: DroneFlybyPredictRequestDto, plan: CameraPlan,
                      tracker: Optional["Tracker"] = None) -> Optional[RequestedViewDto]:
    level = plan.cam_level
    cx, cy = plan.cam_cx, plan.cam_cy
    limit = MAXIMUM_CENTER_DELTA_PIXELS[level]

    class _B:                      # bounds object with the same fields the DTO has
        def __init__(self, lvl):
            self.minimum_center_x, self.maximum_center_x, self.minimum_center_y, self.maximum_center_y = center_bounds_for_level(lvl)

    class _C:
        @staticmethod
        def bounds_for_level(lvl):
            return _B(lvl) if lvl in SOURCE_REGION_SIZES else None
    constraints = _C()

    def clamp_to(level_target, x, y):
        b = constraints.bounds_for_level(level_target)
        if b is None:
            return None
        x = int(min(max(round(x), b.minimum_center_x), b.maximum_center_x))
        y = int(min(max(round(y), b.minimum_center_y), b.maximum_center_y))
        return x, y

    def step_towards(x, y, max_step):
        """Move from the current centre towards (x, y) by at most max_step."""
        dx, dy = x - cx, y - cy
        d = math.hypot(dx, dy)
        if d <= max_step:
            return x, y
        s = max_step / d
        return cx + dx * s, cy + dy * s

    allowed = set(ALLOWED_LEVELS_FROM[level])

    if plan.phase == "init":
        # First frame arrives at L0.  Step to L1 and start the tile tour.
        if 1 in allowed:
            plan.phase = "tiles"
            plan.tile_index = 0
            tx, ty = L1_TILES[0]
            x, y = step_towards(tx, ty, limit * 0.98)
            p = clamp_to(1, x, y)
            return RequestedViewDto(resolution_level=1, center_x=p[0], center_y=p[1])
        return None

    if plan.phase == "tiles":
        tx, ty = L1_TILES[plan.tile_index]
        if level == 1 and abs(cx - tx) < 5 and abs(cy - ty) < 5:
            plan.tile_index += 1
            if plan.tile_index >= len(L1_TILES):
                plan.phase = "sweep"
            else:
                tx, ty = L1_TILES[plan.tile_index]
        if plan.phase == "tiles":
            x, y = step_towards(tx, ty, limit * 0.98)
            p = clamp_to(1, x, y)
            return RequestedViewDto(resolution_level=1, center_x=p[0], center_y=p[1])

    # Dive: one frame at L2 on a track that deserves a closer look.
    if DIVE_ENABLED and tracker is not None and 2 in allowed and request.frame - plan.last_dive_frame >= DIVE_EVERY:
        b2 = constraints.bounds_for_level(2)
        best, best_score = None, 0.0
        for t in tracker.tracks:
            if t.best_level >= 2 or t.conf < REPORT_MIN_CONF:
                continue
            small = max(t.w, t.h) <= DIVE_MAX_SIZE
            unsure = t.conf < DIVE_MIN_CONF
            if not (small or unsure):
                continue
            # Where will it be on the next frame?
            nx_, ny_ = warp_point(t.cx, t.cy)
            if not (DIVE_MARGIN < nx_ < IMAGE_WIDTH - DIVE_MARGIN and DIVE_MARGIN < ny_ < IMAGE_HEIGHT - DIVE_MARGIN):
                continue
            tx = min(max(nx_, b2.minimum_center_x), b2.maximum_center_x)
            ty = min(max(ny_, b2.minimum_center_y), b2.maximum_center_y)
            if math.hypot(tx - cx, ty - cy) > limit * 0.98:
                continue
            # The way back must be legal too: from L2 the limit is 551 px, so
            # only dive where the same centre is also a valid L1 centre.
            b1 = constraints.bounds_for_level(1) if 1 in ALLOWED_LEVELS_FROM[2] else None
            if b1 is None or not (b1.minimum_center_x <= tx <= b1.maximum_center_x
                                  and b1.minimum_center_y <= ty <= b1.maximum_center_y):
                continue
            score = (1.0 if small else 0.0) + (1.0 - t.conf)
            if score > best_score:
                best, best_score = (int(round(tx)), int(round(ty))), score
        if best is not None:
            plan.last_dive_frame = request.frame
            return RequestedViewDto(resolution_level=2, center_x=best[0], center_y=best[1])

    # sweep phase: patrol the top band, zig-zagging in x by the max step.
    lvl = SWEEP_LEVEL
    band_y = SWEEP_Y_BY_LEVEL.get(lvl, SWEEP_Y)
    if level != lvl:
        target = lvl if lvl in allowed else (max(a for a in allowed if a <= lvl) if any(a <= lvl for a in allowed) else min(allowed))
        p = clamp_to(target, *step_towards(cx, band_y, limit * 0.98))
        return RequestedViewDto(resolution_level=target, center_x=p[0], center_y=p[1])

    b = constraints.bounds_for_level(lvl)
    step = int(limit * 0.98)
    # Touch the edge before turning around: objects hugging the frame border
    # are only ever covered by a crop centred on the bound itself.
    if plan.sweep_dir > 0 and cx >= b.maximum_center_x - 2:
        plan.sweep_dir = -1
    elif plan.sweep_dir < 0 and cx <= b.minimum_center_x + 2:
        plan.sweep_dir = 1
    nx = min(max(cx + plan.sweep_dir * step, b.minimum_center_x), b.maximum_center_x)
    x, y = step_towards(nx, band_y, limit * 0.98)
    p = clamp_to(lvl, x, y)
    return RequestedViewDto(resolution_level=lvl, center_x=p[0], center_y=p[1])


# --------------------------------------------------------------------------- #
# Per-sequence state and the entry point
# --------------------------------------------------------------------------- #

_SEQ: Dict[str, Tuple[Tracker, CameraPlan]] = {}


def _state(sequence_id: str, frame_index: int = -1):
    if frame_index == 0 and sequence_id in _SEQ:
        del _SEQ[sequence_id]          # a sequence restarted under the same id
    if sequence_id not in _SEQ:
        if len(_SEQ) > 8:
            _SEQ.clear()
        _SEQ[sequence_id] = (Tracker(), CameraPlan())
    return _SEQ[sequence_id]


RECORD_DIR = os.environ.get("DRONE_RECORD_DIR", os.path.join(HERE, "recordings"))


def _record(request: DroneFlybyPredictRequestDto, response: DroneFlybyPredictResponseDto) -> None:
    """Keep every incoming view and our answer: the validation sequence is the
    only unseen scene we get, and the README allows recording it."""
    try:
        import base64, json
        d = os.path.join(RECORD_DIR, request.sequence_id.replace("/", "_")[:40])
        os.makedirs(d, exist_ok=True)
        v = request.view
        stem = f"{request.frame:04d}_L{v.resolution_level}_{v.center_x}_{v.center_y}"
        with open(os.path.join(d, stem + ".png"), "wb") as f:
            f.write(base64.b64decode(v.image))
        meta = request.model_dump()
        meta["view"]["image"] = None
        meta["response"] = json.loads(response.model_dump_json())
        with open(os.path.join(d, stem + ".json"), "w") as f:
            json.dump(meta, f)
    except Exception:
        logger.exception("recording failed")


def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    response = _predict(request)
    _record(request, response)
    return response


def _predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    tracker, plan = _state(request.sequence_id, request.frame_index)
    if request.camera_command_feedback is not None:
        logger.warning("camera command from frame %s ignored: %s",
                       request.camera_command_feedback.frame, request.camera_command_feedback.reason)

    annotations: List[DroneFlybyPredictionDto] = []
    try:
        image = decode_view(request.view)
        region = request.view.source_region_xyxy
        rx1, ry1, rx2, ry2 = region
        sx = (rx2 - rx1) / request.view.width
        sy = (ry2 - ry1) / request.view.height
        dets = []
        for k, c, x1, y1, x2, y2 in DETECTOR(image):
            dets.append((k, c, rx1 + x1 * sx, ry1 + y1 * sy, rx1 + x2 * sx, ry1 + y2 * sy))
        tracker.step(request.frame, request.view.resolution_level, region, dets, image=image)

        for name, box, conf in tracker.report():
            g = clip_bbox_to_frame(source_bbox_to_global(box, request.original_width, request.original_height))
            if g is None:
                continue
            annotations.append(DroneFlybyPredictionDto(
                object_id=name, bbox=[round(v, 6) for v in g], confidence=round(float(min(max(conf, 0.0), 1.0)), 4)))
        annotations.sort(key=lambda a: a.confidence, reverse=True)
        annotations = annotations[:500]
    except Exception:
        logger.exception("detector/tracker failed on frame %s", request.frame)

    try:
        requested = choose_next_view(request, plan, tracker)
    except Exception:
        logger.exception("camera policy failed on frame %s", request.frame)
        requested = None

    return DroneFlybyPredictResponseDto(
        request_id=request.request_id, frame=request.frame,
        annotations=annotations, requested_view=requested)
