# Challenge 2 — Drone Flyby (CV / real-time)

Folder: `drone-flyby/` · local port **9053** · endpoint `/predict`

## The task
A drone flies straight at 600 m, filming 3840x2160 at **3 fps**. You never get the 4K frame.
You get one **960x540 PNG** of whatever the camera currently points at — and you must answer
for the **whole source frame** anyway.

```
INPUT   the current camera crop
OUTPUT  predictions for the complete current source frame
```
So it is object detection **plus** camera control. Camera state carries over between frames.

## Resolution levels
| level | source region | sent | share |
|---|---|---|---|
| 0 | 3840x2160 (whole frame) | 960x540 | all |
| 1 | 1920x1080 | 960x540 | 1/4 |
| 2 | 960x540 (native, no downsample) | 960x540 | 1/16 |
Cropping a level-0 image yourself does NOT recover level-1/2 detail — it was never sent.

## Camera control
You send the **absolute** desired centre (source px) + level. `requested_view: null` = hold.
- Level changes one step at a time. **0 <-> 2 is impossible directly** (2 frames to go full view).
- Level 0 must use centre (1920, 1080) and is **exempt from the distance limit**.
- Centre bounds: L1 x∈[960,2880] y∈[540,1620]; L2 x∈[480,3360] y∈[270,1890].
- Max centre movement, based on the level you are on **now**: L0 2203 px, L1 1102 px, L2 551 px.
- Out-of-bounds / illegal / too-far commands are **ignored, not clipped**. Camera stays put,
  but the detections in that same response are **still scored**. Reason arrives in
  `camera_command_feedback` on every later frame until you send something legal.
- `requested_view` coords must be **ints** (2500 not 2500.0).
- Read the rules from `camera_constraints` in the request instead of hardcoding.

## Coordinates — the classic point-loser
Boxes in the response are `[x1,y1,x2,y2]` **normalized to the full 3840x2160 source frame**,
not to the 960x540 view. At L1/L2 you must lift local detections:
`utils.view_bbox_to_global(local_bbox, source_region_xyxy, 3840, 2160)`.
Supplied annotations are in source pixels → `utils.source_bbox_to_global`.

## Response
`request_id` (echo exactly), `frame` (echo exactly), `annotations` (<=500), `requested_view`.
Each annotation: `object_id` (exact name), `bbox` (normalized global), `confidence` 0–1.
**No unknown fields** — schema rejects extras.

## 16 classes (case-sensitive, see dtos.OBJECT_CLASSES)
condor, hangar, helicopter, jammer, jet_plane, large_launcher, large_tower,
medium_launcher, medium_plane, mine_roller, small_launcher, small_plane,
small_tower, spacecraft, ta-ta, tank

## Timing
Frames emitted every **333 ms** regardless. Per-request budget **3333 ms** from the POST.
- Slow answers still count for their own frame; what they cost is the frames that passed.
- Only the newest emitted frame is ever sent → a 700 ms round trip loses ~every 2nd frame.
- Unseen frame = frame with no detections; its ground truth still counts → recall loss.
- >3333 ms = abandoned, recorded as error. Gaps in `frame_index` = skipped frames.

## Scoring
**COCO mAP @ IoU 0.50**, macro-averaged per class (faster-coco-eval). 0–1.
No NMS applied by the scorer — **duplicate overlapping boxes are false positives**, suppress
them yourself. Confidence only sets evaluation order.

## Data
`src/helsinki/` — 25 raw 4K frames + per-frame JSON annotations + run_metadata.json.
**One instance of each of the 16 classes**, 13.89 m between frames, instances span several
consecutive frames. Reference/training data only — validation and evaluation scenes differ.

## Local testing
```
python api.py                                   # serve
python local_evaluator.py                       # every frame, no clock → measures detector
python local_evaluator.py --realtime            # 3 fps clock, drops slow frames → measures score
python local_evaluator.py --realtime --simulate-latency-ms 400
python local_evaluator.py --oracle              # must print 1.000
python visualize.py --all                       # draw GT on frames
```
Validation = 249-frame sequence (repeatable, and you may record/keep it).
Evaluation = different 250-frame sequence, **one attempt**.

## OBS
- One invalid box rejects the **entire response**. Need `0 <= x1 < x2 <= 1` strictly.
  `utils.clip_bbox_to_frame` returns None for degenerate boxes; `utils.validate_response` checks all.
- Never raise — catch everything and return a valid (possibly empty) response.
- Warm the model up before the attempt; first inference is slowest and gets no allowance.
