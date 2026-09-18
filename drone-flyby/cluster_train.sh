#!/usr/bin/env bash
# Train the drone detector on the cluster GPU (run inside the salloc on n009).
#   bash cluster_train.sh            # synth data + Helsinki crops -> train yolo11s and yolo11m
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv && ./.venv/bin/pip install -q --upgrade pip
  ./.venv/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu126 || ./.venv/bin/pip install -q torch torchvision
  ./.venv/bin/pip install -q ultralytics fastapi uvicorn "pydantic>=2.7,<3" numpy opencv-python-headless requests "faster-coco-eval>=1.7.2,<2"
fi
./.venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "== building datasets =="
[ -d data/yolo/images/train ] || ./.venv/bin/python build_dataset.py --out data/yolo --l1-per-frame 8 --l2-per-frame 16
[ -d data/synth/images/train ] || ./.venv/bin/python make_synth.py --out data/synth --n 4000 --recordings recordings --jpeg 95
cat > data/combined.yaml <<YAML
path: $(pwd)/data
train: [yolo/images/train, synth/images/train]
val: [yolo/images/val, synth/images/val]
names:
YAML
./.venv/bin/python -c "from dtos import OBJECT_CLASSES as C; print(''.join(f'  {i}: {n}\n' for i,n in enumerate(C)))" >> data/combined.yaml

echo "== baseline on recorded frames (current weights) =="
./.venv/bin/python check_recordings.py --weights weights/best.pt --limit 60 --out debug/check_old || true

# round 2: bash cluster_train.sh round2 <weights.pt>  -> pseudo-label recordings, retrain from those weights
if [ "$1" = "round2" ]; then
  ./.venv/bin/python pseudo_label.py --weights "$2" --recordings recordings --out data/pseudo --conf 0.55 --low 0.15
  cat > data/combined2.yaml <<YAML
path: $(pwd)/data
train: [yolo/images/train, synth/images/train, pseudo/images/train]
val: [yolo/images/val, synth/images/val]
names:
YAML
  ./.venv/bin/python -c "from dtos import OBJECT_CLASSES as C; print(''.join(f'  {i}: {n}\n' for i,n in enumerate(C)))" >> data/combined2.yaml
  ./.venv/bin/python train_yolo.py --model "$2" --data data/combined2.yaml --epochs 25 --imgsz 960 --batch 16 --device 0 --name round2 --workers 0 --amp 1 --lr0 0.003
  ./.venv/bin/python check_recordings.py --weights runs/detect/runs/round2/weights/best.pt --limit 60 --out debug/check_round2 || true
  echo "== round2 done: runs/detect/runs/round2/weights/best.pt =="
  exit 0
fi

for M in yolo11s yolo11m yolo11l; do
  echo "== training $M =="
  ./.venv/bin/python train_yolo.py --model $M.pt --data data/combined.yaml --epochs 40 --imgsz 960 --batch 16 --device 0 --name synth_$M --workers 8 --amp 1
  ./.venv/bin/python check_recordings.py --weights runs/detect/runs/synth_$M/weights/best.pt --limit 60 --out debug/check_$M || true
done
echo "== done. weights: runs/detect/runs/synth_yolo11s/weights/best.pt  runs/detect/runs/synth_yolo11m/weights/best.pt =="
