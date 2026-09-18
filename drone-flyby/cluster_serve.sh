#!/usr/bin/env bash
# Serve the drone endpoint from a cluster GPU node and expose it with cloudflared.
#   bash cluster_serve.sh            # install deps (first time), start server + tunnel
#   bash cluster_serve.sh status     # show tunnel URL + last log lines
#   bash cluster_serve.sh stop
set -e
cd "$(dirname "$0")"
ARCH=$(uname -m)
if [ "$1" = "stop" ]; then pkill -f "cloudflared tunnel --url http://localhost:9053" || true; pkill -f "python api.py" || true; echo stopped; exit 0; fi
if [ "$1" = "status" ]; then grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' tunnel_9053.log | head -1; tail -3 api_9053.log; exit 0; fi

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install -q --upgrade pip
  # torch with CUDA for this architecture (aarch64 wheels carry CUDA 12 on PyPI)
  ./.venv/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu126 || ./.venv/bin/pip install -q torch torchvision
  ./.venv/bin/pip install -q ultralytics fastapi uvicorn "pydantic>=2.7,<3" numpy opencv-python-headless requests "faster-coco-eval>=1.7.2,<2"
fi
./.venv/bin/python - <<'PY'
import torch, ultralytics, platform
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available(), "|", platform.machine())
if torch.cuda.is_available(): print("gpu:", torch.cuda.get_device_name(0))
PY

if [ ! -x ./cloudflared ]; then
  case "$ARCH" in aarch64) CF=cloudflared-linux-arm64;; x86_64) CF=cloudflared-linux-amd64;; esac
  curl -sL "https://github.com/cloudflare/cloudflared/releases/latest/download/$CF" -o cloudflared && chmod +x cloudflared
fi

pkill -f "python api.py" || true
nohup ./.venv/bin/python api.py > api_9053.log 2>&1 &
for i in $(seq 1 60); do sleep 2; curl -s --max-time 2 http://127.0.0.1:9053/api >/dev/null 2>&1 && break; done
curl -s http://127.0.0.1:9053/api; echo
# quick local latency check (same scorer as the competition, 25 sample frames)
./.venv/bin/python local_evaluator.py --realtime 2>&1 | grep -E "round trip|mAP|invalid|skipped"
pkill -f "cloudflared tunnel --url http://localhost:9053" || true
nohup ./cloudflared tunnel --url http://localhost:9053 --no-autoupdate > tunnel_9053.log 2>&1 &
sleep 8
echo "TUNNEL URL: $(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' tunnel_9053.log | head -1)"
