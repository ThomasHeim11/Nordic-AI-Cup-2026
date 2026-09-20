#!/usr/bin/env bash
# One-shot deploy of the CPU drone endpoint on a fresh Ubuntu EC2 box (run ON the box, as ubuntu).
#   ssh -i ~/Downloads/nordic.pem ubuntu@<ip> 'curl -fsSL https://raw.githubusercontent.com/ThomasHeim11/Nordic-AI-Cup-2026/worktree-survival-fast-server/drone-flyby/deploy_aws_cpu.sh | bash -s -- [weights]'
# weights (optional): models/mix2_final_best_openvino_model (default, 124 ms/0.936 on c7i-flex.large) | models/mix2_final_best_int8_openvino_model | models/mix2_final_best.onnx
set -euo pipefail
WEIGHTS=${1:-models/mix2_final_best_openvino_model}
BRANCH=worktree-survival-fast-server
if ! command -v docker >/dev/null; then
  sudo apt-get update -qq && sudo apt-get install -y -qq docker.io git >/dev/null
  sudo systemctl enable --now docker
fi
if [ ! -d ~/Nordic-AI-Cup-2026 ]; then
  git clone -q https://github.com/ThomasHeim11/Nordic-AI-Cup-2026.git ~/Nordic-AI-Cup-2026
fi
cd ~/Nordic-AI-Cup-2026 && git fetch -q origin "$BRANCH" && git checkout -q "$BRANCH" && git pull -q origin "$BRANCH"
cd drone-flyby
sudo docker build -q -f Dockerfile.cpu -t drone-cpu . | tail -1
sudo docker rm -f drone >/dev/null 2>&1 || true
sudo docker run -d --restart unless-stopped --network host -e DRONE_WEIGHTS="$WEIGHTS" -e DRONE_RECORD_DIR=/app/recordings \
  -e DRONE_CONF="${DRONE_CONF:-0.08}" -e DRONE_REPORT_MIN_CONF="${DRONE_REPORT_MIN_CONF:-0.05}" \
  -e DRONE_L0_EVERY="${DRONE_L0_EVERY:-0}" -e DRONE_DIVE_EVERY="${DRONE_DIVE_EVERY:-3}" -e DRONE_SWEEP_LEVEL="${DRONE_SWEEP_LEVEL:-1}" \
  -e DRONE_MISS_DECAY_L1="${DRONE_MISS_DECAY_L1:-0.85}" \
  -e DRONE_MOTION_ONLINE_MIN="${DRONE_MOTION_ONLINE_MIN:-8}" -e DRONE_DISAGREE_PX="${DRONE_DISAGREE_PX:-10}" -e DRONE_SWEEP_FOLLOW="${DRONE_SWEEP_FOLLOW:-1}" \
  -v "$HOME/drone_recordings:/app/recordings" --name drone drone-cpu >/dev/null
for i in $(seq 1 30); do sleep 5; c=$(curl -s -m 3 -o /dev/null -w "%{http_code}" localhost:9053/api || true); [ "$c" = 200 ] && break; done
echo "drone endpoint: http://$(curl -s -m 5 http://checkip.amazonaws.com):9053  (health $c after $((i*5))s, weights $WEIGHTS)"
sudo docker logs drone 2>&1 | grep -iE "loaded|warm|error|Traceback" | tail -3 || true
