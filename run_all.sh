#!/usr/bin/env bash
# Start all three endpoint servers in the background (logs next to each server).
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
(cd "$ROOT/survival-simulator"  && nohup ./.venv/bin/python agent_server.py > agent_server.log 2>&1 &)
(cd "$ROOT/drone-flyby"         && nohup ./.venv/bin/python api.py          > api_9053.log     2>&1 &)
(cd "$ROOT/medical-appointment" && nohup ./.venv/bin/python api.py          > api_9054.log     2>&1 &)
echo "started: survival :9052, drone :9053, medical :9054 (medical needs ~1-2 min to warm up)"
echo "now open tunnels:  cloudflared tunnel --url http://localhost:905X --no-autoupdate"
