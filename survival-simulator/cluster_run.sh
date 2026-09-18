#!/usr/bin/env bash
# Run on the cluster node (inside the salloc).  Sets up a venv and starts the
# evolver with many workers.  Usage:  bash cluster_run.sh [workers] [gens]
set -e
WORKERS=${1:-60}
GENS=${2:-20}
cd "$(dirname "$0")"
python3 -m venv .venv
./.venv/bin/pip install -q --upgrade pip
# requirements.txt pins numpy/scipy builds that need Python 3.11; the node has 3.10
./.venv/bin/pip install -q fastapi numpy pydantic pygame requests scipy shapely uvicorn
export SDL_VIDEODRIVER=dummy
./.venv/bin/python -c "import pygame,numpy,scipy,shapely; print('deps ok on', __import__('platform').machine())"
# 24-seed baseline first, so cluster numbers are comparable with the Mac (889 / 950)
./.venv/bin/python bench_many.py --seeds 201-224 --workers $WORKERS --tag cluster_baseline
# then evolve: 8 seeds per candidate (vs 4 on the Mac) -> much less noise
nohup ./.venv/bin/python evolve.py --gens $GENS --pop 24 --elite 6 --seeds-per-gen 8 --workers $WORKERS --rng-seed 11 --sigma 0.12 > evolve_cluster.log 2>&1 &
echo "evolver started; follow with: tail -f evolve_cluster.log"
echo "best genome lands in src/utils/controllers/hivemind_params.json"
