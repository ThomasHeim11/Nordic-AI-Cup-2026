#!/usr/bin/env bash
# RL pipeline on the cluster node (run inside the salloc, from survival-simulator/).
#   bash rl/cluster_rl.sh bc      # collect 60 teacher episodes, clone, score the clone on 24 maps
#   bash rl/cluster_rl.sh ppo     # start PPO from the clone (background), 60 simulators
#   bash rl/cluster_rl.sh eval rl/weights/ppo/iter_0050.pt   # score a checkpoint on 24 maps
set -e
cd "$(dirname "$0")/.."
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
W=${WORKERS:-60}
if ! ./.venv/bin/python -c "import torch" 2>/dev/null; then
  ./.venv/bin/python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu126 || ./.venv/bin/python -m pip install -q torch
fi
./.venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
case "$1" in
  bc)
    ./.venv/bin/python -m rl.bc_collect --seeds 1000-1059 --workers $W --out rl/data/bc.npz 2>&1 | grep -v -i "pygame\|alsa\|pulse"
    ./.venv/bin/python -m rl.bc_train --data rl/data/bc.npz --epochs 10 --out rl/weights/policy.pt 2>&1 | tail -3
    SURVIVAL_NN=rl/weights/policy.pt ./.venv/bin/python bench_many.py --policy rl.policy_nn --seeds 201-224 --workers $W --tag bc_clone 2>&1 | grep "^\["
    ./.venv/bin/python bench_many.py --seeds 201-224 --workers $W --tag teacher 2>&1 | grep "^\["
    ;;
  ppo)
    nohup ./.venv/bin/python -m rl.ppo --envs $W --rollout 256 --iters 2000 --init rl/weights/policy.pt --out rl/weights/ppo --ckpt-every 5 > rl/ppo.log 2>&1 &
    echo "PPO started; follow with: grep '^iter' rl/ppo.log | tail"
    ;;
  eval)
    SURVIVAL_NN=$2 ./.venv/bin/python bench_many.py --policy rl.policy_nn --seeds 201-224 --workers $W --tag "eval_$(basename $2)" 2>&1 | grep "^\["
    ;;
  *) echo "usage: bash rl/cluster_rl.sh bc|ppo|eval <ckpt>";;
esac
