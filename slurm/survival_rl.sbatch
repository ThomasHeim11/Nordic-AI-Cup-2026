#!/bin/bash
#SBATCH --job-name=surv-rl
#SBATCH --partition=a40q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=23:00:00
#SBATCH --output=%x-%j.log
source ~/slurm/_setup_survival.sh
./.venv/bin/python -c "import torch" 2>/dev/null || ./.venv/bin/python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu126 || ./.venv/bin/python -m pip install -q torch
./.venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
W=56
echo "== behaviour cloning =="
./.venv/bin/python -m rl.bc_collect --seeds 1000-1059 --workers $W --out rl/data/bc.npz 2>&1 | grep -v -i "pygame\|alsa\|pulse"
./.venv/bin/python -m rl.bc_train --data rl/data/bc.npz --epochs 10 --out rl/weights/policy.pt 2>&1 | tail -3
echo "== scoring clone vs teacher on 24 maps =="
SURVIVAL_NN=rl/weights/policy.pt ./.venv/bin/python bench_many.py --policy rl.policy_nn --seeds 201-224 --workers $W --tag bc_clone 2>&1 | grep "^\["
./.venv/bin/python bench_many.py --seeds 201-224 --workers $W --tag teacher 2>&1 | grep "^\["
echo "== PPO =="
./.venv/bin/python -m rl.ppo --envs $W --rollout 256 --iters 2000 --init rl/weights/policy.pt --out rl/weights/ppo --ckpt-every 5 2>&1 | grep -v -i "pygame\|alsa\|pulse"
