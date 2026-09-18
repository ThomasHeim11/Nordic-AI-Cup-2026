#!/bin/bash
#SBATCH --job-name=surv-eval
#SBATCH --partition=a40q
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log
# usage: sbatch ~/slurm/survival_eval_ckpt.sbatch rl/weights/ppo/iter_0050.pt
source ~/slurm/_setup_survival.sh
SURVIVAL_NN=$1 ./.venv/bin/python bench_many.py --policy rl.policy_nn --seeds 201-224 --workers 44 --tag "eval_$(basename $1)" 2>&1 | grep "^\["
