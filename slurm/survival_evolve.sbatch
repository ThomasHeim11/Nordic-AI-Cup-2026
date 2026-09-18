#!/bin/bash
#SBATCH --job-name=surv-evolve
#SBATCH --partition=a40q
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --time=23:00:00
#SBATCH --output=%x-%j.log
source ~/slurm/_setup_survival.sh
./.venv/bin/python evolve.py --gens 20 --pop 24 --elite 6 --seeds-per-gen 8 --workers 44 --rng-seed 21 --sigma 0.12 2>&1 | grep -v -i "pygame\|alsa\|pulse"
