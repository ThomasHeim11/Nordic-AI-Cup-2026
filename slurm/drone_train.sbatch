#!/bin/bash
#SBATCH --job-name=drone-train
#SBATCH --partition=a40q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.log
cd ~/drone-flyby && bash cluster_train.sh
