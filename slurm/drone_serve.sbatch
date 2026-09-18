#!/bin/bash
#SBATCH --job-name=drone-serve
#SBATCH --partition=a40q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00
#SBATCH --output=%x-%j.log
# Serves the drone endpoint + cloudflared tunnel for the job's lifetime; URL in the log.
cd ~/drone-flyby && bash cluster_serve.sh && echo "serving; URL above" && sleep infinity
