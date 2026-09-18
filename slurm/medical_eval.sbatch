#!/bin/bash
#SBATCH --job-name=med-eval
#SBATCH --partition=a40q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.log
cd ~/medical-appointment && bash cluster_medical.sh eval
