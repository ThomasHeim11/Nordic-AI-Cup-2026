#!/bin/bash
#SBATCH --job-name=drone-mix2
#SBATCH --partition=a40q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.log
# Fine-tune the 0.903 Helsinki model on Helsinki + synth2 (validation-domain cutouts, roof/car negatives).
# batch 8 / workers 0: the only setting that has not frozen on the A40 (see status.md).
cd ~/drone-flyby
export OMP_NUM_THREADS=8
./.venv/bin/python train_yolo.py --model weights/best_0903.pt --data data/mix2.yaml --epochs 25 --batch 8 --device 0 --name mix2_s_b8 --workers 0 --lr0 0.003
