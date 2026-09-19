#!/bin/bash
#SBATCH --job-name=drone-mix2
#SBATCH --partition=hgx2q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.log
# Same training as drone_train_mix2.sbatch but on an x86 GPU node (hgx2q/a100q/h200q/dgx2q):
# builds a CUDA venv (.venv_x86) on first use, then fine-tunes on Helsinki + synth2.
cd ~/drone-flyby
if [ ! -x .venv_x86/bin/python ]; then
  python3 -m venv .venv_x86 && .venv_x86/bin/python -m pip install -q --upgrade pip \
  && .venv_x86/bin/python -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
  && .venv_x86/bin/python -m pip install -q ultralytics opencv-python-headless || { echo "venv setup failed"; exit 1; }
fi
.venv_x86/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
export OMP_NUM_THREADS=8
.venv_x86/bin/python train_yolo.py --model weights/best_0903.pt --data data/mix2.yaml --epochs 25 --batch 16 --device 0 --name mix2_s_b8 --workers 4 --lr0 0.003
