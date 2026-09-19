#!/bin/bash
#SBATCH --job-name=drone-eval
#SBATCH --partition=a40q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
# Score the freshly trained weights: what they see on the recorded validation frames
# (vs the old 0.903 model) and the Helsinki pipeline mAP.  Installs the new weights as
# weights/best.pt so drone_serve.sbatch serves them.  Usage: sbatch drone_eval.sbatch [weights]
# On an x86 node: sbatch -p hgx2q --export=ALL,PY=./.venv_x86/bin/python drone_eval.sbatch
cd ~/drone-flyby
W=${1:-runs/detect/runs/mix2_s_b8/weights/best.pt}
PY=${PY:-./.venv/bin/python}
cp "$W" weights/mix2_best.pt
echo "== recordings check (new $W)"; $PY check_recordings.py --weights weights/mix2_best.pt --out debug/check_mix2 2>&1 | grep -E "frames|per class"
echo "== recordings check (old 0.903)"; $PY check_recordings.py --weights weights/best_0903.pt --out debug/check_old 2>&1 | grep -E "frames|per class"
echo "== Helsinki pipeline (new)"; cp weights/mix2_best.pt weights/best.pt; rm -f cache/dets_helsinki.json; $PY eval_inprocess.py 2>&1 | grep -E "mAP|refused"
