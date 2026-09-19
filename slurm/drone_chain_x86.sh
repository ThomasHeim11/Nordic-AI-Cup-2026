#!/bin/bash
# Train + eval on an x86 GPU partition:  bash ~/slurm/drone_chain_x86.sh hgx2q   (or a100q, h200q, dgx2q)
P=${1:-hgx2q}
cd ~
JID=$(sbatch --parsable -p $P ~/slurm/drone_train_x86.sbatch) && echo "train job $JID on $P" && \
sbatch -p $P --dependency=afterok:$JID --export=ALL,PY=./.venv_x86/bin/python ~/slurm/drone_eval.sbatch && squeue -u $USER -p $P
