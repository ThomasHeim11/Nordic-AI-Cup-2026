#!/bin/bash
# Submit drone training, then the evaluation as a dependent job:  bash ~/slurm/drone_chain.sh
cd ~
JID=$(sbatch --parsable ~/slurm/drone_train_mix2.sbatch) && echo "train job $JID" && \
sbatch --dependency=afterok:$JID ~/slurm/drone_eval.sbatch && squeue -u $USER -p a40q
