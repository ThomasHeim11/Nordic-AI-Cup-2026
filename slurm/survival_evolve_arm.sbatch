#!/bin/bash
#SBATCH --job-name=surv-evo-arm
#SBATCH --partition=armq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=120
#SBATCH --time=23:00:00
#SBATCH --output=%x-%j.log
# Continue evolving the hivemind genome on an idle ARM CPU node (n005-n008, 256 threads).
# The population is seeded from the live src/utils/controllers/hivemind_params.json (gen-19 genome).
#   sbatch ~/slurm/survival_evolve_arm.sbatch <rng-seed> <out-name>
#   e.g.  sbatch ~/slurm/survival_evolve_arm.sbatch 7 evo_a     -> ~/survival-simulator/evo_a.json
SEED=${1:-7}; OUT=${2:-evo_$SEED}
source ~/slurm/_setup_survival.sh
./.venv/bin/python evolve.py --gens 30 --pop 32 --elite 8 --seeds-per-gen 8 --workers 110 --rng-seed $SEED --sigma 0.10 --out ~/survival-simulator/$OUT.json 2>&1 | grep -v -i "pygame\|alsa\|pulse"
