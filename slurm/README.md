# SLURM jobs (Simula ex3, partition a40q)

Copy this folder to the cluster once: `scp -P 60441 -r slurm thheim@dnat.simula.no:~/`
Each job expects its code folder in `~` (`~/survival-simulator`, `~/drone-flyby`, `~/medical-appointment`)
— unpack the bundles first. Jobs are independent of any interactive `salloc`.

| job | submit | log |
|---|---|---|
| survival evolver (CPU) | `sbatch ~/slurm/survival_evolve.sbatch` | `surv-evolve-<id>.log` (`grep ^gen`) |
| survival RL: clone → score → PPO (GPU) | `sbatch ~/slurm/survival_rl.sbatch` | `surv-rl-<id>.log` (`grep -E "^\[|^iter"`) |
| score one PPO checkpoint | `sbatch ~/slurm/survival_eval_ckpt.sbatch rl/weights/ppo/iter_0050.pt` | `surv-eval-<id>.log` |
| drone training (GPU) | `sbatch ~/slurm/drone_train.sbatch` | `drone-train-<id>.log` |
| drone train on synth2 → eval chain (GPU) | `bash ~/slurm/drone_chain.sh` | `drone-mix2-<id>.log`, then `drone-eval-<id>.log` |
| drone eval of given weights (GPU) | `sbatch ~/slurm/drone_eval.sbatch [weights]` | `drone-eval-<id>.log` |
| drone serving + tunnel (GPU) | `sbatch ~/slurm/drone_serve.sbatch` | `drone-serve-<id>.log` (`grep TUNNEL`) |
| medical eval with the big LLM (GPU) | `sbatch ~/slurm/medical_eval.sbatch` | `med-eval-<id>.log` |

Useful: `squeue -u $USER` (state), `scancel <id>` (stop), `tail -f <log>` (watch).
Logs land in the directory you ran `sbatch` from (use `cd ~` first).
