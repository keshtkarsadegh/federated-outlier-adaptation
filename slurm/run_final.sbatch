#!/bin/bash
#SBATCH --job-name=foa_final
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log
#
# Final runs for one trainer.
#
# The 24 h limit is deliberately generous: the accounting charges the time a job
# uses, not the time it asked for, so a limit that outlasts the slowest arm
# costs nothing and saves a resubmission.
#   sbatch --partition=$FOA_GPU_PARTITION --account=$FOA_ACCOUNT \
#          slurm/run_final.sbatch DistillationTrainer 1234
#
# Arguments:
#   $1  trainer class name           (default DistillationTrainer)
#   $2  seed, "none" for unseeded    (default none)
#   $3+ extra CLI arguments, e.g. --set T=4 alpha=0.95

# Resolve the repository checkout: submit from the repo root, or export FOA_REPO.
FOA_SLURM_DIR="${FOA_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/slurm}"
source "$FOA_SLURM_DIR/env.sh"

TRAINER="${1:-DistillationTrainer}"
SEED="${2:-none}"
shift 2 2>/dev/null || true

SEED_ARGS=()
if [ "$SEED" != "none" ]; then
    SEED_ARGS=(--seed "$SEED")
fi

python -m federated_outlier_adaptation.cli final \
    --trainer "$TRAINER" \
    --results-dir "$FOA_RESULTS_DIR" \
    "${SEED_ARGS[@]}" "$@"

echo FINAL_DONE
