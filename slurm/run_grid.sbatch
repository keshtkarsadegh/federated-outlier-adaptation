#!/bin/bash
#SBATCH --job-name=foa_grid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=%x_%j.log
#
# Hyperparameter sweep.
#
# The 36 h limit is deliberately generous: a sweep is a whole hyperparameter
# product and the accounting charges the time a job uses, not the time it asked
# for, so a limit that outlasts the slowest configuration costs nothing and
# saves a resubmission. Lower it with `--time=` when the queue is the bottleneck.
#   sbatch --partition=$FOA_GPU_PARTITION --account=$FOA_ACCOUNT \
#          slurm/run_grid.sbatch kd
#
# Arguments:
#   $1  method: ewc | prox | kd | logit | aligned | kd_ewc | extreme
#   $2  extreme case (only for method=extreme): single | double | dual
#   $3+ extra CLI arguments

# Resolve the repository checkout: submit from the repo root, or export FOA_REPO.
FOA_SLURM_DIR="${FOA_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/slurm}"
source "$FOA_SLURM_DIR/env.sh"

METHOD="${1:-kd}"
CASE="${2:-single}"
shift 2 2>/dev/null || true

python -m federated_outlier_adaptation.cli grid \
    --method "$METHOD" \
    --case "$CASE" \
    --results-dir "$FOA_RESULTS_DIR" \
    "$@"

echo GRID_DONE
