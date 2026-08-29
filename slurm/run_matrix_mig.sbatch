#!/bin/bash
#SBATCH --job-name=foa_matrix_mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.log
#
# Optional variant of `run_matrix.sbatch` for the small models.
#
# One task per array element, exactly as `run_matrix.sbatch`, but the element
# asks for a MIG slice (1g.10gb) instead of a whole A100.  The models of this
# study are small - FlexibleCNN is ~1 M parameters and FedAvgCNN is
# of the same order - so a single run fits comfortably into a 10 GB slice, and
# a partition that hands out slices schedules such a job much sooner than one
# that has to free a full accelerator.
#
# Use it when
#   - the tasks are single-run (finals, extreme cases, smoke checks) rather
#     than sweeps that fan out over many configurations,
#   - the worker counts are small (--inner-workers of 2-4), and
#   - the site actually exposes MIG slices; `sinfo -o "%G"` on the GPU
#     partition shows whether `1g.10gb` is offered.  Where it is not, use
#     `run_matrix.sbatch` (one full GPU per task) or `run_matrix_packed.sbatch`
#     (several tasks sharing one full GPU) instead - this script is never
#     required by any plan.
#
#   foa matrix --plan e1_seeds --out $FOA_PROJECT_DIR/jobs/tasks_e1.txt
#   N=$(wc -l < $FOA_PROJECT_DIR/jobs/tasks_e1.txt)
#   sbatch --partition=$FOA_GPU_PARTITION --account=$FOA_ACCOUNT \
#          --array=1-$N%8 slurm/run_matrix_mig.sbatch $FOA_PROJECT_DIR/jobs/tasks_e1.txt
#
# Arguments:
#   $1  task file, one `foa ...` command per line
#
# Everything else - the per-task log, the resume contract of --skip-existing
# and the exit status - is identical to `run_matrix.sbatch`.

# Resolve the repository checkout: submit from the repo root, or export FOA_REPO.
FOA_SLURM_DIR="${FOA_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/slurm}"
source "$FOA_SLURM_DIR/env.sh"

TASK_FILE="${1:?usage: run_matrix_mig.sbatch <task file>}"
INDEX="${SLURM_ARRAY_TASK_ID:-1}"

LOG_DIR="${FOA_MATRIX_LOG_DIR:-${FAL_MATRIX_LOG_DIR:-$FOA_PROJECT_DIR/logs/matrix}}"
mkdir -p "$LOG_DIR"

TASK=$(sed -n "${INDEX}p" "$TASK_FILE")
if [ -z "$TASK" ]; then
    echo "No task at line $INDEX of $TASK_FILE"
    exit 1
fi

TASK_LOG="$LOG_DIR/task_${SLURM_ARRAY_JOB_ID:-local}_${INDEX}.log"

echo "task[$INDEX]: $TASK" | tee "$TASK_LOG"
echo "results=$FOA_RESULTS_DIR host=$(hostname) gpu=mig" | tee -a "$TASK_LOG"

# The task lines call the `foa` console script; run them through the module
# entry point so no PATH assumption is needed.
MODULE_TASK="${TASK/#foa /python -m federated_outlier_adaptation.cli }"

set +e
eval "$MODULE_TASK" 2>&1 | tee -a "$TASK_LOG"
STATUS=${PIPESTATUS[0]}
set -e

echo "exit=$STATUS" | tee -a "$TASK_LOG"
echo MATRIX_TASK_DONE
exit "$STATUS"
