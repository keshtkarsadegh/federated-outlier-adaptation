#!/bin/bash
#SBATCH --job-name=foa_matrix
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.log
#
# Job array over the lines of a task file produced by `foa matrix`.
#
# Time limit: 24 h, the finals budget. A task file of sweeps needs the sweep
# budget instead, so submit those with `--time=36:00:00`. Both limits are
# generous by design - the accounting charges the time a job uses, not the time
# it asked for - so an element that finishes in twenty minutes costs twenty
# minutes.
#
#   foa matrix --plan e1_seeds --out $FOA_PROJECT_DIR/jobs/tasks_e1.txt
#   N=$(wc -l < $FOA_PROJECT_DIR/jobs/tasks_e1.txt)
#   sbatch --partition=$FOA_GPU_PARTITION --account=$FOA_ACCOUNT \
#          --array=1-$N%8 slurm/run_matrix.sbatch $FOA_PROJECT_DIR/jobs/tasks_e1.txt
#
# Arguments:
#   $1  task file, one `foa ...` command per line
#
# Resuming:
#   Every generated task carries --skip-existing, so a task whose summary
#   (final/extreme) or accuracies_points file (grid) is already on disk returns
#   immediately.  Resubmit the very same array after a time-out or a node
#   failure; completed tasks cost seconds and only the missing ones re-run.
#   To retry a subset explicitly, pass a comma/range list to --array, e.g.
#   --array=7,19,23-30%4.

# Resolve the repository checkout: submit from the repo root, or export FOA_REPO.
FOA_SLURM_DIR="${FOA_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/slurm}"
source "$FOA_SLURM_DIR/env.sh"

TASK_FILE="${1:?usage: run_matrix.sbatch <task file>}"
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
echo "results=$FOA_RESULTS_DIR host=$(hostname)" | tee -a "$TASK_LOG"

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
