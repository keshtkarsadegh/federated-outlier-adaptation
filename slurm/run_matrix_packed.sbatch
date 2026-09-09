#!/bin/bash
#SBATCH --job-name=foa_matrix_packed
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.log
#
# WHERE THE LOGS GO.  The --output line above is a relative pattern, and Slurm
# does not expand shell variables in it, so without an --output at submit time
# the log lands in whatever directory you submitted from -- which every
# documented workflow makes the repository checkout, one file per array
# element.  Pass the flag instead, once, after
# `mkdir -p "$FOA_PROJECT_DIR/logs"`:
#
#     --output="$FOA_PROJECT_DIR/logs/%x_%A_%a.log"
#
# Packed job array over the lines of a task file produced by `foa matrix`.
#
# `run_matrix.sbatch` spends one array element - and therefore one whole GPU -
# on one task.  Short tasks (smoke checks, resumed arrays, small pools) leave
# most of that GPU idle, and the queue only grants a handful of elements at a
# time.  This variant packs K consecutive task lines into one array element and
# runs them concurrently on the same GPU, so the element is busy while the
# queue slot is held.
#
#   K = $FOA_TASKS_PER_JOB (default 4)
#
# Array size:
#   ceil(N / K) elements for a task file of N lines.  Element i runs the lines
#   (i-1)*K+1 ... i*K; a trailing element with fewer than K lines is fine.
#
#   foa matrix --plan smoke --out $FOA_PROJECT_DIR/jobs/tasks_smoke.txt
#   N=$(wc -l < $FOA_PROJECT_DIR/jobs/tasks_smoke.txt)
#   K=${FOA_TASKS_PER_JOB:-4}
#   A=$(( (N + K - 1) / K ))
#   sbatch --partition=$FOA_GPU_PARTITION --account=$FOA_ACCOUNT \
#          --output="$FOA_PROJECT_DIR/logs/%x_%A_%a.log" \
#          --export=ALL,FOA_TASKS_PER_JOB=$K --array=1-$A%4 \
#          slurm/run_matrix_packed.sbatch $FOA_PROJECT_DIR/jobs/tasks_smoke.txt
#
# Arguments:
#   $1  task file, one `foa ...` command per line
#
# Logs:
#   One log per *task* under $FOA_MATRIX_LOG_DIR (default
#   $FOA_PROJECT_DIR/logs/matrix), named task_<arrayjob>_<line>.log, exactly as
#   the unpacked script names them, so the two can be mixed freely.
#
# Exit status:
#   The worst status of the K tasks, so a packed element fails whenever any of
#   its tasks failed.  The per-task statuses are echoed individually and also
#   written to the tail of each task log.
#
# Resuming:
#   Every generated task carries --skip-existing, so resubmitting the identical
#   array after a time-out re-runs only what is missing.  To retry a subset,
#   compute which elements hold the failed lines: element = ceil(line / K).
#
# Concurrency note:
#   K tasks share one GPU.  Keep the per-task worker counts low (the smoke plan
#   passes --outer-workers 2 --inner-workers 2) so K x outer x inner processes
#   stay inside --cpus-per-task, and raise --mem before raising K.

# Resolve the repository checkout: submit from the repo root, or export FOA_REPO.
FOA_SLURM_DIR="${FOA_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/slurm}"
source "$FOA_SLURM_DIR/env.sh"

TASK_FILE="${1:?usage: run_matrix_packed.sbatch <task file>}"
INDEX="${SLURM_ARRAY_TASK_ID:-1}"
PER_JOB="${FOA_TASKS_PER_JOB:-${FAL_TASKS_PER_JOB:-4}}"

LOG_DIR="${FOA_MATRIX_LOG_DIR:-${FAL_MATRIX_LOG_DIR:-$FOA_PROJECT_DIR/logs/matrix}}"
mkdir -p "$LOG_DIR"

FIRST=$(( (INDEX - 1) * PER_JOB + 1 ))
LAST=$(( INDEX * PER_JOB ))
TOTAL=$(grep -c '' "$TASK_FILE")
if [ "$LAST" -gt "$TOTAL" ]; then
    LAST="$TOTAL"
fi

echo "packed element $INDEX: lines $FIRST-$LAST of $TOTAL (K=$PER_JOB) host=$(hostname)"
echo "results=$FOA_RESULTS_DIR"

if [ "$FIRST" -gt "$TOTAL" ]; then
    echo "No task at line $FIRST of $TASK_FILE (array larger than the task file)"
    exit 1
fi

STATUS_DIR=$(mktemp -d "${TMPDIR:-/tmp}/foa_packed_${SLURM_ARRAY_JOB_ID:-local}_${INDEX}_XXXXXX")
trap 'rm -rf "$STATUS_DIR"' EXIT

# One background shell per task line; each writes its own log and drops its
# exit status into $STATUS_DIR so the parent can collect it after `wait`.
run_one() {
    local line="$1"
    local task="$2"
    local task_log="$LOG_DIR/task_${SLURM_ARRAY_JOB_ID:-local}_${line}.log"

    {
        echo "task[$line]: $task"
        echo "results=$FOA_RESULTS_DIR host=$(hostname) packed_element=$INDEX"
    } > "$task_log"

    # The task lines call the `foa` console script; run them through the module
    # entry point so no PATH assumption is needed.
    local module_task="${task/#foa /python -m federated_outlier_adaptation.cli }"

    local started
    started=$SECONDS
    set +e
    eval "$module_task" >> "$task_log" 2>&1
    local status=$?
    set -e
    local elapsed=$(( SECONDS - started ))

    echo "exit=$status seconds=$elapsed" >> "$task_log"
    echo "$status" > "$STATUS_DIR/$line"
    echo "task[$line] exit=$status seconds=$elapsed"
}

LINES=()
for (( LINE = FIRST; LINE <= LAST; LINE++ )); do
    TASK=$(sed -n "${LINE}p" "$TASK_FILE")
    if [ -z "$TASK" ]; then
        echo "task[$LINE]: (blank line, skipped)"
        continue
    fi
    LINES+=("$LINE")
    run_one "$LINE" "$TASK" &
done

wait

WORST=0
for LINE in "${LINES[@]}"; do
    STATUS=$(cat "$STATUS_DIR/$LINE" 2>/dev/null || echo 1)
    echo "status[$LINE]=$STATUS"
    if [ "$STATUS" -gt "$WORST" ]; then
        WORST="$STATUS"
    fi
done

echo "exit=$WORST"
echo MATRIX_PACKED_DONE
exit "$WORST"
