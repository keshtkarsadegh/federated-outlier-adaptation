#!/bin/bash
#SBATCH --job-name=foa_study
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.log
#
# Run one line of a task file as one array element.
#
# A task file is a text file: one `foa ...` command per line, blank lines and
# `#` comments ignored.  Array element N runs line N.  Every stage of the study
# is expressed this way, so this one script runs all of them.
#
# THIS SCRIPT IS SITE-NEUTRAL.  It hardcodes no account, no partition and no
# absolute path.  Set the variables below for your site; the two that have no
# safe default are refused rather than guessed.
#
# REQUIRED
#   FOA_PROJECT_DIR   Writable root holding data/, results/, envs/ and logs.
#                     No default: a default pointing at a directory nobody
#                     created turns a missing setup into a silently empty one.
#   FOA_SLURM_DIR     Directory holding env.sh (normally <repo>/slurm).
#                     Defaults to the directory this script lives in.
#
# OPTIONAL
#   FOA_STUDY_DIR     Results root every task line writes into.
#                     Default $FOA_RESULTS_DIR/studies/Digits_study01
#   FOA_RESULTS_DIR   Default $FOA_PROJECT_DIR/results
#   FOA_NIST28_DIR    Packed 28x28 cache. Default $FOA_PROJECT_DIR/data/nist28
#   FOA_NIST_CLASSES  `digits` (10 labels) or `all` (62).  Default `digits`.
#   G0_FOLD           Winning g-0 fold, for the penalties that read its Fisher.
#                     Derived from the study's own g0_selection.json when unset.
#   FOA_DRY_RUN       Any non-empty value: run every check, execute nothing.
#
# GPU.  Add your site's GPU request at submit time, because #SBATCH lines
# cannot be made conditional:
#   sbatch --partition=<gpu partition> --gres=gpu:1 --account=<account> ...
#
# THE GUARD is the point of this file.  A task file is a text file, and text
# files get copied between studies.  Every path-valued flag must name
# $FOA_STUDY_DIR literally - checked BEFORE the variable is expanded, so a
# literal path pasted from another study is caught even though it would have
# expanded perfectly well.
#
# Arguments:
#   $1  task file
#
# Submit with:
#   sbatch --account=$FOA_ACCOUNT --partition=$FOA_GPU_PARTITION --gres=gpu:1 \
#          --array=1-N --output="$FOA_STUDY_DIR/logs/%x_%A_%a.log" \
#          <repo>/slurm/study_phase.sbatch <task file>
#
# WHERE THE LOGS GO.  The #SBATCH --output line above is a relative pattern and
# Slurm does not expand shell variables in it, so without the --output at submit
# time Slurm's per-element log lands in whatever directory you submitted from.
# This script writes its own per-task log under $FOA_STUDY_DIR/logs once it gets
# that far, but a crash BEFORE that point leaves nothing there.  Pass --output:
# your shell expands $FOA_STUDY_DIR, Slurm expands %x/%A/%a.
#
# A dry run walks every check and execs nothing:
#   FOA_DRY_RUN=1 SLURM_ARRAY_TASK_ID=1 bash slurm/study_phase.sbatch <tasks>
#
# Without Slurm, see slurm/run_tasks.sh, which drives this same file.

if [ -z "${FOA_PROJECT_DIR:-}" ]; then
    echo "FOA_PROJECT_DIR is not set." >&2
    echo "It is the writable root holding data/, results/ and envs/. There is" >&2
    echo "no default: one would point at a directory nobody created and turn a" >&2
    echo "missing setup into an empty study. Set it, then resubmit:" >&2
    echo "  export FOA_PROJECT_DIR=/path/to/your/workspace" >&2
    exit 78
fi
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${FOA_SLURM_DIR:=$_here}"
export FOA_PROJECT_DIR FOA_SLURM_DIR
# shellcheck source=env.sh
source "$FOA_SLURM_DIR/env.sh"

export FOA_RESULTS_DIR="${FOA_RESULTS_DIR:-$FOA_PROJECT_DIR/results}"
export FOA_STUDY_DIR="${FOA_STUDY_DIR:-$FOA_RESULTS_DIR/studies/Digits_study01}"
export FOA_NIST28_DIR="${FOA_NIST28_DIR:-$FOA_PROJECT_DIR/data/nist28}"
export FOA_NIST_RESOLUTION=28
export FOA_NIST_CLASSES="${FOA_NIST_CLASSES:-digits}"
export FOA_MODEL=fedavg_cnn

TASK_FILE="${1:?usage: study_phase.sbatch <task file>}"
INDEX="${SLURM_ARRAY_TASK_ID:-1}"

if [ ! -d "$FOA_STUDY_DIR" ]; then
    echo "No study at $FOA_STUDY_DIR." >&2
    echo "Create it with the study's setup stage, or name the right root:" >&2
    echo "  sbatch --export=ALL,FOA_STUDY_DIR=/path/to/studies/<name> ..." >&2
    exit 78
fi

# The winning g-0 fold, for the penalties that read its Fisher.  DERIVED from
# the study's own selection record rather than demanded as an export: a number
# already written down should not also have to be remembered, and a forgotten
# export would expand to an empty path component and fail a whole array at
# startup.  An explicit G0_FOLD still wins.
#
# The derivation is safe here because it touches nothing but G0_FOLD.  Whether a
# task NEEDS the fold cannot be asked yet - $TASK is not read until below - and
# env.sh runs under `set -u`, so asking early is not a wrong answer but an
# immediate abort.  The need-check is after the read, and that ordering is the
# whole reason this comment exists.
if [ -z "${G0_FOLD:-}" ] && [ -f "$FOA_STUDY_DIR/g0_selection.json" ]; then
    G0_FOLD=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['selected_fold'])" \
        "$FOA_STUDY_DIR/g0_selection.json" 2>/dev/null || true)
fi
export G0_FOLD="${G0_FOLD:-}"

LOG_DIR="${FOA_MATRIX_LOG_DIR:-$FOA_STUDY_DIR/logs}"
mkdir -p "$LOG_DIR"

TASK=$(grep -v '^[[:space:]]*\(#\|$\)' "$TASK_FILE" | sed -n "${INDEX}p")
if [ -z "$TASK" ]; then
    echo "No task at index $INDEX of $TASK_FILE"
    exit 1
fi

# ------------------------------------------------- the g-0 fold, if needed
# Now that the task line has been read, we can ask whether it needs the fold.
case "$TASK" in
    *G0_FOLD*)
        if [ -z "$G0_FOLD" ]; then
            echo "REFUSED: this task names \$G0_FOLD and the study has no" >&2
            echo "g0_selection.json to read it from. Export it:" >&2
            echo "  sbatch --export=ALL,G0_FOLD=<k> ..." >&2
            exit 78
        fi
        ;;
esac

# ---------------------------------------------------------------- the guard
PATH_FLAGS=" --results-dir --out --model-path --fold-book --clients-file \
--writers-file --exclude-file --old-book --old-clients-file --outliers-file \
--init-checkpoint --root --data-dir --cache-dir "

PREV=""
BAD=0
for WORD in $TASK; do
    case "$PATH_FLAGS" in
        *" $PREV "*)
            case "$WORD" in
                '$FOA_STUDY_DIR'*) ;;
                -*) ;;   # the flag took no value; the next word is another flag
                *)
                    echo "REFUSED: $PREV $WORD is outside \$FOA_STUDY_DIR." >&2
                    BAD=1
                    ;;
            esac
            ;;
    esac
    PREV="$WORD"
done
if [ "$BAD" -ne 0 ]; then
    echo "A task file names its roots through \$FOA_STUDY_DIR so that one study" >&2
    echo "cannot write into another. Fix the task file." >&2
    exit 78
fi

TASK_LOG="$LOG_DIR/study_${SLURM_ARRAY_JOB_ID:-local}_${INDEX}.log"
echo "task[$INDEX]: $TASK" | tee "$TASK_LOG"
echo "study=$FOA_STUDY_DIR cache=$FOA_NIST28_DIR host=$(hostname)" | tee -a "$TASK_LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$TASK_LOG" || true

MODULE_TASK="${TASK/#foa /python -u -m federated_outlier_adaptation.cli }"

# A dry run walks every check above and stops here.  `bash -n` proves the script
# parses; only running it proves the checks fire in an order where the variables
# they read exist.  That distinction has cost a whole array once.
if [ -n "${FOA_DRY_RUN:-}" ]; then
    echo "DRY RUN would exec: $MODULE_TASK"
    echo "G0_FOLD=$G0_FOLD"
    echo STUDY_TASK_DRY_RUN_OK
    exit 0
fi

set +e
eval "$MODULE_TASK" 2>&1 | tee -a "$TASK_LOG"
STATUS=${PIPESTATUS[0]}
set -e

echo "exit=$STATUS" | tee -a "$TASK_LOG"
echo STUDY_TASK_DONE
exit "$STATUS"
