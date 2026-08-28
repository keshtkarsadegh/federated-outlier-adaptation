#!/bin/bash
# Run a task file without Slurm.
#
#     slurm/run_tasks.sh <task file> [first] [last]
#
# Drives slurm/study_phase.sbatch once per line, in order, on this machine.  It
# is the same script the cluster runs, so the guard, the g-0 fold derivation and
# the logging behave identically - which is the point: a reviewer without a
# scheduler runs the study through the same code path we did, not a second
# implementation that could differ.
#
# Sequential ON PURPOSE.  Each element wants a whole GPU; running the file
# concurrently on one card is slower than running it in order, and two elements
# that write the same results folder would race.  Where the study runs a file
# %1 (chained), sequential is not merely allowed but required - and this is
# always sequential, so a chained file needs no special handling here.
#
# Environment: the same as study_phase.sbatch. FOA_PROJECT_DIR is required.
#   export FOA_PROJECT_DIR=/path/to/workspace
#   export FOA_STUDY_DIR="$FOA_PROJECT_DIR/results/studies/Digits_study01"
#
# A dry run over the whole file, executing nothing - do this first:
#   FOA_DRY_RUN=1 slurm/run_tasks.sh jobs/v4/d01_p18_five.txt
#
# Exit status: 0 only if every element it ran exited 0. It STOPS at the first
# failure rather than carrying on, because later stages read what earlier ones
# wrote and a chain that continues past a failure produces results built on a
# missing input.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_FILE="${1:?usage: run_tasks.sh <task file> [first] [last]}"

if [ ! -f "$TASK_FILE" ]; then
    echo "No task file at $TASK_FILE" >&2
    exit 78
fi

TOTAL=$(grep -c -v '^[[:space:]]*\(#\|$\)' "$TASK_FILE")
FIRST="${2:-1}"
LAST="${3:-$TOTAL}"

if [ "$TOTAL" -eq 0 ]; then
    echo "No task lines in $TASK_FILE" >&2
    exit 78
fi
if [ "$FIRST" -lt 1 ] || [ "$LAST" -gt "$TOTAL" ] || [ "$FIRST" -gt "$LAST" ]; then
    echo "Range $FIRST-$LAST is outside the file's 1-$TOTAL." >&2
    exit 78
fi

echo "== $(basename "$TASK_FILE"): running elements $FIRST-$LAST of $TOTAL"
FAILED=0
for i in $(seq "$FIRST" "$LAST"); do
    echo "-- [$i/$TOTAL]"
    if ! SLURM_ARRAY_TASK_ID="$i" bash "$HERE/study_phase.sbatch" "$TASK_FILE"; then
        echo "FAILED at element $i of $TASK_FILE" >&2
        FAILED=$i
        break
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "== stopped at element $FAILED; later stages read what this one writes." >&2
    exit 1
fi
echo "== $(basename "$TASK_FILE"): $((LAST - FIRST + 1)) element(s) OK"
