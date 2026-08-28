"""
Emit ``study_phase.sbatch`` - the one runner every study's task files go through.

    python tools/make_study_sbatch.py --jobs-dir $FOA_PROJECT_DIR/jobs/v4

Writes one text file: a login-node job.

The script it emits does two things worth keeping in one place. It resolves the
results root from ``FOA_STUDY_DIR`` instead of hard-coding it, and it refuses any
task line whose paths point somewhere else. A task file is a text file, and text
files get copied between studies; the guard is what makes that safe.

The default root must **exist**. A default pointing at a folder nobody created
is worse than no default at all - it turns a missing study into an empty one, and
the run only fails much later, having written real results into a directory that
was never set up.
"""

from __future__ import annotations

import argparse
from pathlib import Path

#: The study this runner defaults to when no root is exported.
DEFAULT_STUDY = "Digits_study01"

SBATCH = r"""#!/bin/bash
#SBATCH --job-name=foa_study
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.log
#
# One study, one results root - enforced, not hoped for.
#
# phase_e.sbatch sets FOA_RESULTS_DIR to the v4 root and every task line under
# it names --results-dir $FOA_RESULTS_DIR/main_v6.  main_v6 is now a symlink to
# study 1, so a study-2 line that inherited that habit would write its results
# into study 1's folder and nobody would notice until the tables disagreed.
#
# This script takes the study root as a variable instead:
#
#   FOA_STUDY_DIR   the root every task line writes into.  Defaults to
#                   Digits_study01; override with --export to run another.
#   G0_FOLD         the winning g-0 fold, for the regularisation cells that
#                   need its Fisher.  Read from the study's own
#                   g0_selection.json when it is not exported, and a task that
#                   needs it and cannot get it is refused rather than run with
#                   an empty path component.
#
# A default that points at a folder nobody created is worse than no default
# at all - it turns a missing study into an empty one.  So the root must
# EXIST: the script refuses to run rather than creating it and writing a
# study into a directory that was never set up.
#
# and then REFUSES to run a task that names any other root.  The guard is the
# point of the file: a task file is a text file, and text files get copied
# between studies.
#
# The ONLY accepted prefix is $FOA_STUDY_DIR.  It used to accept
# $FOA_PROJECT_DIR too, which was too wide by exactly the amount that matters:
# every study lives under $FOA_PROJECT_DIR, so that arm would have let a line
# name another study's folder and pass.  No task line in any study uses any
# other variable - all 37,114 of them go through $FOA_STUDY_DIR - so nothing is
# lost by refusing the rest.
#
# Arguments:
#   $1  task file
#
# Submit with:
#   J=$FOA_PROJECT_DIR/jobs/v4
#   sbatch --account=$FOA_ACCOUNT --array=1-N \
#          --output="$FOA_STUDY_DIR/logs/%x_%A_%a.log" \
#          $J/study_phase.sbatch $J/<tasks>.txt
#
# WHERE THE LOGS GO.  The #SBATCH --output line below is a relative pattern, and
# Slurm does not expand shell variables in it - so without the --output above,
# Slurm's own per-element log lands in whatever directory you submitted from.
# The script writes its own per-task log into $FOA_STUDY_DIR/logs once it gets
# that far, but a crash BEFORE that point leaves nothing there and the only
# trace is Slurm's file in the submitting cwd.  Passing --output at submit time
# is the cheap fix: your shell expands $FOA_STUDY_DIR, Slurm expands %x/%A/%a.
#
# A dry run walks every check and execs nothing:
#   FOA_DRY_RUN=1 SLURM_ARRAY_TASK_ID=1 bash $J/study_phase.sbatch $J/<tasks>.txt
#
# To run a different study:
#   sbatch --export=ALL,FOA_STUDY_DIR=/path/to/studies/<other> ... 

: "${FOA_PROJECT_DIR:?set FOA_PROJECT_DIR to your writable workspace root}"
: "${FOA_SLURM_DIR:=$FOA_PROJECT_DIR/repo_foa/slurm}"
export FOA_PROJECT_DIR FOA_SLURM_DIR
source "$FOA_SLURM_DIR/env.sh"

# env.sh exports FOA_RESULTS_DIR itself, so it is set outright after sourcing.
export FOA_RESULTS_DIR="${FOA_V4_RESULTS_DIR:-$FOA_PROJECT_DIR/results_v4}"
export FOA_STUDY_DIR="${FOA_STUDY_DIR:-$FOA_RESULTS_DIR/studies/Digits_study01}"
export FOA_NIST28_DIR="${FOA_NIST28_DIR:-$FOA_PROJECT_DIR/data/nist28}"
export FOA_NIST_RESOLUTION=28
export FOA_NIST_CLASSES="${FOA_NIST_CLASSES:-digits}"
export FOA_MODEL=fedavg_cnn

TASK_FILE="${1:?usage: study_phase.sbatch <task file>}"
INDEX="${SLURM_ARRAY_TASK_ID:-1}"

if [ ! -d "$FOA_STUDY_DIR" ]; then
    echo "No study at $FOA_STUDY_DIR." >&2
    echo "Create it with the study's setup script, or name the right root:" >&2
    echo "  sbatch --export=ALL,FOA_STUDY_DIR=/path/to/studies/<name> ..." >&2
    exit 78
fi
# The winning g-0 fold, for the regularisation cells that need its Fisher.
# DERIVED here, from the study's own selection record rather than demanded as an
# export: a number that is already written down should not also have to be
# remembered, and a forgotten export would expand to an empty path component and
# fail eighty array elements at startup.  An explicit G0_FOLD still wins.
#
# The derivation is safe here because it touches nothing but G0_FOLD and the
# study directory.  Whether a task NEEDS the fold cannot be asked yet - $TASK is
# not read until below - and env.sh runs under `set -u`, so asking early is not
# a wrong answer, it is an immediate abort.  The need-check is after the read.
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
# Every path-valued flag must land inside this study.  Checked BEFORE the
# variables are expanded, so a literal path copied from another study is caught
# even though it would have expanded perfectly well.
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
case "$TASK" in
    *main_v6*)
        echo "REFUSED: the task names main_v6, a symlink to study 1." >&2
        BAD=1
        ;;
esac
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
# they read exist.  That distinction cost 585 array elements once.
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
"""


def sbatch_text() -> str:
    """The runner, verbatim."""
    return SBATCH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-dir", required=True)
    args = parser.parse_args()
    jobs = Path(args.jobs_dir)
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / "study_phase.sbatch").write_text(sbatch_text())
    print(f"wrote {jobs}/study_phase.sbatch (default study: {DEFAULT_STUDY})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
