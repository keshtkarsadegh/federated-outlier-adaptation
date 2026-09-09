#!/bin/bash
#SBATCH --job-name=foa_prepare_nist28
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log
#
# WHERE THE LOGS GO.  The --output line above is a relative pattern, and Slurm
# does not expand shell variables in it, so without an --output at submit time
# the log lands in whatever directory you submitted from -- which every
# documented workflow makes the repository checkout, one file per job.
# Pass the flag instead, once, after
# `mkdir -p "$FOA_PROJECT_DIR/logs"`:
#
#     --output="$FOA_PROJECT_DIR/logs/%x_%j.log"
#
# Build the packed 28x28, 62-class NIST cache the whole study reads.
#
#   by_write.zip  ->  EMNIST-style conversion  ->  $FOA_NIST28_DIR/{4 files}
#
# The archive is read IN PLACE with zipfile and is never extracted; the output
# is four files (images, labels, writer index, index JSON) whatever the sample
# count, so the job costs four inodes and roughly 650 MB.  On file systems with
# an inode quota that difference is the whole reason the loader works this way.
#
# The character labels come from the SD19 by_class checksum log, joined to the
# by_write paths by MD5.  Both logs are a few tens of megabytes of text;
# by_class.zip itself (~4 GB) is never needed.  See docs/DATA.md.
#
# Resources: 1 task, 4 CPUs, 32 GB, NO GPU, walltime 6 h.  The conversion loop
# is single-threaded (PNG decode -> Gaussian blur -> bounding box -> bicubic
# 28x28) and the label join holds both checksum logs in memory, which is what
# the 32 GB are for.  The job logs a projected total runtime after its first
# 2,000 images, so the real cost is on record within the first minute.
#
# Arguments (both optional):
#   $1  path to by_write.zip   [$FOA_DATA_DIR/nist/by_write.zip]
#   $2  output cache directory [$FOA_NIST28_DIR]
#
# Submit with:
#   sbatch --account=$FOA_ACCOUNT --partition=$FOA_CPU_PARTITION \
#          --output="$FOA_PROJECT_DIR/logs/%x_%j.log" \
#          --export=ALL,FOA_PROJECT_DIR=$FOA_PROJECT_DIR \
#          <repo>/slurm/prepare_nist28.sbatch
#
# Without Slurm, run the same command directly - it needs no GPU:
#   foa prepare-data --dataset nist --zip <by_write.zip> --out <cache dir> \
#       --resolution 28 --classes all

# WHERE env.sh IS.  Slurm COPIES a batch script into its spool directory before
# running it, so ${BASH_SOURCE[0]} is /var/spool/slurmd/job<N>/slurm_script and
# its dirname holds no env.sh.  Deriving the location from the script's own path
# therefore works when you run it directly - every dry run in this repository -
# and fails under sbatch, which is the one case that matters.  It cost a whole
# re-run: 11 of 11 elements exited 127 with "-u: command not found", because
# sourcing env.sh had silently failed and $FOA_PYTHON was empty.
#
# So: an explicit FOA_SLURM_DIR wins, then the script's own directory when it
# really holds env.sh, then the checkout under FOA_PROJECT_DIR.  And if none of
# them does, refuse - an unsourced env.sh leaves every later line running under
# whatever the node happened to provide.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
if [ -z "${FOA_SLURM_DIR:-}" ]; then
    if [ -f "$_here/env.sh" ]; then
        FOA_SLURM_DIR="$_here"
    elif [ -f "${FOA_REPO:-$FOA_PROJECT_DIR/repo_foa}/slurm/env.sh" ]; then
        FOA_SLURM_DIR="${FOA_REPO:-$FOA_PROJECT_DIR/repo_foa}/slurm"
    fi
fi
if [ ! -f "${FOA_SLURM_DIR:-}/env.sh" ]; then
    echo "REFUSED: no env.sh. Looked in FOA_SLURM_DIR=${FOA_SLURM_DIR:-unset}," >&2
    echo "in this script's directory ($_here - which under sbatch is Slurm's" >&2
    echo "spool copy, not the checkout), and in" >&2
    echo "  ${FOA_REPO:-$FOA_PROJECT_DIR/repo_foa}/slurm" >&2
    echo "Set FOA_SLURM_DIR or FOA_REPO and resubmit." >&2
    exit 78
fi
export FOA_SLURM_DIR
# shellcheck source=env.sh
source "$FOA_SLURM_DIR/env.sh"

ZIP="${1:-$FOA_DATA_DIR/nist/by_write.zip}"
OUT="${2:-$FOA_NIST28_DIR}"

if [ ! -f "$ZIP" ]; then
    echo "No archive at $ZIP. See docs/DATA.md for how to fetch it." >&2
    exit 78
fi

echo "archive=$ZIP"
echo "out=$OUT"
mkdir -p "$OUT"

"$FOA_PYTHON" -u -m federated_outlier_adaptation.cli prepare-data \
    --dataset nist \
    --zip "$ZIP" \
    --out "$OUT" \
    --resolution 28 \
    --classes all

echo PREPARE_NIST28_DONE
