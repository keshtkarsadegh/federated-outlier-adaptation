#!/bin/bash
#SBATCH --job-name=foa_prepare_nist28
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log
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
#          <repo>/slurm/prepare_nist28.sbatch
#
# Without Slurm, run the same command directly - it needs no GPU:
#   foa prepare-data --dataset nist --zip <by_write.zip> --out <cache dir> \
#       --resolution 28 --classes all

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${FOA_SLURM_DIR:=$_here}"
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

python -u -m federated_outlier_adaptation.cli prepare-data \
    --dataset nist \
    --zip "$ZIP" \
    --out "$OUT" \
    --resolution 28 \
    --classes all

echo PREPARE_NIST28_DONE
