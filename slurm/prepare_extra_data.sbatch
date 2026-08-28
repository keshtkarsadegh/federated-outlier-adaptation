#!/bin/bash
#SBATCH --job-name=foa_prepare_extra
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
#
# Build the packed cache of an additional dataset. Like the NIST cache, the
# archives are read in place and the result is a single array plus one index
# file, so the inode count stays flat.
#
#   sbatch --partition=$FOA_CPU_PARTITION --account=$FOA_ACCOUNT \
#          slurm/prepare_extra_data.sbatch shakespeare /path/to/pg100.txt
#   sbatch --partition=$FOA_CPU_PARTITION --account=$FOA_ACCOUNT \
#          slurm/prepare_extra_data.sbatch cifar10 /path/to/cifar-10-python.tar.gz
#
# Arguments:
#   $1  dataset: shakespeare | cifar10 | nist   (default shakespeare)
#   $2  raw archive/corpus                      (default under $FOA_DATA_DIR)
#   $3+ extra CLI arguments, e.g. --num-clients 200 --alpha 0.3

FOA_SLURM_DIR="${FOA_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/slurm}"
source "$FOA_SLURM_DIR/env.sh"

DATASET="${1:-shakespeare}"
RAW="${2:-}"
shift 2 2>/dev/null || true

case "$DATASET" in
    shakespeare)
        RAW="${RAW:-${FOA_SHAKESPEARE_DIR:-$FOA_PROJECT_DIR/data/shakespeare}/pg100.txt}"
        OUT="${FOA_SHAKESPEARE_DIR:-$FOA_PROJECT_DIR/data/shakespeare}"
        ;;
    cifar10)
        RAW="${RAW:-${FOA_CIFAR10_DIR:-$FOA_PROJECT_DIR/data/cifar10}/cifar-10-python.tar.gz}"
        OUT="${FOA_CIFAR10_DIR:-$FOA_PROJECT_DIR/data/cifar10}"
        ;;
    nist)
        RAW="${RAW:-$FOA_PROJECT_DIR/data/nist/by_write.zip}"
        OUT="$FOA_CACHE_DIR"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

export MPLBACKEND=Agg

python -m federated_outlier_adaptation.cli prepare-data \
    --dataset "$DATASET" \
    --zip "$RAW" \
    --out "$OUT" \
    "$@"

echo "=== files ==="
ls -l "$OUT"
echo PREPARE_EXTRA_DATA_DONE
