#!/bin/bash
#SBATCH --job-name=foa_prepare_data
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
#
# Build the packed NIST cache directly from by_write.zip. The archive is read
# in place with zipfile; nothing is ever extracted to disk.
#
# Submit with:
#   sbatch --partition=$FOA_CPU_PARTITION --account=$FOA_ACCOUNT \
#          slurm/prepare_data.sbatch /path/to/by_write.zip

# Resolve the repository checkout: submit from the repo root, or export FOA_REPO.
FOA_SLURM_DIR="${FOA_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/slurm}"
source "$FOA_SLURM_DIR/env.sh"

ZIP="${1:-$FOA_PROJECT_DIR/data/nist/by_write.zip}"

python -m federated_outlier_adaptation.cli prepare-data \
    --zip "$ZIP" \
    --labels "$FOA_REPO/data/by_write/digits_labels.json" \
    --out "$FOA_CACHE_DIR"

echo PREPARE_DATA_DONE
