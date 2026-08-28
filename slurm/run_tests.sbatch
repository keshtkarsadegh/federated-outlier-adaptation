#!/bin/bash
#SBATCH --job-name=foa_tests
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.log
#
# Run the CPU test suite and check that the CLI is wired up.
#   sbatch --partition=$FOA_CPU_PARTITION --account=$FOA_ACCOUNT slurm/run_tests.sbatch

# Resolve the repository checkout: submit from the repo root, or export FOA_REPO.
FOA_SLURM_DIR="${FOA_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/slurm}"
source "$FOA_SLURM_DIR/env.sh"

export MPLBACKEND=Agg
python -m pytest -q tests
python -m federated_outlier_adaptation.cli --help

echo TESTS_DONE
