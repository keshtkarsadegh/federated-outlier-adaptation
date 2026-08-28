#!/bin/bash
#SBATCH --job-name=foa_extra_global
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
#
# Global baseline of an additional dataset: pooled global model, Fisher
# information, per-client accuracies, the deterministic client selections and
# the rule-based outlier pool. The NIST artefacts are never written to.
#
#   sbatch --partition=$FOA_GPU_PARTITION --account=$FOA_ACCOUNT \
#          slurm/extra_global_train.sbatch shakespeare
#
# Arguments:
#   $1  provider: shakespeare | cifar10   (default shakespeare)
#   $2  epochs                            (default 100)
#   $3+ extra CLI arguments

FOA_SLURM_DIR="${FOA_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/slurm}"
source "$FOA_SLURM_DIR/env.sh"

PROVIDER="${1:-shakespeare}"
EPOCHS="${2:-100}"
shift 2 2>/dev/null || true

export MPLBACKEND=Agg
export FOA_SHAKESPEARE_DIR="${FOA_SHAKESPEARE_DIR:-$FOA_PROJECT_DIR/data/shakespeare}"
export FOA_CIFAR10_DIR="${FOA_CIFAR10_DIR:-$FOA_PROJECT_DIR/data/cifar10}"

echo "provider=$PROVIDER epochs=$EPOCHS results=$FOA_RESULTS_DIR"

python -m federated_outlier_adaptation.cli global-train \
    --provider "$PROVIDER" \
    --epochs "$EPOCHS" \
    --batch-size 64 \
    --k-values 5 20 50 \
    --results-dir "$FOA_RESULTS_DIR" \
    "$@"

python -m federated_outlier_adaptation.cli combined-train \
    --provider "$PROVIDER" \
    --results-dir "$FOA_RESULTS_DIR"

# The rule-based pool the main experimental setting draws its clients from.
python -m federated_outlier_adaptation.cli select-outliers \
    --provider "$PROVIDER" \
    --mode pool --pool-frac 0.05 \
    --results-dir "$FOA_RESULTS_DIR"

echo EXTRA_GLOBAL_TRAIN_DONE
