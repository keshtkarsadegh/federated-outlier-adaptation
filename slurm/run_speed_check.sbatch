#!/bin/bash
#SBATCH --job-name=foa_speed_check
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
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
# Evaluation speed-up: the same configuration on both evaluation paths.
#
# Runs one configuration twice - once with `--eval-path loader`, the published
# per-round DataLoader passes, and once with `--eval-path cache`, the
# materialised tensors - into two separate result roots, and then compares the
# stored series element by element.
#
# The two runs must agree: for a given pair of models the evaluation is
# deterministic, so the reported maximum absolute difference is the evidence
# that the speed-up costs nothing. What differs is `round_seconds`.
#
#   sbatch --partition=$FOA_GPU_PARTITION --account=$FOA_ACCOUNT \
#          --output="$FOA_PROJECT_DIR/logs/%x_%j.log" \
#          slurm/run_speed_check.sbatch nist 20
#
# Arguments:
#   $1  provider: nist   (default nist)
#   $2  rounds                                   (default 20)
#   $3+ extra CLI arguments
#
# Output:
#   $FOA_SPEED_DIR/<provider>/{loader,cache}/   the two result trees
#   $FOA_SPEED_DIR/<provider>/comparison.json   the full comparison
# The speed roots are separate from the experiment results on purpose: nothing
# this job writes belongs to a reported number.

FOA_SLURM_DIR="${FOA_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/slurm}"
source "$FOA_SLURM_DIR/env.sh"

PROVIDER="${1:-nist}"
ROUNDS="${2:-20}"
shift 2 2>/dev/null || true

SPEED_DIR="${FOA_SPEED_DIR:-$FOA_PROJECT_DIR/results_speed}/$PROVIDER"
mkdir -p "$SPEED_DIR/loader" "$SPEED_DIR/cache"

# The frozen inputs a run reads - writer split, teacher checkpoint, Fisher
# information, client accuracies - are copied (not linked) into both roots, so
# the job never writes anywhere near the experiment results.
seed_root() {
    local target="$1"
    for item in writer_split.json global_model global_results global_clients_results outliers; do
        if [ -e "$FOA_RESULTS_DIR/$item" ] && [ ! -e "$target/$item" ]; then
            cp -r "$FOA_RESULTS_DIR/$item" "$target/"
        fi
    done
}

if [ "$PROVIDER" = "nist" ]; then
    seed_root "$SPEED_DIR/loader"
    seed_root "$SPEED_DIR/cache"
else
    mkdir -p "$SPEED_DIR/loader/$PROVIDER" "$SPEED_DIR/cache/$PROVIDER"
    for side in loader cache; do
        for item in writer_split.json global_model global_results global_clients_results outliers; do
            if [ -e "$FOA_RESULTS_DIR/$PROVIDER/$item" ] && [ ! -e "$SPEED_DIR/$side/$PROVIDER/$item" ]; then
                cp -r "$FOA_RESULTS_DIR/$PROVIDER/$item" "$SPEED_DIR/$side/$PROVIDER/"
            fi
        done
    done
fi

EPOCH_ARGS=()

for EVAL_PATH in loader cache; do
    echo "=== provider=$PROVIDER eval_path=$EVAL_PATH rounds=$ROUNDS ==="
    /usr/bin/time -v python -m federated_outlier_adaptation.cli final \
        --provider "$PROVIDER" \
        --results-dir "$SPEED_DIR/$EVAL_PATH" \
        --trainer DistillationTrainer \
        --parent speedtest_pool_m10_uniform_fedavg \
        --pool-frac 0.05 \
        --clients-per-round 10 \
        --policy uniform \
        --sampler-seed 1 \
        --track-clients \
        --aggregation fedavg \
        --seed 1 \
        --rounds "$ROUNDS" \
        --eval-path "$EVAL_PATH" \
        "${EPOCH_ARGS[@]}" "$@" 2>&1 | tail -n 40
done

echo "=== equality and speed-up ==="
python -m federated_outlier_adaptation.analysis.eval_equality \
    --loader "$SPEED_DIR/loader" \
    --cache "$SPEED_DIR/cache" \
    --out "$SPEED_DIR/comparison.json"

echo SPEED_CHECK_DONE
