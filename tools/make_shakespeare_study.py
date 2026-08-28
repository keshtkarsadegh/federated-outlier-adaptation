"""
Emit the Shakespeare transfer study: the digit winners, unchanged, on text.

    python tools/make_shakespeare_study.py --jobs-dir "$FOA_PROJECT_DIR/jobs/v4"

Writes six task files: a login-node job, no GPU.

The claim this study exists to test
-----------------------------------
The digit study selected a server rule and a client penalty by screening 294
cells over 1,430 runs.  A selection that only holds on the data it was made on
is a tuning result, not a finding.  So the three winners are carried here **with
their hyperparameters untouched** - no re-screen, no re-tune - onto a different
modality, a different architecture and a different task.  Whatever they score
is the transfer, and the point is that nothing was adjusted to help them.

Why there is no cross-validation
--------------------------------
One g-0 and one client split, by the owner's rule for transfer studies.  Five
folds of a second modality would cost five times as much to answer a question
that is qualitative: do the winners hold at all.  The **books are still built
with five folds** - preservation is the mean over five old-fold test partitions
here exactly as in the digit study, which is what keeps that metric comparable -
but only fold 1 is trained on.

The results root indirection
----------------------------
A non-NIST provider writes under ``<results-dir>/<provider>/``.  So a stage
given ``--results-dir $FOA_STUDY_DIR/ginit`` leaves its model at
``$FOA_STUDY_DIR/ginit/shakespeare/global_model``.  That is why every
``--model-path`` below carries the extra component; it is not a typo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from federated_outlier_adaptation.training import five_cells
from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training.study_config import SHAKESPEARE_STUDY01 as CFG

ROOT = "$FOA_STUDY_DIR"
BOOKS = f"{ROOT}/fold_books"
POOLS = f"{ROOT}/outliers"
TABLES = f"{ROOT}/tables"
SETTING = f"--provider {CFG.provider} --model char_lstm"

ELIGIBLE = f"{POOLS}/eligible.json"
COUNTS = f"{POOLS}/writer_counts.json"
GINIT_SCORES = f"{POOLS}/clients_acc_on_global.json"
BAD_POOL = f"{POOLS}/pool_bad.json"
BAD_FLAT = f"{POOLS}/bad_acc_on_g0.json"
OLD_FILE = f"{POOLS}/old_data.json"
COHORT_FILE = f"{POOLS}/{CFG.cohort_file_name}"
ALL_BOOK = f"{BOOKS}/all_users.foldbook.npz"
OLD_BOOK = f"{BOOKS}/old_data.foldbook.npz"
COHORT_BOOK = f"{BOOKS}/{CFG.cohort_book_name}.foldbook.npz"
#: Where a training stage leaves its model, and where the runs look for it.
#:
#: ``--init global --global-name g0`` resolves to ``<results>/<provider>/g0_model``
#: - the provider's own root, not the stage's - so a trained model has to be
#: PROMOTED into that name before any run can start from it. The digit study
#: did the same thing through ``select-fold``, which picked a winning fold and
#: wrote ``g0_model`` beside its record. There is one fold here, so the promotion
#: is a copy; what matters is that it is explicit and carries a checksum, so the
#: model every run anchors on is identified rather than assumed.
STAGE_MODEL = f"{ROOT}/{{stage}}/{CFG.provider}/global_model"
PROVIDER_ROOT = f"{ROOT}/{CFG.provider}"
GINIT = f"{PROVIDER_ROOT}/ginit_model"
G0 = f"{PROVIDER_ROOT}/g0_model"

#: Minimum sequences a user needs to be eligible.
#:
#: Measured, not guessed.  The per-user distribution runs from 2 to 56,944
#: windows (median 1,105); a floor of 100 keeps **1,056 of 1,180 users (89.5%)
#: and 99.9% of all sequences**, and gives the smallest survivor a test
#: partition of about 20 windows.  It is deliberately the same number the digit
#: study used for old-data eligibility, so "eligible" means the same kind of
#: thing in both.  The ranking that cuts the cohort is taken over **all** of a
#: user's rows rather than its test partition, which is what keeps a 20-row
#: user from being selected on measurement noise.
MIN_SEQUENCES = 100

#: The share of the population the coarse detector calls BAD.
BAD_FRACTION = 0.30

#: Old-data draw: the number is the owner's, the seed is recorded here so the
#: draw is reproducible.
OLD_SEED = 20260828

#: Convergence.  Patience 10 / minimum 20 is the digit study's rule; the ceiling
#: is what bounds the cost, and 45 is set from the measured epoch time (below).
EPOCHS, PATIENCE, MIN_EPOCHS = 45, 10, 20

#: The reporting horizon and the local budget, both unchanged from the digit
#: study - a transfer that changed them would not be a transfer.
ROUNDS, LOCAL_EPOCHS, BATCH = 100, 5, 64

FOLD = 1
FOLDS_BUILT = 5


#: The interpreter the runner resolved. Not ``python3``: a batch node's PATH
#: can put an ancient /usr/bin/python3 first, and a guard step that imports this
#: package would then die on syntax rather than on its own assertion.
PY = "$FOA_PYTHON"


def check(body: str) -> str:
    """A guard step: python, one line, no path flags for the runner to police."""
    return f'{PY} -c "{body}"'


def promote(stage: str, name: str) -> str:
    """Copy a stage's model into the name the runs resolve, with its checksum."""
    return (
        f"foa promote-model --results-dir {ROOT} {SETTING}"
        f" --source {STAGE_MODEL.format(stage=stage)}"
        f" --target {PROVIDER_ROOT}/{name}_model --name {name}"
        f" --record {ROOT}/{name}_selection.json"
        f' --rule "single run, fold {FOLD}, no cross-validation"'
    )


# --------------------------------------------------------------------------- #
def population_lines() -> List[str]:
    """Stage 1: who is eligible, and the book every later split descends from."""
    return [
        f"foa writer-counts --results-dir {ROOT} {SETTING} --out {COUNTS}",
        f"foa select-eligible --results-dir {ROOT} {SETTING} --counts {COUNTS}"
        f" --min-samples {MIN_SEQUENCES} --out {ELIGIBLE}",
        f"foa fold-book --results-dir {ROOT} {SETTING} --out {BOOKS}/all_users"
        f" --clients-file {ELIGIBLE} --folds {FOLDS_BUILT} --seed 42"
        f" --train-rate 0.6 --eval-rate 0.2 --tag shk_all_users",
        f"foa check-population --results-dir {ROOT} {SETTING}"
        f" --clients-file {ELIGIBLE} --fold-book {ALL_BOOK} --folds {FOLD}"
        f" --counts {COUNTS} --label all_users_book",
    ]


def ginit_lines() -> List[str]:
    """Stage 2: the coarse detector, trained on every eligible user."""
    return [
        f"foa global-train --results-dir {ROOT}/ginit {SETTING}"
        f" --population file --writers-file {ELIGIBLE}"
        f" --fold-book {ALL_BOOK} --fold {FOLD}"
        f" --epochs {EPOCHS} --early-stopping-patience {PATIENCE}"
        f" --min-epochs {MIN_EPOCHS} --batch-size {BATCH}"
        f" --split-seed 42 --seed 1",
        promote("ginit", "ginit"),
    ]


def pools_lines() -> List[str]:
    """Stage 3: score every eligible user, cut BAD/GOOD, draw the old data."""
    return [
        # score-writers scores on the fold's HELD-OUT parts (val + test) by
        # construction - there is no --part to choose, and that is the right
        # default here: g-init trained on every eligible user, so its ranking is
        # only meaningful on rows it did not fit.
        f"foa score-writers --results-dir {ROOT} {SETTING} --model-path {GINIT}"
        f" --fold-book {ALL_BOOK} --fold {FOLD}"
        f" --batch-size 256 --out {POOLS}/writer_scores.json"
        f" --accuracies-name clients_acc_on_global.json",
        f"foa split-pools --results-dir {ROOT} {SETTING}"
        f" --bad-fraction {BAD_FRACTION} --scores {GINIT_SCORES}"
        f" --out {POOLS}/pools.json --csv {POOLS}/pools.csv",
        f"foa draw-old-data --results-dir {ROOT} {SETTING} --size {CFG.old_size}"
        f" --seed {OLD_SEED} --min-samples {MIN_SEQUENCES}"
        f" --exclude-file {BAD_POOL} --out {OLD_FILE}",
        f"foa fold-book --results-dir {ROOT} {SETTING} --out {BOOKS}/old_data"
        f" --clients-file {OLD_FILE} --folds {FOLDS_BUILT} --seed 42"
        f" --train-rate 0.6 --eval-rate 0.2 --tag shk_old{CFG.old_size}",
        f"foa check-population --results-dir {ROOT} {SETTING}"
        f" --clients-file {OLD_FILE} --expect-size {CFG.old_size}"
        f" --subset-of {POOLS}/pool_good.json"
        f" --fold-book {OLD_BOOK} --folds {' '.join(str(f) for f in range(1, FOLDS_BUILT + 1))}"
        f" --counts {COUNTS} --label old_data",
    ]


def g0_lines() -> List[str]:
    """Stage 4: the shipped model, trained on the old users of fold 1 only."""
    return [
        f"foa global-train --results-dir {ROOT}/g0 {SETTING}"
        f" --population file --writers-file {OLD_FILE}"
        f" --fold-book {OLD_BOOK} --fold {FOLD}"
        f" --epochs {EPOCHS} --early-stopping-patience {PATIENCE}"
        f" --min-epochs {MIN_EPOCHS} --batch-size {BATCH}"
        f" --split-seed 42 --seed 1",
        promote("g0", "g0"),
    ]


def cohort_lines() -> List[str]:
    """Stage 5: g-0 picks the cohort, and the do-nothing row is banked."""
    lines = [
        f"foa score-pool --results-dir {ROOT} {SETTING} --model-path {G0}"
        f" --clients-file {BAD_POOL} --batch-size 256"
        f" --out {POOLS}/bad_scores_on_g0.json --csv {POOLS}/bad_scores_on_g0.csv"
        f" --accuracies-name bad_acc_on_g0.json",
        # --out is not optional here: a non-NIST provider's own outliers
        # directory is <results>/<provider>/outliers, one level deeper than the
        # rest of this chain reads, and the mismatch would not surface until a
        # later stage could not find the cohort.
        f"foa select-outliers --results-dir {ROOT} {SETTING} --mode worst"
        f" --k {CFG.cohort_size} --scores {BAD_FLAT} --require-trainable"
        f" --out {COHORT_FILE}"
        f" --tag shk_cohort{CFG.cohort_size} --force --no-accuracy-table",
        f"foa fold-book --results-dir {ROOT} {SETTING}"
        f" --out {BOOKS}/{CFG.cohort_book_name} --clients-file {COHORT_FILE}"
        f" --folds {FOLDS_BUILT} --seed 42 --train-rate 0.6 --eval-rate 0.2"
        f" --tag shk_cohort{CFG.cohort_size}",
        f"foa check-population --results-dir {ROOT} {SETTING}"
        f" --clients-file {COHORT_FILE} --expect-size {CFG.cohort_size}"
        f" --subset-of {BAD_POOL} --disjoint-from {OLD_FILE}"
        f" --fold-book {COHORT_BOOK} --folds {FOLD}"
        f" --counts {COUNTS} --label cohort --show",
        # The do-nothing row: g-0, untouched, on the cohort's own test rows.
        f"foa evaluate-book --results-dir {ROOT} {SETTING} --model-path {G0}"
        f" --fold-book {COHORT_BOOK} --fold {FOLD} --part test"
        f" --clients-file {COHORT_FILE} --batch-size 256"
        f" --tag cohort_fold{FOLD} --out {ROOT}/g0_perfold_evaluations.json",
    ]
    # Preservation's five partitions, banked the same way the digit study banks
    # them, so the two studies' old-data column means the same thing.
    lines += [
        f"foa evaluate-book --results-dir {ROOT} {SETTING} --model-path {G0}"
        f" --fold-book {OLD_BOOK} --fold {fold} --part test"
        f" --clients-file {OLD_FILE} --batch-size 256"
        f" --tag old_data_fold{fold} --out {ROOT}/g0_evaluations.json"
        for fold in range(1, FOLDS_BUILT + 1)
    ]
    return lines


# --------------------------------------------------------------------------- #
def _run(parent: str, trainer: str, extra: str) -> str:
    """One federated run of this study: fold 1, the shared protocol, one line."""
    return (
        f"foa final --results-dir {ROOT} {SETTING}"
        f" --trainer {trainer} --parent {CFG.tag}_{parent}"
        f" --init global --global-name g0"
        f" --outliers-file {COHORT_FILE}"
        f" --fold-book {COHORT_BOOK} --fold {FOLD}"
        f" --old-book {OLD_BOOK} --old-clients-file {OLD_FILE} --old-fold all"
        f" --policy uniform --clients-per-round {CFG.clients_per_round}"
        f" --sampler-seed {CFG.seed_base}"
        f" --track-clients --rounds {ROUNDS} --epochs {LOCAL_EPOCHS}"
        f" --batch-size {BATCH} --eval-batch-size 256 --save-final-model --seed {FOLD}"
        f" {extra}"
    ).replace(f"--sampler-seed {CFG.seed_base}", "--sampler-seed {seed}")


def runs_lines() -> List[str]:
    """
    Stage 6: the baselines, then the three winners with nothing changed.

    Trainer compatibility on the LSTM, per method:

    * ``trimmed_0p4`` and ``anchor_0p03`` are server rules over state dicts and
      never look at the model - they transfer by construction.
    * ``seq_order_shuffle`` is a client ordering, likewise.
    * ``ntd`` is computed from logits and labels alone, so a next-character head
      is as good as a digit head.
    * ``feature_l2`` needs a penultimate representation.  For the LEAF LSTM that
      is the **final hidden state of the last layer**, ``[B, 256]``, which is
      exactly what ``CharLSTM.penultimate`` returns and what the head consumes.
      It costs a second forward pass per batch, because the model does not take
      ``return_features``.
    * **No Fisher is needed**: none of the three winners uses a Fisher-weighted
      penalty, so this study never reads ``$G0_FOLD``.
    """
    lines = [
        # (a) the centralized bound, both initialisations.
        f"foa isolated-train --results-dir {ROOT} {SETTING} --pooled"
        f" --clients-file {COHORT_FILE}"
        f" --fold-book {COHORT_BOOK} --fold {FOLD}"
        f" --old-book {OLD_BOOK} --old-fold all --old-clients-file {OLD_FILE}"
        f" --init {init}"
        + (f" --init-checkpoint {G0}" if init == "global" else "")
        + f" --epochs {EPOCHS} --early-stopping-patience {PATIENCE}"
        f" --min-epochs {MIN_EPOCHS} --batch-size {BATCH} --eval-batch-size 256"
        f" --out {ROOT}/centralized/centralized_outliers_{init}_fold{FOLD}.json"
        for init in ("global", "scratch")
    ]
    # (b) plain FedAvg, both schedules in the one task.
    lines.append(
        _run("fl_global", "BaseTrainer",
             "--aggregation fedavg --outer-workers 1 --inner-workers 1")
        .format(seed=CFG.seed_base + 1)
    )
    # (c) the three winners, hyperparameters untouched.
    aggs = {c["id"]: c for c in __import__(
        "federated_outlier_adaptation.training.agg_cells",
        fromlist=["screen_cells"]).screen_cells()}
    regs = {c["id"]: c for c in __import__(
        "federated_outlier_adaptation.training.reg_cells",
        fromlist=["screen_cells"]).screen_cells()}
    for index, cell in enumerate(five_cells.configs()):
        if cell["regulariser"] is None:
            continue
        agg, reg = aggs[cell["aggregation"]], regs[cell["regulariser"]]
        extra = f"--aggregation {agg['rule']} --extended-aggregations"
        extra += " --outer-workers 1 --inner-workers 1"
        flags = SL._agg_flags(agg["flags"])
        if flags:
            extra += f" {flags}"
        penalty = SL._reg_set(reg)
        if penalty:
            extra += f" {penalty}"
        lines.append(
            _run(cell["id"], reg["trainer"], extra).format(
                seed=CFG.seed_base + 10 + index
            )
        )
    return lines


STAGES = {
    "p01_population": (population_lines, "who is eligible, and the all-users book"),
    "p02_ginit": (ginit_lines, "the coarse detector, trained on every eligible user"),
    "p03_pools": (pools_lines, "score, cut BAD/GOOD, draw and book the old data"),
    "p04_g0": (g0_lines, "the shipped model, on the old users of fold 1"),
    "p05_cohort": (cohort_lines, "g-0 picks the cohort; the do-nothing row is banked"),
    "p06_runs": (runs_lines, "the baselines and the three transferred winners"),
}

#: Stages whose steps read what the step before wrote, so they submit ``%1``.
CHAINED = {"p01_population", "p03_pools", "p05_cohort"}


def header(name: str, what: str, tasks: List[str]) -> List[str]:
    return [
        "# GENERATED, AND NOT AUTHORISED TO RUN.",
        "#",
        f"# Shakespeare_study01 / {name}: {what}.",
        "#",
        "# THE TRANSFER CLAIM. The three winners below carry the digit study's",
        "# hyperparameters unchanged - no re-screen, no re-tune. A selection",
        "# that only holds on the data it was made on is a tuning result.",
        "#",
        "# NO CROSS-VALIDATION (owner's rule for transfer studies): one g-0, one",
        f"# client split, fold {FOLD}. The books are still built with"
        f" {FOLDS_BUILT} folds,",
        "# because preservation is the mean over five old-fold test partitions",
        "# here exactly as in the digit study - which is what keeps that column",
        "# comparable between the two.",
        "#",
        f"# ELIGIBILITY: >= {MIN_SEQUENCES} sequences. Measured from the corpus:",
        "# per-user counts run 2 .. 56,944 (median 1,105); this floor keeps",
        "# 1,056 of 1,180 users (89.5%) and 99.9% of all sequences.",
        "#",
        "# RESULTS ROOT: a non-NIST provider writes under <results-dir>/shakespeare/,",
        "# which is why every --model-path carries that component.",
        "#",
        ("# SUBMIT %1 - each step reads what the one before it wrote."
         if name in CHAINED else "# The steps are independent; any --array width."),
        "#",
        f"# {len(tasks)} task(s).",
        "#",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-dir", required=True)
    parser.add_argument("--only", default=None, choices=sorted(STAGES))
    args = parser.parse_args()

    out_dir = Path(args.jobs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, (builder, what) in STAGES.items():
        if args.only and name != args.only:
            continue
        tasks = builder()
        path = out_dir / f"{CFG.tag}_{name}.txt"
        path.write_text("\n".join(header(name, what, tasks) + tasks) + "\n")
        chained = " (submit %1)" if name in CHAINED else ""
        print(f"wrote {path.name}: {len(tasks)} task(s){chained}")
        total += len(tasks)
    print(f"{total} tasks over {len(STAGES) if not args.only else 1} stage file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
