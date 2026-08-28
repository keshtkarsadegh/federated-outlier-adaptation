"""
Emit P08, P09 and P10 of the digits study: the three arrangements of the cohort.

    python tools/make_digits_p08.py --jobs-dir $FOA_PROJECT_DIR/jobs/v4
                                    [--cohort-file PATH]

Writes four text files: a login-node job.

The three stages are the ladder the study argues along, and they are deliberately
independent of one another:

    P08  isolated     ten private models - no aggregation, no sharing
    P09  federated    one server model, nine of ten clients per round
    P10  centralized  one model over the pooled cohort, privacy switched off

Each reads only artefacts that already exist - the cohort, its book, the old
book, g-0 - so the three arrays can be submitted at the same time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STUDY = "Digits_study01"
ROOT = "$FOA_STUDY_DIR"
BOOKS = f"{ROOT}/fold_books"
POOLS = f"{ROOT}/outliers"
SETTING = "--resolution 28 --classes digits"
CONVERGE = "--early-stopping-patience 10 --min-epochs 20"

COHORT_FILE = f"{POOLS}/cohort_worst10.json"
OLD_FILE = f"{POOLS}/old_data.json"
COHORT_BOOK = f"{BOOKS}/cohort10.foldbook.npz"
OLD_BOOK = f"{BOOKS}/old_data.foldbook.npz"
G0 = f"{ROOT}/g0_model"

FOLDS = (1, 2, 3, 4, 5)
INITS = ("global", "scratch")
#: Nine of ten per round - this study's 10% dropout.
CLIENTS_PER_ROUND = 9
ROUNDS = 100
LOCAL_EPOCHS = 5

#: Read from the study's own cohort_worst10.json, not retyped from a message.
COHORT = (
    "f3642_03", "f2248_68", "f2297_69", "f2524_58", "f2145_89",
    "f2162_62", "f0619_33", "f2307_62", "f3151_13", "f2304_67",
)

#: Measured from the v2 books.
G0_FOLD = 2
COHORT_ROWS = 1024
COHORT_TRAIN = 579
COHORT_TEST = 222
OLD_TEST = 5297


def _init_flags(init: str) -> str:
    """Where a private or pooled model starts from."""
    return f"--init global --init-checkpoint {G0}" if init == "global" else "--init scratch"


def isolated_lines(cohort) -> list[str]:
    """One private model per (writer, fold, init)."""
    return [
        f"foa isolated-train --results-dir {ROOT} {SETTING}"
        f" --clients-file {COHORT_FILE} --only-client {writer}"
        f" --fold-book {COHORT_BOOK} --fold {fold}"
        f" --old-book {OLD_BOOK} --old-fold all --old-clients-file {OLD_FILE}"
        f" {_init_flags(init)}"
        f" --epochs 100 {CONVERGE} --batch-size 64 --eval-batch-size 256"
        f" --out {ROOT}/isolated/isolated_{init}_{writer}_fold{fold}.json"
        for init in INITS
        for writer in cohort
        for fold in FOLDS
    ]


def fl_lines() -> list[str]:
    """Plain FedAvg on the cohort, both families per task."""
    lines = []
    for index, init in enumerate(INITS):
        for fold in FOLDS:
            seed = 30000 + index * 500 + fold
            start = (
                "--init global --global-name g0" if init == "global"
                else "--init scratch --global-name g0"
            )
            lines.append(
                f"foa final --results-dir {ROOT} {SETTING}"
                f" --model fedavg_cnn --trainer BaseTrainer"
                f" --parent d01_fl_{init}_fold{fold}"
                f" --aggregation fedavg {start}"
                f" --outliers-file {COHORT_FILE}"
                f" --fold-book {COHORT_BOOK} --fold {fold}"
                f" --old-book {OLD_BOOK} --old-clients-file {OLD_FILE}"
                f" --old-fold all"
                f" --policy uniform --clients-per-round {CLIENTS_PER_ROUND}"
                f" --sampler-seed {seed} --track-clients"
                f" --rounds {ROUNDS} --epochs {LOCAL_EPOCHS} --batch-size 64"
                f" --eval-batch-size 256 --save-final-model --seed {fold}"
                f" --outer-workers 2 --inner-workers 1"
            )
    return lines


def centralized_lines() -> list[str]:
    """One model over the pooled cohort - the ceiling with privacy switched off."""
    return [
        f"foa isolated-train --results-dir {ROOT} {SETTING} --pooled"
        f" --clients-file {COHORT_FILE}"
        f" --fold-book {COHORT_BOOK} --fold {fold}"
        f" --old-book {OLD_BOOK} --old-fold all --old-clients-file {OLD_FILE}"
        f" {_init_flags(init)}"
        f" --epochs 100 {CONVERGE} --batch-size 64 --eval-batch-size 256"
        f" --out {ROOT}/centralized/centralized_outliers_{init}_fold{fold}.json"
        for init in INITS
        for fold in FOLDS
    ]


COMMON = [
    "# INDEPENDENT OF THE OTHER TWO. This file reads only artefacts that already",
    "# exist - the cohort, its fold book, the old book and g-0 - and writes only",
    "# its own outputs. P08, P09 and P10 can be submitted at the same time.",
    "#",
    "# THE THREE EVALUATIONS every run reports:",
    "#   own / clients   the cohort's fold-k TEST rows",
    "#   preservation    EVERY fold of the old book, separately (--old-fold all),",
    "#                   reported as five accuracies with their mean and spread.",
    "#                   Five partitions of one population, so the spread across",
    "#                   them is the error bar - merging them would throw it away.",
    "#",
]


def isolated_header(cohort, tasks) -> list[str]:
    return [
        f"# d01_p08.txt - {STUDY}, P08: ISOLATED training.",
        "#",
        "# The arrangement the whole study argues against: each cohort writer",
        "# trains a model of its own, on its own data, sharing nothing. The point",
        "# is to price it, and the price only shows up because each private model",
        "# is scored three ways:",
        "#",
        "#   own    the writer's own fold-k test rows - what isolation WINS on.",
        "#   union  every cohort writer's fold-k test rows - what isolation costs",
        "#          the group. Ten models that are each good at one writer are,",
        "#          collectively, ten models that are bad at nine.",
        "#   old    the old data's five folds - what the private model still knows",
        "#          about the population the shipped model was built for.",
        "#",
        "# TWO STARTING POINTS. --init global starts every private model from the",
        f"# shipped g-0 (winner fold {G0_FOLD}); --init scratch starts from a fresh one. The",
        "# pair separates what the federation learns from what it preserves.",
        "#",
        f"# {len(cohort)} writers x {len(FOLDS)} folds x {len(INITS)} inits = {len(tasks)} tasks.",
        "#",
        "# No checkpoints are written: each model is evaluated in memory and",
        "# dropped. A hundred checkpoints nobody would load again is not an",
        "# artefact, it is an inode bill.",
        "#",
    ] + COMMON + [
        "#   J=$FOA_PROJECT_DIR/jobs/v4",
        f"#   sbatch --account=$FOA_ACCOUNT --array=1-{len(tasks)}%20 \\",
        "#          $J/study_phase.sbatch $J/d01_p08.txt",
        "#",
    ]


def fl_header(tasks) -> list[str]:
    return [
        f"# d01_p09.txt - {STUDY}, P09: NORMAL FEDERATED LEARNING.",
        "#",
        "# The baseline the rest of the study is argued against. Plain FedAvg, no",
        "# regulariser, no anchor - just the ten cohort writers federating among",
        "# themselves, and what that does to a model that already knows the old",
        "# population.",
        "#",
        f"# THE FEDERATION. Ten clients enrolled, {CLIENTS_PER_ROUND} drawn per round - this study's",
        f"# 10% dropout - by a sampler seeded per line, so a fold's participation",
        "# pattern is reproducible and no two lines share one. E=5, batch 64, 100",
        "# rounds.",
        "#",
        "# BOTH FAMILIES PER TASK. --aggregation fedavg runs the parallel schedule",
        "# and the cyclic one in a single job, and skips the two families whose",
        "# rule is the same update written the other way round.",
        "#",
        "# TWO INITS. --init global starts from the shipped g-0; --init scratch",
        "# from a fresh model. --global-name g0 is carried by both so the",
        "# provenance names the same artefact either way.",
        "#",
        f"# {len(INITS)} inits x {len(FOLDS)} folds = {len(tasks)} tasks.",
        "#",
        "# --save-final-model keeps each family's server model for the stages that",
        "# come after. --track-clients keeps the per-client series.",
        "#",
    ] + COMMON + [
        "#   J=$FOA_PROJECT_DIR/jobs/v4",
        f"#   sbatch --account=$FOA_ACCOUNT --array=1-{len(tasks)} \\",
        "#          $J/study_phase.sbatch $J/d01_p09.txt",
        "#",
    ]


def centralized_header(tasks) -> list[str]:
    return [
        f"# d01_p10.txt - {STUDY}, P10: CENTRALIZED ON THE OUTLIERS.",
        "#",
        "# The rung between P08 and P09. P08 trains ten private models sharing",
        "# nothing; P09 federates ten clients that share gradients but never data.",
        "# This trains ONE model on the union of all ten writers' data, with the",
        "# privacy constraint simply switched off.",
        "#",
        "# So it is the ceiling the federated arm is trying to reach without moving",
        "# anything: the gap to P09 is the price of the constraint, and the gap to",
        "# P08 is what pooling buys over isolation.",
        "#",
        f"# The training set is the union of the cohort's fold-k train rows ({COHORT_TRAIN}",
        "# images); the early-stopping set is the union of their fold-k validation",
        "# rows. One convergence rule applied once to one model, identical to P08's,",
        "# so the three rungs differ in the arrangement and not in the stopping",
        "# rule.",
        "#",
        f"# {len(INITS)} inits x {len(FOLDS)} folds = {len(tasks)} tasks.",
        "#",
    ] + COMMON + [
        "#   J=$FOA_PROJECT_DIR/jobs/v4",
        f"#   sbatch --account=$FOA_ACCOUNT --array=1-{len(tasks)} \\",
        "#          $J/study_phase.sbatch $J/d01_p10.txt",
        "#",
    ]


def readme(cohort, iso, fl, cen) -> str:
    total = len(iso) + len(fl) + len(cen)
    return f"""# {STUDY} - P08, P09, P10: three arrangements of the same ten writers

| stage | file | what | tasks | GPU-h |
|---|---|---|---|---|
| P08 | `d01_p08.txt` | isolated: ten private models | {len(iso)} | ~1.2 |
| P09 | `d01_p09.txt` | federated: one server model, {CLIENTS_PER_ROUND} of 10 per round | {len(fl)} | ~1.2 |
| P10 | `d01_p10.txt` | centralized: one model over the pooled cohort | {len(cen)} | ~0.13 |
| | | **total** | **{total}** | **~2.5** |

## Submit all three at once

**There are no inter-stage dependencies.** Each file reads only artefacts that
already exist - `cohort_worst10.json`, `cohort10.foldbook.npz`,
`old_data.foldbook.npz`, `old_data.json`, `g0_model` - and writes only its own
outputs. Nothing here produces an input for anything else here.

```bash
J=$FOA_PROJECT_DIR/jobs/v4
A="sbatch --account=$FOA_ACCOUNT"
SB=$J/study_phase.sbatch

$A --array=1-{len(iso)}%20 $SB $J/d01_p08.txt
$A --array=1-{len(fl)}     $SB $J/d01_p09.txt
$A --array=1-{len(cen)}     $SB $J/d01_p10.txt
```

No exports needed: `study_phase.sbatch` defaults `FOA_STUDY_DIR` to
`studies/{STUDY}` and `FOA_NIST_CLASSES` to `digits`, and refuses any line whose
paths point elsewhere.

## The ladder

**P08 isolated** - each writer trains alone, sharing nothing. The point is to
price that: ten models each good at one writer are, collectively, ten models bad
at nine. Scored three ways - its own test rows (what isolation wins on), the
whole cohort's test rows (what it costs the group), and the old data.

**P09 federated** - ten clients enrolled, {CLIENTS_PER_ROUND} drawn per round,
E=5, batch 64, 100 rounds, both families per task.

**P10 centralized** - one model over the union of all ten writers' fold-k train
rows ({COHORT_TRAIN} images), privacy switched off. The ceiling P09 is trying to
reach without moving data: the gap to P09 is the price of the constraint, the
gap to P08 is what pooling buys over isolation.

All three use the same convergence rule where they converge (patience 10, floor
20, ceiling 100), so the rungs differ in the arrangement and not in the stopping
rule.

## Preservation is five numbers, not one

Every run carries `--old-fold all`: the old book's five test partitions are
scored **separately** and reported as five accuracies with their mean and
standard deviation. They are five partitions of one population, so the spread
across them is the error bar on preservation; merging them into a single set
would report a precision that is not there.

## The data

| | rows |
|---|---|
| cohort, total | {COHORT_ROWS:,} |
| cohort train, per fold | {COHORT_TRAIN} |
| cohort test, per fold | ~{COHORT_TEST} |
| old test, per fold | ~{OLD_TEST:,} |

Both books report an empty `unsplittable` list, so every one of the ten writers
contributes to all three parts of all five folds.

The cohort ({len(cohort)} writers), read from the study's own
`outliers/cohort_worst10.json`:

{", ".join(f"`{w}`" for w in cohort)}

g-0 is the winner of fold {G0_FOLD}; the `--init global` runs of P08 and P10
point at `$FOA_STUDY_DIR/g0_model` and P09 carries `--global-name g0`.

## Cost

The old-data evaluation dominates the two isolated stages: {OLD_TEST:,} rows x 5
folds is about 26.5k images per task, against a few hundred for training. So a
P08 or P10 task is roughly 40-45 s of work plus interpreter and cache startup.

A P09 task is `0.31 + 0.0345 x {CLIENTS_PER_ROUND} x {LOCAL_EPOCHS}` = 1.86 s per round per
family; two families over 100 rounds is ~6.2 min, plus the per-round evaluations
and the final three-category one - about **7.2 min per task**.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-dir", required=True)
    parser.add_argument(
        "--cohort-file", default=None,
        help="Read the cohort from this JSON instead of the recorded list.",
    )
    args = parser.parse_args()

    cohort = list(COHORT)
    if args.cohort_file:
        payload = json.loads(Path(args.cohort_file).read_text())
        cohort = payload["clients"] if isinstance(payload, dict) else list(payload)

    jobs = Path(args.jobs_dir)
    jobs.mkdir(parents=True, exist_ok=True)

    iso, fl, cen = isolated_lines(cohort), fl_lines(), centralized_lines()
    (jobs / "d01_p08.txt").write_text(
        "\n".join(isolated_header(cohort, iso) + iso) + "\n"
    )
    (jobs / "d01_p09.txt").write_text("\n".join(fl_header(fl) + fl) + "\n")
    (jobs / "d01_p10.txt").write_text(
        "\n".join(centralized_header(cen) + cen) + "\n"
    )
    (jobs / "d01_p08_README.md").write_text(readme(cohort, iso, fl, cen))

    print(f"wrote {jobs}/d01_p08.txt: {len(iso)} tasks")
    print(f"wrote {jobs}/d01_p09.txt: {len(fl)} tasks")
    print(f"wrote {jobs}/d01_p10.txt: {len(cen)} tasks")
    print(f"wrote {jobs}/d01_p08_README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
