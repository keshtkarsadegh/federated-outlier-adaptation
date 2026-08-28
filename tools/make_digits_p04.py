"""
Emit P04-P06 of the digits study: the two populations, their books, the figure.

    python tools/make_digits_p04.py --jobs-dir $FOA_PROJECT_DIR/jobs/v4

Writes two text files: a login-node job.

P02 produced the detector - g-init and a score for every writer. P04 cuts the
cohort from that ranking, draws the old data around it, writes the two fold
books every later stage reads, and banks the evidence a reader needs to argue
with the selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

STUDY = "Digits_study01"
ROOT = "$FOA_STUDY_DIR"
BOOKS = f"{ROOT}/fold_books"
POOLS = f"{ROOT}/outliers"
TABLES = f"{ROOT}/tables"
SETTING = "--resolution 28 --classes digits"

COHORT_SIZE = 10
OLD_SIZE = 200
#: Study 2's own seeds, logged here so the draws are reproducible from this file.
OLD_SEED = 20260827
TYPICAL_SEED = 20260828
#: Writers of the reference population drawn for the qualitative figure.
TYPICAL_SIZE = 5
#: Minimum digit rows for the old-data draw - the 62-class study's rule, kept.
OLD_MIN_SAMPLES = 100

COHORT_FILE = f"{POOLS}/cohort_worst{COHORT_SIZE}.json"
OLD_FILE = f"{POOLS}/old_data.json"
TYPICAL_FILE = f"{POOLS}/typical{TYPICAL_SIZE}.json"
COHORT_BOOK = f"{BOOKS}/cohort{COHORT_SIZE}.foldbook.npz"

#: Measured from P02's artefacts, for the header and the cost note.
SCORED_WRITERS = 3580
DIGIT_ROWS = 402953
ELIGIBLE_100 = 2943
PERFECT_WRITERS = 2999
MEDIAN_ROWS = 117
GINIT_FOLD = 3


def lines() -> list[str]:
    return [
        # 1. the cohort: the ten worst under the detector.
        f"foa select-outliers --results-dir {ROOT} {SETTING} --mode worst"
        f" --k {COHORT_SIZE} --require-trainable --tag digits_cohort{COHORT_SIZE} --force",
        # 2. the old data: a draw, not a ranking, around the cohort.
        f"foa draw-old-data --results-dir {ROOT} {SETTING} --size {OLD_SIZE}"
        f" --seed {OLD_SEED} --min-samples {OLD_MIN_SAMPLES} --require-trainable"
        f" --exclude-file {COHORT_FILE} --out {OLD_FILE}",
        # 3-4. the two books every later stage reads its splits from.
        f"foa fold-book --results-dir {ROOT} {SETTING}"
        f" --out {BOOKS}/cohort{COHORT_SIZE} --clients-file {COHORT_FILE}"
        f" --folds 5 --seed 42 --train-rate 0.6 --eval-rate 0.2"
        f" --tag digits_cohort{COHORT_SIZE}",
        f"foa fold-book --results-dir {ROOT} {SETTING} --out {BOOKS}/old_data"
        f" --clients-file {OLD_FILE} --folds 5 --seed 42 --train-rate 0.6"
        f" --eval-rate 0.2 --tag digits_old{OLD_SIZE}",
        # 5. the paper's cohort table - after the book, because it reports the
        #    per-fold split sizes and those do not exist until the book does.
        f"foa cohort-table --results-dir {ROOT} {SETTING}"
        f" --clients-file {COHORT_FILE} --fold-book {COHORT_BOOK}"
        f" --out {TABLES}/cohort_table.json --csv {TABLES}/cohort_table.csv",
        # 6. the reference population of the figure: typical writers, drawn.
        f"foa draw-cohort --results-dir {ROOT} {SETTING} --best"
        f" --k {TYPICAL_SIZE} --seed {TYPICAL_SEED} --require-trainable"
        f" --exclude-file {COHORT_FILE} --tag digits_typical{TYPICAL_SIZE}"
        f" --out {TYPICAL_FILE}",
        # 7. the figure itself: ten digits wide, three glyphs per writer.
        f"foa outlier-figure --results-dir {ROOT} {SETTING}"
        f" --outliers-file {COHORT_FILE} --source-clients-file {TYPICAL_FILE}"
        f" --n-examples 3 --max-columns 10 --batch-size 256"
        f" --out {TABLES}/gallery/cohort_vs_typical.png",
    ]


def header(tasks: list[str]) -> list[str]:
    return [
        f"# d01_p04.txt - {STUDY}, P04-P06: the populations, the books, the figure.",
        "#",
        "# P02 produced the detector: g-init (winner fold {}) and a score for every".format(GINIT_FOLD),
        f"# one of the {SCORED_WRITERS:,} digit writers. This file cuts the cohort from that",
        "# ranking, draws the old data around it, writes the two books, and banks",
        "# the evidence a reader needs to argue with the selection.",
        "#",
        "# THE SCORES SELECT; THEY DO NOT BASELINE. g-init exists to rank writers",
        "# and for nothing else. Every number this study reports is measured",
        "# against g-0, which P07 trains on the old data drawn here.",
        "#",
        f"# LINE 1 cuts the cohort: the {COHORT_SIZE} lowest-scoring writers that can form a",
        "# local training split. --require-trainable is applied before the cut;",
        "# on digits it removes nobody, because every one of the 3,580 writers",
        "# splits (see d01_p02_README.md), but the flag stays so the rule is on",
        "# the command line rather than in a footnote.",
        "#",
        f"# LINE 2 draws the old data: {OLD_SIZE} splittable writers holding at least",
        f"# {OLD_MIN_SAMPLES} digit rows, seeded {OLD_SEED}. A DRAW, not a ranking - taking the",
        "# best or the largest writers would build the preservation reference out",
        f"# of an unrepresentative sample. {ELIGIBLE_100:,} writers clear the {OLD_MIN_SAMPLES}-row bar",
        f"# (median holding is {MEDIAN_ROWS} digit rows), so the draw has room. The cohort is",
        "# excluded: no writer is both an outlier and old data.",
        "#",
        "# LINES 3-4 write the two books: per-writer stratified 60/20/20 over 5",
        "# folds, recorded as row indices, so one fold means the same rows in",
        "# every process that ever opens them.",
        "#",
        "# LINE 5 IS AFTER THE BOOK ON PURPOSE. The cohort table reports each",
        "# writer's per-fold train/val/test sizes, and those do not exist until",
        "# the book does. Running it earlier would silently drop the column that",
        "# makes the table worth banking.",
        "#",
        "# The table answers the two objections a ranking invites on its own:",
        "# were these simply the writers with the least data, and were they the",
        "# ones missing the hard classes? Each row carries the accuracy that",
        "# selected the writer, its rank out of all",
        f"# {SCORED_WRITERS:,}, its digit rows, its per-class counts and its fold splits.",
        "#",
        f"# LINE 6 draws the figure's REFERENCE population: {TYPICAL_SIZE} writers from the",
        f"# TOP of the ranking, seeded {TYPICAL_SEED}. {PERFECT_WRITERS:,} writers score exactly 1.0, so",
        "# 'typical' is the overwhelming majority here and a seeded draw from it",
        "# is well defined. The cohort is excluded so the two rows of the figure",
        "# cannot share a writer.",
        "#",
        "# LINE 7 renders it: ten columns (every digit), three glyphs per writer,",
        "# the reference population's mean digit on top. That top row used to come",
        "# from the legacy writer_split.json, which this study never writes - the",
        "# reference is now named on the command line and recorded in the figure's",
        "# JSON sidecar, so what counts as 'typical' is stated rather than assumed.",
        "#",
        f"# {len(tasks)} tasks, all CPU, minutes each. A CHAIN: every line reads what an",
        "# earlier one wrote.",
        "#",
    ]


def readme(tasks: list[str]) -> str:
    return f"""# {STUDY} - P04-P06: populations, books, figure

P02 produced the **detector**: g-init (winner fold {GINIT_FOLD}) and a score for
each of the {SCORED_WRITERS:,} digit writers. Those scores **select**; they do
not baseline. Every number this study reports is measured against g-0, which
P07 trains on the old data drawn here.

## The chain

| # | task | writes |
|---|---|---|
| 1 | cohort: the {COHORT_SIZE} worst splittable writers | `outliers/cohort_worst{COHORT_SIZE}.json` |
| 2 | old data: seeded draw of {OLD_SIZE}, cohort excluded | `outliers/old_data.json` |
| 3 | cohort fold book | `fold_books/cohort{COHORT_SIZE}.foldbook.npz` |
| 4 | old-data fold book | `fold_books/old_data.foldbook.npz` |
| 5 | the cohort table | `tables/cohort_table.json` + `.csv` |
| 6 | the figure's reference writers | `outliers/typical{TYPICAL_SIZE}.json` |
| 7 | the figure | `tables/gallery/cohort_vs_typical.png` + sidecar |

**Line 5 comes after the books deliberately.** The table reports each writer's
per-fold train/val/test sizes, and those do not exist until the book does.
Running it in the position the plan first suggested would have silently dropped
the column that makes the table worth banking.

## The draws

| draw | size | seed | pool |
|---|---|---|---|
| cohort | {COHORT_SIZE} | none - it is a **ranking**, not a draw | the {SCORED_WRITERS:,} scored writers |
| old data | {OLD_SIZE} | `{OLD_SEED}` | {ELIGIBLE_100:,} writers with >= {OLD_MIN_SAMPLES} digit rows, cohort excluded |
| typical (figure) | {TYPICAL_SIZE} | `{TYPICAL_SEED}` | the top of the ranking, cohort excluded |

The old-data population is a **draw and not a ranking**: taking the best or the
largest writers would build the preservation reference out of an
unrepresentative sample. The median writer holds {MEDIAN_ROWS} digit rows, so
the {OLD_MIN_SAMPLES}-row bar leaves {ELIGIBLE_100:,} eligible - ample for
{OLD_SIZE}.

{PERFECT_WRITERS:,} writers score exactly 1.0 under the detector, so "typical"
is the overwhelming majority and a seeded draw from the top is well defined.

## The cohort table

One row per cohort writer: the accuracy that selected it, **its rank out of all
{SCORED_WRITERS:,}** (not out of ten - "rank 3 of 3,580" is the statement worth
printing), its total digit rows, its per-class counts, and its per-fold
train/val/test sizes. It exists to answer the two objections a ranking invites
on its own: *were these simply the writers with the least data?* and *were they
the ones missing the hard classes?*

## The gallery, and the blocker that was not one

The figure's top row is the reference population's mean digit per class. That
row used to come from `provider.global_client_ids()`, which reads the legacy
`writer_split.json` - a file this study never writes, so the figure could not be
drawn at all.

**The repoint was one line.** Everything else in the renderer already goes
through `provider.build_dataset`, which is book-aware and class-aware. The
reference population is now an argument (`--source-clients-file`), it is
recorded in the figure's JSON sidecar, and the legacy split is still used when
it happens to exist - so the published 62-class figure is unchanged. Where
neither is available the error now says what to pass instead of failing on a
path the figure has no business knowing about.

Ten columns (every digit), three glyphs per cohort writer, one mean row on top.

## Order

```bash
J=$FOA_PROJECT_DIR/jobs/v4
A="sbatch --account=$FOA_ACCOUNT"
SB=$J/study_phase.sbatch

# study_phase.sbatch defaults FOA_STUDY_DIR to studies/{STUDY} and
# FOA_NIST_CLASSES to digits, and refuses to run if that folder is missing.
# No exports are needed.

# a chain - each line reads what the one before it wrote
$A --array=1-{len(tasks)}%1 $SB $J/d01_p04.txt
```

Everything here is CPU work: reading the cache, counting, writing indices and
one PNG. Minutes per task; the whole file is well under an hour and needs no
GPU. It is submitted through the same GPU sbatch for consistency of environment
and logging - if the queue is busy, `--partition=standard96:shared` would do,
but nothing here is worth a second runner.

## What P04-P06 write

```
studies/{STUDY}/
  outliers/cohort_worst{COHORT_SIZE}.json    the cohort, with its rule and accuracies
  outliers/old_data.json            the draw, with seed and eligibility rule
  outliers/typical{TYPICAL_SIZE}.json           the figure's reference writers
  fold_books/cohort{COHORT_SIZE}.foldbook.npz
  fold_books/old_data.foldbook.npz
  tables/cohort_table.json / .csv   the selection's paper trail
  tables/gallery/cohort_vs_typical.png + .json
```
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-dir", required=True)
    args = parser.parse_args()
    jobs = Path(args.jobs_dir)
    jobs.mkdir(parents=True, exist_ok=True)

    tasks = lines()
    (jobs / "d01_p04.txt").write_text("\n".join(header(tasks) + tasks) + "\n")
    (jobs / "d01_p04_README.md").write_text(readme(tasks))
    print(f"wrote {jobs}/d01_p04.txt: {len(tasks)} tasks")
    print(f"wrote {jobs}/d01_p04_README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
