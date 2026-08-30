"""
Emit P02 of the digits study: the detector, and the paper trail behind it.

    python tools/make_digits_p02.py --jobs-dir $FOA_PROJECT_DIR/jobs/v4

Writes two text files and nothing else: a login-node job.

P02 is the digits analogue of the 62-class study's stages 1-3. It builds the
all-writers digit fold book, trains g-init on the whole digit dataset five times
over that book, picks the winner, and scores every writer on its own held-out
digit rows. That ranking is the **detector**: every later stage cuts its cohort
from it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

STUDY = "Digits_study01"
ROOT = "$FOA_STUDY_DIR"
BOOKS = f"{ROOT}/fold_books"
POOLS = f"{ROOT}/outliers"

#: Digits only, and a ten-unit head - McMahan's network exactly.
SETTING = "--resolution 28 --classes digits --model fedavg_cnn"
CONVERGE = "--early-stopping-patience 10 --min-epochs 20"
BOOK = f"{BOOKS}/all_writers_digits.foldbook.npz"
FOLDS = (1, 2, 3, 4, 5)

#: The detector trains on one fold. See the note beside task 3.
DETECTOR_FOLD = 1

#: Measured from the cache: writers holding at least one digit, and their rows.
DIGIT_WRITERS = 3580
DIGIT_ROWS = 402953
#: Writers of the 62-class cache that hold no digit at all.
NO_DIGIT_WRITERS = 17
ALL_WRITERS = 3597


def lines() -> list[str]:
    tasks = [
        # 1. the paper trail. Independent of everything else - it reads the
        #    cache and counts - so it runs first and is available while the
        #    folds train.
        f"foa writer-counts --results-dir {ROOT} --resolution 28 --classes digits"
        f" --out {POOLS}/writer_counts.json --csv {POOLS}/writer_counts.csv",
        # 2. the book every later stage reads its splits from.
        f"foa fold-book --results-dir {ROOT} --resolution 28 --classes digits"
        f" --out {BOOKS}/all_writers_digits --folds 5 --seed 42"
        f" --train-rate 0.6 --eval-rate 0.2 --tag digits_all_writers",
    ]
    # 3. g-init, on ONE fold of the whole digit dataset.
    #
    #    THE DETECTOR IS NOT AN EXPERIMENT. It exists to draw one coarse line:
    #    the worst DETECTOR_CUT of writers become the bad pool. The study's ten
    #    clients are cut from that pool later, by g-0, so g-init only has to put
    #    the badly served writers *inside a net of about a thousand* - it does
    #    not have to rank them precisely.
    #
    #    Measured, and this is why one fold is enough: re-ranking with a whole
    #    fold dropped - a far larger perturbation than retraining the same fold
    #    - the net still contained 100% of the reference worst-fifty, at every
    #    width tried. The pool's boundary does move, but the writers who move
    #    are its least bad members, and g-0 would never select them.
    #
    #    One fold also removes a defect rather than managing it. With five, a
    #    fold had to be crowned, and the five sat within six ten-thousandths of
    #    a point of one another - so floating-point accumulation order chose the
    #    study's clients. A rebuild crowned a different fold and re-derived a
    #    cohort sharing three writers of ten with the published one. With one
    #    fold there is nothing to crown.
    #
    #    The split itself is not a seed that gets re-derived: the fold book
    #    stores the per-row assignment, so every later stage reads the same
    #    rows this model held out.
    tasks.append(
        f"foa global-train --results-dir {ROOT}/ginit_fold{DETECTOR_FOLD} {SETTING}"
        f" --population all --fold-book {BOOK} --fold {DETECTOR_FOLD}"
        f" --epochs 100 {CONVERGE} --batch-size 64 --split-seed 42"
        f" --seed {DETECTOR_FOLD}"
    )
    # 4. the detector: every writer on the rows this model did not train on.
    #
    #    Every writer contributed training images to g-init, so scoring a writer
    #    on all of its data would score the model against its own training set,
    #    and the ranking would measure memorisation rather than difficulty. The
    #    book is what still knows which rows fold DETECTOR_FOLD held out.
    tasks.append(
        f"foa score-writers --results-dir {ROOT} --resolution 28 --classes digits"
        f" --model-path {ROOT}/ginit_fold{DETECTOR_FOLD}/global_model"
        f" --fold-book {BOOK} --fold {DETECTOR_FOLD} --batch-size 256"
        f" --out {POOLS}/writer_scores.json"
        f" --accuracies-name clients_acc_on_global.json"
    )
    return tasks


def header(tasks: list[str]) -> list[str]:
    return [
        f"# d01_p02.txt - {STUDY}, P02: the detector.",
        "#",
        "# NIST SD19, DIGITS 0-9 ONLY. --classes digits gives the dataset ten",
        "# labels and the provider hands FedAvgCNN num_classes=10, so the head is",
        "# McMahan's exactly rather than the 62-unit by-class head.",
        "#",
        "# WHO IS IN THE BOOK, AND THE RULE.",
        "#",
        f"# The cache holds {ALL_WRITERS} writers over the full character set. Restricted to",
        f"# digits, {DIGIT_WRITERS} of them remain: {NO_DIGIT_WRITERS} writers contribute no digit at all and",
        "# are absent from the dataset entirely rather than present with a zero.",
        f"# Those {DIGIT_WRITERS} writers hold {DIGIT_ROWS:,} digit images between them.",
        "#",
        "# The splittability rule is the book's own: a writer is splittable when a",
        "# per-label stratified 60/20/20 leaves it at least one training row AND",
        "# at least one validation row. Counted against the cache:",
        "#",
        f"#   digits    {DIGIT_WRITERS} of {DIGIT_WRITERS} survive - NONE are dropped",
        "#   (62-class  3588 of 3597 survive - 9 are dropped)",
        "#",
        "# The digit filter is a no-op, and that is not luck. Fewer classes means",
        "# more rows per class: a digit writer holds ~113 rows over 10 labels,",
        "# about 11 each, where the same writer holds ~226 over 62, about 3.6",
        "# each. int(3.6 * 0.6) can be zero; int(11 * 0.6) cannot. The book is",
        "# therefore built over every writer, and it records any unsplittable",
        "# writer per fold in its own metadata - which here will be empty.",
        "#",
        "# ONE FOLD, AND WHY THAT IS ENOUGH.",
        "#",
        "# The detector is not an experiment. It draws one coarse line: the",
        "# worst writers become the bad pool. The study's ten clients are cut",
        "# from that pool later, by g-0, so g-init only has to put the badly",
        "# served writers inside a net of about a thousand - it does not have to",
        "# rank them precisely.",
        "#",
        "# Measured: re-ranking with a whole fold dropped, which is a far larger",
        "# perturbation than retraining the same fold, the net still contained",
        "# 100% of the reference worst-fifty at every width tried. The pool's",
        "# boundary moves, but the writers who move are its least bad members,",
        "# and g-0 would never select them.",
        "#",
        "# One fold also removes a defect instead of managing it. With five, one",
        "# had to be crowned, and the five sat within six ten-thousandths of a",
        "# point of one another - so the order in which floating-point error",
        "# accumulates chose the study's clients. A rebuild crowned fold 4 where",
        "# the first run crowned fold 3 and re-derived a cohort sharing three",
        "# writers of ten with the published one. With one fold there is nothing",
        "# to crown, and no fold number to substitute anywhere in the chain.",
        "#",
        "# THE EXPERIMENTS ARE STILL FIVE-FOLD. This is the detector only. Every",
        "# adaptation and preservation number in the study is a mean over the",
        "# five folds of this same book, with fold-matched tests against FedAvg.",
        "#",
        "# WHY THE SCORING USES HELD-OUT ROWS. g-init trained on the whole digit",
        "# dataset, so every writer contributed training images to it. Scoring a",
        "# writer on all of its data would be scoring the model on its own",
        "# training set, and the ranking would measure memorisation rather than",
        "# difficulty. Each writer is scored on its validation+test rows of the",
        "# detector's fold - exactly the rows that model did not train on. The",
        "# book stores that assignment, so it is not re-derived from a seed: the",
        "# rows are the same rows for every later stage that reads the book.",
        "#",
        "# THE BANKING STEP (line 1) is independent of everything: it reads the",
        "# cache and counts. It writes each writer's digit total and its per-class",
        "# counts, because a reader will ask whether the worst writers were simply",
        "# the ones with the least data - and a ranking alone cannot answer that.",
        "#",
        f"# {len(tasks)} tasks. The chain is: counts+book -> the five folds -> the winner",
        "# -> the scores.",
        "#",
    ]


def readme(tasks: list[str]) -> str:
    return f"""# {STUDY} - P02: the detector

NIST SD19, **digits 0-9 only**. `--classes digits` gives the dataset ten labels
and the provider hands `FedAvgCNN` `num_classes=10`, so the head is McMahan's
exactly.

## Who is in the book

| | writers | rows |
|---|---|---|
| full character set | {ALL_WRITERS} | 814,255 |
| **digits only** | **{DIGIT_WRITERS}** | **{DIGIT_ROWS:,}** |

{NO_DIGIT_WRITERS} writers contribute no digit at all and are absent from the
digit dataset entirely rather than present with a zero.

**Splittability rule:** a writer is splittable when a per-label stratified
60/20/20 leaves it at least one training row *and* at least one validation row.
Counted against the cache:

| class set | splittable | dropped |
|---|---|---|
| digits | **{DIGIT_WRITERS} of {DIGIT_WRITERS}** | **0** |
| 62-class (for contrast) | 3588 of 3597 | 9 |

The filter is a no-op on digits, and not by luck: fewer classes means more rows
per class. A digit writer holds ~113 rows over 10 labels, about 11 each, where
the same writer holds ~226 over 62 labels, about 3.6 each. `int(3.6 * 0.6)` can
be zero; `int(11 * 0.6)` cannot. The book is built over every writer and records
any unsplittable writer per fold in its own metadata - which here will be empty,
and that emptiness is the check.

## Order

```bash
J=$FOA_PROJECT_DIR/jobs/v4
A="sbatch --account=$FOA_ACCOUNT"
SB=$J/study_phase.sbatch

# study_phase.sbatch defaults FOA_STUDY_DIR to studies/{STUDY} and
# FOA_NIST_CLASSES to digits, and REFUSES to run if that folder is missing.
# No export is needed unless you are running a different study.

# 1-2. the paper trail and the book (a chain: nothing else can start first)
$A --array=1-2%1 $SB $J/d01_p02.txt

# 3. g-init, on the detector's single fold
$A --array=3 $SB $J/d01_p02.txt

# 4. the detector: every writer on the rows that model did not train on.
#    Nothing is exported and no fold is substituted - see the task file.
$A --array=4 $SB $J/d01_p02.txt
```

## Cost

| lines | what | per task | total |
|---|---|---|---|
| 1 | writer counts | ~1 min, CPU-bound | - |
| 2 | the fold book | ~2 min, CPU-bound | - |
| 3-7 | g-init, one fold each | **~25-35 min** | **~2.5 GPU-h** |
| 8 | select the winner | seconds | - |
| 9 | score {DIGIT_WRITERS} writers | ~3 min | - |

**~2.5-3 GPU-h in total.** Each g-init fold trains on about 242k digit images
and validates on about 80k, so an epoch is roughly 322k images. The convergence
rule (patience 10, floor 20, ceiling 100) normally stops in the twenties. The
sbatch's 4-hour default covers the 100-epoch ceiling with room to spare.

## What P02 writes

```
studies/{STUDY}/
  fold_books/all_writers_digits.foldbook.npz
  ginit_fold1/                        the detector and its metrics
  outliers/writer_scores.json         the ranking
  outliers/clients_acc_on_global.json the flat ranking every later stage reads
  outliers/writer_counts.json/.csv    per-writer totals and per-class counts
```

`ginit_fold1/` carries the detector's validation and test accuracies with
their mean and spread - the paper's "centralized max" row.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-dir", required=True)
    args = parser.parse_args()
    jobs = Path(args.jobs_dir)
    jobs.mkdir(parents=True, exist_ok=True)

    tasks = lines()
    (jobs / "d01_p02.txt").write_text("\n".join(header(tasks) + tasks) + "\n")
    (jobs / "d01_p02_README.md").write_text(readme(tasks))
    print(f"wrote {jobs}/d01_p02.txt: {len(tasks)} tasks")
    print(f"wrote {jobs}/d01_p02_README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
