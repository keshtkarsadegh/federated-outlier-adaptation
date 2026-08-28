"""
Emit the P20 setup chain: the twenty-client cohort, its book, and its baseline.

    python tools/make_digits_p20.py --jobs-dir $FOA_PROJECT_DIR/jobs/v4

Writes one text file: a login-node job.

Why a chain and not a fan-out
-----------------------------
Each step reads what the one before it wrote - the book covers the cohort, the
baseline is scored on the book - so the file is submitted ``%1``.  A fan-out
would race the book against the evaluations and produce either a crash or, far
worse, a baseline scored on whatever book happened to be on disk.

Why the cohort is cut rather than written
-----------------------------------------
``cohort_worst20`` comes out of ``select-outliers`` over the same ranking, with
the same eligibility rule, that produced ``cohort_worst10``.  Cutting it that
way is what makes the twenty-client point the ten-client point *scaled up*
rather than a different population: the first ten writers must come back
identical, and step 2 stops the chain if they do not.  Restating the list here
would remove exactly the check that matters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

STUDY = "Digits_study01"
ROOT = "$FOA_STUDY_DIR"
BOOKS = f"{ROOT}/fold_books"
POOLS = f"{ROOT}/outliers"
SETTING = "--resolution 28 --classes digits"
FOLDS = (1, 2, 3, 4, 5)

BAD_FLAT = f"{POOLS}/bad_acc_on_g0.json"
COHORT10 = f"{POOLS}/cohort_worst10.json"
COHORT20 = f"{POOLS}/cohort_worst20.json"
BOOK20 = f"{BOOKS}/cohort20.foldbook.npz"
G0 = f"{ROOT}/g0_model"
BASELINE = f"{ROOT}/g0_cohort20_evaluations.json"
RUNS = "$FOA_PROJECT_DIR/jobs/v4/d01_p20_runs.txt"
REGEN = f"{ROOT}/tables/p20_runs_regenerated.txt"

#: 3 configurations x 5 folds x 2 participation points.
RUN_TASKS = 30

COHORT_SIZE = 20
#: The book rule every book in this study is cut by, restated nowhere else.
BOOK_FOLDS, BOOK_SEED, TRAIN_RATE, EVAL_RATE = 5, 42, 0.6, 0.2


def extends_check() -> str:
    """Stop unless the worst-20 is the worst-10 plus the next ten."""
    return (
        'python3 -c "import json;'
        f" a=json.load(open('{COHORT20}'));"
        f" b=json.load(open('{COHORT10}'));"
        " a=a['clients'] if isinstance(a,dict) else a;"
        " b=b['clients'] if isinstance(b,dict) else b;"
        " assert len(a)==20, ('size', len(a));"
        " assert a[:10]==b, ('first ten differ', a[:10], b);"
        " assert len(set(a))==20, 'duplicate writer';"
        " print('cohort20 extends cohort10; next ten:', a[10:])\""
    )


def splittable_check() -> str:
    """Stop unless every cohort writer has rows in every part of every fold."""
    return (
        'python3 -c "import json;'
        " from federated_outlier_adaptation.data.fold_book import FoldBook;"
        f" k=FoldBook.load('{BOOK20}');"
        f" a=json.load(open('{COHORT20}'));"
        " a=a['clients'] if isinstance(a,dict) else a;"
        " miss=[w for w in a if not k.covers(w)];"
        " assert not miss, ('uncovered', miss);"
        " thin=[(w,f,p) for w in a for f in (1,2,3,4,5)"
        "   for p in ('train','val','test') if not k.part(f,w,p)];"
        " assert not thin, ('empty partition', thin[:5]);"
        " print('cohort20 book: 20 writers x 5 folds x 3 parts, none empty')\""
    )


def regenerate_check() -> str:
    """
    Re-emit the runs file from the real artefacts and refuse any difference.

    The deployed ``d01_p20_runs.txt`` was written before this chain ran, against
    a cohort derived the same way and a book that did not exist yet.  That is
    reviewable but not yet *verified*: only the real cohort and the real book can
    say the file addresses them.  So the chain regenerates it and stops on any
    difference, rather than letting an array run against lines nothing checked.
    """
    # --out has to land inside the study root: the sbatch guard refuses any
    # path-valued flag that does not, and a temp file would be refused too.
    return (
        f"python3 $FOA_PROJECT_DIR/repo_foa/tools/study_emit.py c20"
        f" --study {STUDY} --root {ROOT} --out {REGEN} --expect {RUN_TASKS}"
        f" && diff -u {RUNS} {REGEN}"
        ' && echo "d01_p20_runs.txt verified against the real cohort and book"'
        ' || { echo "REFUSED: the deployed runs file is not what the real"'
        ' "cohort and book produce; do not submit it." >&2; exit 1; }'
    )


def lines() -> list[str]:
    return [
        # 1. the cohort: the worst twenty of the same g-0 ranking, same rule.
        f"foa select-outliers --results-dir {ROOT} {SETTING} --mode worst"
        f" --k {COHORT_SIZE} --scores {BAD_FLAT} --require-trainable"
        f" --tag digits_cohort{COHORT_SIZE}_v2 --force --no-accuracy-table",
        # 2. and it must extend the ten, or the two sizes are not comparable.
        extends_check(),
        # 3. its book, by the rule every book in this study is cut by.
        f"foa fold-book --results-dir {ROOT} {SETTING}"
        f" --out {BOOKS}/cohort{COHORT_SIZE} --clients-file {COHORT20}"
        f" --folds {BOOK_FOLDS} --seed {BOOK_SEED} --train-rate {TRAIN_RATE}"
        f" --eval-rate {EVAL_RATE} --tag digits_cohort{COHORT_SIZE}_v2",
        # 4. every writer splittable in every fold, or a run would train on air.
        splittable_check(),
    ] + [
        # 5-9. the do-nothing row: the shipped model on the cohort's test rows.
        f"foa evaluate-book --results-dir {ROOT} {SETTING} --model-path {G0}"
        f" --fold-book {BOOK20} --fold {fold} --part test"
        f" --clients-file {COHORT20} --batch-size 256"
        f" --tag cohort20_fold{fold} --out {BASELINE}"
        for fold in FOLDS
    ] + [
        # 10. the runs file, verified against what actually exists.
        regenerate_check(),
    ]


def header(tasks: list[str]) -> list[str]:
    return [
        "# GENERATED, AND NOT AUTHORISED TO RUN.",
        "#",
        "# P20 SETUP: the twenty-client cohort, its fold book, and the",
        "# do-nothing baseline the tables need.",
        "#",
        "# SUBMIT %1. Every step reads what the one before it wrote - the book",
        "# covers the cohort, the baseline is scored on the book - so a fan-out",
        "# would race them and could score a baseline on whichever book",
        "# happened to be on disk.",
        "#",
        "#   1  cohort_worst20.json   the worst 20 of the SAME g-0 ranking and",
        "#                            the SAME eligibility rule that cut the 10",
        "#   2  CHECK                 the first ten are cohort_worst10 exactly,",
        "#                            or the two sizes are different populations",
        "#                            and nothing across them is comparable",
        f"#   3  cohort20.foldbook.npz per-writer stratified {TRAIN_RATE:.0%}/"
        f"{EVAL_RATE:.0%}/{1 - TRAIN_RATE - EVAL_RATE:.0%} over",
        f"#                            {BOOK_FOLDS} folds at seed {BOOK_SEED} -"
        " the rule every book",
        "#                            in this study is cut by",
        "#   4  CHECK                 all 20 covered, and no empty partition in",
        "#                            any fold: a writer with no train rows",
        "#                            would train on nothing and still report",
        "#   5-9 g0_cohort20_evaluations.json  the shipped model on each fold's",
        "#                            test rows, per-writer and pooled: the",
        "#                            do-nothing row every table is read against",
        "#  10  CHECK                 re-emit d01_p20_runs.txt (%d tasks) from" % RUN_TASKS,
        "#                            the real cohort and book, and refuse any",
        "#                            difference",
        "#",
        f"# {len(tasks)} tasks. Steps 1-4 and 10 are CPU; 5-9 score one saved",
        "# model on ~20 writers' test rows and take under a minute each.",
        "#",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-dir", required=True)
    parser.add_argument("--expect", type=int, default=10)
    args = parser.parse_args()

    tasks = lines()
    if len(tasks) != args.expect:
        print(f"FATAL: {len(tasks)} tasks, expected {args.expect}.")
        return 1
    out = Path(args.jobs_dir) / "d01_p20_setup.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header(tasks) + tasks) + "\n")
    print(f"wrote {out} with {len(tasks)} task lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
