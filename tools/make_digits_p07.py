"""
Emit P07 of the digits study: g-0, and the starting line every method is judged against.

    python tools/make_digits_p07.py --jobs-dir $FOA_PROJECT_DIR/jobs/v4

Writes two text files: a login-node job.

g-0 is the shipped model - trained on the two hundred old-data writers drawn in
P04, five-fold cross-validated, early-stopped on validation with the best
weights restored. Everything this study reports is measured against it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

STUDY = "Digits_study01"
ROOT = "$FOA_STUDY_DIR"
BOOKS = f"{ROOT}/fold_books"
POOLS = f"{ROOT}/outliers"
SETTING = "--resolution 28 --classes digits"
CONVERGE = "--early-stopping-patience 10 --min-epochs 20"

OLD_FILE = f"{POOLS}/old_data.json"
COHORT_FILE = f"{POOLS}/cohort_worst10.json"
OLD_BOOK = f"{BOOKS}/old_data.foldbook.npz"
COHORT_BOOK = f"{BOOKS}/cohort10.foldbook.npz"
G0 = f"{ROOT}/g0_model"
FOLDS = (1, 2, 3, 4, 5)

#: Measured from the books P04 wrote.
OLD_WRITERS = 200
OLD_ROWS = 23898
OLD_TRAIN = 13289
OLD_VAL = 5296
OLD_TEST = 5313
COHORT_TEST = 225
COHORT_SIZE = 10


def lines() -> list[str]:
    tasks = [
        f"foa global-train --results-dir {ROOT}/g0_fold{fold} {SETTING}"
        f" --model fedavg_cnn --population file --writers-file {OLD_FILE}"
        f" --fold-book {OLD_BOOK} --fold {fold}"
        f" --epochs 100 {CONVERGE} --batch-size 64 --split-seed 42 --seed {fold}"
        for fold in FOLDS
    ]
    tasks.append(
        f"foa select-fold --results-dir {ROOT} {SETTING} --root {ROOT}"
        f" --prefix g0 --name g0 --folds 1 2 3 4 5"
    )
    # preservation: the reference every later method is measured against
    tasks += [
        f"foa evaluate-book --results-dir {ROOT} {SETTING} --model-path {G0}"
        f" --fold-book {OLD_BOOK} --fold {fold} --part test"
        f" --clients-file {OLD_FILE} --batch-size 256"
        f" --tag old_data_fold{fold} --out {ROOT}/g0_evaluations.json"
        for fold in FOLDS
    ]
    # the adaptation gap: what the shipped model already does for each outlier
    tasks += [
        f"foa evaluate-book --results-dir {ROOT} {SETTING} --model-path {G0}"
        f" --fold-book {COHORT_BOOK} --fold {fold} --part test"
        f" --clients-file {COHORT_FILE} --batch-size 256"
        f" --tag cohort_fold{fold} --out {ROOT}/g0_perfold_evaluations.json"
        for fold in FOLDS
    ]
    return tasks


def header(tasks: list[str]) -> list[str]:
    return [
        f"# d01_p07.txt - {STUDY}, P07: g-0 and the starting line.",
        "#",
        "# g-0 is the SHIPPED model: trained on the 200 old-data writers P04 drew,",
        "# five-fold cross-validated on their book, early-stopped on validation",
        "# with the best weights restored. Every number this study reports is",
        "# measured against it. g-init is not a baseline and never was - it exists",
        "# to rank writers, and its job ended when the cohort was cut.",
        "#",
        f"# THE DATA. {OLD_WRITERS} writers, {OLD_ROWS:,} digit rows per fold:",
        f"# {OLD_TRAIN:,} train, ~{OLD_VAL:,} validation, ~{OLD_TEST:,} test. Both books P04 wrote",
        "# report an EMPTY unsplittable list, so every writer contributes to all",
        "# three parts of all five folds - the check P02 predicted, now confirmed.",
        "#",
        f"# LINES 1-{len(FOLDS)} train one fold each. --population file trains on exactly the",
        "# drawn writers; the fold book supplies their 60/20/20, so these folds are",
        "# the folds every later stage reuses.",
        "#",
        "# THE FISHER COMES FREE, AND IN THE RIGHT PLACE. global-train computes the",
        "# Fisher diagonal after training and writes it to",
        "# <results-dir>/global_results/fisher, so with --results-dir",
        "# $FOA_STUDY_DIR/g0_fold<k> it lands at",
        "#   $FOA_STUDY_DIR/g0_fold<k>/global_results/fisher",
        "# which is exactly where the regularisation stages look. It is computed on",
        "# the TRAIN loader of that fold - the old-data train split - which is what",
        "# EWC wants: the curvature of the model on the data it was fitted to. No",
        "# separate task, and nothing to remember later except the winning fold",
        "# number, which the reg stages take as $G0_FOLD.",
        "#",
        f"# LINE {len(FOLDS) + 1} picks the winner by VALIDATION accuracy - never test, since the",
        "# test halves are what the paper reports on - ties to the lower fold. It",
        "# copies that fold's checkpoint to g0_model and writes g0_selection.json",
        "# with the sha256, the per-fold table and the fold mean and spread.",
        "#",
        f"# LINES {len(FOLDS) + 2}-{2 * len(FOLDS) + 1} ARE PRESERVATION: g-0 on the old data's test rows, one",
        "# fold each, into g0_evaluations.json. This is the reference line every",
        "# later method is measured against - what the shipped model knows before",
        "# anyone adapts it.",
        "#",
        f"# LINES {2 * len(FOLDS) + 2}-{3 * len(FOLDS) + 1} ARE THE ADAPTATION GAP: g-0 on the cohort's test rows,",
        "# per fold AND per writer, into g0_perfold_evaluations.json. That is the",
        f"# do-nothing baseline - what each of the {COHORT_SIZE} outliers already gets from the",
        "# shipped model - and the gap the whole study exists to close. Only",
        f"# ~{COHORT_TEST} rows per fold, so these are seconds of work.",
        "#",
        "# BOTH EVALUATION GROUPS APPEND to a single JSON each, keyed by tag, and",
        "# appending is a read-modify-write: two running at once would each read",
        "# before the other wrote and one result would vanish silently. Submit them",
        "# %1. They are under a minute each, so it costs nothing.",
        "#",
        "# The two files keep the shapes their study-2 counterparts had, so every",
        "# downstream reader works unchanged.",
        "#",
        f"# {len(tasks)} tasks. Chain: the five folds -> the winner -> the evaluations.",
        "#",
    ]


def readme(tasks: list[str]) -> str:
    return f"""# {STUDY} - P07: g-0 and the starting line

g-0 is the **shipped model**: trained on the {OLD_WRITERS} old-data writers P04
drew, five-fold cross-validated on their book, early-stopped on validation with
the best weights restored. Everything this study reports is measured against it.

g-init is not a baseline and never was - it exists to rank writers, and its job
ended when the cohort was cut.

## The data

| | rows |
|---|---|
| old data, per fold | **{OLD_ROWS:,}** |
| train | {OLD_TRAIN:,} |
| validation | ~{OLD_VAL:,} |
| test | ~{OLD_TEST:,} |
| cohort test, per fold | ~{COHORT_TEST} |

Both books P04 wrote report an **empty** unsplittable list, so every writer
contributes to all three parts of all five folds - the check P02 predicted, now
confirmed on the real books.

## The chain

| # | task | writes |
|---|---|---|
| 1-5 | g-0, one fold each | `g0_fold<k>/` incl. its Fisher |
| 6 | the winner, by validation | `g0_model`, `g0_selection.json` |
| 7-11 | preservation: old-data test rows | `g0_evaluations.json` |
| 12-16 | adaptation gap: cohort test rows, per writer | `g0_perfold_evaluations.json` |

Both evaluation groups **append** to a single JSON each, keyed by tag. Appending
is a read-modify-write, so they are submitted `%1`: two at once would each read
before the other wrote and one result would vanish silently. They are under a
minute each.

The two files keep the shapes their study-2 counterparts had, so every
downstream reader works unchanged.

## The Fisher comes free, and in the right place

`global-train` computes the Fisher diagonal after training and writes it to
`<results-dir>/global_results/fisher`. With `--results-dir
$FOA_STUDY_DIR/g0_fold<k>` that is

```
$FOA_STUDY_DIR/g0_fold<k>/global_results/fisher/{{fisher.pt, global_params.pt}}
```

which is exactly where the regularisation stages look. It is computed on the
**train loader of that fold** - the old-data train split - which is what EWC
wants: the curvature of the model on the data it was fitted to.

So there is **no separate Fisher task**, and nothing to remember later except
the winning fold number. The regularisation stages name the directory as

```
--set fisher_path=$FOA_STUDY_DIR/g0_fold$G0_FOLD/global_results/fisher
```

with `$G0_FOLD` exported from the selection:

```bash
G0_FOLD=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['selected_fold'])" \\
        $FOA_V4_RESULTS_DIR/studies/{STUDY}/g0_selection.json)
export G0_FOLD
```

## Order

```bash
J=$FOA_PROJECT_DIR/jobs/v4
A="sbatch --account=$FOA_ACCOUNT"
SB=$J/study_phase.sbatch

# study_phase.sbatch defaults FOA_STUDY_DIR to studies/{STUDY} and
# FOA_NIST_CLASSES to digits. No exports needed for this file.

$A --array=1-5%5   $SB $J/d01_p07.txt    # the five folds
$A --array=6       $SB $J/d01_p07.txt    # the winner
$A --array=7-16%1  $SB $J/d01_p07.txt    # the evaluations, one at a time
```

## Cost

| lines | what | per task | total |
|---|---|---|---|
| 1-5 | a g-0 fold | **~3-4.5 min** | ~0.3 GPU-h |
| 6 | select the winner | seconds | - |
| 7-16 | the ten evaluations | ~45 s | ~0.13 GPU-h |

**~0.45 GPU-h for P07.** An epoch is {OLD_TRAIN:,} train + ~{OLD_VAL:,}
validation images, about 3.6 s at the Phase A calibration; the convergence rule
(patience 10, floor 20, ceiling 100) normally stops in the twenties to
mid-thirties. The Fisher pass adds a per-sample backward over the {OLD_TRAIN:,}
train rows. The 4-hour sbatch default covers the 100-epoch ceiling many times
over.

## What P07 writes

```
studies/{STUDY}/
  g0_fold1..5/                        the five folds, their metrics and Fisher
  g0_model                            the shipped model
  g0_selection.json                   winner, sha256, per-fold table, mean+spread
  g0_evaluations.json                 preservation, per old fold
  g0_perfold_evaluations.json         the do-nothing baseline, per cohort fold
                                      and per outlier
```

Each `g0_fold<k>/` also gets a `writer_split.json` and a `global_model` of its
own - artefacts `global-train` always writes. They are per-fold and harmless;
the study root's `g0_model` is the one that matters.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-dir", required=True)
    args = parser.parse_args()
    jobs = Path(args.jobs_dir)
    jobs.mkdir(parents=True, exist_ok=True)
    tasks = lines()
    (jobs / "d01_p07.txt").write_text("\n".join(header(tasks) + tasks) + "\n")
    (jobs / "d01_p07_README.md").write_text(readme(tasks))
    print(f"wrote {jobs}/d01_p07.txt: {len(tasks)} tasks")
    print(f"wrote {jobs}/d01_p07_README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
