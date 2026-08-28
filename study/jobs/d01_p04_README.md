# Digits_study01 - P04-P06: populations, books, figure

P02 produced the **detector**: g-init (winner fold 3) and a score for
each of the 3,580 digit writers. Those scores **select**; they do
not baseline. Every number this study reports is measured against g-0, which
P07 trains on the old data drawn here.

## The chain

| # | task | writes |
|---|---|---|
| 1 | cohort: the 10 worst splittable writers | `outliers/cohort_worst10.json` |
| 2 | old data: seeded draw of 200, cohort excluded | `outliers/old_data.json` |
| 3 | cohort fold book | `fold_books/cohort10.foldbook.npz` |
| 4 | old-data fold book | `fold_books/old_data.foldbook.npz` |
| 5 | the cohort table | `tables/cohort_table.json` + `.csv` |
| 6 | the figure's reference writers | `outliers/typical5.json` |
| 7 | the figure | `tables/gallery/cohort_vs_typical.png` + sidecar |

**Line 5 comes after the books deliberately.** The table reports each writer's
per-fold train/val/test sizes, and those do not exist until the book does.
Running it in the position the plan first suggested would have silently dropped
the column that makes the table worth banking.

## The draws

| draw | size | seed | pool |
|---|---|---|---|
| cohort | 10 | none - it is a **ranking**, not a draw | the 3,580 scored writers |
| old data | 200 | `20260827` | 2,943 writers with >= 100 digit rows, cohort excluded |
| typical (figure) | 5 | `20260828` | the top of the ranking, cohort excluded |

The old-data population is a **draw and not a ranking**: taking the best or the
largest writers would build the preservation reference out of an
unrepresentative sample. The median writer holds 117 digit rows, so
the 100-row bar leaves 2,943 eligible - ample for
200.

2,999 writers score exactly 1.0 under the detector, so "typical"
is the overwhelming majority and a seeded draw from the top is well defined.

## The cohort table

One row per cohort writer: the accuracy that selected it, **its rank out of all
3,580** (not out of ten - "rank 3 of 3,580" is the statement worth
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

# study_phase.sbatch defaults FOA_STUDY_DIR to studies/Digits_study01 and
# FOA_NIST_CLASSES to digits, and refuses to run if that folder is missing.
# No exports are needed.

# a chain - each line reads what the one before it wrote
$A --array=1-7%1 $SB $J/d01_p04.txt
```

Everything here is CPU work: reading the cache, counting, writing indices and
one PNG. Minutes per task; the whole file is well under an hour and needs no
GPU. It is submitted through the same GPU sbatch for consistency of environment
and logging - if the queue is busy, `--partition=standard96:shared` would do,
but nothing here is worth a second runner.

## What P04-P06 write

```
studies/Digits_study01/
  outliers/cohort_worst10.json    the cohort, with its rule and accuracies
  outliers/old_data.json            the draw, with seed and eligibility rule
  outliers/typical5.json           the figure's reference writers
  fold_books/cohort10.foldbook.npz
  fold_books/old_data.foldbook.npz
  tables/cohort_table.json / .csv   the selection's paper trail
  tables/gallery/cohort_vs_typical.png + .json
```
