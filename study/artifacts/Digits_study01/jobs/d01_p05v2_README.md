# Digits_study01 - revised P05-P07

## The redesign, and why

g-init trained on **every** writer, so its per-writer accuracy is contaminated
by memorisation: a writer can score well because the model remembers it, not
because it is easy. That is survivable in a coarse cut and fatal in a selection.
So the two jobs are split:

| | does | on |
|---|---|---|
| **g-init** | cuts BAD from GOOD at the worst 30% | every writer |
| **g-0** | scores every BAD writer, and the worst 10 become the cohort | writers it never saw |

**No leakage, by construction.** The old data is drawn from **GOOD only**, so
g-0 never trains on a writer it will later score. That is why the draw is
restricted rather than merely cohort-excluded.

| | count |
|---|---|
| scored writers | 3,580 |
| BAD = ceil(0.3 x 3,580) | **1,074** |
| GOOD | 2,506 |
| old data drawn from GOOD | 200 |
| cohort = worst of BAD under g-0 | 10 |

## Where v1 went

**Archived, not deleted** - moved to `$FOA_STUDY_DIR/v1_superseded/`, keeping their relative
paths:

```
g0_fold1..5/  g0_model  g0_selection.json  (v1 winner: fold 5)
g0_evaluations.json  g0_perfold_evaluations.json
outliers/old_data.json  outliers/cohort_worst10.json  outliers/typical5.json
fold_books/old_data.foldbook.npz  fold_books/cohort10.foldbook.npz
tables/cohort_table.json  tables/cohort_table.csv  tables/gallery/
```

Nothing downstream can pick them up: every reader names an exact path in the
study root, and after line 1 those paths are empty until v2 refills them.
Recoverable if a comparison is ever wanted - which is why they are moved rather
than removed.

**Surviving v1 unchanged:** `ginit_*`, `fold_books/all_writers_digits.foldbook.npz`,
`outliers/writer_scores.json`, `outliers/clients_acc_on_global.*`,
`outliers/writer_counts.*`. The coarse detector and the census are still exactly
what they were.

## The chain

| # | task | writes |
|---|---|---|
| 1 | archive v1 | `v1_superseded/` |
| 2 | the coarse cut | `outliers/pools.json` + `.csv`, `pool_bad.json`, `pool_good.json` |
| 3 | old data from GOOD, seed `20260829` | `outliers/old_data.json` |
| 4 | its book | `fold_books/old_data.foldbook.npz` |
| 5-9 | g-0, one fold each | `g0_fold<k>/` incl. its Fisher |
| 10 | the winner, by validation | `g0_model`, `g0_selection.json` (sha256) |
| 11 | **g-0 scores all 1,074 bad writers** | `outliers/bad_scores_on_g0.json` + `.csv` + flat |
| 12 | the cohort: worst 10 of that | `outliers/cohort_worst10.json` |
| 13 | its book | `fold_books/cohort10.foldbook.npz` |
| 14 | the table | `tables/cohort_table.json` + `.csv` |
| 15 | reference writers from GOOD, seed `20260830` | `outliers/typical5.json` |
| 16 | the figure | `tables/gallery/cohort_vs_typical.png` |
| 17-21 | preservation | `g0_evaluations.json` |
| 22-26 | the do-nothing baseline | `g0_perfold_evaluations.json` |

## The selection pass

Line 11 scores g-0 on **every row** of each of the 1,074 bad writers -
about 121,000 images. Every row, not a held-out partition: g-0 has never
seen these writers at all, so there is nothing to hold out from, and using all
of a writer's data makes the estimate as tight as that writer allows. These are
the small, difficult ones; the tightness is the point.

## The table

Each cohort writer carries its **rank within the 1,074-writer bad pool**
(not out of 3,580 - the pool is the population it was selected
from), the g-0 accuracy that selected it, **its g-init accuracy alongside** so
the two rankings can be compared, its per-class counts and its per-fold
train/val/test sizes.

## Order

```bash
J=$FOA_PROJECT_DIR/jobs/v4
A="sbatch --account=$FOA_ACCOUNT"
SB=$J/study_phase.sbatch

$A --array=1-4%1   $SB $J/d01_p05v2.txt   # archive, pools, draw, book
$A --array=5-9%5   $SB $J/d01_p05v2.txt   # the five g-0 folds
$A --array=10-16%1 $SB $J/d01_p05v2.txt   # winner, score-bad, cohort, book, table, typical, figure
$A --array=17-26%1 $SB $J/d01_p05v2.txt   # the baselines, one at a time
```

No exports needed: `study_phase.sbatch` defaults `FOA_STUDY_DIR` to
`studies/Digits_study01` and `FOA_NIST_CLASSES` to `digits`.

## Cost

| lines | what | per task | total |
|---|---|---|---|
| 1-4 | archive, pools, draw, book | seconds to ~2 min, CPU | - |
| 5-9 | a g-0 fold | ~3-4.5 min | ~0.33 GPU-h |
| 10 | the winner | seconds | - |
| 11 | g-0 over 121,000 rows | ~2 min | ~0.03 GPU-h |
| 12-16 | cohort, book, table, typical, figure | ~1-2 min each, CPU | - |
| 17-26 | the ten evaluations | ~45 s | ~0.13 GPU-h |

**~0.55 GPU-h in total**, 30 tasks.
