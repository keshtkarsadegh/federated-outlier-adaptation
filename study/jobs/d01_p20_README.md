# P20 - the twenty-client point of the scaling study

Two files. `d01_p20_setup.txt` (**10 tasks**, submit `%1`) builds the cohort, its
book and the do-nothing baseline; `d01_p20_runs.txt` (**30 tasks**, 100 rounds
each) runs the three winners at both dropout levels.

## The scaling ladder this completes

| clients | drawn | dropout | where |
|---|---|---|---|
| 5 | 4 | 10% | P18 |
| 10 | 9 | 10% | the study proper (P15 / P09) |
| 10 | 8 | 20% | P19 |
| **20** | **18** | **10%** | **this file** |
| **20** | **16** | **20%** | **this file** |

Methods, horizon, shipped model, anchor, teacher, local budget and the three
evaluation categories are held; only the federation moves.

## Participation

```
participants(n, d) = floor((1 - d) * n)

floor((1 - 0.1) * 20) = 18      2 dropped
floor((1 - 0.2) * 20) = 16      4 dropped
```

The same expression gives 9 of 10 for this study and 16 of 20 for the earlier
programme. It lives once, in `training/study_config.py:participants`; no task
file carries a literal.

## The cohort must extend

`cohort_worst20.json` is cut by `select-outliers --mode worst --k 20` over the
**same** ranking (`outliers/bad_acc_on_g0.json`) with the **same** eligibility
rule (`--require-trainable`) that produced `cohort_worst10.json`. Its first ten
writers must therefore come back identical:

```
f3642_03  f2248_68  f2297_69  f2524_58  f2145_89
f2162_62  f0619_33  f2307_62  f3151_13  f2304_67      <- cohort_worst10
f2230_67  f2325_86  f3211_20  f2157_61  f0048_00
f2169_69  f2438_76  f3200_41  f2383_60  f2422_85      <- the next ten
```

**Step 2 of the setup chain stops everything if they do not.** If the worst-20
is not the worst-10 plus the next ten, the twenty-client point is not the
ten-client point scaled up - it is a different population, and every comparison
drawn across the two sizes would be measuring that instead. Nothing downstream
would say so: the tables would fill, the numbers would differ, and the
difference would be read as a size effect. The emitter re-checks it before
writing a line.

## Its own book, cut by the same rule

`fold_books/cohort20.foldbook.npz`. A larger cohort has no choice here - the
ten-client book holds no rows for writers it never covered - so this one is cut
by per-writer stratified 60/20/20 over 5 folds at seed 42, which is the rule
every book in this study is cut by. A different rule would move the rows under
the ten writers both sizes share, and the size comparison would be over two
splits.

Step 4 asserts all 20 writers are covered **and** that no writer has an empty
train, val or test partition in any fold - a writer with no train rows would
train on nothing and still report a number.

## What the trimmed mean does at 18 and 16

The winner's rule is `trimmed_0p4`, dropping `int(f * K)` from each end:

| f | K = 9 | K = 18 | K = 16 |
|---|---|---|---|
| 0.1 | trim 0 -> 9 survive | trim 1 -> 16 | trim 1 -> 14 |
| 0.2 | trim 1 -> 7 | trim 3 -> 12 | trim 3 -> 10 |
| 0.3 | trim 2 -> 5 | trim 5 -> 8 | trim 4 -> 8 |
| **0.4** | trim 3 -> **3** | trim **7** -> **4** | trim **6** -> **4** |

`int(0.4 * 18) = 7`, so 18 - 14 = **4 survive**; `int(0.4 * 16) = 6`, so
16 - 12 = **4 survive**. The rule averages four updates at both points - the
same number it averages at nine of ten (three) to within one - while discarding
78% of an 18-client round against 67% of a 9-client one. The severity of the
trim rises with the federation while the number averaged does not. That is a
property of the rule, not of the size, and it has to be read alongside the
result rather than after it.

Every cell above is produced by `five_cells.trim_survivors` - the server's own
`int(f * K)` expression - and each is checked in the tests against the real
`con_delta_trimmed_mean` on a spread of updates with poisoned ends.

## Winners only

| id | family | aggregation | penalty |
|----|--------|-------------|---------|
| `winner` | concurrent | `trimmed_0p4` | `feature_l2_lam0p1` |
| `balanced` | concurrent | `anchor_0p03` | `ntd_b0p01_t0p5` |
| `sequential` | sequential | `seq_order_shuffle` | `feature_l2_lam0p01` |

**No plain-FedAvg control.** The control's job is to say whether a change is the
method or the participation, and the ten-client pair - 9 of 10 against 8 of 10,
P19 - already answers that with the control included. The question here is
whether the winners hold at a larger size, and paying for the control again
would buy nothing that pair has not already bought.

3 configurations x 5 folds x 2 participation points = **30 tasks**.

## The setup chain (submit `%1`)

Each step reads what the one before it wrote, so a fan-out would race them and
could score a baseline on whichever book happened to be on disk.

| # | what | writes |
|---|------|--------|
| 1 | `select-outliers --mode worst --k 20`, same ranking, same eligibility | `outliers/cohort_worst20.json` |
| 2 | **CHECK** the first ten are `cohort_worst10` exactly | - |
| 3 | `fold-book`, 5 folds, seed 42, 60/20/20 | `fold_books/cohort20.foldbook.npz` |
| 4 | **CHECK** all 20 covered, no empty partition in any fold | - |
| 5-9 | `evaluate-book`, g-0 on each fold's test rows | `g0_cohort20_evaluations.json` |
| 10 | **CHECK** re-emit the 30 runs from the real cohort and book, refuse any difference | `tables/p20_runs_regenerated.txt` |

**Why step 10 exists.** `d01_p20_runs.txt` was emitted before the chain ran,
against a cohort derived the same way and a book that did not exist yet. That is
reviewable but not yet verified - only the real cohort and the real book can say
the file addresses them. So the chain regenerates it and stops on any
difference, rather than letting a 30-element array run against lines nothing
checked. Steps 1-4 and 10 are CPU; 5-9 score one saved model on 20 writers'
test rows.

## Provenance

- Parents: `d01_c20d10_{winner,balanced,sequential}_fold{1..5}` and
  `d01_c20d20_*` - 30 distinct folders.
- Sampler seeds `715001-715025` and `716001-716025`, fresh and **disjoint from
  every seed in every other `d01_p*.txt`** (checked, zero collisions).
- `--seed` is the fold, as everywhere in this study.
- Records: `tables/p20_c20_d10.json`, `tables/p20_c20_d20.json`.

## Cost

Measured from this study's own completed arrays at 100 rounds: P14 reg-full
(m=9, both families) 209 s; P15 combos (m=9, one family) 115 s; P16 extreme
(m~1.3, one family) 53 s. A one-family task costs about `42 + 8.1 m` seconds:

```
15 x (42 + 8.1 x 18) s  +  15 x (42 + 8.1 x 16) s
  = 15 x 188 s  +  15 x 172 s  =  ~5,400 s  =  ~1.5 GPU-h
```

Setup: steps 1-4 and 10 are seconds of CPU; 5-9 are well under a minute each -
call the whole chain **under 0.1 GPU-h**. Total **~1.5 GPU-h**. The 4-hour
per-task limit in `study_phase.sbatch` is ample.

## Verification done

- All 10 setup lines and all 30 run lines execute the real `study_phase.sbatch`
  body under `FOA_DRY_RUN=1`: 40 ok, 0 failed. (`bash -n` alone was not enough
  once; it is not relied on again.)
- All 30 run lines parse against `foa final`'s own parser. Clients-per-round
  histogram exactly `{18: 15, 16: 15}`.
- 30 distinct parents, 30 distinct seeds, zero collisions with any other task
  file. 30 `AnchoredTrainer`, 0 `BaseTrainer` (winners only). No `fisher_path`,
  so neither file depends on `G0_FOLD`.
- Every run line: `cohort_worst20.json`, `cohort20.foldbook.npz`, never
  `cohort10` or `cohort_worst10`; `--init global --global-name g0`,
  `--old-fold all`, `--rounds 100`, `anchor=frozen`.
- Every path-valued flag in the setup chain lands inside `$FOA_STUDY_DIR`,
  asserted in the tests as well as by the sbatch guard - a refusal at element 1
  of a `%1` chain would stop everything behind it.

## To submit

```
J=$FOA_PROJECT_DIR/jobs/v4
sbatch --account=$FOA_ACCOUNT --array=1-10%1 \
       --output="$FOA_STUDY_DIR/logs/%x_%A_%a.log" \
       $J/study_phase.sbatch $J/d01_p20_setup.txt

# after the chain is green (step 10 verifies the runs file):
sbatch --account=$FOA_ACCOUNT --array=1-30 \
       --output="$FOA_STUDY_DIR/logs/%x_%A_%a.log" \
       $J/study_phase.sbatch $J/d01_p20_runs.txt
```

NOT SUBMITTED.
