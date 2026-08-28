# P18 - the selected configurations at five clients

`jobs/v4/d01_p18_five.txt` - **20 tasks**, 100 rounds each. Results first; the
discussion of what they mean comes after they exist.

## The question

Every method comparison in `Digits_study01` was made on **ten** clients with
nine drawn per round. That is one federation size, and a result measured at one
size is a result about that size until something else is measured. This stage
holds the methods fixed and moves the size.

**Varied:** the number of clients (10 -> 5) and, with it, the number drawn per
round (9 -> 4).
**Held:** the fold book, the old-data book, the initial model (g-0), the anchor
(frozen g-0), the teacher, the local budget (5 epochs, batch 64), the horizon
(100 rounds), the sampling policy (uniform), and the three evaluation
categories. Nothing else moves, so a gap against the ten-client runs is about
the size of the federation.

## The cohort

The cohort's **worst five** writers:

```
f3642_03  0.5417
f2248_68  0.7113
f2297_69  0.8235
f2524_58  0.8352
f2145_89  0.8438
```

(g-0 accuracy on every row the writer holds.)

**Derived, not restated.** The emitter re-sorts `outliers/bad_acc_on_g0.json` by
accuracy with ties broken on writer id, then checks that first five against
`outliers/bad_scores_on_g0.json`'s stated ranking **and** against the first five
of `outliers/cohort_worst10.json`. Any disagreement stops the emission: if those
artefacts ever diverge, the five writers here are not the five the ten-client
stages federate, and the comparison this stage exists to make is void. Written
to `outliers/cohort_worst5.json`.

## The fold book is the ten-client one

`fold_books/cohort10.foldbook.npz`, unchanged. The cohort is narrowed by the
**client list alone** - the same machinery the extreme cases used: the runner
keeps only the writers the list names and reads their rows out of the book that
already covers them.

A book re-cut for five writers would give those writers *different rows* than
the ten-client runs gave them, and the comparison would then be over two splits
rather than over two federation sizes. The emitter asserts the book covers all
five before writing a line.

## Participation: 4 of 5

```
participants(n, d) = floor((1 - d) * n)

floor((1 - 0.1) * 5) = 4
```

One expression serves every stage this programme has run - 9 of 10 in this
study, 16 of 20 in the earlier one, **4 of 5** here, 8 of 10 at the P19 dropout
point. A fractional client is not trained on, so the kept count floors. The rule
lives in `training/study_config.py:participants`, every emitter calls it, and no
task file carries a literal.

Four of five is 80%, not 90%. That is what 10% dropout means on a population of
five under the study's own rule, and it is stated here rather than left to be
reconstructed from a float.

**The extreme cases are the one documented exception.** At two clients the rule
keeps one, and dropping a client from a two-client federation is not a
participation study but a coin flip on whether the round happens; at one client
it keeps nobody. Those run at full participation - `--policy all --participation
1.0` - which the extreme emitter states in the line itself, and `participants`
refuses those sizes by name rather than rounding up.

## What the trimmed mean does at four participants

The winner's server rule is `trimmed_0p4`, which drops `int(f * K)` updates from
each end of the ordered coordinate and averages the rest. `K` is now 4, not 9:

| f | K = 9 | K = 4 |
|---|---|---|
| 0.1 | trim 0 -> **9 survive** (not a trimmed mean at all) | trim 0 -> **4 survive** |
| 0.2 | trim 1 -> 7 survive | trim 0 -> **4 survive** |
| 0.3 | trim 2 -> 5 survive | trim 1 -> 2 survive |
| **0.4** | trim 3 -> **3 survive** | trim 1 -> **2 survive** |

So `trimmed_0p4` **still bites** at this size - it is not silently demoted to a
plain mean, which is what happens to `trimmed_0p1` even at nine - but it now
averages two updates where it averaged three, and it discards half the round
rather than two thirds. The numbers in this table are produced by
`five_cells.trim_survivors`, which is the server's own `int(f * K)` expression,
and `tests/test_digits_p18.py` checks each cell against the real
`con_delta_trimmed_mean` on a poisoned-ends vector. They cannot drift apart from
what the code applies.

## The four configurations

| # | id | family | aggregation | penalty | why |
|---|----|--------|-------------|---------|-----|
| 1 | `winner` | concurrent | `trimmed_0p4` | `feature_l2_lam0p1` | the cross's strongest concurrent pair |
| 2 | `balanced` | concurrent | `anchor_0p03` | `ntd_b0p01_t0p5` | the concurrent pair that gave up least preservation |
| 3 | `sequential` | sequential | `seq_order_shuffle` | `feature_l2_lam0p01` | the strongest sequential pair |
| 4 | `control` | **both** | plain `fedavg` | none | the baseline |

Each of the first three names **one** aggregation rule, which lives in exactly
one family, so a task produces one result. The emitter refuses a pair the cross
did not actually run (checked against `tables/p15_combination_grid.json`) and
refuses a rule reported on a schedule it does not belong to.

**Why the control is here.** Without it a change between ten and five clients
could not be attributed: every selected configuration might move together simply
because the federation got smaller, and only an unmodified baseline moving the
same way would show that. It runs `BaseTrainer` with `--aggregation fedavg` and
no penalty, which covers **both** schedules in the one task, as every plain-
FedAvg line of this study does - plain averaging is the same rule on either
loop.

4 configurations x 5 folds = **20 tasks**.

## Evaluation

Unchanged: the three-category `final_evaluation` - pooled clients test, the
per-client column, and the old data's five fold test partitions scored
separately with mean +- sd (`--old-fold all`).

## Provenance

- Parents: `d01_five_{winner,balanced,sequential,control}_fold{1..5}` - 20
  distinct folders.
- Sampler seeds `712001-712035`, fresh and **disjoint from every seed in every
  other `d01_p*.txt`** (checked, zero collisions).
- `--seed` is the fold, as everywhere in this study.
- Records: `outliers/cohort_worst5.json` and `tables/p18_five_client.json`
  (cohort, participation rule, trim survivors, and the configuration table).

## Cost

Measured from this study's own completed arrays at 100 rounds, not assumed:

| stage | m | families | mean elapsed |
|---|---|---|---|
| P14 reg-full | 9 | both | 209 s |
| P15 combos | 9 | one | 115 s |
| P16 extreme | ~1.3 | one | 53 s |

A one-family task costs about `42 + 8.1 m` seconds, so at m = 4 it is ~75 s; the
both-family control is ~1.8x that, ~135 s.

```
15 x 75 s  +  5 x 135 s  =  ~1,790 s  =  ~0.5 GPU-h
```

Budget **1 GPU-h**. The 4-hour per-task limit in `study_phase.sbatch` is ample.

## Verification done

- All 20 lines parse against `foa final`'s own parser.
- All 20 lines execute the real `study_phase.sbatch` body under
  `FOA_DRY_RUN=1`: 20 ok, 0 failed. (`bash -n` alone was not enough once; it is
  not relied on again.)
- 20 distinct parents, 20 distinct seeds, zero collisions with any other task
  file.
- Every line: `--clients-per-round 4`, `--outliers-file .../cohort_worst5.json`,
  `--fold-book .../cohort10.foldbook.npz`, `--init global --global-name g0`,
  `--old-fold all`, `--rounds 100`, `anchor=frozen` wherever a penalty is set.
- No `fisher_path` on any line (none of the three penalties needs one), so this
  file does not depend on `G0_FOLD`.

## To submit

```
J=$FOA_PROJECT_DIR/jobs/v4
sbatch --account=$FOA_ACCOUNT --array=1-20 \
       --output="$FOA_STUDY_DIR/logs/%x_%A_%a.log" \
       $J/study_phase.sbatch $J/d01_p18_five.txt
```

NOT SUBMITTED.
