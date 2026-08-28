# Digits_study01 - P08, P09, P10: three arrangements of the same ten writers

| stage | file | what | tasks | GPU-h |
|---|---|---|---|---|
| P08 | `d01_p08.txt` | isolated: ten private models | 100 | ~1.2 |
| P09 | `d01_p09.txt` | federated: one server model, 9 of 10 per round | 10 | ~1.2 |
| P10 | `d01_p10.txt` | centralized: one model over the pooled cohort | 10 | ~0.13 |
| | | **total** | **120** | **~2.5** |

## Submit all three at once

**There are no inter-stage dependencies.** Each file reads only artefacts that
already exist - `cohort_worst10.json`, `cohort10.foldbook.npz`,
`old_data.foldbook.npz`, `old_data.json`, `g0_model` - and writes only its own
outputs. Nothing here produces an input for anything else here.

```bash
J=$FOA_PROJECT_DIR/jobs/v4
A="sbatch --account=$FOA_ACCOUNT"
SB=$J/study_phase.sbatch

$A --array=1-100%20 $SB $J/d01_p08.txt
$A --array=1-10     $SB $J/d01_p09.txt
$A --array=1-10     $SB $J/d01_p10.txt
```

No exports needed: `study_phase.sbatch` defaults `FOA_STUDY_DIR` to
`studies/Digits_study01` and `FOA_NIST_CLASSES` to `digits`, and refuses any line whose
paths point elsewhere.

## The ladder

**P08 isolated** - each writer trains alone, sharing nothing. The point is to
price that: ten models each good at one writer are, collectively, ten models bad
at nine. Scored three ways - its own test rows (what isolation wins on), the
whole cohort's test rows (what it costs the group), and the old data.

**P09 federated** - ten clients enrolled, 9 drawn per round,
E=5, batch 64, 100 rounds, both families per task.

**P10 centralized** - one model over the union of all ten writers' fold-k train
rows (579 images), privacy switched off. The ceiling P09 is trying to
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
| cohort, total | 1,024 |
| cohort train, per fold | 579 |
| cohort test, per fold | ~222 |
| old test, per fold | ~5,297 |

Both books report an empty `unsplittable` list, so every one of the ten writers
contributes to all three parts of all five folds.

The cohort (10 writers), read from the study's own
`outliers/cohort_worst10.json`:

`f3642_03`, `f2248_68`, `f2297_69`, `f2524_58`, `f2145_89`, `f2162_62`, `f0619_33`, `f2307_62`, `f3151_13`, `f2304_67`

g-0 is the winner of fold 2; the `--init global` runs of P08 and P10
point at `$FOA_STUDY_DIR/g0_model` and P09 carries `--global-name g0`.

## Cost

The old-data evaluation dominates the two isolated stages: 5,297 rows x 5
folds is about 26.5k images per task, against a few hundred for training. So a
P08 or P10 task is roughly 40-45 s of work plus interpreter and cache startup.

A P09 task is `0.31 + 0.0345 x 9 x 5` = 1.86 s per round per
family; two families over 100 rounds is ~6.2 min, plus the per-round evaluations
and the final three-category one - about **7.2 min per task**.
