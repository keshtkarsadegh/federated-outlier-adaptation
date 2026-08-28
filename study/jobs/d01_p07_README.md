# Digits_study01 - P07: g-0 and the starting line

g-0 is the **shipped model**: trained on the 200 old-data writers P04
drew, five-fold cross-validated on their book, early-stopped on validation with
the best weights restored. Everything this study reports is measured against it.

g-init is not a baseline and never was - it exists to rank writers, and its job
ended when the cohort was cut.

## The data

| | rows |
|---|---|
| old data, per fold | **23,898** |
| train | 13,289 |
| validation | ~5,296 |
| test | ~5,313 |
| cohort test, per fold | ~225 |

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
$FOA_STUDY_DIR/g0_fold<k>/global_results/fisher/{fisher.pt, global_params.pt}
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
G0_FOLD=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['selected_fold'])" \
        $FOA_V4_RESULTS_DIR/studies/Digits_study01/g0_selection.json)
export G0_FOLD
```

## Order

```bash
J=$FOA_PROJECT_DIR/jobs/v4
A="sbatch --account=$FOA_ACCOUNT"
SB=$J/study_phase.sbatch

# study_phase.sbatch defaults FOA_STUDY_DIR to studies/Digits_study01 and
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

**~0.45 GPU-h for P07.** An epoch is 13,289 train + ~5,296
validation images, about 3.6 s at the Phase A calibration; the convergence rule
(patience 10, floor 20, ceiling 100) normally stops in the twenties to
mid-thirties. The Fisher pass adds a per-sample backward over the 13,289
train rows. The 4-hour sbatch default covers the 100-epoch ceiling many times
over.

## What P07 writes

```
studies/Digits_study01/
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
