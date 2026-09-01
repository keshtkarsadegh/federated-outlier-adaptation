# Digits_study01 - P02: the detector

NIST SD19, **digits 0-9 only**. `--classes digits` gives the dataset ten labels
and the provider hands `FedAvgCNN` `num_classes=10`, so the head is McMahan's
exactly.

## Who is in the book

| | writers | rows |
|---|---|---|
| full character set | 3597 | 814,255 |
| **digits only** | **3580** | **402,953** |

17 writers contribute no digit at all and are absent from the
digit dataset entirely rather than present with a zero.

**Splittability rule:** a writer is splittable when a per-label stratified
60/20/20 leaves it at least one training row *and* at least one validation row.
Counted against the cache:

| class set | splittable | dropped |
|---|---|---|
| digits | **3580 of 3580** | **0** |
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

# study_phase.sbatch defaults FOA_STUDY_DIR to studies/Digits_study01 and
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
| 9 | score 3580 writers | ~3 min | - |

**~2.5-3 GPU-h in total.** Each g-init fold trains on about 242k digit images
and validates on about 80k, so an epoch is roughly 322k images. The convergence
rule (patience 10, floor 20, ceiling 100) normally stops in the twenties. The
sbatch's 4-hour default covers the 100-epoch ceiling with room to spare.

## What P02 writes

```
studies/Digits_study01/
  fold_books/all_writers_digits.foldbook.npz
  ginit_fold1/                        the detector and its metrics
  outliers/writer_scores.json         the ranking
  outliers/clients_acc_on_global.json the flat ranking every later stage reads
  outliers/writer_counts.json/.csv    per-writer totals and per-class counts
```

`ginit_fold1/` carries the detector's validation and test accuracies with
their mean and spread - the paper's "centralized max" row.
