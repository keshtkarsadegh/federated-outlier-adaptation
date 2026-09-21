# Running the study from nothing

Every artefact this study reports is produced by a script in this repository.
Nothing is copied from a previous run, and nothing is typed at a terminal: a
file made by hand looks exactly like a generated one, and the difference only
becomes visible when a cohort it named silently stops being the cohort the
study selected.

The steps are in dependency order. Each one is checked before it is submitted;
the checks are listed with the step because they exist for failures that
happened, not for tidiness.

```bash
export FOA_PROJECT_DIR=/path/to/workspace
source "$FOA_PROJECT_DIR/repo_foa/slurm/env.sh"
export FOA_STUDY_DIR="$FOA_PROJECT_DIR/results/studies/Digits_study01"
```

Without a scheduler, any task file runs with `slurm/run_tasks.sh <file>`, which
drives the same script the cluster does.

---

## 1. Data

```bash
python tools/fetch_sd19.py  --dest "$FOA_DATA_DIR/nist"
python tools/fetch_mnist.py --dest "$FOA_DATA_DIR/mnist"

foa prepare-data --dataset nist  --zip "$FOA_DATA_DIR/nist/by_write.zip" \
    --out "$FOA_NIST28_DIR" --resolution 28 --classes all
foa prepare-data --dataset mnist --raw-dir "$FOA_DATA_DIR/mnist" --resolution 28
```

Both fetchers verify SHA-256 and refuse a mismatch. SD19 and MNIST are mirrored
widely and the mirrors are not all the same release; a byte-different archive
converts, trains, and reproduces nothing.

**MNIST is not optional.** It is the public proxy set, and without it two of the
eight forgetting signals record `None` for every round of every run, silently.

Verify: the two-directory command in
[`docs/DATA.md`](DATA.md#verify-what-you-downloaded) section 1 - the manifest's
seven entries live in `$FOA_DATA_DIR/nist` and `$FOA_NIST28_DIR`, so no single
`cd` plus `sha256sum -c` checks them all. Laptop is fine — the conversion peaks
at 2.5 GB and takes about ten minutes.

## 2. The detector

```bash
python tools/make_digits_p02.py --jobs-dir "$FOA_STUDY_DIR/jobs"
```

Four tasks: the census, the fold book, one model, one scoring pass.

One fold, deliberately. The detector only has to put the badly served writers
inside a net of about a thousand; the study's clients are cut from that net
later, by g-0. With five folds one had to be crowned, and the five sat within
six ten-thousandths of a point of each other, so floating-point accumulation
order chose the study's clients. With one fold there is nothing to crown.

The experiments remain five-fold. This is the detector alone.

## 3. Cohorts and the shipped model

```bash
python tools/make_digits_p05v2.py --jobs-dir "$FOA_STUDY_DIR/jobs"
```

Thirty tasks. Splits the ranking, draws the 200 source writers, trains g-0 once
per fold, crowns one, scores the bad pool with it, and cuts the worst 5, 10 and
20 from that single ranking - three cuts of one list, each with its own
five-fold book.

The three books agree by construction: a writer's split is seeded on
`sha256(seed | fold | writer)`, so it does not depend on who else is in the
book, and the worst five hold the same rows in all three.

`select-fold` promotes the winning fold's Fisher beside its checkpoint. A Fisher
belongs to one set of weights, and leaving it behind is enough to make the two
Fisher-weighted signals record nothing.

## 4. References

```bash
python tools/make_size_references.py --study-dir "$FOA_STUDY_DIR" \
    --cohort cohort_worst10.json --book cohort10 --tag c10 \
    --seed-base 740000 --per-round 9 8 --out "$FOA_STUDY_DIR/jobs/s03_refs_c10.txt"
```

The four rungs every federated number is read against: do nothing, isolated,
centralized, plain FedAvg. Isolated and centralized carry no rounds, so they are
emitted once per size; only the federated control is emitted per participation
rate.

## 5. The grids

Both grids are searched **once**, at the harder participation rate, and the
winners are carried to the other rates. Searching every rate would multiply the
most expensive stage by the number of rates to answer a question the transfer
already answers.

```bash
python tools/make_digits_p11.py --jobs-dir "$FOA_STUDY_DIR/jobs" --clients-per-round 8
python tools/make_digits_p13.py --jobs-dir "$FOA_STUDY_DIR/jobs" --clients-per-round 8
```

845 and 585 tasks, both at 25 rounds. Nothing from a screen is ever reported: a
25-round number is a ranking signal, and reading it as a result is reading a
race at the quarter mark.

## 6. Selection, and reopening the ranges

```bash
python tools/study_emit.py agg-full --study Digits_study01 --root "$FOA_STUDY_DIR" \
    --out "$FOA_STUDY_DIR/jobs/s05_agg_full.txt" --expect 90 --clients-per-round 8
```

A configuration is chosen by what it added on the new clients less the source
knowledge it spent:

    score = (adaptation - A0) - (P0 - preservation)

with `A0` and `P0` the shipped model's own two accuracies, read from step 3.
Selecting on adaptation alone picks, for every method, the setting that
constrains least, so every winner is the cell closest to plain FedAvg and the
table reports that no method preserves anything.

When a winner sits at the end of a swept row the row is the wrong answer to
give, and the selector says so. Reopen it:

```bash
python tools/make_boundary_ext.py --root "$FOA_STUDY_DIR" --grid agg \
    --label p11/agg-full --clients-per-round 8 --out "$FOA_STUDY_DIR/jobs/ext.txt"
```

Then re-select, and repeat until no winner is on an edge. It can take more than
one round: a reopened range can end at its own new edge. A strength of zero is
where it stops - there is nothing below "no penalty at all".

**The finals must be emitted at the rate their screen was searched at.** The
default is the study's own, and re-running winners at a rate they were not
chosen under produces a file that looks entirely normal.

## 7. Tables

```bash
python tools/report_tables.py --root "$FOA_STUDY_DIR" --what all --csv report/
```

Reads the stored runs and the selection records; trains nothing. Every number
in the manuscript comes from here.

---

## Before submitting anything

```bash
python tools/check_seeds.py     "$FOA_STUDY_DIR"/jobs/*.txt   # ALL of them, together
python tools/check_programme.py "$FOA_STUDY_DIR/jobs"
python tools/freeze_selection.py verify "$FOA_STUDY_DIR"
```

- **check_seeds** - every task that fits a model or draws a sample carries a
  seed, and an unclassified command is a failure so a new subcommand cannot
  slip through. The isolated arm ran unseeded for 110 of 135 reference tasks
  before this existed.

  **Pass every task file at once, not one at a time.** It also cross-checks the
  seed spans, and the second failure it caught was a collision between stages:
  seed blocks are 1000 wide, the 140-cell regularisation screen spans 1400, and
  the combination stage's block sat inside it. Both stages would have run and
  drawn the same client-sampling sequence, with nothing downstream to say so.
- **check_programme** - every file a stage reads is produced by an earlier
  stage. Two inputs had been made by hand and nothing rebuilt them.
- **freeze_selection** - the cohort is the same cohort. GPU arithmetic is not
  bit-deterministic, so the question is not whether the files match but whether
  the decisions do.

## After a stage finishes

```bash
python tools/check_signals.py    --root "$FOA_STUDY_DIR" --all
python tools/describe_cohort.py  --root "$FOA_STUDY_DIR" --all
```

- **check_signals** - all eight forgetting signals are present, in every
  payload, at the horizon the stage ran. Four of the eight were empty across an
  845-task stage: the proxy set was never built and the Fisher was never
  promoted. Every task exited zero. A missing signal looks exactly like a
  present one until something counts them.
- **describe_cohort** - what the cohort actually holds. The verdict it prints -
  whether the clients are label-skewed - is what decides how the not-true
  distillation result may be described, so it is measured rather than
  remembered.

And run **one** task before launching hundreds, then read its output. Four of
the eight signals were empty across an 845-task stage because nothing looked at
a single summary first.

Delete a stage's result folders before re-running it. Every task skips work that
already exists, so a re-run over stale folders is a no-op that reports success.

And delete them before re-running a stage under **changed ranges**, which is a
different hazard with the same cause. Result folders are named by cell id, so a
re-ranged grid leaves the cells it dropped on disk as orphans. Nothing collects
them - the selectors read the catalogue - but anything that reads a *prefix*
does, and that is how a finals table came to be topped by three cells that exist
in no current row. 60 stale regularisation folders and 90 stale combination
folders were removed for exactly this reason.

## Reading the results

```bash
python tools/report_tables.py --root "$FOA_STUDY_DIR" --what all --csv out/
python tools/compare_arms.py  --root "$FOA_STUDY_DIR" --what all --csv out/
```

- **report_tables** - what each arm scored. The manuscript's numbers come from
  here and from nowhere else.
- **compare_arms** - what the DIFFERENCE between two arms is worth, paired by
  fold. Use it for any claim of the form "A beats B": on the combination stage,
  eleven of eighteen pairs beat both their halves on the means and only one of
  the eighteen is positive on every fold. A mean difference smaller than its own
  fold spread is not a finding, and only the paired view shows that.
- **weight_sensitivity** - which winners depend on `w`, the one number in the
  selection rule that is not read off disk. The study selects and reports at
  `w = 1`; this exists because the repository once asserted the winners do not
  depend on it, which was never checked and is false. Publish the sensitivity,
  do not defend the constant.
