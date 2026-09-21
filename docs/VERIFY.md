# Verifying results

Three levels, in increasing cost. The first two need **no GPU and no SD19
download**.

---

## Level 1 — no data, no GPU (minutes)

### The test suite

```bash
pip install -r requirements.txt    # the pins the study ran on
pip install -e . --no-deps         # the `foa` entry point, nothing resolved
pytest -q tests
```

`pip install -e ".[dev]"` installs the same test tooling on top of an
environment that already has the runtime. What it must not be asked to do is
build the runtime: `pyproject.toml` carries loose ranges, so installing from it
alone gives a different pandas, matplotlib and scikit-learn from the ones every
figure and table in this study was rendered on. `requirements.txt` is the pinned
set and is what the numbers below were produced under.

The suite runs on synthetic fixtures and prints its own count - quoting it here
would be one more number to keep true. They pin the study's invariants, not just
the plumbing. The ones that matter most to a reader of the paper:

| Test file | What it pins |
|---|---|
| `test_digits_p18.py` | the participation rule `floor((1-d)*n)` — 9 of 10, 16 of 20, 4 of 5, 8 of 10 — swept against exact integer arithmetic over every size to 100 and every whole-percent rate; the trimmed-mean survivor counts checked against the real aggregation function |
| `test_digits_p20.py` | that the 20-client cohort **extends** the 10-client one; that a larger cohort brings its own book cut by the same rule |
| `test_digits_p11.py`, `test_digits_p13.py` | the screen cell tables and their append-only boundary extensions (deployed files regenerate byte-identical) |
| `test_agg_screen.py`, `test_reg_screen.py` | the selectors: deterministic tie-breaks, one cell per method, refusal to select on no evidence |
| `test_study_runner.py` | the array runner's checks fire in an order where the variables they read exist — it *executes* the script body rather than only parsing it |
| `test_nist28_pipeline.py` | the 28x28 conversion and the packed cache contract |
| `test_selection.py`, `test_outlier_selection.py` | the two-phase g-init/g-0 selection and its no-leakage property |
| `test_release_artifacts.py` | that no absolute machine path survives in the release; that the fetcher's checksums are the documented ones; that the scaling table pins fold 1 explicitly and carries no spread on a single-fold cell |
| `test_paper_figures.py` | that every CSV view a figure, `numbers.tex` or a table reads is shipped and resolves, that round 0 of every trace is one shared point, that no view carries a column its tool no longer writes, and that the signals table's delta column is the difference of the two numbers printed beside it - the one column in the paper not rounded once from full precision, and the reason three of its rows used to be a hundredth away from the subtraction a reader does |
| `test_plateau.py` | what the plateau stopping rule means, on synthetic traces; that its basis is `stopping_table.py`'s own rather than a copy; and that the shipped view still says +1.19p over the fixed horizon on all 83 arms, so a silently regenerated table is a failure and not a diff |
| `test_plateau_holdout.py` | that the plateau's setting is what all 28 hold-out protocols choose - held out by fold, by stage and by seeded random halves of the arms - that none of them loses on the set it did not see, that the two extremes keep the published rounds under every one, and that the three shipped views regenerate byte for byte from an assembled root |
| `test_reviewer_tree.py` | that all twenty-four task files regenerate from their generators byte for byte, and - against an assembled root - that the signals pass over the population `signals_population.txt` names reproduces the two shipped signal extracts byte for byte. Both were promises the documents made and nothing checked |
| `test_shipped_view_membership.py` | that the blend's finals reach every regularisation view a reg-full arm belongs in, that no core view carries an arm of the joint-tuning extension, and that the record generator counts a stage's folders without the ones a later stage lodged under its prefix |
| `test_appendix_tables.py` | the four things an appendix table can say about itself and not keep: that the fairness table's Mean column is the same cohort accuracy the score tables print for the same arm (it was an unweighted mean over clients, and disagreed by up to a point), that it lists every carried arm at every setting (there was no five-client block and one arm of three at twenty), that the selection-axis ranks are computed on the selection score rather than on the stored adaptation ordering and that each shortlist is the first three distinct methods of that ordering, that both schedule blocks of the regularisation table print a no-penalty control, and that all 28 hold-out protocols reach the page |

### The shipped artefacts

> **Provenance paths were normalised for release.** Fields recording which model
> or fold book produced an artefact carried the absolute path of the machine
> that ran the study; they now read `$FOA_STUDY_DIR/...`, the same placeholder
> the task files use, so a reader can expand them. Only path-valued strings were
> rewritten — no accuracy, count, writer id or rule string was touched, and
> `tools/sanitize_artifacts.py --check` re-proves it at any time.

```bash
cd study/artifacts && sha256sum -c SHA256SUMS
cd ../jobs        && sha256sum -c SHA256SUMS
```

Both manifests print their own totals - 19 submission chains, and the derived
artefacts of `study/artifacts`, which grows whenever a view or a record is added
and is therefore not quoted here. **No model checkpoint is among them**, because
this study ships no weights (`study/UPSTREAM.sha256` says so, and the metadata
core's own README names them as the one thing it excludes). What the manifests
prove is that you hold the fold assignments, writer lists, selection records,
task files and experiment definitions that produced the published numbers.

**No rendered image is in either manifest, on purpose.** Every `*.png` under
`study/artifacts` is redrawn by matplotlib from a CSV view that *is* in the
manifest, and a redraw is not byte-stable across machines: on a different
matplotlib the six `signals/signals_pareto_*.png` come out 10-13% different
pixel for pixel and `figures/extreme_stopping.png` about 7.5%, while
`tables/stopping/stopping_extreme.csv` and the signals extracts they are drawn
from regenerate byte for byte. Hashing the picture would fail a reviewer whose
numbers are right, so `tools/artifact_checksums.py` leaves renders out of the
manifest and out of the unlisted-files report both - the data is checksummed and
the pictures are checked by redrawing them and diffing their views
(`docs/REPRODUCE.md` [section 6](REPRODUCE.md#6-which-command-produces-which-claim)).

Six run records are in there too, which looks like an exception and is not:
`signalcheck_*` and the five `d01_extreme_single_*` folds are the six runs of
the frozen signals population that the records asset does not carry, and a pass
cannot be reproduced without the runs it was taken over.
`study/artifacts/Digits_study01/signals_population.txt` says which runs those
are and why.

Both manifests are written from the tree by

```bash
python tools/artifact_checksums.py --dir study/artifacts --check
python tools/artifact_checksums.py --dir study/jobs      --check
```

which is the same comparison plus the one `sha256sum -c` cannot make: a file
that exists and is in no manifest verifies perfectly by never being mentioned.

### The task files parse

Every line of every task file is a real `foa` command. Check them without
running anything:

```bash
python - <<'PY'
import glob, shlex, sys
sys.path.insert(0, "src")
from federated_outlier_adaptation.cli import build_parser
p = build_parser()
n = 0
# The runner expands these before exec; substitute them to parse offline.
SUB = {"$FOA_STUDY_DIR": "/study", "$G0_FOLD": "4", "$GINIT_FOLD": "3"}
for f in sorted(glob.glob("study/artifacts/Digits_study01/jobs/*.txt")):
    for line in open(f):
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith("foa "):
            continue                      # comments and the chain's shell steps
        for k, v in SUB.items():
            line = line.replace(k, v)
        p.parse_args(shlex.split(line)[1:]); n += 1
print(n, "task lines parse")
PY
```

And that the runner accepts them. **`FOA_PROJECT_DIR` has to be exported
first** - it is the writable root every other path is derived from, and the
runner refuses by name without it rather than inventing one. Nothing is written
under it here; `FOA_DRY_RUN=1` executes nothing.

```bash
export FOA_PROJECT_DIR=/path/to/workspace
FOA_DRY_RUN=1 SLURM_ARRAY_TASK_ID=1 bash slurm/study_phase.sbatch \
    study/artifacts/Digits_study01/jobs/s20_combos4.txt
```

### The selection records agree with each other

The cohort of every stage is derived, never restated. You can check the chain
without a GPU:

```bash
python - <<'PY'
import json
J = lambda p: json.load(open(f"study/artifacts/Digits_study01/{p}"))
flat = {k: v for e in J("outliers/bad_acc_on_g0.json") for k, v in e.items()}
order = sorted(flat, key=lambda w: (flat[w], w))
ten    = J("outliers/cohort_worst10.json")["clients"]
twenty = J("outliers/cohort_worst20.json")["clients"]
five   = J("outliers/cohort_worst5.json")["clients"]
assert order[:10] == ten,     "cohort_worst10 is not the worst 10 of the g-0 ranking"
assert twenty[:10] == ten,    "cohort_worst20 does not extend cohort_worst10"
assert five == ten[:5],       "cohort_worst5 is not the first five"
print("cohort chain consistent:", ten[:3], "...")
PY
```

---

## Level 2 — with the cache, no GPU (an hour, mostly download)

Build the cache (`docs/DATA.md`) and check the five index numbers and the
`nist28_images.npy` hash. If that hash matches, your data is byte-identical to
ours, and every split below is checkable:

```bash
python - <<'PY'
import numpy as np, json, os
from federated_outlier_adaptation.data.fold_book import FoldBook
b = FoldBook.load("study/artifacts/Digits_study01/fold_books/cohort10.foldbook.npz")
c = json.load(open("study/artifacts/Digits_study01/outliers/cohort_worst10.json"))["clients"]
for w in c:
    assert b.covers(w), w
    for f in (1, 2, 3, 4, 5):
        for part in ("train", "val", "test"):
            assert b.part(f, w, part), (w, f, part)
print("cohort10 book: 10 writers x 5 folds x 3 parts, none empty")
PY
```

---

## Level 3 — verify a released model reproduces a published row (minutes, 1 GPU)

This is the strongest check that does not re-run training. `foa evaluate-book`
scores a saved model on one part of one fold of a fold book — exactly the
command that produced the study's do-nothing baseline row.

**Verify the g-0 preservation row** (g-0 on the old writers' test rows):

```bash
foa evaluate-book --results-dir "$FOA_STUDY_DIR" --resolution 28 --classes digits \
    --model-path  "$FOA_STUDY_DIR/g0_model" \
    --fold-book   "$FOA_STUDY_DIR/fold_books/old_data.foldbook.npz" --fold 1 --part test \
    --clients-file "$FOA_STUDY_DIR/outliers/old_data.json" \
    --batch-size 256 --tag check_old_fold1 --out /tmp/check.json
```

Compare against the shipped
`study/artifacts/Digits_study01/g0_evaluations.json`, key `old_data_fold1`.

**Verify the do-nothing row** (g-0 on the ten-writer cohort's test rows — the row
the **ten-client** numbers are read against; the other settings have books of
their own, below):

```bash
foa evaluate-book --results-dir "$FOA_STUDY_DIR" --resolution 28 --classes digits \
    --model-path  "$FOA_STUDY_DIR/g0_model" \
    --fold-book   "$FOA_STUDY_DIR/fold_books/cohort10.foldbook.npz" --fold 1 --part test \
    --clients-file "$FOA_STUDY_DIR/outliers/cohort_worst10.json" \
    --batch-size 256 --tag check_cohort_fold1 --out /tmp/check.json
```

Compare against
`study/artifacts/Digits_study01/g0_perfold_evaluations.json`, key `cohort_fold1`. That file also carries the per-writer column, so you can check
an individual writer, not only the pool.

**There are four of these books, and a setting is read against its own.** `A0` is
the shipped model's accuracy on the writers a setting federates, so it moves when
the writers do; `P0` is its accuracy on the source population, which is one
population and so one number — 0.9986, from `g0_evaluations.json` — everywhere.

| book | cohort | keys | fold-mean accuracy | the `A0` of |
|---|---|---|---|---|
| `g0_perfold_evaluations.json` | worst ten | `cohort_fold1`..`cohort_fold5` | 0.8225 | every search stage, and the ten-client carry setting |
| `g0_c5_evaluations.json` | worst five | `c5_cohort_fold1`..`c5_cohort_fold5` | 0.8032 | five clients, one dropped |
| `g0_c20_evaluations.json` | worst twenty | `c20_cohort_fold1`..`c20_cohort_fold5` | 0.8441 | twenty clients, at both dropout rates |
| `g0_extreme_evaluations.json` | the worst two | `extreme_cohort_fold1`..`extreme_cohort_fold5` | 0.7590 | both extreme arrangements |

Each verifies by the command above with its own fold book and clients file. The
20-client one swaps in `fold_books/cohort20.foldbook.npz` and
`outliers/cohort_worst20.json`; the 5-client one `fold_books/cohort5.foldbook.npz`
and `outliers/cohort_worst5.json`.

The extreme book is the one that does not look like the others, because its two
writers are two of the ten and so are cut from the **ten-client** fold book —
only the clients file narrows it:

```bash
foa evaluate-book --results-dir "$FOA_STUDY_DIR" --resolution 28 --classes digits \
    --model-path  "$FOA_STUDY_DIR/g0_model" \
    --fold-book   "$FOA_STUDY_DIR/fold_books/cohort10.foldbook.npz" --fold 1 --part test \
    --clients-file "$FOA_STUDY_DIR/outliers/extreme_double.json" \
    --batch-size 256 --tag extreme_cohort_fold1 --out /tmp/check.json
```

Compare against
`study/artifacts/Digits_study01/g0_extreme_evaluations.json`, key
`extreme_cohort_fold1`; the five folds mean **0.7590**, which is the number both
extreme arrangements' scores subtract. Those five lines were emitted by
`tools/make_size_references.py --only do-nothing` into
`jobs/d01_extreme_references.txt` — `docs/REPRODUCE.md` §6 has the invocation.

**This level needs a `g0_model` and the study does not ship one.** The weights
are outputs of the programme, not records of it: they cannot be diffed, they
answer no question the evaluation books answer, and a committed copy is how a
stale checkpoint outlives the ranking that produced it - which is the reason
`study/UPSTREAM.sha256` stopped listing them and the reason the metadata core's
README names them as the one artefact it excludes. So Level 3 is available to a
reader who has re-run the g-0 stage, or to one the owner has published a
checkpoint to; it is the only level of the four that is.

What ships instead is everything that model was measured with and against: the
evaluation books above, the fold books the rows come from, the cohort records,
and - since the figure pipeline landed - g-0's own per-fold training history in
`g0_fold*/global_results/global_metrics.json`, which is what
`tools/paper_figures/fig_baselines.py` draws the shipped model from.

---

## Level 4 — re-run a stage (hours to days, GPU)

Follow `docs/RUNBOOK.md`. The cheapest meaningful rung is **the extremes**,
`jobs/d01_extreme.txt`: 10 tasks and under an hour of GPU, the smallest stage
of the programme.

**It cannot be the first thing you run.** All ten of its lines carry
`--init global --global-name g0` and load `$FOA_STUDY_DIR/g0_model`, which this
study does not publish - and no stage anywhere in the programme runs a federated
round without g-0, so no cheaper federated rung exists. The shipped model comes
from three short stages first: `s01a_book.txt` (2 tasks, ~1 min),
`s01b_detector.txt` (2 tasks, ~21 min, trains g-init) and `s02_selection.txt`
(29 tasks, ~23 min, trains the five g-0 folds and crowns one). Those are the
scheduler's measured times for the array that produced this study, on one A100.
About **1.75 GPU-hours and 43 tasks** to the cheapest real result. The README's
Quickstart carries the same table.

The most expensive are the screens: the aggregation screen (480 tasks,
~16 GPU-h), the regularisation screen (700 tasks, ~25 GPU-h) and the blend's own
screen (1,170 tasks, ~30 GPU-h). You do not
need either to check the reported winners — the winners and their evidence are
shipped in `study/artifacts/Digits_study01/tables/`, and
`study_emit.py --rank-by {val,test}` will re-derive the ordering from whatever
run folders you do have. `docs/REPRODUCE.md` §3 has every stage with its task
file and its task count, and §9 the GPU-hours quoted here.

---

## What a reviewer can verify without a GPU

- Every test in the suite - the count is the one `pytest -q tests` prints.
- Every task file parses and passes the runner's guard.
- Every derived artefact matches its checksum.
- The cohort chain: worst-5 ⊂ worst-10 ⊂ worst-20, all cut from one ranking.
- The fold books: coverage, and no empty partition in any fold.
- The participation rule and the trimmed-mean survivor counts, against the
  real functions.
- With the cache built (CPU only): that the data is byte-identical to ours.

**What needs a GPU:** producing accuracy numbers. Everything about *how* they
were produced is checkable without one.
