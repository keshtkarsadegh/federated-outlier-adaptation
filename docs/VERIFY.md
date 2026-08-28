# Verifying results

Three levels, in increasing cost. The first two need **no GPU and no SD19
download**.

---

## Level 1 — no data, no GPU (minutes)

### The test suite

```bash
pip install -e ".[dev]"    # or: pip install -r requirements.txt
pytest -q tests
```

1,648 tests, on synthetic fixtures. They pin the study's invariants, not just
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

### The shipped artefacts

```bash
cd study/artifacts && sha256sum -c SHA256SUMS
cd ../jobs        && sha256sum -c SHA256SUMS
```

54 derived artefacts and 40 task files. This proves you hold the fold
assignments, writer lists, selection records and experiment definitions that
produced the published numbers.

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
for f in sorted(glob.glob("study/jobs/*.txt")):
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

And that the runner accepts them:

```bash
FOA_DRY_RUN=1 SLURM_ARRAY_TASK_ID=1 bash slurm/study_phase.sbatch study/jobs/d01_p15.txt
```

### The selection records agree with each other

The cohort of every stage is derived, never restated. You can check the chain
without a GPU:

```bash
python - <<'PY'
import json
J = lambda p: json.load(open(f"study/artifacts/{p}"))
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
b = FoldBook.load("study/artifacts/fold_books/cohort10.foldbook.npz")
c = json.load(open("study/artifacts/outliers/cohort_worst10.json"))["clients"]
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

Compare against the shipped `study/artifacts/g0_evaluations.json`, key
`old_data_fold1`.

**Verify the do-nothing row** (g-0 on the cohort's test rows — the row every
adaptation number is read against):

```bash
foa evaluate-book --results-dir "$FOA_STUDY_DIR" --resolution 28 --classes digits \
    --model-path  "$FOA_STUDY_DIR/g0_model" \
    --fold-book   "$FOA_STUDY_DIR/fold_books/cohort10.foldbook.npz" --fold 1 --part test \
    --clients-file "$FOA_STUDY_DIR/outliers/cohort_worst10.json" \
    --batch-size 256 --tag check_cohort_fold1 --out /tmp/check.json
```

Compare against `study/artifacts/g0_perfold_evaluations.json`, key
`cohort_fold1`. That file also carries the per-writer column, so you can check
an individual writer, not only the pool.

The same pattern verifies the 20-client baseline against
`g0_cohort20_evaluations.json` (`cohort20_fold1`..`fold5`).

Both checkpoints ship in `study/artifacts/models/` (6.6 MB each) and are
covered by `study/artifacts/SHA256SUMS`, so `sha256sum -c` at Level 1 has
already checked them. Point `--model-path` straight at them if you have not
re-run the training:

    --model-path study/artifacts/models/g0_model

Their hashes, for reference:

```
48689ec85d5921b8bd36fa53545fc368f8e9441f4acbec158da36c4e8b09f950  ginit_model
cb06fb83b17b3f85730e3763b70560b293b689068d1adc96774f15bd0f2989ed  g0_model
```

---

## Level 4 — re-run a stage (hours to days, GPU)

Follow `docs/RUNBOOK.md`. The cheapest meaningful rung is **P09**, the plain
FedAvg baseline: 10 tasks, ~0.5 GPU-h, and it produces the 9-of-10 reference
row the whole study is read against.

The most expensive are the two screens: P11 (845 tasks, ~15 GPU-h) and P13
(585 tasks, ~17 GPU-h). You do not need them to check the reported winners —
the winners and their evidence are shipped in `study/artifacts/tables/`, and
`study_emit.py --rank-by {val,test}` will re-derive the ordering from whatever
run folders you do have.

---

## What a reviewer can verify without a GPU

- Every test in the suite (1,648).
- Every task file parses and passes the runner's guard.
- Every derived artefact matches its checksum.
- The cohort chain: worst-5 ⊂ worst-10 ⊂ worst-20, all cut from one ranking.
- The fold books: coverage, and no empty partition in any fold.
- The participation rule and the trimmed-mean survivor counts, against the
  real functions.
- With the cache built (CPU only): that the data is byte-identical to ours.

**What needs a GPU:** producing accuracy numbers. Everything about *how* they
were produced is checkable without one.
