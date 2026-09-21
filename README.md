# Federated outlier adaptation

Adapting a shipped handwriting model to the writers it fails on, without losing
the writers it already serves.

A global model is trained on a population of writers and deployed. A small group
of **outlier writers** — the ones it reads worst — then federate to adapt it.
The question this repository answers is what that adaptation costs: how much
accuracy the outliers gain, how much the original population loses, and which
server rule and which client-side penalty trade those two off best.

The study is **`Digits_study01`**: NIST Special Database 19 by writer, digits
0–9, 28×28, a FedAvg CNN of 1,663,370 parameters, ten outlier clients against
two hundred retained writers, five-fold cross-validation throughout.

---

## What is here

| Path | Contents |
|---|---|
| `src/federated_outlier_adaptation/` | the package and the `foa` CLI (28 subcommands) |
| `tools/` | task-file generators, screen selectors, the SD19 fetcher, the table generator, the figure-view exporters |
| `tools/paper_figures/` | the manuscript's seven figures, its `numbers.tex` and its twelve tables: render only, from the shipped CSV views |
| `slurm/` | the array runner, the data-preparation job, a plain-bash fallback |
| `tests/` | the suite, no GPU and no dataset required - `pytest -q tests` prints the count |
| `study/jobs/` | the submission chains, one TOML per wave, with a `SHA256SUMS` |
| `study/artifacts/` | the derived artefacts needed to *check* results, every task file the study ran among them, with a `SHA256SUMS` |
| `study/UPSTREAM.sha256` | checksums of the source data and the packed cache |
| `docs/` | how to reproduce a claim, the study record, data path, runbook, verification, prior pipeline |

`study/artifacts/` is the part that makes this checkable without re-running
anything: the writer pools and their scores, the three cohorts, all four fold
books, the g-init and g-0 selection records, the baseline evaluations, every
selection table each stage produced, every task file that was submitted, and
the thirty-seven CSV views the manuscript's figures, numbers and tables are
built from — so every figure redraws, and every number and every table
regenerates, from a clone, with no dataset, no GPU and no release asset:

```bash
FOA_PAPER_OUT=/tmp/figures python tools/paper_figures/fig_problem.py
FOA_PAPER_OUT=/tmp/paper   python tools/paper_figures/make_paper_tables.py
```

The views themselves are built from the published records by
`tools/export_traces.py`, `tools/export_baseline_views.py`,
`tools/export_combo_folds.py` and the reporting tools, which is what makes the
figures evidence rather than pictures; [`docs/REPRODUCE.md`](docs/REPRODUCE.md)
§6 maps each figure to its views, its export command and its script, and does
the same for `numbers.tex` and the tables.

**No dataset is redistributed here.** NIST SD19 is downloaded from NIST; this
repository ships the conversion code, the exact command, and the checksums that
prove your cache matches ours.

---

## Quickstart

```bash
git clone https://github.com/keshtkarsadegh/federated-outlier-adaptation.git && cd federated-outlier-adaptation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e . --no-deps
export FOA_ENV="$PWD/.venv"                  # the runner resolves this; see below

pytest -q tests                              # ~2 minutes, no GPU; it prints the count
cd study/artifacts && sha256sum -c SHA256SUMS && cd ../..
```

**`python3`, not `python`.** Ubuntu and Debian ship `/usr/bin/python3` and
no `python` at all, so the one line that has to run before the venv exists is
the one line that cannot say `python`. Inside the activated venv either name
works, which is why the commands elsewhere in this file and throughout `docs/`
say `python`.

**Install from `requirements.txt`, not from `pyproject.toml` alone.** `pip
install -e .` resolves the loose ranges in `pyproject.toml` and gives you a
different pandas, matplotlib and scikit-learn from the ones the study ran on -
the figures and tables then render on a matplotlib this study never used.
`requirements.txt` carries the pins. `pip install -e ".[dev]"` adds pytest,
black and isort to an install that already has the runtime.

**`export FOA_ENV` is not decoration.** `slurm/env.sh` looks for an environment
under `$FOA_PROJECT_DIR/envs/`, which is not where this Quickstart puts one; with
neither found it falls through to whatever `python3` is on PATH, and a system
python that happens to import torch passes its checks. Naming the venv is what
makes the runner use the interpreter you just built. Every job logs which python
it chose - read that line.

That verifies the code and the shipped artefacts. To check that the study's
selection chain is internally consistent — that each cohort really is cut from
the ranking it claims — run the snippet in [`docs/VERIFY.md`](docs/VERIFY.md)
(no GPU, no data).

To get the data and build the cache (no GPU, ~10 min of conversion):

```bash
python tools/fetch_sd19.py --dest "$FOA_DATA_DIR/nist"   # fetch + verify + resume
foa prepare-data --dataset nist --zip "$FOA_DATA_DIR/nist/by_write.zip" \
    --out "$FOA_NIST28_DIR" --resolution 28 --classes all
```

### Running something for real

**No GPU, no data, seconds: the dry run.** `FOA_DRY_RUN=1` walks every line of a
task file, expands it, parses it against the real CLI and executes nothing. It
is the check to run first and it needs nothing but the clone.

```bash
export FOA_PROJECT_DIR=/path/to/workspace
export FOA_STUDY_DIR="$FOA_PROJECT_DIR/results/studies/Digits_study01"
JOBS=study/artifacts/Digits_study01/jobs
FOA_DRY_RUN=1 slurm/run_tasks.sh $JOBS/d01_extreme.txt   # checks, runs nothing
```

**With a GPU and the cache built, the cheapest real rung is still the extremes -
but it is not the first thing you can run.** All ten of its task lines carry
`--init global --global-name g0`: they load `$FOA_STUDY_DIR/g0_model`, and **this
study publishes no weights** (`docs/VERIFY.md` Level 3 says why). No stage
anywhere in the programme runs a federated round without g-0, so there is no
federated smoke test that skips it. The shipped model has to be trained first,
and that is three short stages rather than the whole programme:

| # | task file | tasks | what it does | measured, one A100 |
|---|---|---|---|---|
| 2 | `s01a_book.txt` | 2 | writer census, all-writer fold book | ~1 min |
| 3 | `s01b_detector.txt` | 2 | trains **g-init** and scores every writer with it | **~21 min** |
| 4 | `s02_selection.txt` | 29 | cuts the pools, draws the 200 source writers, trains **g-0** once per fold, crowns one, cuts the three cohorts | **~23 min** |
| 17 | `d01_extreme.txt` | 10 | the extremes themselves | ~1 GPU-h |

```bash
for f in s01a_book s01b_detector s02_selection d01_extreme; do
    slurm/run_tasks.sh $JOBS/$f.txt        # or sbatch --array, as below
done
```

**About one and three quarter GPU-hours to the cheapest real result, 43 tasks.**
The three preparatory stages are cheap because they are centralised trainings on
one GPU with early stopping: g-init stopped after 23 epochs and the five g-0
folds after 20 to 50, read off the shipped `global_metrics.json`. The times are
the scheduler's own, for the array that produced this study.

Two riders. To *read* the extremes against doing nothing you also need
`d01_extreme_references.txt` - five `foa evaluate-book` lines, forward passes,
minutes (`docs/REPRODUCE.md` section 6 has the command that emits it). And
`tools/fetch_mnist.py` is not optional if you want the forgetting signals: two
of the eight are silently empty without the proxy set.

Every task file the study ran ships in the metadata core.

---

## The documents

1. **[`docs/REPRODUCE.md`](docs/REPRODUCE.md)** — given a number in the paper,
   what to run to get it back. The stage graph with its exact task counts, the
   command that regenerates each task file byte for byte, the seed scheme, a
   claim-to-command table covering every result the manuscript states, and the
   figure-to-view-to-script table covering every figure it prints. Start
   here if you want to check something rather than re-run everything.
2. **[`docs/STUDY_RECORD.md`](docs/STUDY_RECORD.md)** — what the study actually
   ran, generated from the study root rather than written: stages, task counts,
   the result folders each produced, and the selections that were made.
3. **[`docs/DATA.md`](docs/DATA.md)** — download SD19, verify it, build the
   packed 28×28 cache, and confirm it byte-for-byte. The numbers that must
   match: **814,255 rows, 3,597 writers, 62 classes; 402,953 digit rows across
   3,580 writers.**
4. **[`docs/RUNBOOK.md`](docs/RUNBOOK.md)** — every stage in order, with its
   task file, its task count, and its **measured** GPU-hours. The whole
   production chain is about **53 GPU-h**.
5. **[`docs/VERIFY.md`](docs/VERIFY.md)** — four levels of checking, two of
   which need neither a GPU nor the dataset, and the `evaluate-book` command
   that proves a released model still reproduces a published row.
6. **[`docs/STOPPING.md`](docs/STOPPING.md)** — what the fixed hundred-round
   horizon cost every arm, what a signal a deployed system is allowed to
   compute could have recovered instead, and the plateau on the cohort's own
   accuracy that recovers **+1.19p of the oracle's 1.48p** without reading
   anything a deployed server has lost.
7. **[`docs/FAIRNESS_AND_COST.md`](docs/FAIRNESS_AND_COST.md)** — who the gain
   reached, per client rather than pooled, and what the programme spent in
   seconds and bytes.
8. **[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)** and
   **[`docs/RESULTS_LAYOUT.md`](docs/RESULTS_LAYOUT.md)** — the earlier
   pipeline, kept because its code is still present. Both carry a superseded
   banner naming what replaced them; neither describes the live study.

---

## How the study is put together

### Everything is a task file

One `foa` command per line; one array element per line; one runner for all of
them. A stage is a text file, which is what makes the experiment definition
reviewable before it costs anything.

```bash
JOBS=study/artifacts/Digits_study01/jobs           # every task file that ran
mkdir -p "$FOA_STUDY_DIR/logs"                     # --output needs it to exist
sbatch --account=$FOA_ACCOUNT --partition=$FOA_GPU_PARTITION --gres=gpu:1 \
       --export=ALL,FOA_PROJECT_DIR=$FOA_PROJECT_DIR,FOA_STUDY_DIR=$FOA_STUDY_DIR \
       --output="$FOA_STUDY_DIR/logs/%x_%A_%a.log" \
       --array=1-N slurm/study_phase.sbatch $JOBS/s20_combos4.txt
slurm/run_tasks.sh $JOBS/s20_combos4.txt           # no scheduler
FOA_DRY_RUN=1 slurm/run_tasks.sh $JOBS/...         # every check, no execution
```

> **`--export` is not optional.** Many sites default to `--export=NONE`, so the
> submitting shell's environment does not reach the job — and `--export=ALL`
> alone is exactly the default being overridden. Name the variables by value.
> Without them `env.sh` refuses at startup, correctly, once per array element:
> a 296-element pair of screens spent about ten GPU-hours saying so.

The runner refuses a task line whose paths point outside the study root —
checked *before* the variable is expanded, so a path pasted from another study
is caught even though it would have expanded perfectly well.

### Selection is two-phase, and cannot leak

`g-init` trains on every writer, so its ranking is contaminated by
memorisation; it is asked only for a coarse BAD/GOOD cut. The two hundred
retained writers are drawn from GOOD. **`g-0`** is trained on those, and it is
`g-0` — a model that has never seen a BAD writer — that scores the BAD pool and
whose worst ten become the cohort. No cohort writer influenced the model that
selected it, by construction.

### Splits are written down, not recomputed

A **fold book** is a persisted CV split: an `int8 [folds, rows]` assignment
array, per-writer stratified 60/20/20 over five folds. Runs read their train,
validation and test rows out of the book rather than re-deriving them, so two
runs of the same fold see the same rows — and a reviewer can check the split
itself, not just the result. The four books ship in
`study/artifacts/Digits_study01/fold_books/`, and `study/artifacts/SHA256SUMS`
carries their hashes.

### Participation is one expression

```
participants(n, dropout) = floor((1 - dropout) * n)
```

Nine of ten for this study, eight of ten at the dropout point, four of five and
eighteen or sixteen of twenty on the scaling ladder. It lives once, in
`training/study_config.py`; no task file carries a literal, and the study
configuration asserts at import that its own `clients_per_round` still equals
the rule. The extremes are the one documented exception: at one or two clients
the rule leaves nobody, and dropping a client from a two-client federation is a
coin flip on whether the round happens, so those cases run at full
participation.

### Every run reports three categories

`final_evaluation` carries the cohort's pooled test accuracy, a per-client
column, and the old data's five fold test partitions scored **separately** with
mean ± sd. Adaptation is read from the first, preservation from the second. A
method that buys one at the other's expense cannot hide it.

### Screens do not choose their own winners

A screen runs at 25 rounds; a selector reads its results and emits the
full-horizon file. The selectors refuse to emit a selection built on no
evidence — a screen that produced nothing would otherwise yield a task file
that looks exactly like a real one — and every emission is checked against a
mandatory `--expect` count, because the array that consumes it is submitted
with a fixed range.

---

## Known gaps

Stated plainly, because a reproducibility claim is only worth what its
exceptions are:

- **`foa prepare-data` defaults to 128 px.** The study needs `--resolution 28`,
  and a 128 px cache will train happily and reproduce nothing. `docs/DATA.md`
  says so twice.
- **The scaling rungs are reported on fold 1 only** (P18, P19, P20) — a
  deliberate design, not an omission: those probes ran one g-0 and one client
  split. P18 and P19 nonetheless have all five folds on disk, so any table over
  them must filter on fold 1 explicitly for *every* rung including the
  ten-client anchor, or it averages five folds for some rows and one for others
  and the difference reads as a size effect. `tools/make_tables.py` bakes that
  rule in and a test pins it; anything else reading those folders must do the
  same.
- **`slurm/prepare_data.sbatch`** is the prior pipeline's 128 px job. Use
  `slurm/prepare_nist28.sbatch`.
- Five generators in `tools/` (`make_stage*.py`, `emit_stage7_hybrid.py`) belong
  to a superseded 62-class study and target a results root that no longer
  exists. They are kept for provenance; the runner refuses their output.

---

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `FOA_PROJECT_DIR` | writable root for data, results, environments | **required** |
| `FOA_STUDY_DIR` | the one results root a study writes into | `$FOA_RESULTS_DIR/studies/Digits_study01` |
| `FOA_RESULTS_DIR` | results root | `$FOA_PROJECT_DIR/results` |
| `FOA_DATA_DIR` | data root | `$FOA_PROJECT_DIR/data` |
| `FOA_NIST28_DIR` | packed 28×28 cache | `$FOA_DATA_DIR/nist28` |
| `FOA_NIST_CLASSES` | `digits` (10) or `all` (62) | `digits` |
| `FOA_NIST_RESOLUTION` | 28 or 128 | `128` — the study needs **28** |
| `FOA_MODEL` | `fedavg_cnn` or `flexible_cnn` | `fedavg_cnn` |
| `G0_FOLD` | winning g-0 fold, for Fisher-weighted penalties | derived from `g0_selection.json` |
| `FOA_DRY_RUN` | run every check, execute nothing | unset |
| `FOA_ACCOUNT`, `FOA_GPU_PARTITION`, `FOA_CPU_PARTITION` | scheduler | unset — pass at submit time |
| `FOA_ENV` | conda prefix or virtualenv to run in - **set it to the Quickstart's `.venv`** | auto: `$FOA_PROJECT_DIR/envs/{foa,fal}`, else a single unambiguous `envs/*`, else `python3` on PATH |
| `FOA_PYTHON` | the interpreter the runner resolved | derived; printed by every job |
| `FOA_MODULES`, `FOA_HTTP_PROXY` | site specifics, both optional | unset |

Nothing site-specific is baked in: no account, no partition, no absolute path.
An unconfigured site gets plain behaviour rather than somebody else's cluster.

**The interpreter is resolved and vetted before anything runs.** Batch nodes
routinely put an old `/usr/bin/python3` first on PATH, and running under it
fails deep inside an import with a `SyntaxError` that reads as a broken source
file. `slurm/env.sh` prefers `$FOA_ENV`, else an environment under
`$FOA_PROJECT_DIR/envs/` - `foa`, then `fal`, then a single unambiguous
`envs/*` - and checks the version and that the package imports, **refusing** with
a message naming the interpreter rather than proceeding.

**What it does not do is guess.** With no `FOA_ENV` and no `envs/` it falls
through to `python3` on PATH, and if that interpreter passes both checks the job
runs under it - which is what happens to a reader who follows the Quickstart,
whose venv lives inside the clone and not under `$FOA_PROJECT_DIR/envs/`. That
is why the Quickstart exports `FOA_ENV`. Every job logs which python it chose,
and that line is the one to read when a result surprises you.

---

## Requirements

Python 3.12, PyTorch (CUDA 12.1 wheels pinned in `requirements.txt`), NumPy,
Pillow, matplotlib, pandas, SciPy. `environment.yml` for conda,
`requirements.txt` for pip - install the pins from it first and then
`pip install -e . --no-deps` for the `foa` entry point, which is what the
[Quickstart](#quickstart) does. `pip install -e .` on its own re-resolves the
loose ranges in `pyproject.toml` and hands you a pandas, matplotlib and
scikit-learn this study never ran on.

A GPU is needed only to produce accuracy numbers. Everything about how they were
produced — the splits, the selections, the task definitions, the participation
arithmetic — is checkable on a laptop.

---

## Before submission - the owner's checklist

Every document here gives public URLs, and none of them resolves while the
repository is private. A reader who cannot execute step 1 cannot execute
anything, so this is the first thing to check and the easiest to forget:

- [ ] the repository is **public** - `gh api repos/<owner>/<repo> --jq .private`
      prints `false`
- [ ] the **`records` release is public** and both assets are attached -
      `curl -fLI https://github.com/<owner>/<repo>/releases/download/records/Digits_study01_records.tar.gz`
      returns 200 unauthenticated
- [ ] the asset's sha256 is still the one `docs/REPRODUCE.md` section 10 quotes;
      it is replaced in place whenever a stage adds records
- [ ] `git log -1` is the commit the manuscript cites

`docs/REPRODUCE.md` section 10 carries the same list beside the URLs it hands
out.

## Citation

See [`CITATION.cff`](CITATION.cff).

## Licence

See [`LICENSE`](LICENSE). NIST SD19 is distributed by NIST under its own terms
and is not included here.
