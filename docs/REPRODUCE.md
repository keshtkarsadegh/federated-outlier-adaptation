# Reproducing any result in `Digits_study01`

This document answers one question: **given a number in the paper, what do I run
to get it back?**

It is written around commands, not values. A document that quotes results goes
stale the first time a stage is re-run; one that names the command that prints
them does not. Every table in the manuscript comes from a tool in `tools/`, and
the mapping is in [§6](#6-which-command-produces-which-claim).

---

## 1. What is reproducible, and in what sense

**Decisions are reproducible; bits are not.** GPU floating-point reduction order
is not deterministic, so re-running a task produces accuracies that differ in the
last places. What must survive is the *decision*: the same cohort, the same
winner, the same ranking. `tools/freeze_selection.py verify` checks exactly that,
and it is the right question - two runs whose files differ byte for byte but
whose selections agree are reproducible for this study's purposes.

Three things ARE bit-exact and are checked as such:

| artefact | why it must be exact | checked by |
|---|---|---|
| the dataset index | the study is defined on it | `UPSTREAM.sha256`, `tools/compare_study_roots.py` |
| the fold books | a different split is a different experiment | seeded on `sha256(seed｜fold｜writer)` |
| the task files | a stage is its lines | regenerate and `diff` |

---

## 2. Environment and data

Python 3.12, PyTorch 2.3.1 / CUDA 12.1.

```bash
git clone <repo> && cd federated-outlier-adaptation
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt && pip install -e . --no-deps

export FOA_PROJECT_DIR=/path/to/workspace          # holds data/ results/ envs/
export FOA_STUDY_DIR=$FOA_PROJECT_DIR/results/studies/Digits_study01
export FOA_NIST28_DIR=$FOA_PROJECT_DIR/data/nist28
export FOA_ACCOUNT=<slurm account>
```

The dataset is never extracted - `by_write.zip` is read in place and becomes one
array plus one JSON index. Unzipping ~400,000 PNGs exhausts the inode quota of
most shared filesystems. See `docs/DATA.md`.

```bash
python tools/fetch_sd19.py            # then verify against UPSTREAM.sha256
python tools/fetch_mnist.py           # the proxy set; without it two of the
                                      # eight forgetting signals are silently empty
```

---

## 3. The stage graph

Each stage is a **task file** - one `foa ...` command per line - run as a Slurm
array where element *N* runs line *N*. One script runs every stage:

```bash
sbatch --account=$FOA_ACCOUNT --partition=<gpu partition> --gres=gpu:1 \
       --array=1-<N>%<concurrency> \
       --export=ALL,FOA_PROJECT_DIR=$FOA_PROJECT_DIR,FOA_STUDY_DIR=$FOA_STUDY_DIR \
       slurm/study_phase.sbatch $FOA_STUDY_DIR/jobs/<file>.txt
```

| # | stage | generator | task file | tasks | rounds |
|---|---|---|---|---|---|
| 1 | dataset index | `tools/fetch_sd19.py` | - | - | - |
| 2 | fold books | `d01_p02.txt` | `d01_p02.txt` | 4 | - |
| 3 | g-init / detector | `s01b_detector.txt` | `s01b_detector.txt` | 2 | - |
| 4 | g-0 + cohorts | `s02_selection.txt` | `s02_selection.txt` | 29 | - |
| 5 | references | `s03_refs_c10.txt`, `s03b_refs_fl.txt` | | 155 | 100 |
| 6 | **aggregation screen** | `tools/make_digits_p11.py` | `s09_agg_screen2.txt` | 480 | 25 |
| 7 | **aggregation finals** | `study_emit.py agg-full` | `s10_agg_full2.txt` | 85 | 100 |
| 8 | **regularisation screen** | `tools/make_digits_p13.py` | `s16_reg_screen3.txt` | 700 | 25 |
| 9 | **regularisation finals** | `study_emit.py reg-full` | `s17_reg_full4.txt` | 70 | 100 |
| 10 | **kd+fisher blend, concurrent** | `study_emit.py reg-hybrid` | `s18_hybrid.txt` | 15 | 100 |
| 11 | **kd+fisher blend, sequential** | `study_emit.py reg-hybrid --hybrid-family sequential` | `s19_hybrid_seq.txt` | 15 | 100 |
| 12 | **combinations** | `study_emit.py combos` | *emitted from both finals* | 90 | 100 |
| 13 | five-client references | `tools/make_size_references.py` | `d01_c5_references.txt` | 75 | 100 |
| 14 | twenty-client references | `tools/make_c20_references.py` | `d01_c20_references.txt` | 235 | 100 |
| 15 | **five clients, one dropped** | `study_emit.py five` | `d01_five.txt` | 20 | 100 |
| 16 | **twenty clients, both dropout levels** | `study_emit.py c20` | `d01_c20.txt` | 30 | 100 |
| 17 | **the extreme cases** | `study_emit.py extreme` | `d01_extreme.txt` | 15 | 100 |
| 18 | **ten clients, one dropped** | `study_emit.py c10d10` | `d01_c10d10.txt` | 20 | 100 |
| 19 | size evaluations, re-scored | *the evaluate-book lines of 13 and 14* | `d01_size_evals_rerun.txt` | 10 | - |

Stages 8-11 have run at the search rate and are reported in
`REG_GRID_RANGES.md` under the 2026-08-31 heading; stage 12 has not been
re-run against the shortlists those stages produced.

**Stages 13-18 are the carry settings and the extremes.** Nothing in them is
searched: they take the arms the combination cross crowned and move the
federation underneath them. Stage 18 is the study's own setting - nine of ten,
the participation every screen and every final drew - and it is a stage of its
own so the other three settings have a middle to be read against rather than a
number borrowed from the selection records. `docs/CARRY_SETTINGS.md` reports
what they found.

**Stage 19 is a repair, not a measurement.** It re-scores g-0 on the five- and
twenty-client cohorts because the two accumulator files those reference stages
wrote had been damaged by a race - see
[§5](#the-accumulator-race-and-why-both-files-were-deleted). It only reads
models, so it carries no sampler seed and changes nothing about the runs it
scores.

Superseded task files live in `jobs/superseded/`. They are kept as a record and
must not be re-run: their seeds and their cells belong to a previous programme.

**Screen then final.** A screen runs 25 rounds and *ranks*; a final runs 100 and
*reports*. Nothing from a screen is a reported number. The distinction is not
cosmetic - see [§7](#7-things-that-look-like-bugs-and-are-not).

---

## 4. Regenerating a stage's task file

Every task file is generated, never hand-edited, and regenerating one must
reproduce it byte for byte:

**No flags.** Every command below is the whole command; a task file that needs
an argument remembered by hand is a task file nobody can regenerate.

```bash
python tools/make_digits_p11.py --jobs-dir /tmp/check     # aggregation screen
python tools/make_digits_p13.py --jobs-dir /tmp/check     # regularisation screen
diff /tmp/check/d01_p11.txt $FOA_STUDY_DIR/jobs/s09_agg_screen2.txt
diff /tmp/check/d01_p13.txt $FOA_STUDY_DIR/jobs/s16_reg_screen3.txt

python tools/study_emit.py agg-full --root $FOA_STUDY_DIR --out /tmp/x.txt --expect 85
python tools/study_emit.py reg-full --root $FOA_STUDY_DIR --out /tmp/x.txt --expect 70
python tools/study_emit.py combos   --root $FOA_STUDY_DIR --out /tmp/x.txt --expect 90

python tools/study_emit.py reg-hybrid --root $FOA_STUDY_DIR --out /tmp/x.txt --expect 15
python tools/study_emit.py reg-hybrid --root $FOA_STUDY_DIR --out /tmp/x.txt --expect 15 --hybrid-family sequential

python tools/study_emit.py five    --root $FOA_STUDY_DIR --out /tmp/x.txt --expect 20
python tools/study_emit.py c10d10  --root $FOA_STUDY_DIR --out /tmp/x.txt --expect 20
python tools/study_emit.py c20     --root $FOA_STUDY_DIR --out /tmp/x.txt --expect 30
python tools/study_emit.py extreme --root $FOA_STUDY_DIR --out /tmp/x.txt --expect 15
```

`diff /tmp/x.txt $FOA_STUDY_DIR/jobs/s17_reg_full4.txt`, `s18_hybrid.txt` and
`s19_hybrid_seq.txt` close the loop on those three. **The blend is a per-family
object**: the two `reg-hybrid` commands build different penalties from different
rows of the same selection record, and each writes its own construction record -
`tables/p14_hybrid_construction.json` and
`tables/p14_hybrid_construction_sequential.json` - so re-running either against
the live study rewrites its own record with the same content and never the other
one's.

`diff /tmp/x.txt $FOA_STUDY_DIR/jobs/s10_agg_full2.txt` closes the loop on the
aggregation finals the same way.

The participation rate these stages emit at is **not** a flag - see
[§5](#5-seeds-and-the-rate-a-grid-is-searched-at). It was one, and the two grids ended up searched at different
rates because it was typed on one command line and not on the other.

---

## 5. Seeds, and the rate a grid is searched at

### The search rate is a constant

Both grids are searched at **one** participation rate, the harder of the two -
eight of ten clients per round rather than nine - and the winners are carried to
the other rates. A configuration chosen where the averaging is noisiest has a
better claim on the easier rate than the reverse would, and searching every rate
would multiply the most expensive stage in the programme by the number of rates.

That rate is `search_clients_per_round` in `StudyConfig`, and the grid emitters -
`make_digits_p11`, `make_digits_p13`, and `study_emit`'s `agg-full`, `reg-full`,
`reg-hybrid` and `combos` - read it from there. `reg-hybrid` is on that list
because the blend is reported beside the finals it is built from: at the study's
own rate it would be priced on an easier loop than the two winners it sits
between. The four stages that deliberately run at other rates (`five`, `drop20`,
`c20`, `extreme`) are unaffected: their rate is the thing they measure.

It used to be a `--clients-per-round` flag with the study's own 9 of 10 as the
default. The aggregation grid was given the flag; the regularisation grid was
not, so it was **searched at 9 of 10 against an aggregation table selected at 8**,
and the two tables are meant to be read against each other. The mismatch then
propagated the way the pipeline is designed to propagate a rate - each stage
matches the stage it selects from - so the regularisation finals and the
combinations inherited it. No error was raised, because no stage was ever told
what rate it should be at, only what rate the stage before it had used.

**Those results were discarded**: 700 screen runs, 70 finals and 34 partial
combinations were deleted rather than reported, and their task files are kept
under `jobs/superseded/` as a record of what ran. The corrected screen is
`s16_reg_screen3.txt`. It differs from the discarded `s11_reg_screen2.txt` in
exactly 701 lines - the 700 task lines and the header sentence that states the
rate - and in nothing else: the sampler seeds are unchanged at 710001-711395,
because that scheme is a function of a cell's index and not of the participation.

An explicit `--clients-per-round` still wins where one is passed, and it now sets
both fields, so a flag given to a grid stage reaches the lines rather than being
overwritten by the constant on exactly the stages the flag is for.

### Seeds

Every task that fits a model or draws a sample carries a seed. Two schemes are in
use, and **the difference is deliberate**:

**Index-based** (every stage but one):

    seed = seed_base + block + index * 10 + fold

where `index` is the row's position in the stage's table. Blocks are 1000 wide.

    2000  agg screen        4000  agg finals       8000  reg finals
    9000  hybrid, both families (offsets 0 and 3 inside the block)
   10000-11395 REG SCREEN (two blocks - 140 cells span 1400)
   12000  five             13000  c10d10           14000  dropout20
   15000/16000  c20        18000  extreme

The reference stages sit in bases of their own rather than in 1000-wide blocks,
because a reference stage seeds per client position and per init rather than per
cell and needs the room:

   40000  ten-client references      720000  twenty-client references
   50000  five-client references (base 750000)

`c10d10` was given 13000 - the gap between `five` at 12000 and `drop20` at
14000 - after checking it against every span above and the combination base at
30000. It is not a cosmetic choice: the three ten-client stages run the same
four configurations, so a shared block would have them drawing the same
client-sampling sequences with nothing in any output to say the three were
correlated.

**Identity-based** (combinations only):

    seed = seed_base + 30000 + (sha256(study｜family｜agg｜reg｜fold) % 900) * 10 + fold

The asymmetry has a reason. A combination's inputs are two *shortlists*, and a
shortlist is re-ranked whenever a selection is revisited. Under the index scheme
that moved seeds: two pairs swapped places in the concurrent list and twenty runs
already on disk stopped matching the file that claimed to emit them. The runs
were fine and the file was fine; they had simply come apart, and nothing said so.
Hashing the pair's identity means a shortlist can be re-ranked, extended or
trimmed and every pair keeps the seed it always had - the same reasoning that
puts the fold split on a hash of its inputs rather than on a position in a book.

Check before every submission, **passing every live task file at once**:

```bash
python tools/check_seeds.py $FOA_STUDY_DIR/jobs/*.txt
```

One file at a time cannot see a cross-stage collision, and a collision is silent:
both stages run, both write results, and the two draw the same client-sampling
sequence with nothing downstream to reveal it. The check reports a seed driving
two *different* tasks; a seed shared by the same task in two files - a smoke file,
a re-run subset - is not a collision.

### The accumulator race, and why both files were deleted

A reference stage scores every fold of a cohort with its own `evaluate-book`
task and points all of them at one accumulator JSON. Those tasks are the
elements of one array, so several of them performed the same read-modify-write
on the same file at the same moment. Without a lock the merge loses whatever
landed between a reader's read and its write: each element adds its own fold to
the file *as it found it*, and the last to write puts back a copy that never saw
the others.

Nothing fails when this happens. Every task exits zero, every task prints its
own accuracy, and the file simply holds fewer folds than the stage ran.
`g0_c5_evaluations.json` ended up with folds 1, 3 and 5; folds 2 and 4 were
computed, printed, and overwritten. `g0_c20_evaluations.json` next to it held
all five - **by luck**, which is the more dangerous artefact of the two, because
it looked exactly like evidence the pattern was sound. Both were deleted and
both were re-scored by stage 19 rather than only the one that visibly lost
folds.

`write_evaluation` now takes an exclusive `flock` on a sidecar `.lock` file
around the whole read-modify-write and swaps the accumulator in with
`os.replace`. The lock is on the sidecar rather than on the accumulator because
the accumulator is replaced by rename, and a lock held on a file that gets
replaced is a lock on an inode nobody else will open. The rename is what makes
the swap atomic for the selection tools, which read these files without taking
the lock. `tests/test_nist28_pipeline.py` pins it with twelve concurrent
processes released from a barrier; against the old code the same test keeps one
key of twelve on this filesystem, and sometimes leaves the file invalid JSON.

---

## 6. Which command produces which claim

| the claim | the command |
|---|---|
| the reference rungs | `report_tables.py --root $FOA_STUDY_DIR --what references` |
| aggregation screen ranking | `report_tables.py --what agg-screen` |
| aggregation winners, full horizon | `report_tables.py --what agg-winners` |
| regularisation screen ranking | `report_tables.py --what reg-screen` |
| regularisation winners, full horizon | `report_tables.py --what reg-winners` |
| combination scores | `report_tables.py --what combos` |
| the carried arms at each federation setting | `report_tables.py --what sizes` |
| the three extreme arrangements | `report_tables.py --what extremes` |
| when the extreme cases should have stopped | `extreme_stopping.py --root $FOA_STUDY_DIR` |
| the extreme-case figure | `extreme_stopping.py --root $FOA_STUDY_DIR --fig $FOA_STUDY_DIR/figures/extreme_stopping.png` |
| does a combination beat its halves? | `compare_arms.py --what combos` |
| penalty vs server rule | `compare_arms.py --what composition` |
| do the winners depend on `w`? | `weight_sensitivity.py --grid both` |
| are all eight signals present? | `check_signals.py --all` |
| are the cohorts label-skewed? | `describe_cohort.py --all` |
| are the seeds sound? | `check_seeds.py jobs/*.txt` |
| is the cohort the same cohort? | `freeze_selection.py verify $FOA_STUDY_DIR` |
| does every stage read what an earlier stage wrote? | `check_programme.py $FOA_STUDY_DIR/jobs` |

All take `--root $FOA_STUDY_DIR`; all accept `--csv <dir>`.
`extreme_stopping.py` is the one exception to the test basis below: it reads the
**validation** columns, because the round it reports is the round the study's own
selection rule would have picked and a selection may only see what a selection is
allowed to see. It says so in its own header line.

**Two bases, and every tool says which it used.**

    SELECTION reads validation   pool_val_accuracies  +  source_val_accuracies
    REPORTING reads test         final_evaluation.clients.accuracy
                                 final_evaluation.old.mean

A selection may only see the halves a selection is allowed to see; the reported
numbers are the test halves. The reporting tools print
`basis: TEST (final_evaluation)` in their header so a pasted table states what it
is. If a selector ever falls back to a test column because a validation series is
missing, it sets `*_is_test_fallback` and the record shows it - a silent fallback
would select on test while claiming validation.

**The selection rule**, one formula for both grids:

    score = (adaptation - A0) - w * (P0 - preservation),    w = 1

`A0` and `P0` are the shipped model's own two accuracies, read from
`g0_perfold_evaluations.json` and `g0_evaluations.json` at selection time, never
written down. Both terms are therefore differences from doing nothing.

---

## 7. Things that look like bugs and are not

**A screen winner at the weak end of its row.** At 25 rounds there is about one
point of forgetting available to protect, so a penalty strong enough to bite reads
as pure cost and selection walks toward "no penalty" - `param_l2`'s screen winner
is `mu = 0` exactly. At 100 rounds the same control is *last* of seven, forgetting
2.8-3.0 points. The screen ranks; it cannot price. Reproduce with
`report_tables.py --what reg-screen` against `--what reg-winners`.

**`fisher_scaled` flat at its top end.** The cap pins the penalty at half the
cross-entropy once the client has drifted, so every `lambda` above the saturation
point is the same run. A winner among those is an arbitrary pick between
duplicates, not a tuned value.

**A combination that beats both halves on the means.** Eleven of eighteen do;
one of eighteen is positive on every fold. The gains are smaller than the fold
spread. Always read `compare_arms.py`, never a table of means, for a claim of the
form "A beats B".

---

## 8. Before re-running anything

```bash
python tools/check_seeds.py     $FOA_STUDY_DIR/jobs/*.txt
python tools/check_programme.py $FOA_STUDY_DIR/jobs
python tools/freeze_selection.py verify $FOA_STUDY_DIR
```

Then run **one** task and read its output before launching hundreds. Four of the
eight signals were empty across an 845-task stage because nothing looked at a
single summary first.

**Delete a stage's result folders before re-running it**, and especially before
re-running it under changed ranges. Folders are named by cell id, so a re-ranged
grid leaves the cells it dropped behind as orphans. The selectors read the
catalogue and ignore them, but anything reading a *prefix* does not - which is how
a finals table came to be topped by three cells that exist in no current row. In
this programme 60 stale regularisation folders and 90 stale combination folders
were removed for exactly this reason.

---

## 9. Cost

| stage | tasks | rounds | GPU-hours |
|---|---|---|---|
| aggregation screen | 480 | 25 | ~16 |
| aggregation finals | 85 | 100 | ~10 |
| regularisation screen | 700 | 25 | ~25 |
| regularisation finals | 70 | 100 | ~8 |
| combinations | 90 | 100 | ~12 |
| five-client point | 20 | 100 | ~2 |
| ten-client point, one dropped | 20 | 100 | ~2 |
| twenty-client pair | 30 | 100 | ~5 |
| extremes | 15 | 100 | ~1 |

One A100 per task, `grete:shared`.
