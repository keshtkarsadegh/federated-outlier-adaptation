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

You do not have to rerun anything to read the study's numbers: what it produced
is published, and [§10](#10-the-published-records) says which part is in this
repository and which part is a release asset.

---

## 3. The stage graph

Each stage is a **task file** - one `foa ...` command per line - run as a Slurm
array where element *N* runs line *N*. One script runs every stage:

```bash
mkdir -p "$FOA_STUDY_DIR/logs"
sbatch --account=$FOA_ACCOUNT --partition=<gpu partition> --gres=gpu:1 \
       --array=1-<N>%<concurrency> \
       --export=ALL,FOA_PROJECT_DIR=$FOA_PROJECT_DIR,FOA_STUDY_DIR=$FOA_STUDY_DIR \
       --output="$FOA_STUDY_DIR/logs/%x_%A_%a.log" \
       slurm/study_phase.sbatch $FOA_STUDY_DIR/jobs/<file>.txt
```

| # | stage | generator | task file | tasks | rounds |
|---|---|---|---|---|---|
| 1 | dataset index | `tools/fetch_sd19.py` | - | - | - |
| 2 | writer counts, all-writer book | `s01a_book.txt` | `s01a_book.txt` | 2 | - |
| 3 | g-init / detector | `s01b_detector.txt` | `s01b_detector.txt` | 2 | - |
| 4 | g-0 + cohorts | `s02_selection.txt` | `s02_selection.txt` | 29 | - |
| 5 | references | `s03_refs_c10.txt`, `s03b_refs_fl.txt` | | 155 | 100 |
| 6 | **aggregation screen** | `tools/make_digits_p11.py` | `s09_agg_screen2.txt` | 480 | 25 |
| 7 | **aggregation finals** | `study_emit.py agg-full` | `s10_agg_full2.txt` | 85 | 100 |
| 8 | **regularisation screen** | `tools/make_digits_p13.py` | `s16_reg_screen3.txt` | 700 | 25 |
| 9 | **regularisation finals** | `study_emit.py reg-full` | `s17_reg_full4.txt` | 70 | 100 |
| 10 | **kd+fisher blend, concurrent** | `study_emit.py reg-hybrid` | `s18_hybrid.txt` | 15 | 100 |
| 11 | **kd+fisher blend, sequential** | `study_emit.py reg-hybrid --hybrid-family sequential` | `s19_hybrid_seq.txt` | 15 | 100 |
| 12 | **combinations** | `study_emit.py combos` | `s20_combos4.txt` | 90 | 100 |
| 13 | five-client references | `tools/make_size_references.py` | `d01_c5_references.txt` | 75 | 100 |
| 14 | twenty-client references | `tools/make_size_references.py` | `d01_c20_references.txt` | 235 | 100 |
| 15 | **five clients, one dropped** | `study_emit.py five` | `d01_five.txt` | 20 | 100 |
| 16 | **twenty clients, both dropout levels** | `study_emit.py c20` | `d01_c20.txt` | 30 | 100 |
| 17 | **the extreme cases** | `study_emit.py extreme` | `d01_extreme.txt` | 10 | 100 |
| 18 | **ten clients, one dropped** | `study_emit.py c10d10` | `d01_c10d10.txt` | 20 | 100 |
| 19 | size evaluations, re-scored | *the evaluate-book lines of 13 and 14* | `d01_size_evals_rerun.txt` | 10 | - |
| 20 | **kd+fisher blend screen** | `tools/make_digits_p21.py` | `s21_blend_screen.txt` | 1170 | 25 |

Stages 8-12 have all run at the search rate and are reported in
`REG_GRID_RANGES.md`. Stage 12 was re-emitted from the shortlists stages 9-11
produced and re-run against them: `tables/p12_agg_top3.json` and
`tables/p13_reg_top3.json` are the two lists it crosses, and every one of the
eighteen combinations on disk is a pair drawn from them. Regenerating
`s20_combos4.txt` reproduces the file that ran byte for byte, which is the
check that the stage and its shortlists have not come apart.

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

### What in `jobs/` is not a stage

Two files in `jobs/` are not stages and are not in the table. `d01_p02.txt` is
byte-identical to `s01_detector.txt`, and `s01_detector.txt` is stages 2 and 3
concatenated before they were split into two array submissions; `d01_p05v2.txt`
is `s02_selection.txt` with the header it was written under. Both are pre-rename
copies, kept as a record. `tools/check_programme.py` names them under NOT WALKED
rather than passing over them, because a checker that examines a subset of a
directory has to say which subset.

The two grid generators write a README beside their task file, and the
current one is not always under the current name. `make_digits_p11.py` writes
`d01_p11_README.md`, which is current and regenerates byte for byte;
`make_digits_p13.py` writes the README saved as `s16_reg_screen3_README.md`,
which is likewise current. Three older copies sit beside them -
`d01_p11_ext_README.md`, `d01_p13_README.md` and `d01_p13_ext_README.md` - and
they describe grids that no longer exist. `d01_p13_README.md` is the one to
watch: it documents a 117-cell, 585-task screen, while the file it appears to
accompany runs 140 cells and 700 tasks. Read the README the generator writes,
not the one with the matching stem.

`tools/make_c20_references.py` emitted an earlier attempt at stage 14 and does
**not** reproduce the file that ran: 484 of its lines differ, because its
`isolated-train` lines carry no `--seed`. It is bannered as superseded. Stage 14
is `make_size_references.py`, the same generator as stages 5 and 13.

Superseded task files live in `jobs/superseded/`. They are kept as a record and
must not be re-run: their seeds and their cells belong to a previous programme.

**Stage 20 is the screen the blend never had.** Stages 10 and 11 built the
kd+fisher blend from the two winners stage 9 named and swept `mix` alone, so its
`lam` and its `T` were inherited rather than searched - and it then held the best
mean score in both schedules. Stage 20 screens its own grid over the rows its two
parents were screened over, at the ranking horizon, in both schedules.
`REG_GRID_RANGES.md` records the grid and the unit conversion it needs.

**Screen then final.** A screen runs 25 rounds and *ranks*; a final runs 100 and
*reports*. Nothing from a screen is a reported number. The distinction is not
cosmetic - see [§7](#7-things-that-look-like-bugs-and-are-not).

### Extension: joint tuning of the best pair

**The two stages below are not part of the programme above.** They are an
extension, added after every stage in the table had run, and nothing the paper
reports reads anything they produce: the shortlists, the combination cross, the
crowning, the carry settings and the paper views are all unchanged and are not
re-emitted. They are listed here - in their own table, deliberately outside the
core one - because they run under the same runner, draw from the same seed
scheme and have to be reproducible on the same terms as everything else.

| # | extension stage | generator | task file | tasks | rounds |
|---|---|---|---|---|---|
| E1 | **joint screen of the leading pair** | `tools/make_digits_p23.py` | `s23_combo_screen.txt` | 1080 | 25 |
| E2 | **its winners at the full horizon** | `study_emit.py combo-tune-full` | `s24_combo_full.txt` | 10 | 100 |
| E3 | **joint screen of the SELECTED pair** | `tools/make_digits_p25.py` | `s25_combo_screen_selected.txt` | 1080 | 25 |
| E4 | **its winners at the full horizon** | `study_emit.py combo-tune-full-selected` | `s26_combo_full_selected.txt` | 10 | 100 |

**What it asks.** Stage 12 crossed the top three server rules with the top three
penalties per schedule, and every pair in it is a rule at the coefficients it
won on *alone* beside a penalty at the coefficients it won on *alone*. The two
shortlists were selected independently and nothing in the cross ever moved the
two together, so `COMBINATIONS.md`'s finding - that the two halves do not
measurably compose - is measured at one point of a joint grid. E1 screens that
grid for the pair that leads each schedule by TEST score of
`tables/paper/agg-winners.csv` and `tables/paper/reg-winners.csv`: `eta_0p95`
with the KD+EWC blend on the parallel schedule, `seq_delta_capped` with the same
blend on the cyclic one. Every non-grid flag is copied from the shipped `s20`
line that pairs the same two halves, so a line here differs from one there in
the horizon and in the coefficients and in nothing else.

The grid is the owner's four dials - the EWC coefficient, the KD coefficient,
the KD temperature and the blend weight - plus the server step on the parallel
side, where the rule has a coefficient to move at all. `REG_GRID_RANGES.md`
records the ranges and the mapping onto the two numbers `AnchoredTrainer`
actually takes; `s23_combo_screen_README.md` records both again beside the file.

E2 applies the study's own selection rule - gain less spend at `w = 1`, on the
validation columns, one winner per schedule, ties broken on the cell id - to
E1's cells and re-runs the two it crowns at 100 rounds. Its record is
`tables/p23_combo_tune_winners.json`, which says in its own text that it is not
part of the core programme; its boundary report is on the owner's dials rather
than on the two coefficients the trainer takes, because those are functions of
all four dials and an edge in one of them would name a row nobody searched.

**E3 asks E1's question about the pair the study actually selected.** E1 tunes
the pair that leads each schedule by TEST score, which is the owner's documented
departure from ranking on validation. The aggregation shortlist itself was cut
on VALIDATION - `tables/p12_agg_top3.json` says so in its own `rank_by` field -
and on the parallel schedule the two orderings disagree about which rule comes
first: the validation head is `anchor_h2`, the test head is `eta_0p95`. So E1's
answer was measured on a rule the programme did not choose, and E3 screens the
joint grid of the pair it did: `anchor_h2` on the parallel schedule and
`seq_fedavg` on the cyclic one, each beside the same KD+EWC blend E1 used.

The rule half is read out of that record on the basis the record names, never
typed, and the penalty half is read out of `make_digits_p23.THE_PAIR` - so the
two joint grids differ in exactly one place and can be read against each other,
which is the only reason to run the second. The four penalty rows are the same
four rows at the same values and resolve through the same
`reg_cells.combo_tune_lam_mix`; what is new is the rule axis. On the parallel
side that axis is the anchor's half-life `h` in {1R, 2R, 4R}, and **4R is
outside anything the programme screened**: `agg_cells.ANCHOR_HALFLIVES_R` stops
at 2R because beyond it the anchor never acts inside the run, so the selected
setting sat at the top edge of its own row and a bracket around it has no
neighbour above inside the screened range. `REG_GRID_RANGES.md` section 10
records that and what it costs; `s25_combo_screen_selected_README.md` records it
again beside the file. `seq_fedavg` has no coefficient at all, so the cyclic half
of this grid is a third the size, exactly as E1's is.

E4 applies the same selection rule E2 does - gain less spend at `w = 1`, on the
validation columns, one winner per schedule, ties broken on the cell id - to E3's
cells, and re-runs the two it crowns at 100 rounds. It is the same function over
a different catalogue and a different folder stem, and it writes a record of its
own: `tables/p25_combo_tune_selected_winners.json`, keyed
`combo-tune-selected/<schedule>` so it cannot be merged with E2's by accident.
Nothing it produces supersedes E1 or E2, and neither of those is re-emitted.

**What E3 chose is worth reading beside the note above.** The parallel schedule's
winner takes `h = 4R` - the half-life outside the screened row - so once the
penalty beside it moves, the selected server rule is wanted *weaker* than
anything the aggregation screen offered. Eight of the two winners' nine dials sit
at an end of their row; `s26_combo_full_selected_README.md` lists them and says
what they do and do not settle.

---

## 4. Regenerating a stage's task file

Every task file is generated, never hand-edited, and regenerating one must
reproduce it byte for byte:

**No flags.** Every command below is the whole command; a task file that needs
an argument remembered by hand is a task file nobody can regenerate.

```bash
python tools/make_digits_p11.py --jobs-dir /tmp/check     # aggregation screen
python tools/make_digits_p13.py --jobs-dir /tmp/check     # regularisation screen
python tools/make_digits_p21.py --jobs-dir /tmp/check     # the blend's screen
diff /tmp/check/d01_p11.txt $FOA_STUDY_DIR/jobs/s09_agg_screen2.txt
diff /tmp/check/d01_p13.txt $FOA_STUDY_DIR/jobs/s16_reg_screen3.txt
diff /tmp/check/d01_p11_README.md $FOA_STUDY_DIR/jobs/d01_p11_README.md
diff /tmp/check/d01_p13_README.md $FOA_STUDY_DIR/jobs/s16_reg_screen3_README.md
diff /tmp/check/s21_blend_screen.txt \
     study/artifacts/Digits_study01/jobs/s21_blend_screen.txt
diff /tmp/check/s21_blend_screen_README.md \
     study/artifacts/Digits_study01/jobs/s21_blend_screen_README.md

python tools/study_emit.py agg-full --root $FOA_STUDY_DIR --out /tmp/x.txt --expect 85
python tools/study_emit.py reg-full --root $FOA_STUDY_DIR --out /tmp/x.txt --expect 70
python tools/study_emit.py combos   --root $FOA_STUDY_DIR --out /tmp/x.txt --expect 90

python tools/study_emit.py reg-hybrid --root $FOA_STUDY_DIR --out /tmp/x.txt --expect 15
python tools/study_emit.py reg-hybrid --root $FOA_STUDY_DIR --out /tmp/x.txt --expect 15 --hybrid-family sequential

python tools/study_emit.py five    --root $FOA_STUDY_DIR --out /tmp/x.txt --expect 20
python tools/study_emit.py c10d10  --root $FOA_STUDY_DIR --out /tmp/x.txt --expect 20
python tools/study_emit.py c20     --root $FOA_STUDY_DIR --out /tmp/x.txt --expect 30
python tools/study_emit.py extreme --root $FOA_STUDY_DIR --out /tmp/x.txt --expect 10
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

### The extension's task files

Not stages of the programme - see [the extension
subsection](#extension-joint-tuning-of-the-best-pair) - but regenerated on the
same terms, and byte for byte:

```bash
python tools/make_digits_p23.py --jobs-dir /tmp/check     # the joint screen
diff /tmp/check/s23_combo_screen.txt \
     study/artifacts/Digits_study01/jobs/s23_combo_screen.txt
diff /tmp/check/s23_combo_screen_README.md \
     study/artifacts/Digits_study01/jobs/s23_combo_screen_README.md

python tools/study_emit.py combo-tune-full --root $FOA_STUDY_DIR \
    --out /tmp/x.txt --expect 10
diff /tmp/x.txt study/artifacts/Digits_study01/jobs/s24_combo_full.txt

python tools/make_digits_p25.py --jobs-dir /tmp/check   # the selected pair
diff /tmp/check/s25_combo_screen_selected.txt \
     study/artifacts/Digits_study01/jobs/s25_combo_screen_selected.txt
diff /tmp/check/s25_combo_screen_selected_README.md \
     study/artifacts/Digits_study01/jobs/s25_combo_screen_selected_README.md

python tools/study_emit.py combo-tune-full-selected --root $FOA_STUDY_DIR \
    --out /tmp/y.txt --expect 10
diff /tmp/y.txt study/artifacts/Digits_study01/jobs/s26_combo_full_selected.txt
```

### The reference stages, which do take arguments

Stages 5, 13 and 14 are the exception to the no-flags rule above, and the
exception is deliberate: one generator serves every federation size, so the
size has to be named. The three invocations that reproduce the three shipped
files byte for byte are:

```bash
python tools/make_size_references.py --study-dir "$FOA_STUDY_DIR" \
    --cohort cohort_worst10.json --book cohort10 --per-round 9 8 \
    --tag c10 --seed-base 740000 --out /tmp/check/s03_refs_c10.txt

python tools/make_size_references.py --study-dir "$FOA_STUDY_DIR" \
    --cohort cohort_worst5.json  --book cohort5  --per-round 4 \
    --tag c5  --seed-base 750000 --out /tmp/check/d01_c5_references.txt

python tools/make_size_references.py --study-dir "$FOA_STUDY_DIR" \
    --cohort cohort_worst20.json --book cohort20 --per-round 18 16 \
    --tag c20 --seed-base 720000 --out /tmp/check/d01_c20_references.txt
```

The seed base is the argument that must not be guessed. A reference stage seeds
per client position and per init rather than per cell, so its seeds occupy a
range of their own - 40000, 750000 and 720000 - and a base typed differently
would produce a file that looks right and draws different clients. The bases are
listed with every other seed block in [§5](#seeds).

The participation rate the GRID stages emit at is **not** a flag - see
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
| who the gain reached: the per-client spread of every headline stage | `fairness_cost.py --root $FOA_STUDY_DIR --what fairness` |
| what the programme cost: seconds, rounds and bytes | `fairness_cost.py --root $FOA_STUDY_DIR --what cost` |
| when the extreme cases should have stopped | `extreme_stopping.py --root $FOA_STUDY_DIR` |
| the extreme-case figure | `extreme_stopping.py --root $FOA_STUDY_DIR --fig $FOA_STUDY_DIR/figures/extreme_stopping.png` |
| what the fixed horizon cost every arm | `stopping_table.py --root $FOA_STUDY_DIR` |
| what a permitted signal would have delivered | `stopping_table.py --root $FOA_STUDY_DIR --csv $FOA_STUDY_DIR/tables/stopping` |
| what a plateau on the cohort's own accuracy would have delivered | `plateau_rule.py --root $FOA_STUDY_DIR` |
| the four plateau views, and the grid the setting was fixed on | `plateau_rule.py --root $FOA_STUDY_DIR --out $FOA_STUDY_DIR/tables/stopping` |
| does that plateau setting survive being chosen where it is not measured? | `plateau_holdout.py --root $FOA_STUDY_DIR --out $FOA_STUDY_DIR/tables/stopping` |
| do the eight signals track forgetting? | `foa signals --root $FOA_STUDY_DIR` |
| does a combination beat its halves? | `compare_arms.py --what combos` |
| penalty vs server rule | `compare_arms.py --what composition` |
| does a blend beat the two cells it was built from? | `compare_arms.py --what blends` |
| do the winners depend on `w`? | `weight_sensitivity.py --grid both` |
| are all eight signals present? | `check_signals.py --all` |
| are the cohorts label-skewed? | `describe_cohort.py --all` (reads the dataset: needs `FOA_NIST28_DIR`) |
| are the seeds sound? | `check_seeds.py $FOA_STUDY_DIR/jobs/*.txt` |
| is the cohort the same cohort? | `freeze_selection.py verify $FOA_STUDY_DIR` |
| does every stage read what an earlier stage wrote? | `check_programme.py $FOA_STUDY_DIR/jobs` |
| how much data does each cohort writer hold, and does it hold every class? | `describe_cohort.py --all --csv $FOA_STUDY_DIR/tables/` (same) |
| the whole paper bundle, one command | `report_tables.py --what all --csv $FOA_STUDY_DIR/tables/paper/` |
| the six per-round views, and the two rounds the extreme figure marks | `export_traces.py --root $FOA_STUDY_DIR --out <dir>` |
| g-0's own training history, and the isolated clients | `export_baseline_views.py --root $FOA_STUDY_DIR --out <dir>` |
| each combination against BOTH of its halves, by fold | `export_combo_folds.py --root $FOA_STUDY_DIR --out <dir>` |
| does the schedule matter? six pairs, each schedule's own selected arm against the other's, cyclic minus parallel and paired by fold | `export_schedule_views.py --root $FOA_STUDY_DIR --csv <dir>` -> `schedule_pairs.csv` |
| the study's eight strongest arms across the three reporting stages, and which schedule ran each | the same command -> `schedule_top.csv` |
| what each selection saw: the VALIDATION ordering every shortlist was cut on, beside the same runs' test ordering, per schedule | `export_selection_axis.py --root $FOA_STUDY_DIR --csv <dir>` -> `selection_axis.csv` |
| the client recipe the methods section states, read back off every stored payload | `export_recipe_view.py --root $FOA_STUDY_DIR --csv <dir>` -> `recipe.csv` |
| the two extracts the forgetting-signals section is written on | `export_signals_summary.py --root $FOA_STUDY_DIR --out <dir>` |
| the five writers the two selection rankings disagree about | `export_decouple_example.py --root $FOA_STUDY_DIR --out <dir>` |
| every figure in the manuscript | [below](#and-which-command-produces-which-figure) - two commands per figure |
| every number the manuscript sets | `paper_figures/make_numbers.py` - [below](#and-which-command-produces-the-numbers-and-the-tables) |
| every table the manuscript sets | `paper_figures/make_paper_tables.py` - [below](#and-which-command-produces-the-numbers-and-the-tables) |
| what this study actually ran, read off disk | `study_record.py --root $FOA_STUDY_DIR > docs/STUDY_RECORD.md` |
| does a task file still regenerate byte for byte? | [§4](#4-regenerating-a-stages-task-file) - every live generator |

`selection_axis.csv` and `recipe.csv` are **core** for the same reason the two
schedule views are. The first is the other half of a sentence the manuscript
already sets: every shipped table is TEST and every shortlist that decided which
arms those tables contain was cut on VALIDATION, and the two orderings are not
the same ordering - on the parallel schedule the rule the combination cross was
built from is not the rule that leads Table 2. Each arm in it is the head of a
record's own ranking, read under the basis that record names in `rank_by`, and
the shortlist is the record's own `top` rather than the first three rows of the
ranking: a composite competes for a slot of its own, and reading the head of the
ranking instead would print a shortlist the study never carried. The second
carries the six client-side constants of the methods paragraph - the learning
rate, the weight decay, the two early-stopping settings that are off, the batch
size and the local epochs - each read out of `config` in every stored payload and
refused if the programme does not agree on one value; `tests/test_recipe.py`
holds the paragraph, the code and the view to each other.

The two schedule views are **core**, which is why they are in the table above
rather than beside it. Every stage of the programme selected and reported the two
schedules separately - correctly, because a rule that exists on only one of them
cannot be ranked against a rule that exists on only the other - and the cost is
that the comparison a reader makes first is spread across three files sorted by
score. `schedule_pairs.csv` pairs it arm by arm, and which arm each schedule
contributes is read and not typed: the rule from the head of
`tables/p12_agg_top3.json`'s own ranking, under the basis that record says it
selected on, the blend from `make_digits_p23.THE_PAIR`, the logit and NTD cells
from `tables/p13_reg_method_winners.json`, which crowns per method AND per
schedule, the crowned pair and the arm carried beside it out of
`tables/p15_stage_winner.json`, and the control out of `agg_cells`. Every row is
an arm a selection record names - the control is the fixed control, which is
what the selecting is measured against - and no row is ranked into the table by
the numbers the table then prints. Both views are TEST, because both report.

### The extensions' own views

**Nothing above reads these four, and nothing the manuscript reports does
either.** They exist so that the two extensions of
[§3](#extension-joint-tuning-of-the-best-pair) can be read on the basis the
tables of §6 are measured on - and against each other - rather than off their
task files' READMEs, and they are written into `tables/paper/` beside the
manuscript bundle under names that begin `extension_`, so a reader sorting that
directory can see at a glance which four of its files are not the programme.
They are cut by one reader over a `Grid` per extension, because two copies of
that reading could drift apart in a way that looks like a result.

| the claim | the command |
|---|---|
| the jointly-tuned pair against the rule alone, the penalty alone, the untuned pair of those same halves and the pair the cross crowned, per schedule | `export_extension_views.py --root $FOA_STUDY_DIR --csv <dir>` -> `extension_combo_tune.csv` |
| what the joint screen ranked: its top five cells per schedule, their dials, and which of those dials sit at the end of a row | the same command -> `extension_combo_screen.csv` |
| the same two claims for the pair the study SELECTED, whose rule half is the head of `p12_agg_top3.json`'s own validation ranking | the same command -> `extension_combo_tune_selected.csv`, `extension_combo_screen_selected.csv` |

Which arms any of them carries is read and not typed: the pair comes from the
grid's own generator - `make_digits_p23.THE_PAIR` or `make_digits_p25.THE_PAIR` -
the untuned line from `THE_SHIPPED_LINES` beside it, the tuned cell from that
grid's own record (`p23_combo_tune_winners.json`,
`p25_combo_tune_selected_winners.json`), and the crowned pair from
`tables/p15_stage_winner.json`, through the one reader this repository has of
that record. The horizon views are on TEST because they report; the screen views
are on VALIDATION because a screen is a selection, and each is ranked by
`study_emit`'s own rule so that its first row is the cell its record crowned.
The penalty-alone and crowned-pair rows are the same arms in both horizon views
and the rule, untuned and tuned rows are not, because the second grid moves the
rule half and nothing else.

The reporting tools all take `--root $FOA_STUDY_DIR` and all accept
`--csv <dir>`. `check_seeds.py`, `check_programme.py` and
`freeze_selection.py` are the exception: each takes its path as a positional
argument, as written above.
`foa signals` writes to `$FOA_STUDY_DIR/signals/` - three CSVs, a summary JSON
and one Pareto plot per method - and `stopping_table.py --csv` writes one CSV per
stage plus `stopping_all.csv`. `plateau_rule.py --out` writes four more into
that same directory - `plateau_arms.csv`, `plateau_stages.csv`,
`plateau_grid.csv` and `plateau_extremes.csv` - and `plateau_holdout.py --out`
three more beside them: `plateau_holdout_protocols.csv`,
`plateau_holdout_extremes.csv` and `plateau_holdout_grids.csv`.
`docs/STOPPING.md` reads all of it.
`fairness_cost.py --csv` writes one CSV per stage plus a per-client detail CSV
beside each, and `cost_stages.csv` and `cost_arms.csv`. `docs/FAIRNESS_AND_COST.md`
reads all of it.

**The four stopping tools are the exception to the test basis below**:
`extreme_stopping.py`, `stopping_table.py`, `plateau_rule.py` and
`plateau_holdout.py` all read the **validation** columns, because the round they
report is the round the study's own selection rule would have picked, and a
selection may only see what a selection is allowed to see. All four say so in
their own header lines. `stopping_table.py` imports the fold mean, the oracle
round and the score from `extreme_stopping.py`, and the drift and stopping
semantics from `analysis/forgetting_signals.py`; `plateau_rule.py` imports its
whole basis - arm reader, fold mean, score, oracle round and the one-signal
rule - from `stopping_table.py`; `plateau_holdout.py` imports the rule and its
sixteen-cell grid from `plateau_rule.py`, and the arm reader from
`stopping_table.py` through it. So the five cannot come apart:
`stopping_table.py --stage extreme` reproduces `extreme_stopping.py` row for
row, `plateau_arms.csv` carries `stopping_all.csv`'s own `final_score` arm for
arm, and the `in-sample` row of `plateau_holdout_protocols.csv` reproduces
`plateau_stages.csv`'s `all` row. All three are checks rather than claims, and
`tests/test_plateau.py` and `tests/test_plateau_holdout.py` pin the last two.

**Two bases, and every tool says which it used.**

    SELECTION reads validation   pool_val_accuracies  +  source_val_accuracies
    REPORTING reads test         final_evaluation.clients.accuracy
                                 final_evaluation.clients.per_client
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

### And which command produces which figure

The figures were the last part of the manuscript still built by hand. They are
now the same shape as the tables: a tool reads the records and writes a CSV
*view*, and a script reads the view and renders. Nothing in
`tools/paper_figures/` opens a run folder, computes an accuracy or knows where
the study lives, so a figure cannot disagree with the records it is drawn from
without the view between them disagreeing first.

The ten views ship in `study/artifacts/Digits_study01/tables/paper_figures/`, so
the second column runs against a clone with no study root assembled at all. The
first column is what proves them: run against an assembled root ([§10](#10-the-published-records))
each export reproduces its shipped views byte for byte.

| figure | its views | regenerate the views | render |
|---|---|---|---|
| `fig_problem` | `traces_control.csv`, `references.csv` | `export_traces.py --what traces_control`, `report_tables.py --what references` | `python tools/paper_figures/fig_problem.py` |
| `fig_aggs` | `traces_aggfull.csv` | `export_traces.py --what traces_aggfull` | `python tools/paper_figures/fig_aggs.py` |
| `fig_regs` | `traces_regfull.csv` | `export_traces.py --what traces_regfull` | `python tools/paper_figures/fig_regs.py` |
| `fig_blends` | `traces_blends.csv` | `export_traces.py --what traces_blends` | `python tools/paper_figures/fig_blends.py` |
| `fig_combo` | `traces_combo.csv` | `export_traces.py --what traces_combo` | `python tools/paper_figures/fig_combo.py` |
| `fig_extremes` | `traces_extreme.csv`, `plateau_extremes.csv` | `export_traces.py --what traces_extreme`, `plateau_rule.py --out` | `python tools/paper_figures/fig_extremes.py` |
| `fig_baselines` | `g0_training.csv`, `isolated_clients.csv`, `references.csv` | `export_baseline_views.py`, `report_tables.py --what references` | `python tools/paper_figures/fig_baselines.py` |

`export_traces.py --what all` writes the first six plus `extreme_stop_rounds.csv`
in one pass - the last of those is read by `make_numbers.py` rather than by a
figure now; every export takes `--root $FOA_STUDY_DIR --out <dir>` like the
reporting tools take `--csv`.

`FOA_PAPER_OUT` says where the PDF and its draft caption are written - without it
they land beside the script, which is what the manuscript build wants and not
what a source checkout does. `FOA_PAPER_DATA` says where the views are read from,
one directory or several separated the way `PATH` is; unset, the scripts find the
two shipped bundles on their own. `tools/paper_figures/README.md` is the longer
version of this table.

**WHICH ARMS A FIGURE DRAWS IS READ, NOT TYPED.** Every arm list comes out of the
frozen selection records - `tables/p12_agg_top3.json` for the server rules,
`p13_reg_method_winners.json` for one arm per penalty family,
`p14_hybrid_construction*.json` for each blend and its two parents,
`p15_stage_winner.json` for the crowned pair, which is split back into its two
halves through the same two shortlists the cross was emitted from. Naming them in
the exporter instead would let a figure outlive the selection that put the arm in
it, which is the failure `compare_arms.py` documents at length for the blends.

**THE FIGURES ARE ON THE VALIDATION BASIS**, `pool_val_accuracies` against
`source_val_accuracies`, and round 0 of every trace view is the shipped model's
own two accuracies so that all the arms of a figure begin at one shared point.
They are therefore not the test-set numbers the tables of §6 report and the two
must not be quoted against each other; every caption says so. `fig_baselines` is
the exception and is on test throughout, because the isolation records store test
evaluations only.

### And which command produces the numbers and the tables

One step further along the same shape. `numbers.tex` - the file every
quantitative sentence of the manuscript reads its value from - and the
fourteen `tables/*.tex` it sets are written from the views too, by two generators that sit
beside the figure scripts and compute nothing:

```bash
cp /path/to/manuscript/numbers.tex /tmp/paper/     # rewritten in place: see below
FOA_PAPER_OUT=/tmp/paper python tools/paper_figures/make_numbers.py
FOA_PAPER_OUT=/tmp/paper python tools/paper_figures/make_paper_tables.py
```

`numbers.tex` is written into `$FOA_PAPER_OUT` and the tables into
`$FOA_PAPER_OUT/tables/`. `--check` on either writes nothing and exits non-zero
when what is on disk disagrees with the views, which is the form the manuscript
build runs and the reason a stale table cannot reach a PDF. Neither generator
will write into a source checkout: with no `FOA_PAPER_OUT` set they say so and
stop, rather than drop LaTeX into the tracked tree. `FOA_PAPER_DATA` overrides
where the views are read from, exactly as it does for the figures.

Thirty-two views feed the two of them and all thirty-two ship here, spread over
four directories because four different tools write them:

| where | what is in it | written by |
|---|---|---|
| `tables/paper/` | the reference rungs, both winner tables, the combinations, the four carry settings, the extremes, the two screens, the fairness and cost views, the two signal extracts, the decoupling view, the two schedule views, the selection axis, the client recipe | `report_tables.py --what all --csv`, `fairness_cost.py --what all --csv`, `export_signals_summary.py`, `export_decouple_example.py`, `export_schedule_views.py`, `export_selection_axis.py`, `export_recipe_view.py` |
| `tables/paper_figures/` | the per-round traces, `extreme_stop_rounds.csv`, `isolated_clients.csv`, `combos_folds.csv` | `export_traces.py`, `export_baseline_views.py`, `export_combo_folds.py` |
| `tables/stopping/` | `stopping_all.csv`, `stopping_extreme.csv` and the per-stage rest; the four `plateau_*.csv` beside them, three of which the generators read - `plateau_stages.csv` builds `plateau.tex`, `plateau_arms.csv` and `plateau_extremes.csv` fill macros, and `fig_extremes.py` marks its stopping rounds from the last - while `docs/STOPPING.md` reads all four; the three `plateau_holdout_*.csv` beside them answer that rule's own objection and feed neither generator and no figure | `stopping_table.py --csv`, `plateau_rule.py --out`, `plateau_holdout.py --out` |
| `tables/` | `blends.csv`, `composition.csv`, `weight_sensitivity_*.csv`, `cohort_composition.csv`, `cohort_table.csv` | `compare_arms.py --csv`, `weight_sensitivity.py --csv`, `describe_cohort.py --csv`, and the cohort stage |

`tables/paper/` also holds the extension's four views. They feed neither
generator and no figure, they are named `extension_*` for that reason, and the
thirty-two above are thirty-two without them.

**THE PAPER BUNDLE IS SEARCHED FIRST, AND THAT MATTERS.** `tables/combos.csv` is
the raw grid dump the combination stage left behind and `tables/paper/combos.csv`
is the view the manuscript quotes. They carry one name and different rows, so the
order the four directories are tried in is part of the contract, not a detail;
`_data_dirs()` in both generators says so at the point where it fixes the order.

**`numbers.tex` IS REWRITTEN IN PLACE, NOT WRITTEN FROM NOTHING.** The file
carries `% BEGIN GENERATED` and `% END GENERATED` and only the block between them
is touched: the macro *names* are read out of that block and each is filled from
the registry the script builds from the views. So a copy of the manuscript's own
`numbers.tex` has to be in `$FOA_PAPER_OUT` before the command is run. Every
definition it writes carries a trailing comment naming the CSV file, the row and
the column the value came from, and a name with no registry entry is written back
as `\TBD{unmapped}`, reported on stdout, and exits non-zero - so an unsourced
number is loud rather than silent.

**SIX DEFINITIONS SIT ABOVE THE MARKER AND ARE NOT GENERATED.** They are
hand-maintained on purpose, they are outside the block the generator rewrites,
and `numbers.tex` says which is which in a comment of its own:

| what | how many | why it is not generated |
|---|---|---|
| `\nBadFraction`, `\nDigitRows`, `\nDigitWriters`, `\nOldSize`, `\nLocalEpochs`, `\nBatchSize` | 6 | protocol constants: configuration the study was **given**, not a measurement it produced, so no view carries them. `tests/test_recipe.py` holds each to its own source in `src/`, in `outliers/pools.json` or in the shipped task lines |

There were fifty-nine on 2026-09-20, and the fifty-three that have gone were
retired rather than moved. The forty-six `\pub...` macros were facts of the
earlier published single-seed runs; this programme re-ran every quantity they
carried, five folds deep, and no page of the manuscript called one. The learning
rate and the weight decay moved INTO the generated block when `recipe.csv`
landed, and the rest - the proxy-set constants, the good pool, the per-writer
sample counts - were macros nothing set. A macro no page calls is a macro no
reader can check: `\TBD{}` cannot fire on a name nobody writes.

The other 138 macros are generated, down from 205 in the same pass: the template
now defines what the manuscript sets plus the handful the tables and the figure
captions set for it, and `make_numbers.py` reports any registry entry with no
macro between the markers, so the registry was pruned to match rather than left
to answer names that had gone. `tables/sensitivity_agg.tex` and
`tables/sensitivity_reg.tex` are the two table files `make_paper_tables.py` does
not write: a third generator in the manuscript checkout does, from the two
`weight_sensitivity_*.csv` that ship here.

**THREE VIEWS ARE EXTRACTS OF OTHER RECORDS, AND EACH HAS A COMMAND.**
`signals_summary_extract.csv` and `signals_extras_extract.csv` join what `foa
signals` wrote into `$FOA_STUDY_DIR/signals/` to the stop `stopping_table.py`
reads off the same arms, and `decouple_example.csv` is the five-writer rank
comparison the cohort-selection section quotes. All three still ship, so that
`make_numbers.py` runs against a clone with no study root, and all three are now
rebuilt from the records byte for byte:

```bash
python tools/export_signals_summary.py  --root "$FOA_STUDY_DIR" --out <dir>
python tools/export_decouple_example.py --root "$FOA_STUDY_DIR" --out <dir>
```

The second reads only `outliers/`, so it runs against
`study/artifacts/Digits_study01/` in a bare clone. The first needs more than the
unpacked records: the three CSVs it joins them to are `foa signals`' own output,
which is derived, is 195 MB, and is published in neither half - so on a freshly
assembled reviewer tree it stops and says the files are not there, and the
`foa signals` pass of [§6](#6-which-command-produces-which-claim) is what puts
them there. Neither writes into the study tree.

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

**A combination that beats both halves on the means.** Seven of eighteen do;
two of eighteen are positive on every fold. The gains are smaller than the fold
spread. Always read `compare_arms.py`, never a table of means, for a claim of the
form "A beats B".

**A kd+fisher blend that beats both of its parents on the means.** Six of six
do. Paired by fold, five of six clear their KD parent on every fold and **none**
clears its Fisher parent - the three concurrent blends each lose fold 1 to
`fisher_lam8` by more than their whole mean gain. The blends do hold the best
mean score and the best preservation in their table; that is the claim the folds
support, and `compare_arms.py --what blends` is where it is checked.

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
| extremes | 10 | 100 | ~1 |
| blend screen | 1170 | 25 | ~30 |
| blend finals | 10 | 100 | <1 |

One A100 per task, `grete:shared`. Those are the hours that were **booked**;
`fairness_cost.py --what cost` reports the hours the round loop actually spent
inside them, read from the runner's own timer, and
[`docs/FAIRNESS_AND_COST.md`](FAIRNESS_AND_COST.md#4-what-it-cost) reads the two
against each other. The table above is the submission plan and the measured one
is read off disk, so two rows differ: the regularisation finals are 110 folders
rather than 70 once the two hybrid emissions and the blend's finals are counted
with them, the regularisation screen is 1,870 once the blend's own screen is,
and the 50 size-reference tasks appear in neither row here.

**The extension is not in this table**, for the same reason it is not in the
stage table of [§3](#3-the-stage-graph): its joint screen booked ~32 GPU-h and
measured 18.4, and its finals are under one, but nothing the paper reports reads
either, so they are priced in `s23_combo_screen_README.md` and
`s24_combo_full_README.md` and are not added to the programme's total.

---

## 10. The published records

**In git - `study/artifacts/Digits_study01/`, 8.8 MB.** The metadata core: the
frozen cohort, the g-0 evaluation books both baselines are measured against, the
fold books, the outlier and cohort records, every shipped table and CSV - the ten
views the manuscript's figures are drawn on among them - every task file that was
submitted, and the figures. That is enough to check any claim
in the manuscript against a record, and small enough to diff. Its own `README.md`
maps each artefact to the tool that reads it and names everything excluded, with
the command that regenerates it. Absolute machine paths are rewritten to
`$FOA_STUDY_DIR` and friends by `tools/sanitize_artifacts.py`; that this stayed
true is asserted by `tests/test_release_artifacts.py`.

**A release asset - `Digits_study01_records.tar.gz`, 151,177,192 bytes.** Every
`accuracies_*.json` and `summary_0.json` under the 4,945 `d01_*` run folders plus
the 380 reference-rung JSONs: 14,390 files, 1.3 GB unpacked. The blend's two
stages and the extensions' four are in it with the rest: the extensions' views
are computed from their records, so an asset that carried only the core would
publish a table nobody could rebuild. These are the numbers
every table is computed from. They are an asset rather than a tracked directory
because git is the wrong place for half a gigabyte of machine output nobody will
diff, and they carry no machine paths either - the same sanitiser is run over a
staged copy before packing. `tools/build_records_asset.py` builds it, decides
membership with the same predicate every reader uses, and packs deterministically,
so the checksum is a property of the records rather than of the day they were
packed.

    sha256 11456ca21081aa553ef6d12f1630916c45c86182d8700a6a18b394baed56cc88

published beside the archive as `Digits_study01_records.tar.gz.sha256`. Both are
attached to the repository's records release, which carries no version number:
<https://github.com/keshtkarsadegh/federated-outlier-adaptation/releases/tag/records>.
The records are not the source and do not move with it: whenever a stage adds
records, the archive is replaced in place and its checksum with it. So the
sha256 above is the one to verify against, and a copy that fails it was fetched
before the last stage landed.
Paths inside are relative to the study root, so the asset unpacks straight over
one -
but the asset is only half of a study root. The other half is the metadata core
above, which travels in the clone rather than in the archive. Neither half is a
study root on its own, and nothing runs until the two are brought together.

### Assembling the reviewer tree

The asset carries the 4,945 `d01_*` run folders and the reference rungs and
nothing else; the core carries the selection records, fold books, task files and
tables that every tool reads alongside them. Both are laid out relative to the
study root, so assembling one is two copies into an empty directory:

```bash
export FOA_STUDY_DIR=/path/to/Digits_study01        # any empty directory
mkdir -p "$FOA_STUDY_DIR"

curl -fLO https://github.com/keshtkarsadegh/federated-outlier-adaptation/releases/download/records/Digits_study01_records.tar.gz
curl -fLO https://github.com/keshtkarsadegh/federated-outlier-adaptation/releases/download/records/Digits_study01_records.tar.gz.sha256
sha256sum -c Digits_study01_records.tar.gz.sha256   # 151,177,192 bytes
tar -xzf Digits_study01_records.tar.gz -C "$FOA_STUDY_DIR"
cp -r study/artifacts/Digits_study01/. "$FOA_STUDY_DIR/"
```

The result is 14,555 files - the asset's 14,390 plus this directory's 165 -
across 4,970 top-level entries, and every row of
[§6](#6-which-command-produces-which-claim) runs against it from the repository
root. `report_tables.py --what all --csv` reproduces `tables/paper/*.csv` byte
for byte from it, and so do `compare_arms.py --csv`, `stopping_table.py --csv`,
`plateau_rule.py --out`, `plateau_holdout.py --out`,
`weight_sensitivity.py --csv` and the three
`export_*.py` of
[§6](#and-which-command-produces-which-figure) for their own tables and views -
which is the check that the two halves were assembled correctly.

Two things behave differently on such a tree, and neither is a defect.
`describe_cohort.py` reads the dataset rather than the records, so it needs
`FOA_NIST28_DIR` and the `fetch_sd19.py` step in [§2](#2-environment-and-data);
every other row of §6 runs without it. And `foa signals` recomputes the derived
signal files over **six runs fewer** than the machine that ran the study saw:
a one-task smoke run that predates the stage and was never part of it, and the
five runs of the one-client extreme arrangement the study does not define. The
difference is named rather than the two totals, which move whenever a stage is
added and say nothing when they do. Neither belongs to a shipped table. Every selected arm,
oracle and gap comes out identical - only the `num_candidates` and `num_allowed`
populations shift.

**Not published: the weights.** `g0_model`, the five `g0_fold*/global_model` and
`global_results/fisher_g0` - 51 MB of checkpoints, which answer no question the
records do not. Every stage that consumes them is reproducible from §3 onwards.

**`compare_arms.py --what blends` exists, and needs both halves of the above.**
It differences each kd+fisher blend against each of the two cells it was built
from, fold by fold. The parents come from `tables/p14_hybrid_construction.json`
and `tables/p14_hybrid_construction_sequential.json`, which are in the tree; the
scores come from the `d01_regfull_*` records, which are in the asset. Unpack the
asset and

    python tools/compare_arms.py --root "$FOA_STUDY_DIR" --what blends

reproduces the claim `docs/REG_GRID_RANGES.md` now makes: best mean score and
best preservation, five of six blends clearing their KD parent on every fold, and
none of the six clearing its Fisher parent.
