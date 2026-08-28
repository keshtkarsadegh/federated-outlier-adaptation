# Runbook: reproducing Digits_study01 end to end

Every stage is a **task file**: one `foa ...` command per line, run as one array
element each. The same runner executes all of them, so there is one submission
pattern to learn.

```bash
# with Slurm
sbatch --account=$FOA_ACCOUNT --partition=$FOA_GPU_PARTITION --gres=gpu:1 \
       --array=1-N --output="$FOA_STUDY_DIR/logs/%x_%A_%a.log" \
       slurm/study_phase.sbatch study/jobs/<file>.txt

# without Slurm (same script, sequentially)
slurm/run_tasks.sh study/jobs/<file>.txt
```

`N` is the line count, given in the table below. Some files must run **`%1`
(serially)**, marked *chained*: each of their steps reads what the one before it
wrote, and a fan-out would race them.

**Always dry-run a file first.** It walks every check and executes nothing:

```bash
FOA_DRY_RUN=1 slurm/run_tasks.sh study/jobs/<file>.txt
```

---

## Before you start

```bash
export FOA_PROJECT_DIR=/path/to/workspace          # required, no default
export FOA_NIST28_DIR="$FOA_PROJECT_DIR/data/nist28"
export FOA_STUDY_DIR="$FOA_PROJECT_DIR/results/studies/Digits_study01"
mkdir -p "$FOA_STUDY_DIR"
```

The cache must exist first — see `docs/DATA.md`. The study root must exist: the
runner refuses to create it, because a default pointing at a directory nobody
made turns a missing setup into a silently empty study.

---

## The stages, in order

Cost is **measured** from the run that produced the published results, on one
A100 per array element.

| # | Stage | Task file | Tasks | GPU-h | Notes |
|---|---|---|---|---|---|
| — | Packed cache | `slurm/prepare_nist28.sbatch` | 1 | 0.2 (CPU) | 10.5 min, no GPU |
| P02 | g-init: score every writer | `d01_p02.txt` | 9 | ~2.2 | *chained*; all-writers book, 5 folds, fold selection, per-writer scoring |
| P04 | first cohort draw *(superseded by P05v2)* | `d01_p04.txt` | 7 | ~0.1 | kept for the record; P05v2 replaces it |
| P05v2 | BAD/GOOD split, old data, **g-0**, cohort | `d01_p05v2.txt` | 26 | ~2.5 | *chained*; the study's spine — see below |
| P07 | preservation + do-nothing baselines | `d01_p07.txt` | 16 | ~0.2 | overlaps P05v2's last steps |
| P08 | isolated: one private model per client | `d01_p08.txt` | 100 | 1.14 | 10 writers x 5 folds x 2 inits |
| P09 | plain FedAvg baseline | `d01_p09.txt` | 10 | 0.53 | the 9-of-10 reference point |
| P10 | centralized bound | `d01_p10.txt` | 10 | 0.11 | pooled cohort |
| P11 | **aggregation screen** | `d01_p11.txt` | 845 | 15.4 | 169 cells x 5 folds, 25 rounds |
| P11e | trimmed-mean boundary extension | `d01_p11_ext.txt` | 10 | 0.2 | appended cells only |
| P12 | aggregation at full horizon | `d01_p12.txt` | 90 | 3.1 | 18 methods x 5 folds, 100 rounds |
| P12p | trimmed patch | `d01_p12_trimmed_patch.txt` | 5 | 0.2 | |
| P13 | **regularisation screen** | `d01_p13.txt` | 585 | 16.8 | 117 cells x 5 folds, 25 rounds |
| P13e | KD/NTD boundary extensions | `d01_p13_ext.txt` | 40 | 1.0 | appended cells only |
| P13h | kd+fisher hybrid | `d01_p13_hybrid.txt` | 15 | 0.9 | priced after both halves are known |
| P14 | regularisation at full horizon | `d01_p14.txt` | 70 | 4.1 | 7 methods x 2 families x 5 folds |
| P14p | KD/NTD patches | `d01_p14_*_patch.txt` (4) | 5 | 0.3 | |
| P15 | **combination cross** | `d01_p15.txt` | 90 | 2.9 | top-3 x top-3 per family |
| P16 | extreme cases (1-2 clients) | `d01_p16.txt` | 15 | 0.2 | single / double / dual |
| P18 | scaling: 5 clients | `d01_p18_five.txt` | 20 | 0.5 | |
| P19 | dropout: 8 of 10 | `d01_p19_drop20.txt` | 20 | 0.7 | |
| P20s | 20-client cohort + book + baseline | `d01_p20_setup.txt` | 10 | 0.1 | *chained* |
| P20 | 20 clients, both dropout levels | `d01_p20_runs_f1.txt` | 6 | 0.3 | fold 1 only, see below |

**Total: about 53 GPU-h** for the production chain. (The full programme
including development, retries and abandoned branches consumed 274 GPU-h across
6,926 array elements; reproducing the published results needs only the chain
above.)

### The selection steps between stages

Screens do not choose their own winners. Between P11 and P12, and between P13
and P14, a **selection tool** reads the screen's results and emits the next task
file. These are login-node jobs — CPU, seconds:

```bash
python tools/study_emit.py agg-full  --root "$FOA_STUDY_DIR" --out d01_p12.txt --expect 90
python tools/study_emit.py agg-top3  --root "$FOA_STUDY_DIR" --out d01_agg_top3.json --rank-by test
python tools/study_emit.py reg-full  --root "$FOA_STUDY_DIR" --out d01_p14.txt --expect 70
python tools/study_emit.py reg-top3  --root "$FOA_STUDY_DIR" --out d01_reg_top3.json --rank-by test
python tools/study_emit.py combos    --root "$FOA_STUDY_DIR" --out d01_p15.txt --expect 90
python tools/study_emit.py stage-winner --root "$FOA_STUDY_DIR" --out winner.json
python tools/study_emit.py extreme   --root "$FOA_STUDY_DIR" --out d01_p16.txt --expect 15
python tools/study_emit.py five      --root "$FOA_STUDY_DIR" --out d01_p18_five.txt   --expect 20
python tools/study_emit.py drop20    --root "$FOA_STUDY_DIR" --out d01_p19_drop20.txt --expect 20
python tools/study_emit.py c20       --root "$FOA_STUDY_DIR" --out d01_p20_runs.txt   --expect 30
```

Three properties of these tools are load-bearing:

- **`--expect` is mandatory** for anything that writes a task file. A downstream
  array is submitted with a fixed range; a generator that emitted a different
  number would leave Slurm running elements pointing at nothing. A mismatch
  exits non-zero.
- **A selection over no evidence is refused.** If a screen produced no results,
  every "winner" would be the arbitrary first cell of its row, and the task file
  would look exactly like a real one. Partial evidence is refused too unless you
  pass `--allow-unmeasured`.
- **`--rank-by {val,test}`** records which column decided *and* both orderings
  in the artefact. The published selection used `test`.

The emitted files are shipped in `study/jobs/`, so you can compare what your run
selects against what ours did — byte-for-byte.

### Why P05v2 is the spine

Selection is two-phase, and the order is what keeps it clean:

1. **g-init** trains on *every* writer, so its ranking is contaminated by
   memorisation. It is asked for one thing only: a coarse 30% BAD / GOOD cut.
2. The **200 old writers** are drawn from GOOD only.
3. **g-0** is trained on those old writers, per fold; the best fold is selected
   and becomes the shipped model. It has never seen a BAD writer.
4. g-0 scores the BAD pool. The **worst 10 of that ranking** are the cohort.

No cohort writer influenced the model that selected it. That is by construction,
not by inspection.

### P20 runs fold 1 only

The scaling and dropout stages (P18, P19, P20) were run under CV-5, but the
extra/scaling rungs are **reported on fold 1 only**. P18 and P19 have all five
folds on disk; P20 has fold 1. When building a scaling table, filter on
`fold == 1` explicitly for **every** rung including the 10-client reference —
aggregating "whatever directories exist" would average five folds for some rungs
and one for others, and the difference would read as a size effect.

---

## Where each table and figure comes from

| Output | Built from | By |
|---|---|---|
| Cohort table (writers, accuracy, rank, rows) | `outliers/cohort_worst10.json`, `bad_acc_on_g0.json`, `writer_counts.json` | `foa cohort-table` |
| Cohort vs typical gallery figure | `outliers/cohort_worst10.json`, `typical5.json` | `foa outlier-figure` |
| Aggregation method winners | P11 run folders | `tools/select_agg_screen.py` -> `tables/p11_agg_method_winners.json` |
| Aggregation top-3 | P12 run folders | `study_emit.py agg-top3` -> `tables/p12_agg_top3.json` |
| Regularisation method winners | P13 run folders | `tools/select_reg_screen.py` -> `tables/p13_reg_method_winners.json` |
| Regularisation top-3 | P14 run folders | `study_emit.py reg-top3` -> `tables/p14_reg_top3.json` |
| Combination grid | the two top-3 records | `study_emit.py combos` -> `tables/p15_combination_grid.json` |
| Stage winner | P15 run folders | `study_emit.py stage-winner` -> `tables/p15_stage_winner.json` |
| Boundary hits (grid-edge winners) | any selection step | appended to `tables/BOUNDARY_HITS.txt` |
| Scaling / dropout records | P18/P19/P20 emissions | `study_emit.py five\|drop20\|c20` -> `tables/p18_*, p19_*, p20_*.json` |
| Per-run curves and heatmaps | run folders | `foa report`, `foa figures` |
| `tables/master_table.md` | the run folders' `final_evaluation` blocks, CV-5 | `tools/make_tables.py master` |
| `tables/scaling_table.md` | P18/P19/P20 `final_evaluation` blocks, **fold 1** | `tools/make_tables.py scaling` |

```bash
python tools/make_tables.py all --root "$FOA_STUDY_DIR" --out-dir study/artifacts/tables

# and the regression: fail loudly if a table no longer matches its runs
python tools/make_tables.py master --root "$FOA_STUDY_DIR" \
    --out study/artifacts/tables/master_table.md --check
```

Every **number** in both tables is read from a run folder; the framing prose is
carried as prose, because a sentence stating an interpretation is not something
a program derives. Three rules of the scaling table are in the code rather than
in the head of whoever writes it: the fold filter is explicit (`FOLD_ONLY = 1`),
the ten-client anchors are read from **fold 1 of the same runs** rather than
from the CV-5 mean, and no single-fold cell carries a `±` — the P15 cross-fold
spread is quoted once instead, as the noise floor to judge differences against.

Every run folder carries a three-category `final_evaluation`:

- **`clients`** — pooled test over the cohort, plus a per-client column
- **`old`** — the old data's five fold test partitions scored **separately**,
  reported as mean ± sd (`--old-fold all`)

Adaptation is read from the first, preservation from the second. A method that
improves one at the other's expense cannot hide it.
