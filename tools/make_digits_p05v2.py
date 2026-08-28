"""
Emit the revised P05-P07 chain: the coarse cut, the shipped model, the real cohort.

    python tools/make_digits_p05v2.py --jobs-dir $FOA_PROJECT_DIR/jobs/v4

Writes two text files: a login-node job.

The redesign moves the selection off g-init and onto g-0. g-init trained on
every writer, so its ranking is contaminated by memorisation; it is now asked
only for a coarse split - who is worth looking at - and the shipped model, which
never saw those writers, does the actual choosing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

STUDY = "Digits_study01"
ROOT = "$FOA_STUDY_DIR"
BOOKS = f"{ROOT}/fold_books"
POOLS = f"{ROOT}/outliers"
TABLES = f"{ROOT}/tables"
ARCHIVE = f"{ROOT}/v1_superseded"
SETTING = "--resolution 28 --classes digits"
CONVERGE = "--early-stopping-patience 10 --min-epochs 20"
FOLDS = (1, 2, 3, 4, 5)

GINIT_SCORES = f"{POOLS}/clients_acc_on_global.json"
BAD_POOL = f"{POOLS}/pool_bad.json"
BAD_SCORES = f"{POOLS}/bad_scores_on_g0.json"
BAD_FLAT = f"{POOLS}/bad_acc_on_g0.json"
OLD_FILE = f"{POOLS}/old_data.json"
COHORT_FILE = f"{POOLS}/cohort_worst10.json"
TYPICAL_FILE = f"{POOLS}/typical5.json"
OLD_BOOK = f"{BOOKS}/old_data.foldbook.npz"
COHORT_BOOK = f"{BOOKS}/cohort10.foldbook.npz"
G0 = f"{ROOT}/g0_model"

COHORT_SIZE = 10
OLD_SIZE = 200
TYPICAL_SIZE = 5
BAD_FRACTION = 0.30
OLD_MIN_SAMPLES = 100
#: Fresh seeds for the v2 draws, logged here so both are reproducible.
OLD_SEED = 20260829
TYPICAL_SEED = 20260830

#: Measured from P02/P04.
SCORED_WRITERS = 3580
BAD_COUNT = 1074
GOOD_COUNT = 2506
BAD_ROWS = 121000
GINIT_FOLD = 3
V1_G0_FOLD = 5

#: What v1 left behind, moved out of the way rather than deleted.
V1_ARTEFACTS = (
    "g0_fold1", "g0_fold2", "g0_fold3", "g0_fold4", "g0_fold5",
    "g0_model", "g0_selection.json",
    "g0_evaluations.json", "g0_perfold_evaluations.json",
    "outliers/old_data.json", "outliers/cohort_worst10.json",
    "outliers/typical5.json",
    "fold_books/old_data.foldbook.npz", "fold_books/cohort10.foldbook.npz",
    "tables/cohort_table.json", "tables/cohort_table.csv", "tables/gallery",
)


def archive_line() -> str:
    """Move every v1 artefact a v2 reader could mistake for its own."""
    moves = " ".join(f'"$D/{name}"' for name in V1_ARTEFACTS)
    return (
        f'D={ROOT}; A={ARCHIVE}; mkdir -p "$A/outliers" "$A/fold_books" '
        f'"$A/tables" && for F in {moves}; do '
        '[ -e "$F" ] && mv -f "$F" "$A/${F#$D/}" && echo "archived ${F#$D/}"; '
        'done; echo "v1 artefacts archived under $A"; exit 0'
    )


def lines() -> list[str]:
    tasks = [
        # 1. get v1 out of the way before anything can read it by mistake.
        archive_line(),
        # 2. the coarse cut - the only thing g-init is asked for now.
        f"foa split-pools --results-dir {ROOT} {SETTING}"
        f" --bad-fraction {BAD_FRACTION} --scores {GINIT_SCORES}"
        f" --require-trainable --out {POOLS}/pools.json --csv {POOLS}/pools.csv",
        # 3. the old data, drawn from GOOD only - which is what keeps the
        #    later scoring clean.
        f"foa draw-old-data --results-dir {ROOT} {SETTING} --size {OLD_SIZE}"
        f" --seed {OLD_SEED} --min-samples {OLD_MIN_SAMPLES} --require-trainable"
        f" --exclude-file {BAD_POOL} --out {OLD_FILE}",
        # 4. its book.
        f"foa fold-book --results-dir {ROOT} {SETTING} --out {BOOKS}/old_data"
        f" --clients-file {OLD_FILE} --folds 5 --seed 42 --train-rate 0.6"
        f" --eval-rate 0.2 --tag digits_old{OLD_SIZE}_v2",
    ]
    # 5-9. g-0 v2, one fold each; each writes its own Fisher.
    tasks += [
        f"foa global-train --results-dir {ROOT}/g0_fold{fold} {SETTING}"
        f" --model fedavg_cnn --population file --writers-file {OLD_FILE}"
        f" --fold-book {OLD_BOOK} --fold {fold}"
        f" --epochs 100 {CONVERGE} --batch-size 64 --split-seed 42 --seed {fold}"
        for fold in FOLDS
    ]
    tasks += [
        # 10. the shipped model.
        f"foa select-fold --results-dir {ROOT} {SETTING} --root {ROOT}"
        f" --prefix g0 --name g0 --folds 1 2 3 4 5",
        # 11. the selection pass: g-0 on every row of every bad writer.
        f"foa score-pool --results-dir {ROOT} {SETTING} --model-path {G0}"
        f" --clients-file {BAD_POOL} --batch-size 256"
        f" --out {BAD_SCORES} --csv {POOLS}/bad_scores_on_g0.csv"
        f" --accuracies-name bad_acc_on_g0.json",
        # 12. the cohort: the worst ten of THAT ranking.
        f"foa select-outliers --results-dir {ROOT} {SETTING} --mode worst"
        f" --k {COHORT_SIZE} --scores {BAD_FLAT} --require-trainable"
        f" --tag digits_cohort{COHORT_SIZE}_v2 --force --no-accuracy-table",
        # 13. its book.
        f"foa fold-book --results-dir {ROOT} {SETTING}"
        f" --out {BOOKS}/cohort{COHORT_SIZE} --clients-file {COHORT_FILE}"
        f" --folds 5 --seed 42 --train-rate 0.6 --eval-rate 0.2"
        f" --tag digits_cohort{COHORT_SIZE}_v2",
        # 14. the paper's table, ranked within the bad pool, carrying both scores.
        f"foa cohort-table --results-dir {ROOT} {SETTING}"
        f" --clients-file {COHORT_FILE} --fold-book {COHORT_BOOK}"
        f" --scores {BAD_FLAT} --extra-scores ginit={GINIT_SCORES}"
        f' --rule "the {COHORT_SIZE} worst of the bad pool under the shipped model g-0"'
        f" --out {TABLES}/cohort_table.json --csv {TABLES}/cohort_table.csv",
        # 15. the figure's reference writers: the easiest of the GOOD pool.
        f"foa draw-cohort --results-dir {ROOT} {SETTING} --best"
        f" --k {TYPICAL_SIZE} --seed {TYPICAL_SEED} --scores {GINIT_SCORES}"
        f" --require-trainable --exclude-file {BAD_POOL}"
        f" --tag digits_typical{TYPICAL_SIZE}_v2 --out {TYPICAL_FILE}",
        # 16. the figure.
        f"foa outlier-figure --results-dir {ROOT} {SETTING}"
        f" --outliers-file {COHORT_FILE} --source-clients-file {TYPICAL_FILE}"
        f" --n-examples 3 --max-columns 10 --batch-size 256"
        f" --out {TABLES}/gallery/cohort_vs_typical.png",
    ]
    # 17-21 preservation, 22-26 the do-nothing baseline.
    tasks += [
        f"foa evaluate-book --results-dir {ROOT} {SETTING} --model-path {G0}"
        f" --fold-book {OLD_BOOK} --fold {fold} --part test"
        f" --clients-file {OLD_FILE} --batch-size 256"
        f" --tag old_data_fold{fold} --out {ROOT}/g0_evaluations.json"
        for fold in FOLDS
    ]
    tasks += [
        f"foa evaluate-book --results-dir {ROOT} {SETTING} --model-path {G0}"
        f" --fold-book {COHORT_BOOK} --fold {fold} --part test"
        f" --clients-file {COHORT_FILE} --batch-size 256"
        f" --tag cohort_fold{fold} --out {ROOT}/g0_perfold_evaluations.json"
        for fold in FOLDS
    ]
    return tasks


def header(tasks: list[str]) -> list[str]:
    return [
        f"# d01_p05v2.txt - {STUDY}, revised P05-P07: the coarse cut, g-0, the cohort.",
        "#",
        "# THE REDESIGN. g-init trained on every writer, so its per-writer",
        "# accuracy is contaminated by memorisation - a writer can score well",
        "# because the model remembers it, not because it is easy. That is",
        "# survivable in a COARSE CUT and fatal in a SELECTION, so the two jobs",
        "# are now split:",
        "#",
        f"#   g-init  cuts the population at the worst {BAD_FRACTION:.0%} - BAD vs GOOD, and",
        "#           nothing else. Its job ends there.",
        "#   g-0     is trained on GOOD writers only, then scores every BAD",
        "#           writer it has never seen. Those scores pick the cohort.",
        "#",
        "# NO LEAKAGE, BY CONSTRUCTION. The old data is drawn from GOOD, so g-0",
        "# never trains on a writer it will later score. That is the whole reason",
        "# the draw is restricted rather than merely cohort-excluded.",
        "#",
        "# LINE 1 ARCHIVES v1. The previous g-0 (winner fold "
        f"{V1_G0_FOLD}), its evaluations, the",
        "# old-data draw, the cohort, both books, the table and the gallery are",
        "# all superseded - they were built around a selection that no longer",
        "# stands. They are MOVED, not deleted, to",
        f"#   {ARCHIVE}/",
        "# keeping their relative paths. Nothing downstream can pick them up:",
        "# every reader names an exact path in the study root, and after this",
        "# line those paths are empty until v2 refills them. Recoverable if a",
        "# comparison is ever wanted.",
        "#",
        "# WHAT SURVIVES v1: ginit_*, the all-writers book, writer_scores.json,",
        "# clients_acc_on_global.json and writer_counts.*. The coarse detector and",
        "# the census are still exactly what they were.",
        "#",
        f"# LINE 2 cuts the pools at ceil({BAD_FRACTION} * {SCORED_WRITERS}) = {BAD_COUNT}: BAD is the worst",
        f"# {BAD_COUNT}, GOOD the remaining {GOOD_COUNT}. Banked with the rule, the cut rank and",
        "# the accuracy there, plus both lists as ordinary pool files.",
        "#",
        f"# LINE 3 draws {OLD_SIZE} old writers from GOOD only, seeded {OLD_SEED}.",
        f"# LINE 4 books them. LINES 5-9 train g-0, five folds, each writing its",
        "# own Fisher to g0_fold<k>/global_results/fisher - the path the",
        "# regularisation stages name. LINE 10 picks the winner by VALIDATION.",
        "#",
        f"# LINE 11 IS THE SELECTION PASS: g-0 on EVERY row of each of the {BAD_COUNT}",
        f"# bad writers - about {BAD_ROWS:,} images, one GPU task, minutes. Every row,",
        "# not a held-out partition: g-0 has never seen these writers at all, so",
        "# there is nothing to hold out from, and using all of a writer's data is",
        "# what makes the estimate as tight as that writer allows. These are the",
        "# small difficult ones; the tightness matters.",
        "#",
        f"# LINE 12 cuts the cohort - the worst {COHORT_SIZE} of that ranking, not of g-init's.",
        "# LINE 13 books them. LINE 14 banks the table: each writer's rank within",
        f"# the {BAD_COUNT}-writer bad pool, the g-0 accuracy that selected it, its g-init",
        "# accuracy alongside so the two rankings can be compared, its per-class",
        "# counts and its per-fold split sizes.",
        "#",
        "# LINES 15-16 redraw the figure's reference writers from GOOD and render",
        "# it again. LINES 17-26 are the baselines: preservation on the five old",
        "# folds, and the do-nothing baseline per cohort writer. Same two",
        "# filenames and shapes as before, so every downstream reader is",
        "# unchanged.",
        "#",
        "# BOTH EVALUATION GROUPS APPEND, keyed by tag - submit them %1.",
        "#",
        f"# {len(tasks)} tasks. Chain: archive -> pools -> draw -> book -> folds 1-5",
        "# -> winner -> score-bad -> cohort -> book -> table/typical/gallery",
        "# -> evaluations.",
        "#",
    ]


def readme(tasks: list[str]) -> str:
    return f"""# {STUDY} - revised P05-P07

## The redesign, and why

g-init trained on **every** writer, so its per-writer accuracy is contaminated
by memorisation: a writer can score well because the model remembers it, not
because it is easy. That is survivable in a coarse cut and fatal in a selection.
So the two jobs are split:

| | does | on |
|---|---|---|
| **g-init** | cuts BAD from GOOD at the worst {BAD_FRACTION:.0%} | every writer |
| **g-0** | scores every BAD writer, and the worst {COHORT_SIZE} become the cohort | writers it never saw |

**No leakage, by construction.** The old data is drawn from **GOOD only**, so
g-0 never trains on a writer it will later score. That is why the draw is
restricted rather than merely cohort-excluded.

| | count |
|---|---|
| scored writers | {SCORED_WRITERS:,} |
| BAD = ceil({BAD_FRACTION} x {SCORED_WRITERS:,}) | **{BAD_COUNT:,}** |
| GOOD | {GOOD_COUNT:,} |
| old data drawn from GOOD | {OLD_SIZE} |
| cohort = worst of BAD under g-0 | {COHORT_SIZE} |

## Where v1 went

**Archived, not deleted** - moved to `{ARCHIVE}/`, keeping their relative
paths:

```
g0_fold1..5/  g0_model  g0_selection.json  (v1 winner: fold {V1_G0_FOLD})
g0_evaluations.json  g0_perfold_evaluations.json
outliers/old_data.json  outliers/cohort_worst10.json  outliers/typical5.json
fold_books/old_data.foldbook.npz  fold_books/cohort10.foldbook.npz
tables/cohort_table.json  tables/cohort_table.csv  tables/gallery/
```

Nothing downstream can pick them up: every reader names an exact path in the
study root, and after line 1 those paths are empty until v2 refills them.
Recoverable if a comparison is ever wanted - which is why they are moved rather
than removed.

**Surviving v1 unchanged:** `ginit_*`, `fold_books/all_writers_digits.foldbook.npz`,
`outliers/writer_scores.json`, `outliers/clients_acc_on_global.*`,
`outliers/writer_counts.*`. The coarse detector and the census are still exactly
what they were.

## The chain

| # | task | writes |
|---|---|---|
| 1 | archive v1 | `v1_superseded/` |
| 2 | the coarse cut | `outliers/pools.json` + `.csv`, `pool_bad.json`, `pool_good.json` |
| 3 | old data from GOOD, seed `{OLD_SEED}` | `outliers/old_data.json` |
| 4 | its book | `fold_books/old_data.foldbook.npz` |
| 5-9 | g-0, one fold each | `g0_fold<k>/` incl. its Fisher |
| 10 | the winner, by validation | `g0_model`, `g0_selection.json` (sha256) |
| 11 | **g-0 scores all {BAD_COUNT:,} bad writers** | `outliers/bad_scores_on_g0.json` + `.csv` + flat |
| 12 | the cohort: worst {COHORT_SIZE} of that | `outliers/cohort_worst10.json` |
| 13 | its book | `fold_books/cohort10.foldbook.npz` |
| 14 | the table | `tables/cohort_table.json` + `.csv` |
| 15 | reference writers from GOOD, seed `{TYPICAL_SEED}` | `outliers/typical5.json` |
| 16 | the figure | `tables/gallery/cohort_vs_typical.png` |
| 17-21 | preservation | `g0_evaluations.json` |
| 22-26 | the do-nothing baseline | `g0_perfold_evaluations.json` |

## The selection pass

Line 11 scores g-0 on **every row** of each of the {BAD_COUNT:,} bad writers -
about {BAD_ROWS:,} images. Every row, not a held-out partition: g-0 has never
seen these writers at all, so there is nothing to hold out from, and using all
of a writer's data makes the estimate as tight as that writer allows. These are
the small, difficult ones; the tightness is the point.

## The table

Each cohort writer carries its **rank within the {BAD_COUNT:,}-writer bad pool**
(not out of {SCORED_WRITERS:,} - the pool is the population it was selected
from), the g-0 accuracy that selected it, **its g-init accuracy alongside** so
the two rankings can be compared, its per-class counts and its per-fold
train/val/test sizes.

## Order

```bash
J=$FOA_PROJECT_DIR/jobs/v4
A="sbatch --account=$FOA_ACCOUNT"
SB=$J/study_phase.sbatch

$A --array=1-4%1   $SB $J/d01_p05v2.txt   # archive, pools, draw, book
$A --array=5-9%5   $SB $J/d01_p05v2.txt   # the five g-0 folds
$A --array=10-16%1 $SB $J/d01_p05v2.txt   # winner, score-bad, cohort, book, table, typical, figure
$A --array=17-26%1 $SB $J/d01_p05v2.txt   # the baselines, one at a time
```

No exports needed: `study_phase.sbatch` defaults `FOA_STUDY_DIR` to
`studies/{STUDY}` and `FOA_NIST_CLASSES` to `digits`.

## Cost

| lines | what | per task | total |
|---|---|---|---|
| 1-4 | archive, pools, draw, book | seconds to ~2 min, CPU | - |
| 5-9 | a g-0 fold | ~3-4.5 min | ~0.33 GPU-h |
| 10 | the winner | seconds | - |
| 11 | g-0 over {BAD_ROWS:,} rows | ~2 min | ~0.03 GPU-h |
| 12-16 | cohort, book, table, typical, figure | ~1-2 min each, CPU | - |
| 17-26 | the ten evaluations | ~45 s | ~0.13 GPU-h |

**~0.55 GPU-h in total**, {len(tasks)} tasks.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-dir", required=True)
    args = parser.parse_args()
    jobs = Path(args.jobs_dir)
    jobs.mkdir(parents=True, exist_ok=True)
    tasks = lines()
    (jobs / "d01_p05v2.txt").write_text("\n".join(header(tasks) + tasks) + "\n")
    (jobs / "d01_p05v2_README.md").write_text(readme(tasks))
    print(f"wrote {jobs}/d01_p05v2.txt: {len(tasks)} tasks")
    print(f"wrote {jobs}/d01_p05v2_README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
