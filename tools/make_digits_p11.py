"""
Emit P11 of the digits study: the aggregation screen.

    python tools/make_digits_p11.py --jobs-dir $FOA_PROJECT_DIR/jobs/v4

Writes two text files: a login-node job.

P09 fixed the server rule at plain FedAvg and varied nothing, so it cannot say
how much of the adaptation - or of the forgetting - was the rule's doing. P11
varies the rule and nothing else.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from federated_outlier_adaptation.training import agg_cells
from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training.study_config import DIGITS_STUDY01 as _STUDY

#: The grid is searched at ONE participation rate and the winners are carried to
#: the others. Searching every rate would multiply the most expensive stage in
#: the programme by the number of rates, to answer a question the transfer
#: already answers: whether a configuration chosen under heavier dropout still
#: holds when fewer clients drop.
#:
#: The search runs at the HARDER rate - two of ten dropped rather than one - so
#: the winners are chosen where the averaging is noisiest and the anchor matters
#: most. A rule that survives 8-of-10 has a better claim on 9-of-10 than the
#: reverse would.
CFG = _STUDY

#: Sampler seeds of the screen. Well clear of P09's 30001-30505 and of every
#: hand-written seed in the earlier digit stages, so no two lines anywhere in
#: this study draw the same participation pattern.
SEED_OFFSET = 2000

#: Measured from the study's own books.
COHORT_TRAIN = 579
COHORT_TEST = 222
OLD_TEST = 5297
G0_FOLD = 2


def lines() -> list[str]:
    """One line per (cell, fold), at the screening horizon."""
    return [
        SL.agg_line(
            CFG, cell, fold, SL.SCREEN_ROUNDS,
            CFG.seed_base + SEED_OFFSET + index * 10 + fold,
        )
        for index, cell in enumerate(agg_cells.screen_cells())
        for fold in SL.FOLDS
    ]


#: How many cells the screen ran before the boundary extension was added.
SCREENED_CELLS = 169


def ext_lines() -> list[str]:
    """
    The boundary extension: the cells appended to the table after the screen.

    ``lines()`` enumerates the whole table, and the extension cells were
    *appended* to it, so the first ``SCREENED_CELLS * 5`` lines are byte for
    byte the ones already submitted and the tail is exactly the new work. Slicing
    is what keeps the two files consistent - regenerating the extension can never
    disagree with the screen it extends.
    """
    return lines()[SCREENED_CELLS * len(SL.FOLDS):]


def screened_lines() -> list[str]:
    """The lines the original screen ran - unchanged, and asserted so."""
    return lines()[: SCREENED_CELLS * len(SL.FOLDS)]


def header(tasks: list[str]) -> list[str]:
    # The cells this FILE runs, which is the table as it stood when the screen
    # was submitted - the boundary extension was appended afterwards and lives
    # in d01_p11_ext.txt. Counting the whole table here would describe the file
    # as something it is not.
    cells = agg_cells.screen_cells()[:SCREENED_CELLS]
    by_path: dict = {}
    for cell in cells:
        by_path[cell["path"]] = by_path.get(cell["path"], 0) + 1
    return [
        f"# d01_p11.txt - {CFG.name}, P11: WHAT THE AGGREGATION RULE IS WORTH.",
        "#",
        "# P09 fixed the server rule at plain FedAvg and varied nothing, so it",
        "# cannot say how much of the adaptation - or of the forgetting - was the",
        "# rule's doing. This file varies the rule and nothing else: the same ten",
        f"# cohort writers, {CFG.clients_per_round} of {CFG.cohort_size} per round, E=5, batch 64, the same",
        f"# cohort fold book, g-0 (winner fold {G0_FOLD}) as the only initialisation. A",
        "# difference between two lines is the rule.",
        "#",
        "# SCREENING HORIZON: 25 ROUNDS, NOT 100. Nothing in this file is a",
        "# reported number. A 25-round result is a RANKING SIGNAL - reading it as",
        "# a result is reading a race at the quarter mark. The winners are re-run",
        "# at 100 rounds by P12, which is gated separately and NOT yet authorised.",
        "#",
        f"# {len(cells)} cells x {len(SL.FOLDS)} folds = {len(tasks)} tasks:",
        f"#   concurrent {by_path.get('concurrent', 0):>4} cells",
        f"#   sequential {by_path.get('sequential', 0):>4} cells",
        f"#   control    {by_path.get('control', 0):>4} cells",
        "#",
        "# ONE FAMILY OR TWO, AS THE CELL SAYS. A named rule lives in exactly one",
        "# (scenario, metadata) family, so those lines produce one result. The two",
        "# control cells name the 'fedavg' variant, which deliberately spans both",
        "# schedules and skips the two families whose rule is the same update",
        "# written the other way round. That is the table's own definition and it",
        "# is followed rather than second-guessed.",
        "#",
        "# EVERY LINE reports the standard three categories into its",
        "# accuracies_0.json under 'final_evaluation': the pooled clients test on",
        "# the cohort's fold-k rows, the per-client column, and the old data's",
        "# FIVE fold test partitions scored separately with their mean and spread.",
        "# Preservation is five numbers because five partitions of one population",
        "# is what the error bar is made of; merging them would report a precision",
        "# that is not there.",
        "#",
        "# SEEDS. Each line's sampler seed is",
        f"#   {CFG.seed_base} + {SEED_OFFSET} + cell_index * 10 + fold",
        "# which is deterministic, distinct per line, and well clear of P09's",
        "# range - so no two lines anywhere in this study draw the same",
        "# participation pattern.",
        "#",
        "# INDEPENDENT OF EVERYTHING ELSE. This file reads only artefacts that",
        "# already exist - the cohort, its book, the old book, g-0 - and writes",
        "# only its own folders. It needs no other stage to have run first beyond",
        "# P05v2/P07, and produces no input for P08, P09 or P10.",
        "#",
        "#   J=$FOA_PROJECT_DIR/jobs/v4",
        f"#   sbatch --account=$FOA_ACCOUNT --array=1-{len(tasks)}%40 \\",
        "#          $J/study_phase.sbatch $J/d01_p11.txt",
        "#",
        "# AFTER IT COMPLETES, the selection is a CPU command - see",
        "# d01_p11_README.md. It writes tables/p11_agg_method_winners.json and",
        "# emits the P12 task file, which must not be submitted without the",
        "# owner's go.",
        "#",
    ]


def readme(tasks: list[str]) -> str:
    cells = agg_cells.screen_cells()[:SCREENED_CELLS]
    per_round = 0.31 + 0.0345 * CFG.clients_per_round * 5
    screen_s = per_round * SL.SCREEN_ROUNDS
    task_s = screen_s + 6 + 2 + 40
    return f"""# {CFG.name} - P11: the aggregation screen

P09 fixed the server rule at plain FedAvg and varied nothing, so it cannot say
how much of the adaptation - or of the forgetting - was the rule's doing. P11
varies the rule and nothing else.

{CFG.describe()}. g-0 (winner fold {G0_FOLD}) is the only initialisation.

| | |
|---|---|
| cells | **{len(cells)}** |
| folds | {len(SL.FOLDS)} |
| **tasks** | **{len(tasks)}** |
| horizon | {SL.SCREEN_ROUNDS} rounds |

## A ranking horizon, not a reporting one

Nothing in this file is a reported number. A 25-round result is a ranking
signal; reading it as a result is reading a race at the quarter mark. The
winners are re-run at {SL.FULL_ROUNDS} rounds by **P12, which is gated
separately and is not yet authorised**.

## Submit

```bash
J=$FOA_PROJECT_DIR/jobs/v4
sbatch --account=$FOA_ACCOUNT --array=1-{len(tasks)}%40 \\
       $J/study_phase.sbatch $J/d01_p11.txt
```

No exports needed: `study_phase.sbatch` defaults `FOA_STUDY_DIR` to
`studies/{CFG.name}` and `FOA_NIST_CLASSES` to `digits`, and refuses any line
whose paths point elsewhere.

This file is independent of P08, P09 and P10 - it reads only the cohort, its
book, the old book and g-0, and writes only its own folders - so it can run
alongside them.

## After the screen: the selection

A CPU command, minutes, no GPU:

```bash
P=$FOA_PROJECT_DIR
S=$P/results_v4/studies/{CFG.name}
cd $P/repo_foa && PYTHONPATH=$PWD/src $P/envs/fal/bin/python tools/study_emit.py \\
    agg-full --study {CFG.name} --root $S \\
    --out $P/jobs/v4/d01_p12.txt --expect {len(SL.AGG_METHODS) * len(SL.FOLDS)}
```

It writes `tables/p11_agg_method_winners.json` - each method's best cell with
its adaptation, spread and preservation - and emits `d01_p12.txt`.

**No method filtering.** Every one of the {len(SL.AGG_METHODS)} methods gets its
best coefficients re-run, so the full stage compares eighteen methods rather
than one winner against nothing. That is {len(SL.AGG_METHODS)} x
{len(SL.FOLDS)} = **{len(SL.AGG_METHODS) * len(SL.FOLDS)} lines**, and
`--expect` makes the emitter exit non-zero if it produces any other number.

**`d01_p12.txt` IS NOT AUTHORISED TO RUN.** The emitter marks it in its own
header. It exists so the file is ready for review, not so it can be submitted.

### Grid-edge winners

A winner sitting at the lowest or highest value of its row means the optimum may
lie outside the range searched. Those are written to
`tables/BOUNDARY_HITS.txt` and printed - and they **never halt** anything: the
run is still the best of what was tried, and the fact is worth reporting rather
than acting on automatically.

## Method grouping

The full stage runs one line per **method** at its best coefficients, so
"method" is a definition, not an intuition. The {len(cells)} cells group into
{len(SL.AGG_METHODS)}: a coefficient row like `eta_*` is one method at several
settings, while `weight_uniform` and `weight_capped` are two - a weighting rule
is not a coefficient of another weighting rule - and the two controls are
likewise distinct, because a control with the oracle stop rule armed is not a
setting of the control without it.

## Seeds

Each line's sampler seed is `{CFG.seed_base} + {SEED_OFFSET} + cell_index * 10 + fold`:
deterministic, distinct per line, and well clear of P09's 30001-30505, so no two
lines anywhere in this study draw the same participation pattern.

## Cost

At `0.31 + 0.0345 x {CFG.clients_per_round} x 5 = {per_round:.2f}` s per round per family:

| | seconds |
|---|---|
| training, {SL.SCREEN_ROUNDS} rounds, one family | {screen_s:.0f} |
| per-round evaluations | ~6 |
| final three-category evaluation | ~2 |
| interpreter, torch, cache, two books | ~40 |
| **wall per task** | **~{task_s:.0f} (~{task_s / 60:.1f} min)** |

The two control cells run both families and cost about twice the training part.

**{len(tasks)} tasks is roughly {len(tasks) * task_s / 3600:.0f} GPU-h**, about
{len(tasks) * task_s / 3600 / 40 * 60:.0f} minutes of wall clock at
`--array=...%40`.

Startup is a large share of a short task here: the cohort is ten small writers,
so a screening run trains on {COHORT_TRAIN} images per round-epoch while the
evaluation reads ~{OLD_TEST:,} old test rows per round. Batching the five folds
of a cell into one element would cut the total substantially - but one task per
cell-fold keeps a failure to one element and matches every other stage.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-dir", required=True)
    parser.add_argument(
        "--clients-per-round", type=int, default=None, metavar="M",
        help="Participation the grid is searched at. Defaults to the study's "
             "own rate; the programme searches at the harder rate and carries "
             "the winners to the others.",
    )
    args = parser.parse_args()
    if args.clients_per_round is not None:
        import dataclasses
        global CFG
        CFG = dataclasses.replace(CFG, clients_per_round=args.clients_per_round)
    jobs = Path(args.jobs_dir)
    jobs.mkdir(parents=True, exist_ok=True)
    screened = screened_lines()
    ext = ext_lines()
    (jobs / "d01_p11.txt").write_text(
        "\n".join(header(screened) + screened) + "\n"
    )
    (jobs / "d01_p11_ext.txt").write_text(
        "\n".join(ext_header(ext) + ext) + "\n"
    )
    (jobs / "d01_p11_README.md").write_text(readme(screened))
    (jobs / "d01_p11_ext_README.md").write_text(ext_readme(ext))
    print(f"wrote {jobs}/d01_p11.txt: {len(screened)} tasks")
    print(f"wrote {jobs}/d01_p11_ext.txt: {len(ext)} tasks")
    print(f"wrote {jobs}/d01_p11_README.md")
    print(f"wrote {jobs}/d01_p11_ext_README.md")
    return 0




# --------------------------------------------------------------------------- #
# the boundary extension
# --------------------------------------------------------------------------- #
#: What the trimmed mean actually does at this study's nine clients per round.
#: ``trim = int(f * K)`` and the rule only bites when ``trim > 0``.
TRIM_SURVIVORS = {0.1: 9, 0.2: 7, 0.3: 5, 0.4: 3}
CLIENTS = 9


def ext_header(tasks: list[str]) -> list[str]:
    return [
        f"# d01_p11_ext.txt - {CFG.name}, P11 BOUNDARY EXTENSION: the trim row.",
        "#",
        "# WHY. The screen's trimmed-mean winner was trimmed_0p2, which sits on",
        "# the HIGH edge of the row that was searched, [0.1, 0.2]. A winner at the",
        "# end of its row means the optimum may lie outside it, so the row is",
        "# widened rather than the result being reported as if the range had been",
        "# a choice.",
        "#",
        f"# WHAT TRIMMING ACTUALLY HAPPENS AT {CLIENTS} CLIENTS. The rule drops",
        "# int(f * K) updates from each end per coordinate, and only bites when",
        f"# that is at least one. At K = {CLIENTS}:",
        "#",
        f"#   f=0.1  int(0.9)=0  ->  drops NOTHING, all {TRIM_SURVIVORS[0.1]} updates survive",
        f"#   f=0.2  int(1.8)=1  ->  drops 1 each end, {TRIM_SURVIVORS[0.2]} survive",
        f"#   f=0.3  int(2.7)=2  ->  drops 2 each end, {TRIM_SURVIVORS[0.3]} survive",
        f"#   f=0.4  int(3.6)=3  ->  drops 3 each end, {TRIM_SURVIVORS[0.4]} survive",
        "#",
        "# READ THAT AGAIN: at nine clients, trim_frac=0.1 is NOT a trimmed mean.",
        "# int(0.1 * 9) is zero, so trimmed_0p1 computes the plain coordinate-wise",
        "# mean and is arithmetically identical to con_delta_eta at eta=1. The row",
        "# the boundary rule saw was therefore effectively [no trimming, drop one]",
        "# - trimmed_0p2 was the only cell in it that trimmed anything at all.",
        "# That makes the extension more clearly warranted than the edge alone",
        "# suggested, and it is worth stating in the write-up rather than leaving",
        "# a reader to rediscover it from the aggregation source.",
        "#",
        "# 0.5 is not reachable: ServerState refuses trim_fraction outside [0, 0.5),",
        "# and int(0.5 * 9) = 4 would leave a single update, which is not a mean of",
        "# anything. So 0.4 is the top of the row that can exist here.",
        "#",
        "# THE CELLS ARE APPENDED TO THE TABLE, NOT INSERTED INTO THE TRIM ROW.",
        "# A screening seed is a function of a cell's index, so inserting would",
        "# have re-seeded every cell after the trim row and made the screen that",
        "# has already run unreproducible from the table describing it. Appended,",
        f"# the {SCREENED_CELLS * len(SL.FOLDS)} lines of d01_p11.txt are byte for byte what was submitted",
        "# and these ten are the whole of the new work. The selector groups by id",
        "# prefix, so it sees all four trim cells as siblings regardless of where",
        "# they sit, and boundary detection now evaluates the widened range.",
        "#",
        f"# {len(TRIM_SURVIVORS) - 2} cells x {len(SL.FOLDS)} folds = {len(tasks)} tasks. Same protocol as the screen:",
        f"# 25 rounds, {CFG.clients_per_round} of {CFG.cohort_size} per round, digits, g-0 init, --old-fold all,",
        "# fresh sampler seeds continuing the screen's own scheme.",
        "#",
        "#   J=$FOA_PROJECT_DIR/jobs/v4",
        f"#   sbatch --account=$FOA_ACCOUNT --array=1-{len(tasks)} \\",
        "#          $J/study_phase.sbatch $J/d01_p11_ext.txt",
        "#",
        "# AFTER THESE RUN, re-run the selection - see d01_p11_ext_README.md. If a",
        "# new trimmed winner emerges the delta is five full-horizon tasks, emitted",
        "# as d01_p12_trimmed_patch.txt and gated exactly as P12 was.",
        "#",
    ]


def ext_readme(tasks: list[str]) -> str:
    per_round = 0.31 + 0.0345 * CFG.clients_per_round * 5
    task_s = per_round * SL.SCREEN_ROUNDS + 6 + 2 + 40
    return f"""# {CFG.name} - P11 boundary extension: the trim row

The screen's trimmed-mean winner was `trimmed_0p2`, sitting on the **high edge**
of the row that was searched, `[0.1, 0.2]`. A winner at the end of its row means
the optimum may lie outside it, so the row is widened: `trim_frac` 0.3 and 0.4,
**{len(tasks)} tasks** (2 cells x 5 folds), same protocol as the screen.

## What trimming actually happens at nine clients

The rule drops `int(f * K)` updates from each end per coordinate, and only bites
when that is at least one. Measured against the implementation at K = {CLIENTS}:

| `trim_frac` | `int(f x 9)` | dropped each end | **updates surviving** |
|---|---|---|---|
| 0.1 | 0 | 0 | **{TRIM_SURVIVORS[0.1]}** |
| 0.2 | 1 | 1 | **{TRIM_SURVIVORS[0.2]}** |
| 0.3 | 2 | 2 | **{TRIM_SURVIVORS[0.3]}** |
| 0.4 | 3 | 3 | **{TRIM_SURVIVORS[0.4]}** |

**At nine clients, `trim_frac=0.1` is not a trimmed mean.** `int(0.1 x 9)` is
zero, so `trimmed_0p1` computes the plain coordinate-wise mean and is
arithmetically identical to `con_delta_eta` at `eta=1`. The row the boundary
rule saw was therefore effectively *[no trimming, drop one]* - `trimmed_0p2` was
the only cell in it that trimmed anything at all.

That makes the extension more clearly warranted than the edge alone suggested,
and it belongs in the write-up rather than being left for a reader to
rediscover from the aggregation source.

`0.5` is not reachable: `ServerState` refuses `trim_fraction` outside `[0, 0.5)`,
and `int(0.5 x 9) = 4` would leave a single update, which is not a mean of
anything. **0.4 is the top of the row that can exist here.**

## Appended, not inserted

A screening seed is a function of the cell's index in the table, so inserting
0.3 and 0.4 into the trim row would have re-seeded every cell after it - and the
screen that has already run would no longer be reproducible from the table that
describes it.

They are **appended**. The {SCREENED_CELLS * len(SL.FOLDS)} lines of
`d01_p11.txt` are byte for byte what was submitted, and these ten are the whole
of the new work. The selector groups by id prefix, so it sees all four trim
cells as siblings regardless of position, and boundary detection now evaluates
the widened range `[0.1, 0.2, 0.3, 0.4]`.

## Submit

```bash
J=$FOA_PROJECT_DIR/jobs/v4
sbatch --account=$FOA_ACCOUNT --array=1-{len(tasks)} \\
       $J/study_phase.sbatch $J/d01_p11_ext.txt
```

## After they run: re-select

```bash
P=$FOA_PROJECT_DIR
S=$P/results_v4/studies/{CFG.name}
cd $P/repo_foa && PYTHONPATH=$PWD/src $P/envs/fal/bin/python tools/study_emit.py \\
    agg-full --study {CFG.name} --root $S \\
    --out $P/jobs/v4/d01_p12.txt --expect {len(SL.AGG_METHODS) * len(SL.FOLDS)}
```

This re-runs the whole per-method selection over the widened table. Two
outcomes:

- **The trimmed winner is unchanged** - the extension confirmed the edge was the
  optimum, `d01_p12.txt` is regenerated identical, and nothing needs re-running.
  `tables/BOUNDARY_HITS.txt` will still record `trimmed_0p4` as an edge if it
  wins, which is then a genuine finding rather than an artefact of a short row.
- **A new trimmed winner emerges** - only the trimmed method's five lines change.
  Emit just those:

```bash
cd $P/repo_foa && PYTHONPATH=$PWD/src $P/envs/fal/bin/python tools/study_emit.py \\
    agg-trimmed-patch --study {CFG.name} --root $S \\
    --out $P/jobs/v4/d01_p12_trimmed_patch.txt --expect 5
```

**{len(SL.FOLDS)} tasks**, not 90. The emitter compares the new winner against
the cell P12 was built from and says which case it is; the patch file is stamped
`NOT AUTHORISED` exactly as `d01_p12.txt` was, because a re-run is the owner's
call.

## Cost

At `{per_round:.2f}` s per round per family, 25 rounds plus evaluation and
startup is about **{task_s:.0f} s per task**. {len(tasks)} tasks is roughly
**{len(tasks) * task_s / 3600:.2f} GPU-h** - minutes of wall clock. The patch, if
it is needed, is 5 tasks at the full horizon: about
{5 * (per_round * SL.FULL_ROUNDS + 48) / 3600:.2f} GPU-h.
"""


if __name__ == "__main__":
    raise SystemExit(main())
