"""
Emit P13 of the digits study: the regularisation screen.

    python tools/make_digits_p13.py --jobs-dir $FOA_PROJECT_DIR/jobs/v4

Writes two text files: a login-node job.

P11 asked what the *server* rule is worth. P13 asks the other half - what the
**client-side penalty** is worth - under the same protocol with plain FedAvg on
the server, changing only the term added to the client's loss.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from federated_outlier_adaptation.training import reg_cells
from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training.study_config import DIGITS_STUDY01 as _STUDY

#: Searched at one participation rate, like the aggregation grid, and at the
#: same one - two of ten dropped. The winners are carried to the other rates.
#: Searching both grids at the same rate is what lets their tables be read
#: against each other.
#:
#: The rate comes from ``search_clients_per_round``, not from a flag.  This grid
#: is the reason the constant exists: it was searched at 9 of 10 because the
#: flag the aggregation grid had been given was not given to it, and the
#: mismatch reached the finals and the combinations before anyone read a rate
#: off a line.
import dataclasses as _dc

CFG = _dc.replace(_STUDY, clients_per_round=_STUDY.search_clients_per_round)

#: Sampler seeds of the screen. Disjoint from every earlier range in this study:
#: P09 30001-30505, P11 702001-703685, P11-ext 703691-703705, P12 704001-704175,
#: and the superseded P13 screen at 706001-707255.
#:
#: A seed here is a function of a cell's INDEX in the table, so re-ranging the
#: grid moves every seed at or after the first changed row. A fresh window is
#: what stops the new screen from reusing the old one's seeds under different
#: cells - the same reasoning that put the fold split on a hash of its inputs
#: rather than on a position.
SEED_OFFSET = 10000

#: The winning g-0 fold, whose Fisher the fisher-weighted cells read. The runner
#: derives it from the study's own g0_selection.json; recorded here for the
#: header and the README.
G0_FOLD = 2


def lines() -> list[str]:
    """One line per (cell, fold), at the screening horizon."""
    return [
        SL.reg_line(
            CFG, cell, fold, SL.SCREEN_ROUNDS,
            CFG.seed_base + SEED_OFFSET + index * 10 + fold,
        )
        for index, cell in enumerate(reg_cells.screen_cells())
        for fold in SL.FOLDS
    ]


#: The whole table. There is nothing to hold apart any more: the first screen's
#: edge probes are inside the rows now, so this file IS the grid.
SCREENED_CELLS = len(reg_cells.screen_cells())


def header(tasks: list[str]) -> list[str]:
    screened = reg_cells.screen_cells()
    grouped: dict = {}
    for cell in screened:
        grouped.setdefault(cell["method"], []).append(cell)
    fisher_cells = sum(1 for c in screened if c["needs_fisher"])
    return [
        f"# d01_p13.txt - {CFG.name}, P13: WHAT THE REGULARISER IS WORTH.",
        "#",
        "# P11 asked what the SERVER rule is worth. This asks the other half:",
        "# what the CLIENT-SIDE PENALTY is worth. Every line runs P11's protocol -",
        f"# ten cohort writers, {CFG.clients_per_round} of {CFG.cohort_size} per round, E=5, batch 64, the",
        "# cohort fold book, g-0 as the only initialisation, plain FedAvg on the",
        "# server - and changes only the term added to the client's loss.",
        "#",
        "# THE ANCHOR IS THE FROZEN g-0 IN EVERY CELL. That is the premise: the",
        "# penalty measures distance from the model the study is trying not to",
        "# forget.",
        "#",
        "# BOTH SCHEDULES PER TASK. --aggregation fedavg runs the parallel family",
        "# and the cyclic one in one job, as this stage has always run. A penalty",
        "# that helps a server average and one that helps a sequential walk are",
        "# different findings, so the selector ranks them separately.",
        "#",
        f"# {len(screened)} cells x {len(SL.FOLDS)} folds = {len(tasks)} tasks:",
    ] + [
        f"#   {method:<14} {len(group):>3} cells   {len(group) * len(SL.FOLDS):>4} tasks"
        for method, group in grouped.items()
    ] + [
        "#",
        "# THE HYBRID IS NOT IN THE SCREEN. A blend of two penalties is only worth",
        "# pricing once each one's own strength is known; screening it here would",
        "# mean guessing three coefficients before any of them was measured. Its",
        "# three cells are emitted afterwards from the kd and fisher winners.",
        "#",
        "# SCREENING HORIZON: 25 ROUNDS, NOT 100. Nothing here is a reported",
        "# number - a 25-round result is a ranking signal. The winners are re-run",
        "# at 100 rounds by P14, which is gated separately and NOT authorised.",
        "#",
        f"# THE FISHER. {fisher_cells} cells ({fisher_cells * len(SL.FOLDS)} tasks) need the Fisher diagonal of the",
        f"# shipped model. g-0 v2 won on FOLD {G0_FOLD}, and its Fisher is on disk at",
        f"#   $FOA_STUDY_DIR/g0_fold{G0_FOLD}/global_results/fisher",
        "# (verified: fisher.pt and global_params.pt, 6.7 MB each, for all five",
        "# folds). The lines name it as g0_fold$G0_FOLD/... and study_phase.sbatch",
        "# DERIVES G0_FOLD from the study's own g0_selection.json - so NO EXPORT",
        "# IS NEEDED. A number already written down should not also have to be",
        "# remembered. If the record is ever missing, a task that needs the fold",
        "# is REFUSED (exit 78) rather than run with an empty path component.",
        "# Export G0_FOLD only to override the record deliberately.",
        "#",
        "# EVERY LINE reports the standard three categories: the pooled clients",
        "# test on the cohort's fold-k rows, the per-client column, and the old",
        "# data's FIVE fold test partitions scored separately with their mean and",
        "# spread.",
        "#",
        "# SEEDS.",
        f"#   {CFG.seed_base} + {SEED_OFFSET} + cell_index * 10 + fold",
        "# disjoint from P09, P11, the P11 extension, P12 and the superseded P13.",
        "#",
        "#   J=$FOA_PROJECT_DIR/jobs/v4",
        f"#   sbatch --account=$FOA_ACCOUNT --array=1-{len(tasks)}%40 \\",
        "#          $J/study_phase.sbatch $J/d01_p13.txt",
        "#",
    ]


def readme(tasks: list[str]) -> str:
    screened = reg_cells.screen_cells()
    grouped: dict = {}
    for cell in screened:
        grouped.setdefault(cell["method"], []).append(cell)
    fisher_cells = sum(1 for c in screened if c["needs_fisher"])
    per_round = 0.31 + 0.0345 * CFG.clients_per_round * 5
    # both families, and the teacher forward pass the output-space penalties add
    train_s = 2 * per_round * 1.4 * SL.SCREEN_ROUNDS
    task_s = train_s + 6 + 2 + 40
    rows = "\n".join(
        f"| {method} | {len(group)} | {len(group) * len(SL.FOLDS)} |"
        for method, group in grouped.items()
    )
    return f"""# {CFG.name} - P13: the regularisation screen

P11 asked what the **server** rule is worth. P13 asks the other half - what the
**client-side penalty** is worth - under the same protocol with plain FedAvg on
the server, changing only the term added to the client's loss.

The anchor is the **frozen g-0** in every cell. That is the premise: the penalty
measures distance from the model the study is trying not to forget.

| method | cells | tasks |
|---|---|---|
{rows}
| **total** | **{len(screened)}** | **{len(tasks)}** |

Both schedules run per task (`--aggregation fedavg`), as this stage always has.
A penalty that helps a server average and one that helps a sequential walk are
different findings, so the selector ranks them separately.

## The Fisher - no export needed

{fisher_cells} cells ({fisher_cells * len(SL.FOLDS)} tasks) need the Fisher
diagonal of the shipped model. **g-0 v2 won on fold {G0_FOLD}**, and I checked
the artefacts are on disk before trusting the path: `fisher.pt` and
`global_params.pt`, 6.7 MB each, present for **all five** folds at
`g0_fold<k>/global_results/fisher`.

The lines name it as `$FOA_STUDY_DIR/g0_fold$G0_FOLD/global_results/fisher`, and
**`study_phase.sbatch` now derives `G0_FOLD` from the study's own
`g0_selection.json`**. So no export is required.

That is the safer of the two options you offered: a number that is already
written down should not also have to be remembered, and a forgotten export would
expand to an empty path component and fail {fisher_cells * len(SL.FOLDS)} array
elements at startup. If the selection record is ever missing, a task that needs
the fold is **refused with exit 78** rather than run with a broken path. Export
`G0_FOLD` only to override the record deliberately.

## A ranking horizon

25 rounds, not 100. Nothing here is a reported number; a 25-round result is a
ranking signal. Winners are re-run at 100 rounds by **P14, which is gated
separately and is not authorised**.

## Submit

```bash
J=$FOA_PROJECT_DIR/jobs/v4
sbatch --account=$FOA_ACCOUNT --array=1-{len(tasks)}%40 \\
       $J/study_phase.sbatch $J/d01_p13.txt
```

## After the screen

Two CPU commands, minutes each, both stamped so the files they write cannot be
mistaken for authorised work.

```bash
P=$FOA_PROJECT_DIR
S=$P/results_v4/studies/{CFG.name}
cd $P/repo_foa && PYTHONPATH=$PWD/src $P/envs/fal/bin/python tools/study_emit.py \\
    reg-full --study {CFG.name} --root $S \\
    --out $P/jobs/v4/d01_p14.txt --expect {SL.counts(CFG)["reg_full"]}
```

**Per family, not per method.** Each penalty's best cell is chosen within each
schedule, giving {len(SL.REG_METHODS)} methods x {len(SL.FAMILIES)} families x
{len(SL.FOLDS)} folds = **{SL.counts(CFG)["reg_full"]} lines**. The full-horizon
parents carry the family they were selected for - the same cell often wins in
both, and without the tag two tasks would name one folder and race into it.

Then the blend, once both halves are known:

```bash
cd $P/repo_foa && PYTHONPATH=$PWD/src $P/envs/fal/bin/python tools/study_emit.py \\
    reg-hybrid --study {CFG.name} --root $S \\
    --out $P/jobs/v4/d01_p14_hybrid.txt --expect {len(reg_cells.HYBRID_MIXES) * len(SL.FOLDS)}
```

The hybrid is deliberately **not** in the screen: a blend is only worth pricing
once each penalty's own strength is known. Screening it would have meant
guessing the KD half's `lam` and `T` and the Fisher half's `lam` before any was
measured - a three-dimensional grid for a question that only becomes meaningful
after the two one-dimensional ones are settled. It costs
{len(reg_cells.HYBRID_MIXES)} cells x {len(SL.FOLDS)} folds =
**{len(reg_cells.HYBRID_MIXES) * len(SL.FOLDS)} tasks** instead of hundreds.
`mix = 1` would reproduce the KD winner exactly, which is what makes three blend
points readable as a line between the two methods.

Every emitter takes `--expect` and exits non-zero on any other count, and every
file it writes is stamped `NOT YET AUTHORISED`.

Grid-edge winners go to `tables/BOUNDARY_HITS.txt` and to stdout and never halt
anything.

## Cost

At `0.31 + 0.0345 x {CFG.clients_per_round} x 5 = {per_round:.2f}` s per round per family, with
both families per task and roughly 40% added for the teacher forward pass the
output-space penalties need on every batch:

| | seconds |
|---|---|
| training, {SL.SCREEN_ROUNDS} rounds, two families, +40% | {train_s:.0f} |
| per-round evaluations | ~6 |
| final three-category evaluation | ~2 |
| interpreter, torch, cache, two books | ~40 |
| **wall per task** | **~{task_s:.0f} (~{task_s / 60:.1f} min)** |

**{len(tasks)} tasks is roughly {len(tasks) * task_s / 3600:.0f} GPU-h**, about
{len(tasks) * task_s / 3600 / 40 * 60:.0f} minutes of wall clock at `%40`.

The Fisher-weighted cells are cheaper than that estimate - a Fisher penalty is a
parameter-space term with no teacher pass - and the kd/ntd/logit/feature cells
are the ones carrying the 40%. Taking the whole file at the higher rate is the
honest way round.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-dir", required=True)
    parser.add_argument(
        "--clients-per-round", type=int, default=None, metavar="M",
        help="Participation the grid is searched at; defaults to the study's.",
    )
    args = parser.parse_args()
    if getattr(args, "clients_per_round", None) is not None:
        import dataclasses
        global CFG
        CFG = dataclasses.replace(CFG, clients_per_round=args.clients_per_round)
    jobs = Path(args.jobs_dir)
    jobs.mkdir(parents=True, exist_ok=True)
    screened = lines()
    (jobs / "d01_p13.txt").write_text(
        "\n".join(header(screened) + screened) + "\n"
    )
    (jobs / "d01_p13_README.md").write_text(readme(screened))
    print(f"wrote {jobs}/d01_p13.txt: {len(screened)} tasks")
    print(f"wrote {jobs}/d01_p13_README.md")
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
