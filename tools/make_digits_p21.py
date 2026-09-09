"""
Emit P21 of the digits study: the kd+fisher blend's own screen.

    python tools/make_digits_p21.py

Writes two files beside the study's other task files: the stage and its README.

P13 screened seven penalties and left the blend out on purpose - a mixture is
only worth pricing once each half's own strength is known - and P14's
``reg-hybrid`` then built it from the two winners and swept ``mix`` alone.  So
the blend reached the finals with ``lam`` and ``T`` inherited rather than
searched, and went on to hold the best mean score in both schedules.  It is the
only penalty in the study reported at coefficients nothing measured.  This is
the screen it never had.
"""

from __future__ import annotations

import argparse
import dataclasses as _dc
import json
from pathlib import Path

from federated_outlier_adaptation.training import reg_cells
from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training.study_config import DIGITS_STUDY01 as _STUDY

#: The stage's name, and the stem of both files it writes.
STAGE = "s21_blend_screen"

#: Searched at the rate every grid in this study is searched at - two of ten
#: dropped, not the study's own nine of ten - so this table can be read against
#: the screen whose winners the blend is built from.  The rate comes from
#: ``search_clients_per_round``, never from a flag.
CFG = _dc.replace(_STUDY, clients_per_round=_STUDY.search_clients_per_round)

#: Sampler seeds of this screen: 722001-724335.  Disjoint from every range the
#: study has drawn - the last of them is the combination stage at 730351-738984 -
#: and wide enough for 234 cells, which span 2340 and therefore run through three
#: of the thousand-wide blocks the scheme assumes.  That span is the reason the
#: window is opened at 22000 rather than at the next free thousand: a block
#: chosen for its number rather than for its width is how the combination stage
#: came to draw seeds the regularisation screen had already used.
SEED_OFFSET = 22000

def _selected_fold() -> int:
    """
    The winning g-0 fold, read from the record rather than remembered.

    Every cell here reads the shipped model's Fisher, and ``study_phase.sbatch``
    resolves ``$G0_FOLD`` from the study's own ``g0_selection.json`` at run
    time - so the only thing this number does is appear in a header and a
    README.  That is exactly the kind of number that goes stale without anything
    failing: ``make_digits_p13.py`` carries it as a literal 2, and the record
    shipped beside it says 1.  Reading it keeps the prose and the run in
    agreement, and a missing record stops the emission rather than printing a
    guess.
    """
    record = (Path(__file__).resolve().parents[1] / "study" / "artifacts"
              / CFG.name / "g0_selection.json")
    return int(json.loads(record.read_text())["selected_fold"])


#: The winning g-0 fold, whose Fisher every cell here reads.
G0_FOLD = _selected_fold()


def lines() -> list[str]:
    """One line per (cell, fold), at the screening horizon."""
    return [
        SL.reg_line(
            CFG, cell, fold, SL.SCREEN_ROUNDS,
            CFG.seed_base + SEED_OFFSET + index * 10 + fold,
        )
        for index, cell in enumerate(reg_cells.blend_cells())
        for fold in SL.FOLDS
    ]


#: The whole grid: 13 strengths x 6 temperatures x 3 mixes.
SCREENED_CELLS = len(reg_cells.blend_cells())


def _rows() -> list[tuple]:
    """``(EWC lambda, the blend lam it becomes at each mix)`` for the README."""
    return [
        (lam_ewc, tuple(reg_cells.blend_lam_of_ewc(lam_ewc, mix)
                        for mix in reg_cells.BLEND_MIXES))
        for lam_ewc in reg_cells.BLEND_EWC_LAMS
    ]


def header(tasks: list[str]) -> list[str]:
    lams = reg_cells.BLEND_EWC_LAMS
    temps = reg_cells.BLEND_TEMPERATURES
    mixes = reg_cells.BLEND_MIXES
    return [
        f"# {STAGE}.txt - {CFG.name}, P21: THE BLEND'S OWN SCREEN.",
        "#",
        "# THE ONE PENALTY THAT WAS NEVER SCREENED. P13 left the kd+fisher",
        "# blend out of the regularisation screen deliberately, and P14 then",
        "# built it from the kd and fisher winners and swept mix alone. Its lam",
        "# and its T were INHERITED, not searched - and it went on to hold the",
        "# best mean score in both schedules. This stage searches them.",
        "#",
        "# THE PROTOCOL IS P13'S, UNCHANGED. Ten cohort writers,",
        f"# {CFG.clients_per_round} of {CFG.cohort_size} per round, E=5, batch 64, the cohort fold book,",
        "# g-0 as the only initialisation, plain FedAvg on the server, the",
        "# frozen g-0 as the anchor. Only the penalty's coefficients move.",
        "#",
        "# BOTH SCHEDULES PER TASK. --aggregation fedavg runs the parallel",
        "# family and the cyclic one in one job, as the screen it extends does.",
        "# The selector ranks them separately.",
        "#",
        "# THE GRID, IN ITS PARENTS' UNITS.",
        "#   strength  the EWC row's own thirteen lambdas",
        f"#             {{{', '.join(f'{v:g}' for v in lams)}}}",
        f"#   T         the kd row's own six temperatures",
        f"#             {{{', '.join(f'{v:g}' for v in temps)}}}",
        f"#   mix       {{{', '.join(f'{v:g}' for v in mixes)}}}",
        f"#   = {len(lams)} x {len(temps)} x {len(mixes)} = {SCREENED_CELLS} cells"
        f" x {len(SL.FOLDS)} folds = {len(tasks)} tasks",
        "#",
        "# WHY THE STRENGTH IS WRITTEN AS AN EWC LAMBDA AND CONVERTED. The",
        "# objective is lam * (mix * KD + (1 - mix) * Fisher), so the two",
        "# halves' coefficients are locked in the ratio mix : (1 - mix) and one",
        "# lam cannot carry both parents' rows. It can carry one exactly, and",
        "# the Fisher half is the one that needs it: the emitted blends run",
        "# their Fisher term at lam * (1 - mix) = 0.083, where the concurrent",
        "# selection chose lambda = 8. So each cell names the EWC lambda its",
        "# Fisher half actually receives and the line carries",
        "#   lam = lambda_ewc / (1 - mix)",
        "# the same conversion the kd row makes from alpha.",
        "#",
        "# ALPHA IS NOT A KNOB HERE. AnchoredTrainer has no alpha; the kd row",
        "# is WRITTEN in alpha and converted to lam by (1 - alpha) / alpha. In",
        "# the blend that same coefficient is lam * mix, so alpha has no",
        "# independent existence and is not gridded.",
        "#",
        "# SCREENING HORIZON: 25 ROUNDS, NOT 100. Nothing here is a reported",
        "# number - a 25-round result is a ranking signal. The winners are",
        "# re-run at 100 rounds by a later stage, which is gated separately and",
        "# NOT authorised.",
        "#",
        f"# THE FISHER. Every cell needs the Fisher diagonal of the shipped",
        f"# model. g0_selection.json names FOLD {G0_FOLD}, whose Fisher is at",
        f"#   $FOA_STUDY_DIR/g0_fold{G0_FOLD}/global_results/fisher",
        "# The lines name it as g0_fold$G0_FOLD/... and study_phase.sbatch",
        "# DERIVES G0_FOLD from the study's own g0_selection.json, so NO EXPORT",
        "# IS NEEDED. If the record is ever missing, a task that needs the fold",
        "# is REFUSED (exit 78) rather than run with an empty path component.",
        "#",
        "# SEEDS.",
        f"#   {CFG.seed_base} + {SEED_OFFSET} + cell_index * 10 + fold = "
        f"{CFG.seed_base + SEED_OFFSET + 1}-"
        f"{CFG.seed_base + SEED_OFFSET + (SCREENED_CELLS - 1) * 10 + len(SL.FOLDS)}",
        "# disjoint from every range this study has drawn.",
        "#",
        f"#   sbatch --account=$FOA_ACCOUNT --array=1-{len(tasks)}%20 \\",
        f"#          slurm/study_phase.sbatch <jobs>/{STAGE}.txt",
        "#",
    ]


def readme(tasks: list[str]) -> str:
    lams = reg_cells.BLEND_EWC_LAMS
    temps = reg_cells.BLEND_TEMPERATURES
    mixes = reg_cells.BLEND_MIXES
    per_round = 0.31 + 0.0345 * CFG.clients_per_round * 5
    # both families, and the teacher forward pass the KD half adds
    train_s = 2 * per_round * 1.4 * SL.SCREEN_ROUNDS
    task_s = train_s + 6 + 2 + 40
    rate = f"{CFG.clients_per_round} of {CFG.cohort_size}"
    seeds_lo = CFG.seed_base + SEED_OFFSET + 1
    seeds_hi = (CFG.seed_base + SEED_OFFSET
                + (SCREENED_CELLS - 1) * 10 + len(SL.FOLDS))
    span = (SCREENED_CELLS - 1) * 10
    gpu_h = len(tasks) * task_s / 3600
    gpu_h_measured = len(tasks) * 93.6 / 3600
    conversion = "\n".join(
        f"| {lam_ewc:g} | "
        + " | ".join(f"{reg_cells.blend_lam_of_ewc(lam_ewc, m):.6g}"
                     for m in mixes)
        + " |"
        for lam_ewc in lams
    )
    return f"""# {CFG.name} - P21: the kd+fisher blend's own screen

The blend is the one penalty in this study that was **never screened**. P13 left
it out of the regularisation screen on purpose - a mixture is only worth pricing
once each half's own strength is known - and `study_emit.py reg-hybrid` then
built it from the kd and fisher winners and swept `mix` alone. Its `lam` and its
`T` were **inherited from the KD half**, not searched. It then held the best mean
score in both schedules and the nine highest preservation figures in the table,
which is exactly why leaving its coefficients unsearched is no longer tolerable:
the strongest method in the study is the one whose grid nobody ran.

This stage runs that grid, at the same ranking horizon and under the same
protocol as every other penalty: {rate} per round, E=5, batch 64, the cohort
fold book, g-0 as the only initialisation, plain FedAvg on the server, the frozen
g-0 as the anchor, both schedules per task.

## The grid

    lambda_ewc in {{{', '.join(f'{v:g}' for v in lams)}}}
    T          in {{{', '.join(f'{v:g}' for v in temps)}}}
    mix        in {{{', '.join(f'{v:g}' for v in mixes)}}}
    = {len(lams)} x {len(temps)} x {len(mixes)} = {SCREENED_CELLS} cells x {len(SL.FOLDS)} folds = {len(tasks)} tasks

**The same ranges the parents were screened over, not a reduced set.** The
strength row is `fisher`'s own thirteen lambdas from `REG_GRID_RANGES.md` §2 -
the row that covers the earlier work's `[0.2, 1, 4, 8, 10, 20, 100, 200]` and
resolves 1-10 - and the temperature row is `kd`'s own six from §6, the row that
stops at 8 because preservation was measured saturating there. Neither is
narrowed for this stage. A blend screened over less than its parents were would
answer a smaller question than the one being asked.

## The scale question, and how it is resolved

`AnchoredTrainer` computes

    lam * (mix * KD + (1 - mix) * Fisher)

so `lam` is **not** a free third knob: the two halves' coefficients are locked in
the ratio `mix : (1 - mix)`, and a single `lam` cannot put both parents' rows on
both halves at once. It can put one of them on one half exactly, and the Fisher
half is the one that needs it.

The blends that ran inherit `lam` from the **KD** half - 0.111111 in the parallel
schedule, 0.010101 in the cyclic one - so their Fisher term runs at
`lam * (1 - mix)`: 0.083 down to 0.028 where the concurrent selection chose
`lambda = 8`, and 0.0076 down to 0.0025 where the sequential one chose 0.1. That
is two decades below the row EWC was screened over, and it is the same fact
`REG_GRID_RANGES.md` already records as *"`mix = 0` does not reproduce the Fisher
winner"*. The blend has never been run with a Fisher term of the strength its
Fisher parent was selected at.

So the strength axis is written as the **EWC lambda the Fisher half receives**
and converted on the way to the line:

    lam = lambda_ewc / (1 - mix)

which is the same move the `kd` row makes in writing itself in `alpha` and
emitting `(1 - alpha) / alpha`: a cell names the quantity whose range was argued
for, not the number the trainer happens to take. The KD half then rides at
`lambda_ewc * mix / (1 - mix)`, which at `mix = 0.25` runs 0.033 to 333 and at
`mix = 0.75` runs 0.3 to 3000 - so the bottom of the row places both halves
inside their parents' live regions and the top is EWC-dominant by construction.

| `lambda_ewc` | `lam` at mix 0.25 | at mix 0.5 | at mix 0.75 |
|---|---|---|---|
{conversion}

**The two inherited points are below this row.** 0.111111 and 0.010101 are KD-half
lambdas; expressed as the Fisher-half coefficient they are 0.083 and below, under
the EWC row's floor of 0.1. That is the finding this stage exists to act on
rather than a gap to be papered over: the row starts where EWC's row starts,
because that is the range EWC was screened over. The three cells that ran keep
their full-horizon results and still compete in `reg-top3`; nothing here
supersedes them.

## Alpha is not a knob in the blend

`AnchoredTrainer` has no `alpha` parameter. `alpha` is the literature's
parameterisation of the `kd` space, converted to the trainer's `lam` by
`(1 - alpha) / alpha` because the two objectives have the same minimiser. Inside
the blend the KD term's coefficient is `lam * mix`, so `alpha` has no independent
existence and gridding it would be gridding `lam` twice. It is omitted, and the
grid stays three-dimensional.

## A ranking horizon

25 rounds, not 100. Nothing here is a reported number; a 25-round result is a
ranking signal. The winners are re-run at 100 rounds by a later stage, which is
**gated separately and is not authorised**.

## The Fisher - no export needed

Every cell reads the Fisher diagonal of the shipped model. The study's own
`g0_selection.json` names **fold {G0_FOLD}**, and this stage reads that number
rather than restating it. The lines name the directory as
`$FOA_STUDY_DIR/g0_fold$G0_FOLD/global_results/fisher`, which
`slurm/study_phase.sbatch` derives from the study's own `g0_selection.json`. So
no export is required; a task that needs the fold when the record is missing is
refused with exit 78 rather than run with an empty path component.

## Seeds

    {CFG.seed_base} + {SEED_OFFSET} + cell_index * 10 + fold

= {seeds_lo}-{seeds_hi}, disjoint from every range this study has drawn.
{SCREENED_CELLS} cells span {span}, so this stage runs through three of the
thousand-wide blocks the scheme assumes - which is why the window was opened on
width rather than on the next free thousand.

## Submit

```bash
sbatch --account=$FOA_ACCOUNT --partition=<gpu partition> --gres=gpu:1 \\
       --array=1-{len(tasks)}%20 \\
       --export=ALL,FOA_PROJECT_DIR=$FOA_PROJECT_DIR,FOA_STUDY_DIR=$FOA_STUDY_DIR \\
       slurm/study_phase.sbatch study/artifacts/{CFG.name}/jobs/{STAGE}.txt
```

## Cost

At `0.31 + 0.0345 x {CFG.clients_per_round} x 5 = {per_round:.2f}` s per round per family, with both
families per task and roughly 40% added for the teacher forward pass the KD half
needs on every batch:

| | seconds |
|---|---|
| training, {SL.SCREEN_ROUNDS} rounds, two families, +40% | {train_s:.0f} |
| per-round evaluations | ~6 |
| final three-category evaluation | ~2 |
| interpreter, torch, cache, two books | ~40 |
| **wall per task** | **~{task_s:.0f} (~{task_s / 60:.1f} min)** |

**{len(tasks)} tasks is roughly {gpu_h:.0f} GPU-h** by that model, about
{gpu_h / 20 * 60:.0f} minutes of wall clock at `%20`. The model is the one P13's
README used and it ran high: that screen's 700 tasks cost 18.2 GPU-h measured,
94 s each against the {task_s:.0f} s predicted, which puts this stage nearer
**{gpu_h_measured:.0f} GPU-h**. Both are quoted because a stage this size should
be authorised against the pessimistic one.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs-dir", default=None,
        help="Where to write; defaults to the study's shipped jobs directory.",
    )
    args = parser.parse_args()
    jobs = Path(args.jobs_dir) if args.jobs_dir else (
        Path(__file__).resolve().parents[1]
        / "study" / "artifacts" / CFG.name / "jobs"
    )
    jobs.mkdir(parents=True, exist_ok=True)
    tasks = lines()
    (jobs / f"{STAGE}.txt").write_text("\n".join(header(tasks) + tasks) + "\n")
    (jobs / f"{STAGE}_README.md").write_text(readme(tasks))
    print(f"wrote {jobs}/{STAGE}.txt: {len(tasks)} tasks")
    print(f"wrote {jobs}/{STAGE}_README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
