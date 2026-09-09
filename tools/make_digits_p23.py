"""
Emit P23 of the digits study: an EXTENSION, not a stage of the programme.

    python tools/make_digits_p23.py

Writes two files beside the study's other task files: the stage and its README.

WHAT THIS IS NOT.  It is not a missing piece of the study.  Every stage the
paper reports has run, and the combination cross has already answered the
question it was built to answer.  This is an extension bolted on afterwards, and
it is named as one everywhere it appears - in the task file's header, in the
chain manifest, in the selection record and in ``REPRODUCE.md``, which keeps it
out of the core stage table on purpose.

WHAT IT ASKS.  The combination stage crossed two shortlists at ONE setting each.
A pair there is a server rule at the coefficients it won on alone beside a
penalty at the coefficients it won on alone, and nothing in that cross ever moved
the two together.  So "the two halves do not measurably compose" is a statement
about one point of a joint grid: the point where each half is best in the other's
absence.  This screens the joint grid of the pair that leads each schedule -
the best rule and the best penalty by the TEST score of
``tables/paper/{agg-winners,reg-winners}.csv`` - at the same 25-round ranking
horizon every screen in this study uses.
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
STAGE = "s23_combo_screen"

#: Searched at the rate every grid in this study is searched at - eight of ten,
#: not the study's own nine - so this table can be read against the two screens
#: whose winners it tunes.  The rate comes from ``search_clients_per_round``,
#: never from a flag.
CFG = _dc.replace(_STUDY, clients_per_round=_STUDY.search_clients_per_round)

#: Sampler seeds of this extension: 760001-762155.  Opened at 60000 rather than
#: at the next free thousand for the reason P21's was opened at 22000: 216 cells
#: span 2160 and a stage no longer fits in the thousand-wide block the scheme
#: assumes.  The highest seed any shipped file draws is 750105, so this window
#: also puts a visible gap between the programme's blocks and the extension's -
#: a stage that is not part of the study should not be interleaved with the ones
#: that are.
SEED_OFFSET = 60000


def _selected_fold() -> int:
    """
    The winning g-0 fold, read from the record rather than remembered.

    Every cell here reads the shipped model's Fisher, and ``study_phase.sbatch``
    resolves ``$G0_FOLD`` from the study's own ``g0_selection.json`` at run time -
    so the only thing this number does is appear in a header and a README.  That
    is exactly the kind of number that goes stale without anything failing, which
    is why P21 started reading it and this reads it the same way.
    """
    record = (Path(__file__).resolve().parents[1] / "study" / "artifacts"
              / CFG.name / "g0_selection.json")
    return int(json.loads(record.read_text())["selected_fold"])


#: The winning g-0 fold, whose Fisher every cell here reads.
G0_FOLD = _selected_fold()

#: The pair this extension tunes, per schedule, and where each half came from.
#: Read off ``tables/paper/agg-winners.csv`` and ``tables/paper/reg-winners.csv``
#: by TEST score, which is the owner's documented departure from ranking on
#: validation and is what those two views are ordered by.
THE_PAIR = {
    "concurrent": ("eta_0p95", "hybrid_seq_mix0p75"),
    "sequential": ("seq_delta_capped", "hybrid_seq_mix0p75"),
}

#: The untuned pair already on disk in ``s20_combos4.txt``, whose every non-grid
#: flag this stage copies.  Named so that a reader can diff one of these lines
#: against one of those and see that only the horizon and the coefficients moved.
THE_SHIPPED_LINES = {
    "concurrent": "eta_0p95_hybrid_seq_mix0p5",
    "sequential": "seq_delta_capped_hybrid_mix0p5",
}


def lines() -> list[str]:
    """One line per (cell, fold), at the screening horizon."""
    return [
        SL.combo_tune_line(
            CFG, cell, fold, SL.SCREEN_ROUNDS,
            CFG.seed_base + SEED_OFFSET + index * 10 + fold,
        )
        for index, cell in enumerate(reg_cells.combo_tune_cells())
        for fold in SL.FOLDS
    ]


#: The whole grid: 3 x 3 x 2 x 3 penalty settings, times the rule knob each
#: schedule has - three server steps on the parallel side, none on the cyclic.
SCREENED_CELLS = len(reg_cells.combo_tune_cells())

#: Dial settings the deduplication removed because they hand the trainer the
#: same (lam, mix, T) under the same rule.  Zero here; computed, not asserted.
DUPLICATES = reg_cells.combo_tune_duplicates()


def _by_family() -> dict:
    """``{schedule: cells}``, for the header and the README."""
    return reg_cells.combo_tune_by_family()


def _mapping_rows() -> str:
    """A README table of dials -> coefficients, one row per (c_ewc, c_kd, m)."""
    rows = []
    for c_ewc in reg_cells.CTUNE_EWC_COEFFS:
        for c_kd in reg_cells.CTUNE_KD_COEFFS:
            for mix in reg_cells.CTUNE_MIXES:
                lam, lam_mix = reg_cells.combo_tune_lam_mix(c_ewc, c_kd, mix)
                rows.append(
                    f"| {c_ewc:g} | {c_kd:g} | {mix:g} | {lam:.6g} | "
                    f"{lam_mix:.6g} |"
                )
    return "\n".join(rows)


def header(tasks: list[str]) -> list[str]:
    families = _by_family()
    ewc = reg_cells.CTUNE_EWC_COEFFS
    kd = reg_cells.CTUNE_KD_COEFFS
    temps = reg_cells.CTUNE_TEMPERATURES
    mixes = reg_cells.CTUNE_MIXES
    etas = reg_cells.CTUNE_SERVER_ETAS
    dials = len(ewc) * len(kd) * len(temps) * len(mixes)
    return [
        f"# {STAGE}.txt - {CFG.name}, P23: AN EXTENSION, NOT A STAGE OF THE",
        "# PROGRAMME. The study is complete without it and no table, shortlist",
        "# or crossing the paper reports reads anything it produces.",
        "#",
        "# THE QUESTION THE CROSS COULD NOT ASK. s20 crossed two shortlists at",
        "# ONE setting each: a rule at the coefficients it won on alone, beside",
        "# a penalty at the coefficients IT won on alone. Neither half was ever",
        "# chosen in the other's presence, so the study's finding - that the two",
        "# do not measurably compose - is measured at one point of a joint grid.",
        "# This screens that grid, for the pair that leads each schedule.",
        "#",
        "# THE PAIR, by TEST score of tables/paper/{agg-winners,reg-winners}.csv:",
    ] + [
        f"#   {family:<11} rule {rule:<17} penalty {penalty}"
        for family, (rule, penalty) in THE_PAIR.items()
    ] + [
        "#",
        "# EVERY NON-GRID FLAG IS COPIED FROM THE SHIPPED s20 LINES that pair",
        "# these two halves -",
    ] + [
        f"#   {family:<11} {cell}"
        for family, cell in THE_SHIPPED_LINES.items()
    ] + [
        "# - so a line here differs from one there in the horizon and in the",
        "# coefficients, and in nothing else. The fisher_path convention is s19's.",
        "#",
        "# ONE LINE IS ONE SCHEDULE. Each cell names a single aggregation rule,",
        "# which lives in exactly one family, so a task produces one result - the",
        "# combination stage's shape, not the screens'. A --aggregation fedavg",
        "# line would run the cyclic family under a rule chosen for the parallel",
        "# one.",
        "#",
        "# THE GRID, IN THE OWNER'S DIALS.",
        f"#   c_ewc     {{{', '.join(f'{v:g}' for v in ewc)}}}",
        f"#   c_kd      {{{', '.join(f'{v:g}' for v in kd)}}}",
        f"#   T         {{{', '.join(f'{v:g}' for v in temps)}}}",
        f"#   m         {{{', '.join(f'{v:g}' for v in mixes)}}}",
        f"#   eta_s     {{{', '.join(f'{v:g}' for v in etas)}}} - PARALLEL ONLY;"
        " seq_delta_capped",
        "#             has no coefficient, so the cyclic half is a third the size",
        f"#   = {len(ewc)} x {len(kd)} x {len(temps)} x {len(mixes)} = {dials}"
        f" penalty settings"
        f" -> {len(families['concurrent'])} parallel + "
        f"{len(families['sequential'])} cyclic = {SCREENED_CELLS} cells",
        f"#     x {len(SL.FOLDS)} folds = {len(tasks)} tasks",
        "#",
        "# HOW THE DIALS REACH THE TRAINER. AnchoredTrainer computes",
        "#   lam * (mix * KD + (1 - mix) * Fisher)",
        "# and the intended penalty is",
        "#   m * c_kd * KD + (1 - m) * c_ewc * Fisher",
        "# so, term by term,",
        "#   lam = m * c_kd + (1 - m) * c_ewc      mix = m * c_kd / lam",
        "# The trainer's mix is therefore NOT the owner's m except where the two",
        "# coefficients are equal: it is the KD half's share of the total weight,",
        "# and it moves with c_ewc and c_kd as well as with m. A grid that wrote",
        "# m straight into mix would sweep a different object than the one it",
        "# named, and the emitted line would carry a legal coefficient and run.",
        "#",
        f"# DEDUPLICATION: {DUPLICATES} cell(s) removed. Two dial settings collide",
        "# when they hand the trainer the same (lam, mix, T) under the same rule -",
        "# two folders, two seeds, one experiment, and nothing in the records to",
        "# say so. The check is on the coefficients, not on the dials.",
        "#",
        "# SCREENING HORIZON: 25 ROUNDS, NOT 100. Nothing here is a reported",
        "# number - a 25-round result is a ranking signal. The winners are re-run",
        "# at 100 rounds by s24_combo_full.txt.",
        "#",
        "# THE FISHER. Every cell needs the Fisher diagonal of the shipped model.",
        f"# g0_selection.json names FOLD {G0_FOLD}, whose Fisher is at",
        f"#   $FOA_STUDY_DIR/g0_fold{G0_FOLD}/global_results/fisher",
        "# The lines name it as g0_fold$G0_FOLD/... and study_phase.sbatch DERIVES",
        "# G0_FOLD from the study's own g0_selection.json, so NO EXPORT IS NEEDED.",
        "# If the record is ever missing, a task that needs the fold is REFUSED",
        "# (exit 78) rather than run with an empty path component.",
        "#",
        "# SEEDS.",
        f"#   {CFG.seed_base} + {SEED_OFFSET} + cell_index * 10 + fold = "
        f"{CFG.seed_base + SEED_OFFSET + 1}-"
        f"{CFG.seed_base + SEED_OFFSET + (SCREENED_CELLS - 1) * 10 + len(SL.FOLDS)}",
        "# disjoint from every range this study has drawn; the highest seed any",
        "# shipped file draws is 750105.",
        "#",
        f"#   sbatch --account=$FOA_ACCOUNT --array=1-{len(tasks)}%20 \\",
        f"#          slurm/study_phase.sbatch <jobs>/{STAGE}.txt",
        "#",
    ]


def readme(tasks: list[str]) -> str:
    families = _by_family()
    ewc = reg_cells.CTUNE_EWC_COEFFS
    kd = reg_cells.CTUNE_KD_COEFFS
    temps = reg_cells.CTUNE_TEMPERATURES
    mixes = reg_cells.CTUNE_MIXES
    etas = reg_cells.CTUNE_SERVER_ETAS
    dials = len(ewc) * len(kd) * len(temps) * len(mixes)
    per_round = 0.31 + 0.0345 * CFG.clients_per_round * 5
    # ONE family per task, and the teacher forward pass the KD half adds.
    train_s = per_round * 1.4 * SL.SCREEN_ROUNDS
    task_s = train_s + 6 + 2 + 40
    seeds_lo = CFG.seed_base + SEED_OFFSET + 1
    seeds_hi = (CFG.seed_base + SEED_OFFSET
                + (SCREENED_CELLS - 1) * 10 + len(SL.FOLDS))
    gpu_h = len(tasks) * task_s / 3600
    gpu_h_measured = len(tasks) * 47 / 3600
    return f"""# {CFG.name} - P23: joint tuning of the best pair (**an extension**)

**This is not a stage of the study.** Every stage the paper reports has run, and
`COMBINATIONS.md` already records what the combination cross found. This is an
extension added afterwards, it is named as one in the task file's header, in its
chain manifest and in its selection record, and `REPRODUCE.md` keeps it out of
the core stage table deliberately. Nothing in the core programme reads anything
it produces: the shortlists, the cross and the crowning are unchanged and are not
re-emitted.

## The question the cross could not ask

`s20_combos4.txt` crossed the top three server rules with the top three penalties
per schedule. A pair there is **a rule at the coefficients it won on alone beside
a penalty at the coefficients it won on alone** - the two shortlists were
selected independently, and nothing in the cross ever moved the two together. So
the study's finding, that the two halves do not measurably compose, is measured
at exactly one point of a joint grid: the point where each half is best *in the
other's absence*. That is the honest reading of what was run, and it is also the
reading with the obvious gap in it.

This extension screens the joint grid, for the pair that leads each schedule.

## The pair

Read by **TEST** score off the two shipped views - `tables/paper/agg-winners.csv`
and `tables/paper/reg-winners.csv` - which is the owner's documented departure
from ranking on validation and is the order those views carry.

| schedule | best rule alone | best penalty alone |
|---|---|---|
| parallel (`concurrent`) | `eta_0p95` - `con_delta_eta`, score 0.0775 | `hybrid_seq_mix0p75` - the KD+EWC blend, score 0.1050 |
| cyclic (`sequential`) | `seq_delta_capped` - score 0.0702 | `hybrid_seq_mix0p75` - score 0.1069 |

The penalty is the same object in both schedules and the rule is not, which is
why the grid is not the same size on both sides: `eta_0p95` is one setting of a
coefficient row and `seq_delta_capped` is a rule with no coefficient at all.

**Every non-grid flag is copied from the shipped `s20` lines that pair these two
halves** - `eta_0p95_hybrid_seq_mix0p5` and `seq_delta_capped_hybrid_mix0p5` - so
a line here differs from one there in the horizon and in the four (or five)
coefficients, and in nothing else: same cohort, same books, same rate, same
`--outer-workers 1 --inner-workers 1`, same `--extended-aggregations`, and the
`fisher_path` convention `s19_hybrid_seq.txt` established.

## The grid

    c_ewc  in {{{', '.join(f'{v:g}' for v in ewc)}}}      EWC coefficient
    c_kd   in {{{', '.join(f'{v:g}' for v in kd)}}}      KD coefficient
    T      in {{{', '.join(f'{v:g}' for v in temps)}}}         KD temperature
    m      in {{{', '.join(f'{v:g}' for v in mixes)}}}      blend weight
    eta_s  in {{{', '.join(f'{v:g}' for v in etas)}}}      PARALLEL ONLY

    {len(ewc)} x {len(kd)} x {len(temps)} x {len(mixes)} = {dials} penalty settings
      x {len(etas)} server steps = {len(families['concurrent'])} parallel cells
      x 1 (no rule knob)        = {len(families['sequential'])} cyclic cells
      = {SCREENED_CELLS} cells x {len(SL.FOLDS)} folds = {len(tasks)} tasks at 25 rounds

### Where the values come from

**`c_ewc`.** EWC's own selections in this study sat at `lambda = 8` on the
parallel schedule and `lambda = 0.1` on the cyclic one - two decades apart, which
already says the coefficient is not the same object in the two loops. What
settles the range is the blend's own screen: `tables/p21_blend_winners.json`
records `lambda_ewc = 0.1` chosen in **both** schedules, and 0.1 is the *bottom*
of the thirteen-value row `REG_GRID_RANGES.md` section 2 was screened over. A
winner on the floor of a row is a statement that the live region is at or below
it, so this row brackets the floor - 0.05 under it, 0.1 at it, 0.3 above - rather
than reaching back up to 8. Nothing here re-searches the standalone EWC row; that
selection stands.

**`c_kd`.** The `kd` row is written in the literature's `alpha` and emits
`lam = (1 - alpha) / alpha`, so its two winners - `kd_T0p25_a0p9` on the parallel
schedule and `kd_T2_a0p99` on the cyclic one - are `lam = 0.111` and
`lam = 0.0101`. This row brackets the first (0.05 below, 0.11 at it, 0.2 above)
and steps down toward the second. As with `c_ewc`, the bracket is placed on a
measured winner rather than on a decade chosen by eye.

**`T`.** The two temperatures the `kd` row's own winners sat at: 0.25 and 2. Not
the full six-point row of section 6. This grid pays for four other axes and the
two values here are measured points rather than a bracket around a guess -
and the blend's own screen chose 0.5 and 0.25, both inside the interval they
span.

**`m`.** The three the emitted blends swept, so this axis can be read directly
against `s18`, `s19` and the `hybrid_*` rows of `reg-winners.csv`.

**`eta_s`.** `eta_0p95` is a setting of a coefficient row, so the rule half of
the parallel pair *has* a knob and this grid moves it: 0.95 is the winner, 0.9 is
one step below, and 1.0 is the plain full step - so the row also says what the
rule is worth at all when the penalty beside it moves. `seq_delta_capped` has no
coefficient, which is why the cyclic half is a third the size.

## How the dials reach the trainer

`AnchoredTrainer` computes

    lam * ( mix * KD + (1 - mix) * Fisher )

and the penalty this grid names is

    m * c_kd * KD + (1 - m) * c_ewc * Fisher

Matching term by term gives `lam * mix = m * c_kd` and
`lam * (1 - mix) = (1 - m) * c_ewc`, hence

    lam = m * c_kd + (1 - m) * c_ewc
    mix = m * c_kd / lam

which is :func:`reg_cells.combo_tune_lam_mix`, sitting beside
`blend_lam_of_ewc` and doing the other half of that function's job: where the
blend's own screen wrote the strength in **one** parent's units and let the other
half ride, this names **both** coefficients and solves for the pair the trainer
takes.

**The trainer's `mix` is not the owner's `m`.** It is the KD half's share of the
total penalty weight, and it moves with `c_ewc` and `c_kd` as well as with `m` -
the two coincide only where the two coefficients are equal, which on this grid is
`c_ewc = c_kd = 0.05` alone. A grid that wrote `m` straight into `mix` would be
sweeping a different object than the one it named, and nothing in the emitted
line would say so: the line would carry a legal coefficient and run.

| `c_ewc` | `c_kd` | `m` | `lam` | trainer `mix` |
|---|---|---|---|---|
{_mapping_rows()}

(`T` and `eta_s` pass through untouched; the table is the same for both
temperatures and, on the parallel side, for all three server steps.)

## Deduplication

**{DUPLICATES} cell(s) removed.** Two dial settings collide when they hand the
trainer the same `(lam, mix, T)` under the same rule - two folders, two seeds and
one experiment, with nothing in the run records to say so. The check is on the
coefficients rather than on the dials, because the dials are what differ. It
finds nothing on this grid: a collision needs `m * c_kd` and `(1 - m) * c_ewc` to
repeat *together*, and the nine `(m, c_kd)` products here are all distinct. The
check is emitted anyway, so that a widened row cannot quietly pay twice.

## A ranking horizon

25 rounds, not 100. Nothing here is a reported number; a 25-round result is a
ranking signal. The winners are re-run at 100 rounds by `s24_combo_full.txt`,
which `study_emit.py combo-tune-full` emits under the same rule every other
selection in this study uses.

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

= {seeds_lo}-{seeds_hi}, disjoint from every range this study has drawn. {SCREENED_CELLS}
cells span {(SCREENED_CELLS - 1) * 10}, more than the thousand-wide block the scheme assumes, so
the window is opened on width rather than on the next free thousand - P21's
lesson. The highest seed any shipped file draws is 750105, so the extension's
blocks also sit visibly clear of the programme's rather than interleaved with
them.

## Submit

```bash
sbatch --account=$FOA_ACCOUNT --partition=<gpu partition> --gres=gpu:1 \\
       --array=1-{len(tasks)}%20 \\
       --export=ALL,FOA_PROJECT_DIR=$FOA_PROJECT_DIR,FOA_STUDY_DIR=$FOA_STUDY_DIR \\
       slurm/study_phase.sbatch study/artifacts/{CFG.name}/jobs/{STAGE}.txt
```

## Cost

At `0.31 + 0.0345 x {CFG.clients_per_round} x 5 = {per_round:.2f}` s per round per family, with **one**
family per task - a combination line names one rule - and roughly 40% added for
the teacher forward pass the KD half needs on every batch:

| | seconds |
|---|---|
| training, {SL.SCREEN_ROUNDS} rounds, one family, +40% | {train_s:.0f} |
| per-round evaluations | ~6 |
| final three-category evaluation | ~2 |
| interpreter, torch, cache, two books | ~40 |
| **wall per task** | **~{task_s:.0f} (~{task_s / 60:.1f} min)** |

**{len(tasks)} tasks is roughly {gpu_h:.0f} GPU-h** by that model, about
{gpu_h / 20 * 60:.0f} minutes of wall clock at `%20`. The model ran low on P13,
whose two-family 25-round tasks cost 94 s measured against 78 s predicted; the
same ratio on a one-family task puts this nearer **{gpu_h_measured:.0f} GPU-h**.
Both are quoted because a stage this size should be authorised against the
pessimistic one.
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
