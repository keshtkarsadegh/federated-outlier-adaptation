"""
Emit P25 of the digits study: an EXTENSION, not a stage of the programme.

    python tools/make_digits_p25.py

Writes two files beside the study's other task files: the stage and its README.

WHAT THIS IS NOT.  It is not a missing piece of the study, and it is not a
correction to P23.  Every stage the paper reports has run; P23's screen and its
finals stand exactly as they are and are not re-emitted.  This is a second
extension bolted on afterwards, and it is named as one everywhere it appears -
in the task file's header, in the chain manifest, in the selection record and in
``REPRODUCE.md``, which keeps both out of the core stage table on purpose.

WHAT IT ASKS, AND WHY A SECOND TIME.  P23 screened the joint grid of the pair
that leads each schedule **by TEST score**, which is the owner's documented
departure from ranking on validation.  That is a defensible pair to tune and it
is not the pair the study selected: ``tables/p12_agg_top3.json`` says in its own
``rank_by`` that the shortlist was cut on VALIDATION, and on the parallel
schedule the two orderings disagree about which rule comes first - the
validation head is ``anchor_h2``, the test head is ``eta_0p95``.  So P23's
answer - that moving the two halves together finds nothing the point of
independent bests did not already have - was measured on a rule the programme
did not choose.  This screens the joint grid of the pair it DID choose.

THE ONE THING THAT MOVES.  The rule half, and nothing else.  The penalty half is
read out of P23's own ``THE_PAIR`` rather than re-derived here, the four penalty
dials are the same four rows at the same three, three, two and three values, and
the mapping onto the trainer's coefficients is the same function.  Two grids
that differed in two places could not be read against each other, and reading
them against each other is the whole point of running the second one.
"""

from __future__ import annotations

import argparse
import dataclasses as _dc
import json
from pathlib import Path

import make_digits_p23 as _p23

from federated_outlier_adaptation.training import reg_cells
from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training.study_config import DIGITS_STUDY01 as _STUDY

#: The half-lives the PROGRAMME screened, for this stage's rule row to be read
#: against.  Imported rather than retyped: the point of that row is which of its
#: three values sit inside what a stage actually searched, and that is a fact
#: about ``agg_cells`` and not about this file.
from federated_outlier_adaptation.training.agg_cells import (  # noqa: E402
    ANCHOR_HALFLIVES_R as AGG_HALFLIVES,
)

#: The stage's name, and the stem of both files it writes.
STAGE = "s25_combo_screen_selected"

#: Searched at the rate every grid in this study is searched at - eight of ten,
#: not the study's own nine - so this table can be read against P23's and
#: against the two screens whose winners both of them tune.  The rate comes from
#: ``search_clients_per_round``, never from a flag.
CFG = _dc.replace(_STUDY, clients_per_round=_STUDY.search_clients_per_round)

#: Sampler seeds of this extension: 770001-772155.  Opened at 70000 for P23's
#: reason and one of its own: 216 cells span 2160, so a stage no longer fits in
#: the thousand-wide block the scheme assumes, and P23's own two blocks now run
#: to 764015.  A ten-thousand step past P23's screen keeps this stage's window
#: clear of both of P23's and of everything the programme drew.
SEED_OFFSET = 70000

#: Where the study's own records live, read rather than restated.
_TABLES = (Path(__file__).resolve().parents[1] / "study" / "artifacts"
           / CFG.name / "tables")

#: The record the aggregation shortlist was cut from, and the only place this
#: stage learns which rule each schedule selected.
AGG_SHORTLIST = "p12_agg_top3.json"


def _selected_fold() -> int:
    """
    The winning g-0 fold, read from the record rather than remembered.

    Every cell here reads the shipped model's Fisher, and ``study_phase.sbatch``
    resolves ``$G0_FOLD`` from the study's own ``g0_selection.json`` at run time -
    so the only thing this number does is appear in a header and a README.  That
    is exactly the kind of number that goes stale without anything failing, which
    is why P21 started reading it and P23 and this read it the same way.
    """
    record = (Path(__file__).resolve().parents[1] / "study" / "artifacts"
              / CFG.name / "g0_selection.json")
    return int(json.loads(record.read_text())["selected_fold"])


#: The winning g-0 fold, whose Fisher every cell here reads.
G0_FOLD = _selected_fold()


def selected_rule(family: str) -> str:
    """
    The server rule one schedule's screen SELECTED, from its own record.

    The record holds both orderings and names the one it selected on in
    ``rank_by``, so the list is chosen by that field rather than by naming a
    basis here - which is the same reading ``export_schedule_views.selected_rule``
    does for the shipped schedule views, and it is done that way for the same
    reason: a shortlist re-cut on the other basis would move this stage's whole
    grid without anything failing.  The record's ``top`` is the same shortlist as
    a set and carries no order, which is why the head of ``rankings`` is read.
    """
    record = json.loads((_TABLES / AGG_SHORTLIST).read_text())
    basis = record.get("rank_by")
    ranking = ((record.get("rankings") or {}).get(family) or {}).get(basis) or []
    if not ranking:
        raise SystemExit(
            f"FATAL: {AGG_SHORTLIST} ranks no {family} rule under {basis!r}. "
            "Which rule this schedule selected is written there and nowhere "
            "else; a grid built around a rule nobody selected is this stage's "
            "one unrecoverable failure."
        )
    return ranking[0]["id"]


#: The pair this extension tunes, per schedule, and where each half came from.
#:
#: The RULE is the head of ``p12_agg_top3.json``'s own ranking, on the basis
#: that record says it selected on.  The PENALTY is P23's, read out of its
#: module rather than retyped: this grid moves the rule half and nothing else,
#: so the two joint grids differ in one place and can be read against each
#: other.  (The validation shortlist's own penalty leaders are different arms
#: again - ``logit_l2_lam0p001`` parallel, ``ntd_b0p01_t2`` cyclic, neither of
#: them a KD+EWC blend - so tuning one of those jointly would need a different
#: mapping and would be a different grid, not this one widened.)
THE_PAIR = {
    family: (selected_rule(family), _p23.THE_PAIR[family][1])
    for family in SL.FAMILIES
}

#: The untuned pair already on disk in ``s20_combos4.txt``, whose every non-grid
#: flag this stage copies.  Named so that a reader can diff one of these lines
#: against one of those and see that only the horizon and the coefficients moved.
THE_SHIPPED_LINES = {
    "concurrent": "anchor_h2_hybrid_seq_mix0p5",
    "sequential": "seq_fedavg_hybrid_mix0p5",
}


def _check_the_grid_is_the_selected_pair() -> None:
    """
    The grid moves the rule the record selected, or this stage does not emit.

    ``reg_cells.CTUNE_SEL_RULES`` names the rule as an ``--aggregation`` and a
    knob, because that is what a line needs; the record names it as a cell id.
    Nothing makes those two agree except this check, and a disagreement would be
    silent: every line would still be legal, still parse and still run, and the
    stage would report a joint grid around an arm nobody chose.
    """
    for family, (rule, _penalty) in THE_PAIR.items():
        spec = reg_cells.CTUNE_SEL_RULES[family]
        ids = {spec.agg_id.format(reg_cells._fmt(value)) for value in spec.values} \
            if spec.knob is not None else {spec.agg_id}
        if rule not in ids:
            raise SystemExit(
                f"FATAL: {AGG_SHORTLIST} selected {rule!r} for the {family} "
                f"schedule and this grid moves {sorted(ids)}. The grid is built "
                "around the selected rule or it is built around nothing."
            )


_check_the_grid_is_the_selected_pair()


def lines() -> list[str]:
    """One line per (cell, fold), at the screening horizon."""
    return [
        SL.combo_tune_line(
            CFG, cell, fold, SL.SCREEN_ROUNDS,
            CFG.seed_base + SEED_OFFSET + index * 10 + fold,
            tags=SL.CTUNE_SEL_TAGS,
        )
        for index, cell in enumerate(reg_cells.combo_tune_selected_cells())
        for fold in SL.FOLDS
    ]


#: The whole grid: 3 x 3 x 2 x 3 penalty settings, times the rule knob each
#: schedule has - three anchor half-lives on the parallel side, none on the
#: cyclic, where the selected rule is cyclic FedAvg and has no coefficient.
SCREENED_CELLS = len(reg_cells.combo_tune_selected_cells())

#: Dial settings the deduplication removed because they hand the trainer the
#: same (lam, mix, T) under the same rule.  Zero here; computed, not asserted.
DUPLICATES = reg_cells.combo_tune_selected_duplicates()


def _by_family() -> dict:
    """``{schedule: cells}``, for the header and the README."""
    return reg_cells.combo_tune_selected_by_family()


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


def _anchor_rows() -> str:
    """A README table of half-life -> the coefficient each horizon emits."""
    rows = []
    for halflife in reg_cells.CTUNE_SEL_ANCHOR_HALFLIVES_R:
        values = []
        for rounds in (SL.SCREEN_ROUNDS, SL.FULL_ROUNDS):
            resolved = SL.resolve_agg_flags(
                {"anchor_halflife_r": halflife}, CFG, rounds)
            values.append(resolved["server_anchor"])
        screened = ("yes" if halflife in AGG_HALFLIVES else "**no**")
        rows.append(f"| {halflife:g}R | {values[0]:.6g} | {values[1]:.6g} | "
                    f"{screened} |")
    return "\n".join(rows)


def header(tasks: list[str]) -> list[str]:
    families = _by_family()
    ewc = reg_cells.CTUNE_EWC_COEFFS
    kd = reg_cells.CTUNE_KD_COEFFS
    temps = reg_cells.CTUNE_TEMPERATURES
    mixes = reg_cells.CTUNE_MIXES
    halflives = reg_cells.CTUNE_SEL_ANCHOR_HALFLIVES_R
    dials = len(ewc) * len(kd) * len(temps) * len(mixes)
    return [
        f"# {STAGE}.txt - {CFG.name}, P25: AN EXTENSION, NOT A STAGE OF THE",
        "# PROGRAMME. The study is complete without it and no table, shortlist",
        "# or crossing the paper reports reads anything it produces.",
        "#",
        "# WHY A SECOND JOINT SCREEN. P23 screened the joint grid of the pair",
        "# that leads each schedule by TEST score - the owner's documented",
        "# departure from ranking on validation. That is not the pair the study",
        f"# SELECTED: {AGG_SHORTLIST} says in its own rank_by that the",
        "# shortlist was cut on VALIDATION, and on the parallel schedule the two",
        "# orderings disagree about which rule comes first. So P23's finding was",
        "# measured on a rule the programme did not choose. This screens the",
        "# joint grid of the pair it did.",
        "#",
        f"# THE PAIR, rule by the head of {AGG_SHORTLIST}'s own ranking:",
    ] + [
        f"#   {family:<11} rule {rule:<17} penalty {penalty}"
        for family, (rule, penalty) in THE_PAIR.items()
    ] + [
        "#",
        "# THE PENALTY HALF IS P23'S, UNCHANGED, and is read out of its module",
        "# rather than retyped. This grid moves the rule half and nothing else,",
        "# so the two joint grids differ in one place and can be read against",
        "# each other - which is the only reason to run the second one.",
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
        f"#   h         {{{', '.join(f'{v:g}R' for v in halflives)}}} - PARALLEL"
        " ONLY; seq_fedavg has",
        "#             no coefficient, so the cyclic half is a third the size",
        f"#   = {len(ewc)} x {len(kd)} x {len(temps)} x {len(mixes)} = {dials}"
        f" penalty settings"
        f" -> {len(families['concurrent'])} parallel + "
        f"{len(families['sequential'])} cyclic = {SCREENED_CELLS} cells",
        f"#     x {len(SL.FOLDS)} folds = {len(tasks)} tasks",
        "#",
        "# THE ANCHOR ROW REACHES OUTSIDE WHAT THE PROGRAMME SEARCHED, and says",
        f"# so. agg_cells.ANCHOR_HALFLIVES_R is"
        f" {{{', '.join(f'{v:g}R' for v in AGG_HALFLIVES)}}} and stops at 2R",
        "# because beyond it the anchor never acts inside the run - so the",
        "# selected setting sat at the TOP EDGE of its own row and a bracket",
        "# around it has no neighbour above inside anything a stage screened.",
        "# 4R is that neighbour, and it is a WEAKER intervention than the whole",
        "# screened row rather than a stronger one.",
        "#",
        "# HOW THE HALF-LIFE REACHES THE TRAINER. A cell stores h as a multiple",
        "# of the run length R and study_lines.resolve_agg_flags emits",
        "#   --server-anchor  lambda_s = 1 - 2 ** (-1 / (h * rounds))",
        "# so a setting means the same intervention on this 25-round screen and",
        "# at the 100-round horizon its winner is re-run at. A cell that stored",
        "# the coefficient would be two different rules at the two horizons.",
        "#",
        "# HOW THE PENALTY DIALS REACH THE TRAINER. AnchoredTrainer computes",
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
        "# at 100 rounds by s26_combo_full_selected.txt.",
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
        "# disjoint from every range this study has drawn, P23's two blocks",
        "# included: the highest seed any shipped file draws is 764015.",
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
    halflives = reg_cells.CTUNE_SEL_ANCHOR_HALFLIVES_R
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
    rule_par, penalty_par = THE_PAIR["concurrent"]
    rule_cyc, penalty_cyc = THE_PAIR["sequential"]
    return f"""# {CFG.name} - P25: joint tuning of the SELECTED pair (**an extension**)

**This is not a stage of the study, and it is not a correction to P23.** Every
stage the paper reports has run, `COMBINATIONS.md` already records what the
combination cross found, and P23's screen and finals stand exactly as they are
and are not re-emitted. This is a second extension, it is named as one in the
task file's header, in its chain manifest and in its selection record, and
`REPRODUCE.md` keeps both out of the core stage table deliberately.

## Why a second joint screen

P23 screened the joint grid of the pair that leads each schedule **by TEST
score** - the owner's documented departure from ranking on validation, and the
order the two shipped winners views carry. That is a defensible pair to tune.
It is not the pair the study selected.

`tables/{AGG_SHORTLIST}` says in its own `rank_by` field that the aggregation
shortlist was cut on **validation**, and on the parallel schedule the two
orderings disagree about which rule comes first:

| schedule | validation head (selected) | test head (P23 tuned this) |
|---|---|---|
| parallel (`concurrent`) | `{rule_par}` | `eta_0p95` |
| cyclic (`sequential`) | `{rule_cyc}` | `seq_delta_capped` |

So P23's finding - that moving the two halves together finds nothing the point
of independent bests did not already have - was measured on a rule the
programme did not choose. This screens the joint grid of the pair it did.

## The pair

| schedule | selected rule | penalty |
|---|---|---|
| parallel (`concurrent`) | `{rule_par}` - the server anchor at half-life h = 2R | `{penalty_par}` - the KD+EWC blend |
| cyclic (`sequential`) | `{rule_cyc}` - cyclic FedAvg, parameter-free | `{penalty_cyc}` - the same blend |

The rule is read from the head of `{AGG_SHORTLIST}`'s own ranking, on the basis
that record names, rather than typed here: a shortlist re-cut on the other basis
would move this whole grid without anything failing. **The penalty half is
P23's, unchanged**, and is read out of `make_digits_p23.THE_PAIR` rather than
re-derived - this grid moves the rule half and nothing else, so the two joint
grids differ in exactly one place and can be read against each other, which is
the only reason to run the second one.

(The validation shortlist's own *penalty* leaders are different arms again -
`logit_l2_lam0p001` on the parallel schedule and `ntd_b0p01_t2` on the cyclic
one, neither of them a KD+EWC blend. Tuning one of those jointly would need a
different mapping onto the trainer's coefficients and would be a different grid,
not this one widened. It is not run and it is not claimed.)

**Every non-grid flag is copied from the shipped `s20` lines that pair these two
halves** - `{THE_SHIPPED_LINES['concurrent']}` and
`{THE_SHIPPED_LINES['sequential']}` - so a line here differs from one there in
the horizon and in the four (or five) coefficients, and in nothing else: same
cohort, same books, same rate, same `--outer-workers 1 --inner-workers 1`, same
`--extended-aggregations`, and the `fisher_path` convention
`s19_hybrid_seq.txt` established.

## The grid

    c_ewc  in {{{', '.join(f'{v:g}' for v in ewc)}}}      EWC coefficient
    c_kd   in {{{', '.join(f'{v:g}' for v in kd)}}}      KD coefficient
    T      in {{{', '.join(f'{v:g}' for v in temps)}}}         KD temperature
    m      in {{{', '.join(f'{v:g}' for v in mixes)}}}      blend weight
    h      in {{{', '.join(f'{v:g}R' for v in halflives)}}}       PARALLEL ONLY

    {len(ewc)} x {len(kd)} x {len(temps)} x {len(mixes)} = {dials} penalty settings
      x {len(halflives)} anchor half-lives = {len(families['concurrent'])} parallel cells
      x 1 (no rule knob)      = {len(families['sequential'])} cyclic cells
      = {SCREENED_CELLS} cells x {len(SL.FOLDS)} folds = {len(tasks)} tasks at 25 rounds

The four penalty rows are **P23's four penalty rows**, at the same values, for
the reason above; `REG_GRID_RANGES.md` section 9 records where each of them came
from and that reading is not repeated here. What is new is the rule axis.

### The anchor row, and where it leaves the programme

`{rule_par}` is one setting of a coefficient row, so the parallel pair's rule
half *has* a knob and this grid moves it. The row is the selected value with one
neighbour each side - and the neighbour above is **outside anything the study
screened**, which is a fact about the selection and is stated rather than
smoothed over.

`agg_cells.ANCHOR_HALFLIVES_R` is
{{{', '.join(f'{v:g}R' for v in AGG_HALFLIVES)}}} and stops at 2R deliberately:
the coefficient is `lambda_s = 1 - 2 ** (-1/h)` in rounds, so at h = 2R the
displacement from g-0 never halves inside the run at all, and nothing slower
than that is an intervention. **The rule the study selected therefore sat on the
top edge of its own row.** A bracket around it has no neighbour above inside the
screened range, so 4R steps outside it, and 4R is a *weaker* intervention than
every value the screen tried rather than a stronger one. The row is:

| h | `--server-anchor` at 25 rounds | at 100 rounds | inside the screened row? |
|---|---|---|---|
{_anchor_rows()}

A cell stores `h` and not the coefficient, and
`study_lines.resolve_agg_flags` turns it into `--server-anchor` at the horizon
the line runs - which is why one setting means the same intervention on this
screen and in the finals its winner is re-run in. A cell that stored the
coefficient would be two different rules at the two horizons, which is the
failure that row of `resolve_agg_flags` exists to document.

`{rule_cyc}` has no coefficient at all, which is why the cyclic half of this
grid is a third the size - the same asymmetry P23 had, arriving from a different
rule.

## How the penalty dials reach the trainer

`AnchoredTrainer` computes

    lam * ( mix * KD + (1 - mix) * Fisher )

and the penalty this grid names is

    m * c_kd * KD + (1 - m) * c_ewc * Fisher

Matching term by term gives `lam * mix = m * c_kd` and
`lam * (1 - mix) = (1 - m) * c_ewc`, hence

    lam = m * c_kd + (1 - m) * c_ewc
    mix = m * c_kd / lam

which is :func:`reg_cells.combo_tune_lam_mix`, the same function P23's grid
resolves through and not a second copy of it.

**The trainer's `mix` is not the owner's `m`.** It is the KD half's share of the
total penalty weight, and it moves with `c_ewc` and `c_kd` as well as with `m` -
the two coincide only where the two coefficients are equal, which on this grid
is `c_ewc = c_kd = 0.05` alone. A grid that wrote `m` straight into `mix` would
be sweeping a different object than the one it named, and nothing in the emitted
line would say so: the line would carry a legal coefficient and run.

| `c_ewc` | `c_kd` | `m` | `lam` | trainer `mix` |
|---|---|---|---|---|
{_mapping_rows()}

(`T` and `h` pass through untouched; the table is the same for both temperatures
and, on the parallel side, for all three half-lives.)

## Deduplication

**{DUPLICATES} cell(s) removed.** Two dial settings collide when they hand the
trainer the same `(lam, mix, T)` under the same rule - two folders, two seeds and
one experiment, with nothing in the run records to say so. The check is on the
coefficients rather than on the dials, because the dials are what differ. It
finds nothing here, for P23's reason: a collision needs `m * c_kd` and
`(1 - m) * c_ewc` to repeat *together*, and the nine `(m, c_kd)` products are
all distinct. The check is emitted anyway, so that a widened row cannot quietly
pay twice.

## A ranking horizon

25 rounds, not 100. Nothing here is a reported number; a 25-round result is a
ranking signal. The winners are re-run at 100 rounds by
`s26_combo_full_selected.txt`, which `study_emit.py combo-tune-full-selected`
emits under the same rule every other selection in this study uses.

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

= {seeds_lo}-{seeds_hi}, disjoint from every range this study has drawn and from
both of P23's: {SCREENED_CELLS} cells span {(SCREENED_CELLS - 1) * 10}, more than the thousand-wide block
the scheme assumes, so the window is opened on width rather than on the next
free thousand - P21's lesson, and P23's. The highest seed any shipped file draws
is 764015, so this block sits clear of the programme's and of the first
extension's rather than interleaved with either.

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
{gpu_h / 20 * 60:.0f} minutes of wall clock at `%20`. P23's screen of the same
shape booked that and measured 18.4 GPU-h, which is nearer the **{gpu_h_measured:.0f} GPU-h** the
measured per-task cost of a one-family 25-round task gives. Both are quoted
because a stage this size should be authorised against the pessimistic one.
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
