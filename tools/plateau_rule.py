#!/usr/bin/env python3
"""
The stopping rule a server could actually have run: a plateau on its own cohort.

`stopping_table.py` prices the fixed horizon against an oracle and against the
eight forgetting signals, and its verdict is that one signal fixed in advance
buys +0.46p over the horizon. This tool asks a smaller question that turns out
to be the better one. The eight signals are proxies for forgetting, and
forgetting lives on data the server no longer has - but the *cohort's own*
validation accuracy is not a proxy for anything. The clients hold it, the server
already reads it every round to report a number, and nothing about it is
forbidden the way `source_val_accuracies` is forbidden.

    python tools/plateau_rule.py --root "$FOA_STUDY_DIR"
    python tools/plateau_rule.py --root "$FOA_STUDY_DIR" \\
        --out "$FOA_STUDY_DIR/tables/stopping"

THE RULE, IN ONE PARAGRAPH. Let ``c(t)`` be the fold-mean cohort validation
accuracy at round ``t`` - `pool_val_accuracies`, the adaptation half - and
``b(t)`` its best value so far. The rule FIRES at the first round where ``c`` has
failed to exceed ``b + eps`` for ``k`` consecutive rounds. It does not report the
round it fired on: the server has been keeping the checkpoint of the best cohort
round all along, so what it deploys is round ``t_kept = argmax c(t)`` over
``t <= t_fire``, earliest round on ties. An arm the rule never fires on runs the
full budget and is scored at its last round, exactly as the tables report it.
The primary setting is ``k = 20``, ``eps = 0``.

THREE THINGS THIS IS NOT. It is not the oracle: the oracle maximises the study's
*score*, which is a function of the source population, and this maximises the
cohort accuracy, which is not. It is not a per-arm choice: one ``(k, eps)`` is
fixed for every arm of every stage. And it is not free of the horizon - a rule
that fires late still spent the rounds it took to fire, which is why the fire
round is reported beside the kept round rather than folded into it.

WHAT IT DELIVERS, over the 83 hundred-round arms:

    fixed horizon        7.74p
    plateau, kept best   8.93p     +1.19p, fires on 50 arms
    the one signal rule  8.20p     +0.46p
    oracle (not a rule)  9.22p

so the plateau recovers 1.19 of the 1.48 points the oracle prices - four fifths
of what stopping is worth at all - against the signals' 0.46. It gains where the
horizon costs most: +11.57p on the two extreme arrangements, where `dual` fires
at round 28 and keeps round 8 for 16.60p against the horizon's -0.59p. One arm
of eighty-three is hurt, by half a point.

WHY THE SETTING IS NOT FITTED TO THE EXTREMES. ``k = 20``, ``eps = 0`` is the
best cell of the sixteen on the mean over all 83 arms, and the two extremes are
the arms with by far the most to gain - so the obvious objection is that the
setting was chosen by them. It was not: choosing the cell on the 81
NON-EXTREME arms alone, with the extremes held out entirely, picks ``k = 20``,
``eps = 0`` as well. `plateau_grid.csv` carries the column that check is made on,
and `tests/test_plateau.py` pins it.

WHAT IT WRITES, into the stopping bundle beside `stopping_all.csv`, because it
is the same question read a different way:

    plateau_arms.csv      one line per arm at the primary setting
    plateau_stages.csv    per stage and over all arms: fixed, rule, gain, fires
    plateau_grid.csv      the sixteen settings x two variants, for the record
    plateau_extremes.csv  the two extreme arrangements against oracle and proxy

BASIS AND CONVENTIONS ARE `stopping_table.py`'S, BY IMPORT RATHER THAN BY
RESTATEMENT. The arm reader, the fold mean, the score, the oracle round and the
one-signal rule all come from that module, so the bases of the two tables are
identical by construction and cannot drift apart: the `final_score` column here
is the `final_score` column there, arm for arm.
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))

from stopping_table import (  # noqa: E402
    STAGES,
    stage_baselines,
    Verdict,
    arm_of,
    baselines,
    judge,
    one_rule,
    read_arms,
)

from federated_outlier_adaptation.analysis.forgetting_signals import (  # noqa: E402
    DEFAULT_DELTAS,
)

#: The patience grid, in rounds. Three is the shortest patience that is not
#: simply "one bad round", and twenty is a fifth of the horizon; past that a
#: rule cannot fire early enough to be worth anything on a hundred rounds.
PATIENCES = (3, 5, 10, 20)

#: The improvement margin, in accuracy. Zero asks only that the cohort get
#: better; the three positive values ask that it get better by enough to be
#: worth another round of the source model.
MARGINS = (0.0, 0.001, 0.0025, 0.005)

#: What the paper reports. Chosen on the mean over every arm, and the same cell
#: the 79 non-extreme arms choose on their own - see the header.
PRIMARY_PATIENCE = 20
PRIMARY_MARGIN = 0.0

#: The two things a server can do when the rule fires. STOP_AT_FIRE deploys the
#: round it stopped on; CHECKPOINT_BEST deploys the best cohort round it has
#: been keeping. The second is what a server does anyway - it is cheaper to keep
#: one checkpoint than to explain why the last round was deployed instead - and
#: it is the variant the paper reports.
STOP_AT_FIRE = "stop_at_fire"
CHECKPOINT_BEST = "checkpoint_best"
VARIANTS = (STOP_AT_FIRE, CHECKPOINT_BEST)

#: The label the stage rows use for the row that is every arm at once.
ALL = "all"


# ------------------------------------------------------------------ the rule
def fire_round(cohort: Sequence[float], patience: int, margin: float) -> Optional[int]:
    """
    The zero-based round the plateau rule fires on, or None if it never does.

    Round 0 is the shipped model before any client has trained, so it is the
    first best-so-far rather than a round the rule may fire on: the count of
    consecutive non-improvements starts at round 1. A round improves when it
    exceeds the best so far by more than ``margin``, and only an improvement
    resets the count - a run that oscillates below its own best is a run that
    has plateaued, however lively the trace looks.
    """
    best = cohort[0]
    missed = 0
    for index in range(1, len(cohort)):
        if cohort[index] > best + margin:
            best = cohort[index]
            missed = 0
            continue
        missed += 1
        if missed >= patience:
            return index
    return None


def kept_round(cohort: Sequence[float], fired: Optional[int]) -> int:
    """
    The zero-based round the server deploys: the best cohort round it has seen.

    Ties go to the EARLIEST round. Two rounds that reach the same cohort
    accuracy are not equally good, because the later one spent more rounds of
    the source model arriving at it - the same tie-break `stopping_table.py`
    applies when two permitted rules reach the same score.
    """
    window = list(cohort[: (len(cohort) if fired is None else fired + 1)])
    return window.index(max(window))


def scored_round(cohort: Sequence[float], fired: Optional[int], variant: str) -> int:
    """The zero-based round an arm is scored at, under one variant of the rule."""
    if fired is None:
        return len(cohort) - 1
    return fired if variant == STOP_AT_FIRE else kept_round(cohort, fired)


# ----------------------------------------------------------------- the arms
def verdicts(root: Path, tag: str, min_folds: int) -> List[Tuple[str, Verdict]]:
    """
    ``(stage, verdict)`` for every hundred-round arm, in the tables' own order.

    Read through `stopping_table.read_arms` and judged by
    `stopping_table.judge`, so an arm that appears here appears in
    `stopping_all.csv` with the same final score and the same oracle round.
    """
    refs = stage_baselines(root)
    found: List[Tuple[str, Verdict]] = []
    for stem, _title in STAGES:
        # Each stage against its own cohort's shipped-model accuracy.
        a0, p0 = refs[stem.strip("_")]
        for (cell, family), stored in sorted(read_arms(root, f"{tag}_{stem}").items()):
            arm = arm_of(cell, family, stored)
            if arm is None or arm.folds < min_folds or arm.rounds < 2:
                continue
            found.append((stem.strip("_"), judge(arm, a0, p0, DEFAULT_DELTAS)))
    return found


def cohort_of(verdict: Verdict) -> List[float]:
    """The trace the rule reads: the cohort's own validation accuracy, scored."""
    return list(verdict.arm.adaptation[: len(verdict.values)])


def outcome(stage: str, verdict: Verdict, patience: int, margin: float,
            variant: str) -> dict:
    """One arm under one setting: where it fired, what it kept, what that cost."""
    cohort = cohort_of(verdict)
    fired = fire_round(cohort, patience, margin)
    index = scored_round(cohort, fired, variant)
    final = verdict.values[-1]
    return {
        "stage": stage,
        "cell": verdict.arm.cell,
        "family": verdict.arm.family,
        "folds": verdict.arm.folds,
        "rounds": len(verdict.values),
        "fired": int(fired is not None),
        "fire_round": None if fired is None else fired + 1,
        "kept_round": index + 1,
        "kept_score": verdict.values[index],
        "final_score": final,
        "gain": verdict.values[index] - final,
        "oracle_round": verdict.star,
        "oracle_score": verdict.values[verdict.star - 1],
    }


def outcomes(staged: Sequence[Tuple[str, Verdict]], patience: int, margin: float,
             variant: str) -> List[dict]:
    return [outcome(stage, verdict, patience, margin, variant)
            for stage, verdict in staged]


# ---------------------------------------------------------------- the tables
ARM_COLUMNS = ("stage", "cell", "family", "folds", "rounds", "fire_round",
               "kept_round", "kept_score", "final_score", "gain")

STAGE_COLUMNS = ("patience", "margin", "variant", "stage", "arms", "fixed_mean",
                 "rule_mean", "gain", "fires", "oracle_mean")

GRID_COLUMNS = ("patience", "margin", "variant", "arms", "mean_score", "mean_gain",
                "fires", "extreme_mean_gain", "nonextreme_mean_score",
                "worst_arm", "worst_arm_gain", "arms_hurt")

EXTREME_COLUMNS = ("cell", "family", "fire_round", "kept_round", "kept_score",
                   "final_score", "gain", "oracle_round", "oracle_score",
                   "proxy_round", "proxy_score")

#: The stage the extreme table is cut from, and the order its rows are written
#: in - the pair the manuscript prints, and not a glob over whatever ran.
EXTREME_STAGE = "extreme"


def stage_rows(rows: Sequence[dict], patience: int, margin: float,
               variant: str) -> List[dict]:
    """
    Per stage and then over every arm at once, at one setting.

    The all-arms row is the mean over ARMS rather than over stages, so a stage
    with twenty-six arms weighs twenty-six times a stage with one - the same
    weighting `stopping_table.one_rule` chooses its rule under, and the reason
    the two tables' `all` rows can be read against each other.
    """
    order = [stem.strip("_") for stem, _ in STAGES] + [ALL]
    out: List[dict] = []
    for stage in order:
        priced = rows if stage == ALL else [r for r in rows if r["stage"] == stage]
        if not priced:
            continue
        out.append({
            "patience": patience,
            "margin": margin,
            "variant": variant,
            "stage": stage,
            "arms": len(priced),
            "fixed_mean": st.mean(r["final_score"] for r in priced),
            "rule_mean": st.mean(r["kept_score"] for r in priced),
            "gain": st.mean(r["gain"] for r in priced),
            "fires": sum(r["fired"] for r in priced),
            "oracle_mean": st.mean(r["oracle_score"] for r in priced),
        })
    return out


def grid_row(rows: Sequence[dict], patience: int, margin: float, variant: str) -> dict:
    """
    One cell of the sixteen, under one variant.

    `nonextreme_mean_score` is the column the honesty split is read on: it is
    what the setting would have been chosen by had the two extreme arrangements
    - the arms with the most to gain - never been measured at all.
    """
    worst = min(rows, key=lambda r: (r["gain"], r["stage"], r["cell"], r["family"]))
    extreme = [r for r in rows if r["stage"] == EXTREME_STAGE]
    ordinary = [r for r in rows if r["stage"] != EXTREME_STAGE]
    return {
        "patience": patience,
        "margin": margin,
        "variant": variant,
        "arms": len(rows),
        "mean_score": st.mean(r["kept_score"] for r in rows),
        "mean_gain": st.mean(r["gain"] for r in rows),
        "fires": sum(r["fired"] for r in rows),
        "extreme_mean_gain": st.mean(r["gain"] for r in extreme) if extreme else None,
        "nonextreme_mean_score": st.mean(r["kept_score"] for r in ordinary),
        "worst_arm": f"{worst['stage']}/{worst['cell']}/{worst['family']}",
        "worst_arm_gain": worst["gain"],
        "arms_hurt": sum(1 for r in rows if r["gain"] < 0),
    }


def grid_rows(staged: Sequence[Tuple[str, Verdict]]) -> List[dict]:
    """Every setting, both variants, in the order the grids are declared."""
    return [grid_row(outcomes(staged, patience, margin, variant), patience, margin, variant)
            for patience in PATIENCES
            for margin in MARGINS
            for variant in VARIANTS]


def best_cell(grid: Sequence[dict], variant: str, column: str) -> dict:
    """
    The setting one column would have chosen, ties to the smaller patience.

    Used twice and never to select anything the paper reports: once on
    `mean_score` to confirm the primary setting is the best cell, and once on
    `nonextreme_mean_score` to make the honesty split - both of which are checks
    on a setting fixed in this file rather than a search that produces one.
    """
    return max((row for row in grid if row["variant"] == variant),
               key=lambda row: (row[column], -row["patience"], -row["margin"]))


def extreme_rows(rows: Sequence[dict], staged: Sequence[Tuple[str, Verdict]],
                 rule: Optional[Tuple[str, float]]) -> List[dict]:
    """
    The two extreme arrangements, against the oracle and against the proxy rule.

    The proxy columns are the one-signal rule `stopping_table.py` fixes for the
    whole study, recomputed here from the same verdicts rather than read back out
    of `stopping_all.csv`, so this table cannot quote a rule the other stopped
    choosing.
    """
    stops = {(verdict.arm.cell, verdict.arm.family): verdict
             for stage, verdict in staged if stage == EXTREME_STAGE}
    out: List[dict] = []
    for row in rows:
        if row["stage"] != EXTREME_STAGE:
            continue
        verdict = stops[(row["cell"], row["family"])]
        index = None if rule is None else verdict.stops.get(rule)
        out.append({
            "cell": row["cell"],
            "family": row["family"],
            "fire_round": row["fire_round"],
            "kept_round": row["kept_round"],
            "kept_score": row["kept_score"],
            "final_score": row["final_score"],
            "gain": row["gain"],
            "oracle_round": row["oracle_round"],
            "oracle_score": row["oracle_score"],
            "proxy_round": None if index is None else index + 1,
            "proxy_score": None if index is None else verdict.values[index],
        })
    return out


def write_csv(rows: Sequence[dict], columns: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})
    print(f"  -> {path}  ({len(rows)} rows)")


# ------------------------------------------------------------------ printing
def _p(value: Optional[float]) -> str:
    """A score in points, or a dash where the rule produced none."""
    return "-" if value is None else f"{value * 100:.2f}p"


def _n(value) -> str:
    """A round, or a dash where the rule never fired."""
    return "-" if value is None else str(value)


def show_arms(rows: Sequence[dict]) -> None:
    """One line per arm: where it fired, what it kept, what that was worth."""
    width = max([len(r["cell"]) for r in rows] + [len("cell")]) + 2
    print(f"\nTHE PLATEAU RULE, ARM BY ARM  (k={PRIMARY_PATIENCE}, "
          f"eps={PRIMARY_MARGIN:g}, checkpoint kept)")
    print(f"{'stage':<10}{'cell':<{width}}{'sched':<10}{'fires':>7}{'kept':>7}"
          f"{'score':>10}{'fixed':>10}{'vs fixed':>11}")
    print("-" * (width + 55))
    for row in rows:
        print(f"{row['stage']:<10}{row['cell']:<{width}}{row['family']:<10}"
              f"{_n(row['fire_round']):>7}{row['kept_round']:>7}"
              f"{_p(row['kept_score']):>10}{_p(row['final_score']):>10}"
              f"{_p(row['gain']):>11}")


def show_stages(rows: Sequence[dict]) -> None:
    """What the rule delivered, stage by stage, against the horizon it replaces."""
    print("\nWHAT THE PLATEAU RULE DELIVERS, BY STAGE")
    print("  the oracle column is printed to price the gain, never as a method: "
          "it reads the source population.")
    print(f"  {'stage':<12}{'arms':>6}{'fixed':>10}{'plateau':>10}{'vs fixed':>11}"
          f"{'fires':>8}{'oracle':>10}")
    for row in rows:
        label = "ALL" if row["stage"] == ALL else row["stage"]
        print(f"  {label:<12}{row['arms']:>6}{_p(row['fixed_mean']):>10}"
              f"{_p(row['rule_mean']):>10}{_p(row['gain']):>11}"
              f"{row['fires']:>8}{_p(row['oracle_mean']):>10}")


def show_grid(grid: Sequence[dict]) -> None:
    """The sixteen settings under both variants, and the honesty split."""
    print("\nEVERY SETTING, BOTH VARIANTS")
    print(f"  {'variant':<16}{'k':>4}{'eps':>8}{'mean':>9}{'vs fixed':>10}"
          f"{'fires':>7}{'extremes':>10}{'non-extr':>10}{'worst':>9}  worst arm")
    for row in grid:
        print(f"  {row['variant']:<16}{row['patience']:>4}{row['margin']:>8g}"
              f"{_p(row['mean_score']):>9}{_p(row['mean_gain']):>10}"
              f"{row['fires']:>7}{_p(row['extreme_mean_gain']):>10}"
              f"{_p(row['nonextreme_mean_score']):>10}"
              f"{_p(row['worst_arm_gain']):>9}  {row['worst_arm']}")
    for variant in VARIANTS:
        chosen = best_cell(grid, variant, "mean_score")
        honest = best_cell(grid, variant, "nonextreme_mean_score")
        print(f"  {variant}: best on all arms k={chosen['patience']} "
              f"eps={chosen['margin']:g}; best with the extremes held out "
              f"k={honest['patience']} eps={honest['margin']:g}")


def show_extremes(rows: Sequence[dict]) -> None:
    """The pair the horizon costs most, against the two bounds beside it."""
    print("\nTHE EXTREME ARRANGEMENTS")
    print(f"  {'case':<10}{'fires':>7}{'kept':>7}{'plateau':>10}{'fixed':>10}"
          f"{'vs fixed':>11}{'r*':>5}{'oracle':>10}{'proxy r':>9}{'proxy':>10}")
    for row in rows:
        print(f"  {row['cell']:<10}{_n(row['fire_round']):>7}{row['kept_round']:>7}"
              f"{_p(row['kept_score']):>10}{_p(row['final_score']):>10}"
              f"{_p(row['gain']):>11}{row['oracle_round']:>5}"
              f"{_p(row['oracle_score']):>10}{_n(row['proxy_round']):>9}"
              f"{_p(row['proxy_score']):>10}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="The study root ($FOA_STUDY_DIR).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Directory the four CSV views are written into. "
                         "Without it nothing is written and the tables are "
                         "only printed.")
    ap.add_argument("--study-tag", default="d01")
    ap.add_argument("--min-folds", type=int, default=5,
                    help="Arms with fewer folds than this are not reported.")
    args = ap.parse_args()

    staged = verdicts(args.root, args.study_tag, args.min_folds)
    if not staged:
        print("no hundred-round arm on disk under this root.")
        return 1

    rows = outcomes(staged, PRIMARY_PATIENCE, PRIMARY_MARGIN, CHECKPOINT_BEST)
    stages = stage_rows(rows, PRIMARY_PATIENCE, PRIMARY_MARGIN, CHECKPOINT_BEST)
    grid = grid_rows(staged)
    extremes = extreme_rows(rows, staged, one_rule([v for _, v in staged]))

    show_arms(rows)
    show_stages(stages)
    show_grid(grid)
    if extremes:
        show_extremes(extremes)

    if args.out:
        print()
        write_csv(rows, ARM_COLUMNS, args.out / "plateau_arms.csv")
        write_csv(stages, STAGE_COLUMNS, args.out / "plateau_stages.csv")
        write_csv(grid, GRID_COLUMNS, args.out / "plateau_grid.csv")
        write_csv(extremes, EXTREME_COLUMNS, args.out / "plateau_extremes.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
