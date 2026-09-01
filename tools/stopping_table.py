#!/usr/bin/env python3
"""
What the fixed horizon cost every arm, and how much of it a signal could save.

Every stage in this study runs to a fixed hundred-round horizon and reports the
last round. `extreme_stopping.py` showed what that costs on the three extreme
arrangements - up to twenty-seven points of score spent after the best trade had
already been reached. This tool asks the same question of every arm of every
hundred-round stage, and adds the column the extreme tool does not have: what a
rule that respects the study's constraint would actually have delivered.

    python tools/stopping_table.py --root "$FOA_STUDY_DIR"
    python tools/stopping_table.py --root "$FOA_STUDY_DIR" --stage extreme
    python tools/stopping_table.py --root "$FOA_STUDY_DIR" --csv out/

THREE COLUMNS, AND ONLY ONE OF THEM IS DEPLOYABLE.

    FINAL      the last round: what the tables report.
    ORACLE     the round with the best score, chosen by looking at the whole
               run. It reads `source_val_accuracies`, which is the source
               population - the data a shipped system no longer has. It is a
               bound on what stopping is worth, never a method.
    STOPPED    the first round at which a PERMITTED signal's drift exceeds a
               budget. The eight signals of `runners/forgetting_signals` are
               what a server may compute with no source data at all, so a rule
               built on one of them is a rule that could have run online.

The oracle is reported to price the horizon; the stopped column is reported to
say how much of that price a constrained system could have avoided.

ONE RULE FOR THE WHOLE STUDY. The best (signal, budget) *per arm* is printed
because it bounds what the eight signals contain, but it is not deployable
either: choosing it needs the source split, exactly as the oracle round does. So
the last column fixes a SINGLE (signal, budget) for every arm of every stage -
chosen by the mean score it stops at - and reports what that one rule delivers
arm by arm. That number is the honest one, and it is allowed to be worse than
the horizon: whichever way it comes out is the finding.

BASIS AND CONVENTIONS. Everything here reads the VALIDATION columns -
`pool_val_accuracies` against `source_val_accuracies` - because a stopping rule
is a selection and a selection may only see what a selection is allowed to see.
The score is the study's own rule,

    score = (adaptation - A0) - (P0 - preservation)

with A0 and P0 read from the study's shipped-model evaluations through
`report_tables.baselines`. Rounds are one-based over the stored series, whose
first entry is the shipped model before any client has trained - the same
numbering `extreme_stopping.py` prints, and the reason a permitted rule can
never stop before round 2. The fold mean, the oracle round and the drift-stop
semantics are all imported rather than restated, from `extreme_stopping` and
from `analysis.forgetting_signals`, so the three tools cannot drift apart: run
this with `--stage extreme` and the three rows reproduce `extreme_stopping.py`
round for round.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))

from extreme_stopping import fold_mean, oracle_round, scores  # noqa: E402
from report_tables import baselines, cell_and_family  # noqa: E402

from federated_outlier_adaptation.analysis.forgetting_signals import (  # noqa: E402
    DEFAULT_DELTAS,
    RunTrajectory,
    stop_round,
)
from federated_outlier_adaptation.runners.forgetting_signals import (  # noqa: E402
    SIGNAL_KEYS,
)

#: The hundred-round stages, in the order they are reported: the three stages
#: that searched something, then the four federations the winners were carried
#: into, then the extremes. The twenty-five-round screens are deliberately
#: absent - a screen ranks and does not report, and a stopping round read off
#: one would be a claim about a horizon nothing was measured at.
STAGES = (
    ("aggfull_", "AGGREGATION WINNERS (full horizon)"),
    ("regfull_", "REGULARISATION WINNERS (full horizon)"),
    ("combo_", "COMBINATIONS: server rule x client penalty"),
    ("c10d10_", "TEN CLIENTS, ONE DROPPED - the study's own setting"),
    ("five_", "FIVE CLIENTS, ONE DROPPED"),
    ("c20d10_", "TWENTY CLIENTS, TWO DROPPED"),
    ("c20d20_", "TWENTY CLIENTS, FOUR DROPPED"),
    ("extreme_", "THE EXTREME CASES (full participation)"),
)

#: The two series every decision here is made on. Both are validation halves.
SERIES = ("pool_val_accuracies", "source_val_accuracies")


# --------------------------------------------------------------- an arm
@dataclass
class Arm:
    """One configuration's fold-mean traces, and the signals beside them."""

    cell: str
    family: str
    folds: int
    adaptation: List[float]
    preservation: List[float]
    signals: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def rounds(self) -> int:
        return min(len(self.adaptation), len(self.preservation))


def read_arms(root: Path, prefix: str) -> Dict[Tuple[str, str], Dict[str, List[List[float]]]]:
    """``{(cell, family): {series: [per-fold list, ...]}}`` from the summaries."""
    found: Dict[Tuple[str, str], Dict[str, List[List[float]]]] = defaultdict(
        lambda: defaultdict(list))
    for path in sorted(glob.glob(f"{root}/{prefix}*/**/summary_0.json", recursive=True)):
        cell, family = cell_and_family(Path(path), prefix)
        if cell is None:
            continue
        body = next(iter(json.loads(Path(path).read_text()).values()))
        for name in tuple(SERIES) + tuple(SIGNAL_KEYS):
            values = [v for v in (body.get(name) or []) if v is not None]
            if values:
                found[(cell, family)][name].append(values)
    return {key: dict(value) for key, value in found.items()}


def arm_of(cell: str, family: str, stored: Dict[str, List[List[float]]]) -> Optional[Arm]:
    """The fold-mean traces of one arm, or None when it carries no trade to read."""
    if not stored.get("pool_val_accuracies") or not stored.get("source_val_accuracies"):
        return None
    return Arm(
        cell=cell,
        family=family,
        folds=len(stored["pool_val_accuracies"]),
        adaptation=fold_mean(stored["pool_val_accuracies"]),
        preservation=fold_mean(stored["source_val_accuracies"]),
        signals={key: fold_mean(stored[key]) for key in SIGNAL_KEYS if stored.get(key)},
    )


# ----------------------------------------------------------- the three answers
def trajectory(arm: Arm) -> RunTrajectory:
    """
    The analysis module's own trajectory object, over this arm's fold means.

    Built rather than reimplemented so that `drift` and `stop_round` mean here
    exactly what they mean in `foa signals`: a distance is read as its rise and
    an accuracy as its drop, both from the run's own first entry, so one budget
    means the same thing for every signal.
    """
    rounds = arm.rounds
    return RunTrajectory(
        name=f"{arm.cell}/{arm.family}",
        source="fold-mean",
        method=arm.family,
        adaptation=list(arm.adaptation[:rounds]),
        source_val=list(arm.preservation[:rounds]),
        signals={key: list(values) for key, values in arm.signals.items()},
    )


def rule_stops(arm: Arm, deltas: Sequence[float]) -> Dict[Tuple[str, float], int]:
    """The zero-based round each permitted ``(signal, budget)`` rule stops at."""
    traj = trajectory(arm)
    stops: Dict[Tuple[str, float], int] = {}
    for signal in SIGNAL_KEYS:
        drift = traj.drift(signal)
        if all(value is None for value in drift):
            continue
        for delta in deltas:
            stops[(signal, delta)] = stop_round(drift, delta)
    return stops


@dataclass
class Verdict:
    """One arm's three answers: the horizon, the oracle, and every rule."""

    arm: Arm
    values: List[float]
    star: int
    stops: Dict[Tuple[str, float], int]

    def at(self, index: int) -> dict:
        """Round label, both accuracies and the score, at a zero-based index."""
        return {
            "round": index + 1,
            "adaptation": self.arm.adaptation[index],
            "preservation": self.arm.preservation[index],
            "score": self.values[index],
        }

    @property
    def final(self) -> dict:
        return self.at(len(self.values) - 1)

    @property
    def oracle(self) -> dict:
        return self.at(self.star - 1)


def judge(arm: Arm, a0: float, p0: float, deltas: Sequence[float]) -> Verdict:
    """Score every round of an arm and locate the oracle stop among them."""
    values = scores(arm.adaptation, arm.preservation, a0, p0)
    star, _, _ = oracle_round(arm.adaptation, arm.preservation, a0, p0)
    return Verdict(arm=arm, values=values, star=star, stops=rule_stops(arm, deltas))


def best_rule(verdict: Verdict) -> Optional[Tuple[str, float]]:
    """
    The ``(signal, budget)`` this arm would have wanted, had it been allowed to
    choose one per arm.

    Ties go to the rule that stops EARLIER: two rules that reach the same score
    are not equally good, because the later one spent more rounds of the source
    model arriving at it. Not deployable - picking the best rule per arm needs
    the same source split the oracle round needs - and printed as a bound on
    what the eight signals contain.
    """
    if not verdict.stops:
        return None
    return max(
        verdict.stops,
        key=lambda rule: (verdict.values[verdict.stops[rule]], -verdict.stops[rule]),
    )


def one_rule(verdicts: Sequence[Verdict]) -> Optional[Tuple[str, float]]:
    """
    The single ``(signal, budget)`` a deployment would have to fix in advance.

    Scored as the mean, over every arm, of the score it stops at; ties go to the
    rule that stops earlier on average. Only rules that are defined on every arm
    are eligible, because a rule that is silent on a stage has not been priced on
    it. The mean is taken over arms rather than over runs so that a stage with
    twenty arms does not outvote a stage with three by weight of arithmetic
    alone; A0 and P0 shift every arm's score by the same constant, so which rule
    wins does not depend on them at all.
    """
    totals: Dict[Tuple[str, float], List[Tuple[float, int]]] = defaultdict(list)
    for verdict in verdicts:
        for rule, index in verdict.stops.items():
            totals[rule].append((verdict.values[index], index))
    complete = {rule: pairs for rule, pairs in totals.items() if len(pairs) == len(verdicts)}
    if not complete:
        return None
    return max(
        complete,
        key=lambda rule: (
            st.mean(score for score, _ in complete[rule]),
            -st.mean(index for _, index in complete[rule]),
        ),
    )


def row_of(verdict: Verdict, stage: str, rule: Optional[Tuple[str, float]]) -> dict:
    """One arm's whole line: final, oracle, its own best rule, and the one rule."""
    best = best_rule(verdict)
    final, oracle = verdict.final, verdict.oracle
    row = {
        "stage": stage,
        "cell": verdict.arm.cell,
        "family": verdict.arm.family,
        "folds": verdict.arm.folds,
        "rounds": len(verdict.values),
        "final_round": final["round"],
        "final_adaptation": final["adaptation"],
        "final_preservation": final["preservation"],
        "final_score": final["score"],
        "oracle_round": oracle["round"],
        "oracle_adaptation": oracle["adaptation"],
        "oracle_preservation": oracle["preservation"],
        "oracle_score": oracle["score"],
        "oracle_cost": oracle["score"] - final["score"],
    }
    for tag, choice in (("best", best), ("one", rule)):
        if choice is None or choice not in verdict.stops:
            row.update({f"{tag}_{k}": None for k in
                        ("signal", "delta", "round", "adaptation", "preservation",
                         "score", "vs_final")})
            continue
        at = verdict.at(verdict.stops[choice])
        row.update({
            f"{tag}_signal": choice[0],
            f"{tag}_delta": choice[1],
            f"{tag}_round": at["round"],
            f"{tag}_adaptation": at["adaptation"],
            f"{tag}_preservation": at["preservation"],
            f"{tag}_score": at["score"],
            f"{tag}_vs_final": at["score"] - final["score"],
        })
    return row


COLUMNS = (
    "stage", "cell", "family", "folds", "rounds",
    "final_round", "final_adaptation", "final_preservation", "final_score",
    "oracle_round", "oracle_adaptation", "oracle_preservation", "oracle_score",
    "oracle_cost",
    "best_signal", "best_delta", "best_round", "best_adaptation",
    "best_preservation", "best_score", "best_vs_final",
    "one_signal", "one_delta", "one_round", "one_adaptation",
    "one_preservation", "one_score", "one_vs_final",
)


# ------------------------------------------------------------------ printing
def _p(value: Optional[float]) -> str:
    """A score in points, or a dash when the rule was not defined."""
    return "-" if value is None else f"{value * 100:.2f}p"


def _n(value) -> str:
    """A budget or a round, or a dash when there is none."""
    return "-" if value is None else (f"{value:g}" if isinstance(value, float) else str(value))


def show_horizon(rows: Sequence[dict], title: str, a0: float, p0: float) -> None:
    """The fixed horizon against the oracle stop, one line per arm."""
    width = max([len(r["cell"]) for r in rows] + [len("cell")]) + 2
    print(f"\n{title}")
    print(f"reference: the shipped model  adapt {a0:.4f}  preserve {p0:.4f}")
    print("basis: VALIDATION (pool_val_accuracies against source_val_accuracies), "
          "which is what a stopping rule is allowed to read.")
    print(f"{'cell':<{width}}{'sched':<10}{'adapt':>8}{'preserve':>10}{'final':>9}"
          f"{'r*':>5}{'adapt':>8}{'preserve':>10}{'oracle':>9}{'cost':>9}")
    print("-" * (width + 78))
    for r in rows:
        print(f"{r['cell']:<{width}}{r['family']:<10}"
              f"{r['final_adaptation']:>8.4f}{r['final_preservation']:>10.4f}"
              f"{_p(r['final_score']):>9}"
              f"{r['oracle_round']:>5}"
              f"{r['oracle_adaptation']:>8.4f}{r['oracle_preservation']:>10.4f}"
              f"{_p(r['oracle_score']):>9}{_p(r['oracle_cost']):>9}")


def show_rules(rows: Sequence[dict], title: str) -> None:
    """What a permitted signal delivered, per arm and under the one fixed rule."""
    width = max([len(r["cell"]) for r in rows] + [len("cell")]) + 2
    print(f"\n{title} - WHAT A PERMITTED SIGNAL DELIVERS")
    print(f"{'cell':<{width}}{'sched':<10}{'best signal':<28}{'budget':>9}"
          f"{'r':>5}{'stopped':>9}{'vs final':>10}"
          f"{'r':>7}{'one rule':>10}{'vs final':>10}")
    print("-" * (width + 98))
    for r in rows:
        print(f"{r['cell']:<{width}}{r['family']:<10}"
              f"{_n(r['best_signal']):<28}{_n(r['best_delta']):>9}"
              f"{_n(r['best_round']):>5}{_p(r['best_score']):>9}"
              f"{_p(r['best_vs_final']):>10}"
              f"{_n(r['one_round']):>7}{_p(r['one_score']):>10}"
              f"{_p(r['one_vs_final']):>10}")
    silent = sum(1 for r in rows if r["best_round"] == r["rounds"])
    if silent:
        print(f"  {silent} of {len(rows)} arms are best served by a rule that never "
              f"fires - the horizon is already their best round, so no earlier stop "
              f"can beat it. The signal named on those lines is the first rule in the "
              f"grid that stays silent, not a rule that did any work.")


def write_csv(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in COLUMNS})
    print(f"  -> {path}")


def show_gaps(rows: Sequence[dict]) -> None:
    """
    How far the horizon sits below the oracle, stage by stage.

    This is the table that decides whether stopping is a footnote or a finding:
    a stage whose worst arm loses a tenth of a point to the oracle does not need
    a stopping rule, and a stage whose median arm loses ten does.
    """
    print("\nWHAT THE HORIZON COSTS, BY STAGE (oracle score minus final score)")
    print(f"  {'stage':<12}{'arms':>6}{'min':>10}{'median':>10}{'max':>10}"
          f"{'r* median':>12}")
    by_stage: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_stage[row["stage"]].append(row)
    for stage, entries in by_stage.items():
        costs = sorted(r["oracle_cost"] for r in entries)
        stars = sorted(r["oracle_round"] for r in entries)
        print(f"  {stage:<12}{len(entries):>6}{_p(costs[0]):>10}"
              f"{_p(st.median(costs)):>10}{_p(costs[-1]):>10}"
              f"{st.median(stars):>12.0f}")


def show_one_rule(rows: Sequence[dict], rule: Optional[Tuple[str, float]]) -> None:
    """The deployable answer: one rule, fixed in advance, priced per stage."""
    if rule is None:
        print("\nno permitted rule is defined on every arm; nothing to fix in advance.")
        return
    signal, delta = rule
    print(f"\nTHE ONE RULE, FIXED FOR THE WHOLE STUDY: stop when {signal} "
          f"has drifted more than {delta:g} from its round-1 value")
    print("  it is chosen on the mean score it stops at, over every arm of every "
          "stage, and it is the only column here a deployed system could run.")
    print("  the mean is what the rule is chosen on; the worst arm is what it "
          "costs somebody, and it is printed beside the mean for that reason.")
    print(f"  {'stage':<12}{'arms':>6}{'final':>10}{'one rule':>11}"
          f"{'vs final':>11}{'worst arm':>12}{'best arm':>11}{'oracle':>10}"
          f"{'r median':>11}")

    def line(label: str, priced: Sequence[dict]) -> None:
        gaps = sorted(r["one_vs_final"] for r in priced)
        print(f"  {label:<12}{len(priced):>6}"
              f"{_p(st.mean(r['final_score'] for r in priced)):>10}"
              f"{_p(st.mean(r['one_score'] for r in priced)):>11}"
              f"{_p(st.mean(gaps)):>11}{_p(gaps[0]):>12}{_p(gaps[-1]):>11}"
              f"{_p(st.mean(r['oracle_score'] for r in priced)):>10}"
              f"{st.median(r['one_round'] for r in priced):>11.0f}")

    by_stage: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_stage[row["stage"]].append(row)
    for stage, entries in by_stage.items():
        priced = [r for r in entries if r["one_score"] is not None]
        if priced:
            line(stage, priced)
    line("ALL", [r for r in rows if r["one_score"] is not None])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--study-tag", default="d01")
    ap.add_argument("--stage", action="append", default=None,
                    choices=[stem.strip("_") for stem, _ in STAGES],
                    help="Stage to report; repeatable. Default: all eight.")
    ap.add_argument("--deltas", type=float, nargs="*", default=None,
                    help="Budget grid the permitted rules are swept over "
                         "(default: the grid foa signals uses).")
    ap.add_argument("--min-folds", type=int, default=5,
                    help="Arms with fewer folds than this are not reported.")
    ap.add_argument("--csv", type=Path, default=None, help="Also write CSVs here.")
    args = ap.parse_args()

    deltas = tuple(args.deltas) if args.deltas else DEFAULT_DELTAS
    a0, p0 = baselines(args.root)
    wanted = [(stem, title) for stem, title in STAGES
              if args.stage is None or stem.strip("_") in args.stage]

    staged: List[Tuple[str, str, List[Verdict]]] = []
    for stem, title in wanted:
        prefix = f"{args.study_tag}_{stem}"
        verdicts = []
        for (cell, family), stored in sorted(read_arms(args.root, prefix).items()):
            arm = arm_of(cell, family, stored)
            if arm is None or arm.folds < args.min_folds or arm.rounds < 2:
                continue
            verdicts.append(judge(arm, a0, p0, deltas))
        staged.append((stem.strip("_"), title, verdicts))

    every = [verdict for _, _, verdicts in staged for verdict in verdicts]
    if not every:
        print("no hundred-round arm on disk under this root.")
        return 1
    rule = one_rule(every)

    rows: List[dict] = []
    for stage, title, verdicts in staged:
        if not verdicts:
            print(f"\n{title}\n  nothing on disk yet")
            continue
        stage_rows = [row_of(verdict, stage, rule) for verdict in verdicts]
        stage_rows.sort(key=lambda r: -r["final_score"])
        show_horizon(stage_rows, title, a0, p0)
        show_rules(stage_rows, title)
        rows += stage_rows
        if args.csv:
            write_csv(stage_rows, args.csv / f"stopping_{stage}.csv")

    show_gaps(rows)
    show_one_rule(rows, rule)
    if args.csv:
        write_csv(rows, args.csv / "stopping_all.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
