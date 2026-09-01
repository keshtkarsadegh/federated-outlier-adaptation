#!/usr/bin/env python3
"""
The per-round views the manuscript's accuracy-over-rounds figures are drawn on.

Every table in this study is a number per arm. The six figures are a number per
arm PER ROUND, and until now the CSVs behind them were cut by hand at a
terminal, which is the one thing `report_tables.py` exists to make impossible.
This tool cuts them from the same stored runs, so a figure in the manuscript is
reachable from the published records by a command rather than by a memory of
one.

    python tools/export_traces.py --root "$FOA_STUDY_DIR" --out paper/data
    python tools/export_traces.py --root "$FOA_STUDY_DIR" --out paper/data \\
        --what traces_extreme

WHAT A ROW IS. Round 0 carries the shipped model's own two accuracies - `P0` on
the source population, `A0` on the cohort - read through
``report_tables.baselines``, so every arm of a figure starts from one shared
point and the curves show only what adaptation then spends and buys. Rounds 1
onward are the fold-mean over the five folds of `source_val_accuracies` and
`pool_val_accuracies`, truncated to the shortest fold by
``extreme_stopping.fold_mean``.

VALIDATION, NOT TEST. These are the halves a selection is allowed to see, and
they are not the test-set numbers the tables report. The two must not be quoted
against each other, and the figures say so in their captions.

WHICH ARMS ARE DRAWN IS READ, NOT TYPED. Every arm list comes out of the frozen
records under `tables/`:

    traces_aggfull        `p12_agg_top3.json`      the three server rules
    traces_regfull        `p13_reg_method_winners.json`  one arm per penalty family
    traces_blends         `p14_hybrid_construction*.json`  each blend and its two parents
    traces_combo          `p15_stage_winner.json`  the crowned pair, split back into halves
    traces_extreme        `extreme_stopping.CASES`  the three arrangements
    extreme_stop_rounds   `stopping_table`, whole-study rule

Naming them here instead would let a figure outlive the selection that put the
arm in it - the failure `compare_arms.py` documents at length for the blends,
where a second naming of the same cells left every paired row empty while the
runs sat on disk.

THE CONTROL AND THE TWO SCHEDULES are the only things stated rather than read.
The control is plain federated averaging, `control_fedavg`, which is the arm
every server rule was selected over; the aggregation, combination and extreme
views are drawn on the parallel (concurrent) schedule and the regularisation
view on the cyclic (sequential) one, which is what the manuscript reports. That
the parallel choice still matches the record is checked, not assumed: the
crowned combination's own schedule has to be the one named here or this tool
refuses.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))

from extreme_stopping import CASES, fold_mean  # noqa: E402
from report_tables import baselines, cell_and_family  # noqa: E402

#: The two series every curve is drawn from. Both are validation halves.
SERIES = ("pool_val_accuracies", "source_val_accuracies")

#: How a selection record's family name maps onto the schedule a run folder is
#: told apart by. ``cell_and_family`` answers "parallel"/"cyclic"; the records
#: were written with the runner's own "concurrent"/"sequential".
SCHEDULE = {"concurrent": "parallel", "sequential": "cyclic"}

#: The unprotected arm every server rule was selected over. Not in any
#: shortlist, because a shortlist is what a screen chose and this is what it
#: chose against.
CONTROL = "control_fedavg"

#: The schedule each group of figures is drawn on, named because the manuscript
#: names it. Checked against the crowned combination's own family below.
AGG_FAMILY = "concurrent"
REG_FAMILY = "sequential"


# ------------------------------------------------------------------ records
class Records:
    """The stored round series of one study root, read once per prefix."""

    def __init__(self, root: Path, tag: str) -> None:
        self.root = root
        self.tag = tag
        self._cache: Dict[str, dict] = {}

    def prefix(self, stem: str) -> str:
        return f"{self.tag}_{stem}_"

    def _read(self, prefix: str) -> dict:
        """``{(cell, schedule): {series: [per-fold list, ...]}}`` under a prefix."""
        found: Dict[Tuple[str, str], Dict[str, List[List[float]]]] = defaultdict(
            lambda: defaultdict(list))
        pattern = f"{self.root}/{prefix}*/**/summary_0.json"
        for path in sorted(glob.glob(pattern, recursive=True)):
            cell, family = cell_and_family(Path(path), prefix)
            if cell is None:
                continue
            body = next(iter(json.loads(Path(path).read_text()).values()))
            for name in SERIES:
                values = [v for v in (body.get(name) or []) if v is not None]
                if values:
                    found[(cell, family)][name].append(values)
        return found

    def traces(self, prefix: str) -> dict:
        if prefix not in self._cache:
            self._cache[prefix] = self._read(prefix)
        return self._cache[prefix]

    def arm(self, stem: str, cell: str, family: str) -> Tuple[List[float], List[float]]:
        """One arm's two fold-mean series, ``(source, cohort)``."""
        prefix = self.prefix(stem)
        stored = self.traces(prefix).get((cell, SCHEDULE[family]))
        if not stored or not all(stored.get(name) for name in SERIES):
            raise SystemExit(
                f"FATAL: no {family} run of {cell!r} under {self.root}/{prefix}*. "
                "The arm is named by a selection record, so a missing one means "
                "the records and the runs have come apart - it is not guessed."
            )
        return (fold_mean(stored["source_val_accuracies"]),
                fold_mean(stored["pool_val_accuracies"]))

    def table(self, name: str) -> dict:
        path = self.root / "tables" / name
        if not path.is_file():
            raise SystemExit(
                f"FATAL: {path} is missing. Which arms a figure draws is written "
                "there and nowhere else; inferring them from folder names would "
                "draw a figure the selection never chose."
            )
        return json.loads(path.read_text())


# --------------------------------------------------------------- arm lists
def agg_arms(records: Records, family: str) -> List[str]:
    """The three server rules, in the order the shortlist ranked them.

    ``top`` holds the same three as a set; the ranking is what carries their
    order, and a figure's legend is read in order.
    """
    ranking = records.table("p12_agg_top3.json")["rankings"][family]["val"]
    return [entry["id"] for entry in ranking[:3]]


def reg_arms(records: Records, family: str) -> List[str]:
    """One arm per penalty family: that family's own winner under this schedule."""
    winners = records.table("p13_reg_method_winners.json")
    return [body["winner"] for key, body in winners.items()
            if key.endswith(f"/{family}")]


def blend_groups(records: Records) -> List[Tuple[str, List[str]]]:
    """Each schedule's two parents and the three mixtures built from them."""
    # study_emit is what names a blend's cells; naming them a second time here
    # is how the folders on disk stop being findable. Imported inside the view,
    # as compare_arms does, so the other views keep running in a clone where
    # that module's package is not installed.
    from study_emit import _hybrid_records, hybrid_id

    groups: List[Tuple[str, List[str]]] = []
    for family, path in _hybrid_records(records.root):
        if not path.is_file():
            continue
        record = json.loads(path.read_text())
        mixes = record.get("mixes")
        if not mixes:
            raise SystemExit(
                f"FATAL: {path} names no mixes, so which blend cells it produced "
                "cannot be known."
            )
        groups.append((family, [record["kd_winner"], record["fisher_winner"]]
                       + [hybrid_id(family, mix) for mix in mixes]))
    if not groups:
        raise SystemExit(
            f"FATAL: no blend construction record under {records.root / 'tables'}."
        )
    return groups


def crowned(records: Records) -> Tuple[str, str, str, str]:
    """``(family, combination, server rule, penalty)`` of the crowned pair.

    The winner is one id; which two cells it was built from is recovered from
    the two shortlists the cross was emitted from, so the halves a figure draws
    beside it are the halves the stage actually crossed.
    """
    winner = records.table("p15_stage_winner.json")["rankings"]["val"][0]["id"]
    agg_top = records.table("p12_agg_top3.json")["top"]
    reg_top = records.table("p13_reg_top3.json")
    for family in SCHEDULE:
        for agg_id in agg_top.get(family, []):
            for reg_id in reg_top.get(family, []):
                if f"{agg_id}_{reg_id}" == winner:
                    return family, winner, agg_id, reg_id
    raise SystemExit(
        f"FATAL: the crowned combination {winner!r} is not a pair from "
        "p12_agg_top3.json x p13_reg_top3.json. The cross is defined by the "
        "shortlists it was emitted from, and splitting the id on an underscore "
        "instead would invent halves no stage ran."
    )


# ------------------------------------------------------------------ writing
def write_traces(path: Path, columns: Sequence[Tuple[str, List[float], List[float]]],
                 a0: float, p0: float) -> None:
    """One CSV: round 0 the shipped model, then the fold means round by round."""
    rounds = min(min(len(src), len(cohort)) for _, src, cohort in columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["round"]
        for name, _, _ in columns:
            header += [f"{name}_src", f"{name}_cohort"]
        writer.writerow(header)
        # ROUND 0 IS THE SHIPPED MODEL, one shared point for every arm.
        row: List[object] = [0]
        for _ in columns:
            row += [p0, a0]
        writer.writerow(row)
        for index in range(rounds):
            row = [index + 1]
            for _, src, cohort in columns:
                row += [src[index], cohort[index]]
            writer.writerow(row)
    print(f"  -> {path}")


#: The renaming the stopping view carries: the stopping table speaks of an arm's
#: adaptation and preservation, the figures of its cohort and source panels.
STOP_COLUMNS = (
    ("arm", "cell"), ("family", "family"), ("folds", "folds"), ("rounds", "rounds"),
    ("oracle_round", "oracle_round"), ("oracle_src", "oracle_preservation"),
    ("oracle_cohort", "oracle_adaptation"), ("oracle_score", "oracle_score"),
    ("rule_signal", "one_signal"), ("rule_delta", "one_delta"),
    ("rule_round", "one_round"), ("rule_src", "one_preservation"),
    ("rule_cohort", "one_adaptation"), ("rule_score", "one_score"),
    ("final_score", "final_score"),
)


def stop_rounds(root: Path, tag: str, a0: float, p0: float) -> List[dict]:
    """
    The two rounds `fig_extremes` marks, for each extreme arrangement.

    Both come out of `stopping_table` rather than from a rule restated here: the
    oracle round is the round maximising the study's own selection score along
    the trace, and the rule round is the round the ONE permitted rule fires -
    the single `(signal, budget)` chosen over every arm of every hundred-round
    stage. Choosing it on the extremes alone would be choosing a rule on the
    three runs it is about to be reported on, so the whole pool is read.
    """
    import stopping_table as stopping

    staged = []
    for stem, _ in stopping.STAGES:
        verdicts = []
        for (cell, family), stored in sorted(
                stopping.read_arms(root, f"{tag}_{stem}").items()):
            arm = stopping.arm_of(cell, family, stored)
            if arm is None or arm.folds < 5 or arm.rounds < 2:
                continue
            verdicts.append(stopping.judge(arm, a0, p0, stopping.DEFAULT_DELTAS))
        staged.append((stem.strip("_"), verdicts))

    every = [verdict for _, verdicts in staged for verdict in verdicts]
    if not every:
        raise SystemExit("FATAL: no hundred-round arm on disk under this root.")
    rule = stopping.one_rule(every)
    rows = {row["cell"]: row
            for stage, verdicts in staged if stage == "extreme"
            for row in (stopping.row_of(verdict, stage, rule) for verdict in verdicts)}

    out = []
    for case in CASES:
        if case not in rows:
            raise SystemExit(f"FATAL: the extreme arrangement {case!r} has not run.")
        row = rows[case]
        entry = {name: row[source] for name, source in STOP_COLUMNS}
        # A0 and P0 travel with the rows: a score is a difference from doing
        # nothing, and a reader who cannot see the two numbers it is measured
        # against cannot check it.
        entry.update({"a0": a0, "p0": p0})
        out.append(entry)
    return out


def write_stop_rounds(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}")


# --------------------------------------------------------------------- views
VIEWS = ("traces_control", "traces_aggfull", "traces_regfull", "traces_blends",
         "traces_combo", "traces_extreme", "extreme_stop_rounds")


def build(records: Records, what: str, a0: float, p0: float) -> Optional[list]:
    """The columns of one trace view, in the order the figure argues them."""
    if what == "traces_control":
        return [(CONTROL,) + records.arm("aggfull", CONTROL, AGG_FAMILY)]

    if what == "traces_aggfull":
        arms = [CONTROL] + agg_arms(records, AGG_FAMILY)
        return [(arm,) + records.arm("aggfull", arm, AGG_FAMILY) for arm in arms]

    if what == "traces_regfull":
        return [(arm,) + records.arm("regfull", arm, REG_FAMILY)
                for arm in reg_arms(records, REG_FAMILY)]

    if what == "traces_blends":
        columns = []
        for family, arms in blend_groups(records):
            columns += [(arm,) + records.arm("regfull", arm, family) for arm in arms]
        return columns

    if what == "traces_combo":
        family, winner, agg_id, reg_id = crowned(records)
        if family != AGG_FAMILY:
            raise SystemExit(
                f"FATAL: the crowned combination is a {family} pair, and the "
                f"aggregation views here are drawn on {AGG_FAMILY}. The two "
                "cannot be plotted on one canvas."
            )
        return [(CONTROL,) + records.arm("aggfull", CONTROL, family),
                (agg_id,) + records.arm("aggfull", agg_id, family),
                (reg_id,) + records.arm("regfull", reg_id, family),
                (winner,) + records.arm("combo", winner, family)]

    if what == "traces_extreme":
        return [(case,) + records.arm("extreme", case, AGG_FAMILY) for case in CASES]

    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="The study root ($FOA_STUDY_DIR).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Directory the CSV views are written into.")
    ap.add_argument("--what", default="all", choices=("all",) + VIEWS)
    ap.add_argument("--study-tag", default="d01")
    args = ap.parse_args()

    a0, p0 = baselines(args.root)
    print(f"reference: the shipped model  adapt {a0:.4f}  preserve {p0:.4f}")
    print("basis: VALIDATION (pool_val_accuracies against source_val_accuracies), "
          "which is what a selection is allowed to read.")

    records = Records(args.root, args.study_tag)
    for what in (VIEWS if args.what == "all" else (args.what,)):
        if what == "extreme_stop_rounds":
            write_stop_rounds(args.out / f"{what}.csv",
                              stop_rounds(args.root, args.study_tag, a0, p0))
            continue
        write_traces(args.out / f"{what}.csv", build(records, what, a0, p0), a0, p0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
