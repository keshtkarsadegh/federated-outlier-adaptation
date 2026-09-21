#!/usr/bin/env python3
"""
The axis the study selected on, beside the axis it reports on.

Every shipped table is TEST. Every shortlist that decided which arms those
tables contain was cut on VALIDATION, because a selection may only see the rows
the protocol lets it see. Both halves of that sentence are in the manuscript and
neither has a view: `p12_agg_top3.json`, `p14_reg_top3.json` and
`p15_stage_winner.json` carry the validation ordering that actually did the
choosing, and they carry it inside JSON records nothing the paper sets ever
reads.

    python tools/export_selection_axis.py --root "$FOA_STUDY_DIR" \\
        --csv "$FOA_STUDY_DIR/tables/paper/"

WHY IT MATTERS THAT THE TWO ORDERS DIFFER. On the parallel schedule the
aggregation record's validation order puts the server anchor first and its test
order puts the server step first, so the rule the study carried into the
combination cross is not the rule that leads the table the paper prints. That is
not an error and it is not a coincidence either: it is what selecting on one
axis and reporting on another looks like from the inside, and a reader who only
ever sees the test table has no way to tell that the two disagree. The cyclic
side agrees at the head and disagrees below it, which is the other half of the
same point.

WHAT IS READ, NOT TYPED. Every arm here is the head of a record's own ranking,
read under the basis the record names in its `rank_by` field, and no arm is
named in this file. `rank_by` is checked rather than assumed: a record re-cut on
test would silently turn this view into two copies of one column, and the
refusal below says so instead.

THE SHORTLIST IS THE RECORD'S OWN, AND IS NOT THE HEAD OF ITS RANKING. Each
screen record carries a `top` field, and on the parallel penalty side that field
is not the first three rows of the validation order: the hybrid competes for a
slot of its own, which is what that record's `rule` says and what put an arm
ranked fifth into the cross ahead of the arms ranked third and fourth. Reading
the head of the ranking instead would print a shortlist the study never carried,
so `top` is what is read and the validation rank is printed beside each arm - a
rank of 5 in a three-arm block is the own-slot rule, visible.

THE NUMBERS ARE THE RECORDS' OWN. Each entry of a ranking carries the fold-mean
validation adaptation it was ranked by and the fold-mean test adaptation of the
same runs, so both columns of this view come out of one file and cannot drift
apart. Preservation and score are deliberately NOT here: `agg-winners.csv`,
`reg-winners.csv` and `combos.csv` already carry them per cell, the table that
sets this view joins them by cell, and a second copy is how two numbers begin.

Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

#: How a selection record's family name maps onto the schedule a table is read
#: by; the records were written with the runner's own "concurrent"/"sequential"
#: and every shipped view says "parallel"/"cyclic".
SCHEDULE = {"concurrent": "parallel", "sequential": "cyclic"}

#: How many arms of each schedule the crowning contributes. The two screens
#: contribute the shortlist they actually cut, which each record carries in its
#: own `top` field; the crowning cut no shortlist - it crowned - so its depth is
#: named here, and three matches what a screen carried.
DEPTH = 3

#: The three records, the stage each one selected, and the order this view
#: states them in - which is the order the stages ran.
RECORDS = (("aggregation", "p12_agg_top3.json"),
           ("regularisation", "p14_reg_top3.json"),
           ("combination", "p15_stage_winner.json"))

#: The record that says which schedule ran each combination. A pair id is two
#: cell ids joined by an underscore and both halves carry underscores of their
#: own, so which grid emitted it is read off that grid rather than split out of
#: the string.
COMBINATION_GRID = "p15_combination_grid.json"

COLUMNS = ("stage", "schedule", "val_rank", "cell",
           "val_adaptation", "test_adaptation", "test_rank", "ranked", "crowned")


def _record(root: Path, name: str) -> dict:
    """One selection record, or a refusal that names the file."""
    path = Path(root) / "tables" / name
    if not path.is_file():
        raise SystemExit(
            f"FATAL: no {path}. The validation ordering this view exists to "
            "print is a selection result; recomputing it here would report an "
            "order the study never selected on."
        )
    return json.loads(path.read_text())


def _basis(record: dict, name: str) -> str:
    """The basis a record says it ranked on, refusing anything but validation."""
    basis = record.get("rank_by")
    if basis != "val":
        raise SystemExit(
            f"FATAL: {name} says it ranked on {basis!r}. This view prints the "
            "VALIDATION order beside the test one, and a record cut on test "
            "would make both columns the same column."
        )
    return basis


def _entries(ranking: List[dict]) -> Dict[str, dict]:
    return {entry["id"]: entry for entry in ranking}


def _rows(stage: str, schedule: str, val: List[dict], test: List[dict],
          shortlist: List[str], crowned: str) -> List[dict]:
    """One block: the arms that went forward, at their place in both orderings."""
    val_place = {entry["id"]: rank for rank, entry in enumerate(val, start=1)}
    test_place = {entry["id"]: rank for rank, entry in enumerate(test, start=1)}
    entries = _entries(val)
    out = []
    for cell in sorted(shortlist, key=lambda c: val_place.get(c, len(val) + 1)):
        if cell not in val_place or cell not in test_place:
            raise SystemExit(
                f"FATAL: {stage}/{schedule} carries {cell!r} forward and does "
                "not rank it on both axes. The two orderings are two readings "
                "of one set of runs; an arm in only one of them means they are "
                "not."
            )
        entry = entries[cell]
        out.append({"stage": stage, "schedule": schedule,
                    "val_rank": val_place[cell], "cell": cell,
                    "val_adaptation": "%.6f" % float(entry["adaptation"]),
                    "test_adaptation": "%.6f" % float(entry["adaptation_test"]),
                    "test_rank": test_place[cell], "ranked": len(val),
                    "crowned": "yes" if cell == crowned else "no"})
    return out


def screen_rows(root: Path, stage: str, name: str) -> List[dict]:
    """One shortlist record's two orderings, per schedule."""
    record = _record(root, name)
    basis = _basis(record, name)
    out: List[dict] = []
    for family, schedule in SCHEDULE.items():
        rankings = (record.get("rankings") or {}).get(family) or {}
        val, test = rankings.get(basis) or [], rankings.get("test") or []
        shortlist = (record.get("top") or {}).get(family) or []
        if not val or not test or not shortlist:
            raise SystemExit(
                f"FATAL: {name} carries no {family} ranking or no {family} "
                "shortlist. Each schedule selected separately and the other "
                "one's order is not a substitute for it."
            )
        out += _rows(stage, schedule, val, test, shortlist, crowned="")
    return out


def crowning_rows(root: Path, stage: str, name: str) -> List[dict]:
    """
    The crowning's own two orderings, split back into the two schedules.

    The cross ranked both schedules in one list - it had to, because it crowned
    one pair out of the eighteen - so the per-schedule order this view prints is
    that one list filtered, never re-ranked. Filtering preserves order, and
    re-ranking here would be a third selection nobody ran.
    """
    record = _record(root, name)
    basis = _basis(record, name)
    families = {}
    for family, pairs in (_record(root, COMBINATION_GRID).get("pairs") or {}).items():
        for agg_id, reg_id in pairs:
            families[f"{agg_id}_{reg_id}"] = family
    if not families:
        raise SystemExit(
            f"FATAL: {COMBINATION_GRID} names no pair, so which schedule ran "
            "each combination cannot be read and would have to be guessed from "
            "the identifier."
        )
    rankings = record.get("rankings") or {}
    out: List[dict] = []
    for family, schedule in SCHEDULE.items():
        def mine(key):
            return [e for e in (rankings.get(key) or [])
                    if families.get(e["id"]) == family]
        ordered = mine(basis)
        out += _rows(stage, schedule, ordered, mine("test"),
                     [entry["id"] for entry in ordered[:DEPTH]],
                     crowned=record.get("winner") or "")
    return out


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}  ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="The study root ($FOA_STUDY_DIR).")
    ap.add_argument("--csv", required=True, type=Path,
                    help="Directory the CSV view is written into.")
    args = ap.parse_args()

    rows: List[dict] = []
    for stage, name in RECORDS:
        rows += (crowning_rows(args.root, stage, name) if stage == "combination"
                 else screen_rows(args.root, stage, name))
    disagree = sum(1 for r in rows if int(r["val_rank"]) != int(r["test_rank"]))
    print("basis: col val_adaptation is VALIDATION, col test_adaptation is TEST;")
    print("the ranks are the two orderings of the same runs, per schedule.")
    print(f"{disagree} of {len(rows)} arms sit at a different place in the two.")
    write_csv(args.csv / "selection_axis.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
