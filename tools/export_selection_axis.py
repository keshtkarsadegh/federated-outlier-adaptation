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

THE RANK THAT CHOSE IS THE RANK ON THE SCORE, AND IT IS NOT THE RANK THE
RECORDS STORE. Each record's ``rankings`` block is written by
``study_emit.both_rankings``, which orders by ADAPTATION alone, on both axes.
The choosing was not done there: ``agg_top3`` and ``reg_top3`` call
``ranked_by_trade``, which orders by the selection score of Eq. 2 --- what an
arm gained on the cohort less what it spent on the source population, at
``w = 1``. The two orders are different orders. This view therefore carries
BOTH, per arm and per axis: ``val_rank`` is the record's stored adaptation
place and ``val_score_rank`` is the place in the ordering the selection
actually cut on. A view that printed only the first, as this one did, showed a
shortlist sitting at ranks 1, 2 and 5 and left the reader to guess why.

WHAT IS READ, NOT TYPED. Every arm here is an arm a record ranked, read under
the basis the record names in its `rank_by` field, and no arm is named in this
file. `rank_by` is checked rather than assumed: a record re-cut on test would
silently turn this view into two copies of one column, and the refusal below
says so instead.

EVERY RANKED ARM IS A ROW, NOT ONLY THE SHORTLIST. A shortlist of three whose
ranks read 1, 2 and 5 cannot be reconciled against a table that prints three
rows: the arms at 3 and 4 have to be on the page for the rule that skipped them
to be visible. So the view carries each record's whole ranked population and
marks the arms that went forward in ``role``. The rule that does the skipping
is one cell per METHOD --- a family's three slots are three ideas, not three
settings of one --- with the composite penalty competing for a slot of its own,
and ``method`` is carried per row so that rule is checkable from the view alone
rather than from this docstring.

THE PRESERVATION AND THE SCORE ARE READ FROM THE RUNS, because the records do
not store them. Both axes of both quantities come out of one read of one set of
run folders --- validation from ``pool_val_accuracies`` and
``source_val_accuracies``, test from ``final_evaluation`` --- and the
adaptation that read produces is asserted against the adaptation the record
stored, arm for arm. That assertion is the whole safety of this file: it is
what says the runs being scored here are the runs the selection scored.

THE BASELINES ARE THE SHIPPED MODEL'S OWN, read through ``report_tables`` from
``g0_perfold_evaluations.json`` and ``g0_evaluations.json``, which is where
``study_emit`` read them when it selected. A score is
``(adaptation - A0) - (P0 - preservation)``, and both terms are therefore
differences from doing nothing.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[1]
for entry in (str(Path(__file__).resolve().parent), str(REPO / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import report_tables  # noqa: E402

#: How a selection record's family name maps onto the schedule a table is read
#: by; the records were written with the runner's own "concurrent"/"sequential"
#: and every shipped view says "parallel"/"cyclic".
SCHEDULE = {"concurrent": "parallel", "sequential": "cyclic"}

#: The three records, the stage each one selected, the run-folder stem its arms
#: were run under, and the order this view states them in - which is the order
#: the stages ran.
RECORDS = (("aggregation", "p12_agg_top3.json", "aggfull"),
           ("regularisation", "p14_reg_top3.json", "regfull"),
           ("combination", "p15_stage_winner.json", "combo"))

#: The record that says which schedule ran each combination. A pair id is two
#: cell ids joined by an underscore and both halves carry underscores of their
#: own, so which grid emitted it is read off that grid rather than split out of
#: the string.
COMBINATION_GRID = "p15_combination_grid.json"

COLUMNS = ("stage", "schedule", "cell", "method", "role", "ranked",
           "val_adaptation", "val_preservation", "val_score",
           "val_rank", "val_score_rank",
           "test_adaptation", "test_preservation", "test_score",
           "test_rank", "test_score_rank", "crowned")

#: How close a run read here has to be to the adaptation the record stored
#: before the two are the same measurement. They are written from the same
#: float, so anything but a rounding difference is a different set of runs.
AGREEMENT = 1e-9


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


def _last(series):
    """The final round of a stored series, or None when there is none."""
    values = [v for v in (series or []) if v is not None]
    return values[-1] if values else None


def read_axes(root: Path, prefix: str) -> Dict[tuple, dict]:
    """
    Both axes of every arm under a run-folder prefix, fold-averaged.

    VALIDATION IS ``pool_val_accuracies`` AND ``source_val_accuracies`` - the
    two series a selection is allowed to see - and TEST is the
    ``final_evaluation`` block the reporting tables read. The two are read in
    one pass over one set of folders so that an arm's two readings cannot come
    from two different populations of runs.

    Keyed by ``(cell, schedule)``, exactly as ``report_tables.read_runs`` keys
    its own output: a run folder holds both schedules and the folder's own tag
    decides which half the study selected.
    """
    found: Dict[tuple, Dict[str, list]] = defaultdict(
        lambda: {"val_adaptation": [], "val_preservation": [],
                 "test_adaptation": [], "test_preservation": []})
    for path in sorted(glob.glob(f"{root}/{prefix}*/**/summary_0.json",
                                 recursive=True)):
        cell, family = report_tables.cell_and_family(Path(path), prefix)
        if cell is None or not report_tables.in_study(str(Path(path))):
            continue
        body = next(iter(json.loads(Path(path).read_text()).values()))
        final = body.get("final_evaluation") or {}
        values = {
            "val_adaptation": _last(body.get("pool_val_accuracies")),
            "val_preservation": _last(body.get("source_val_accuracies")),
            "test_adaptation": (final.get("clients") or {}).get("accuracy"),
            "test_preservation": (final.get("old") or {}).get("mean"),
        }
        for key, value in values.items():
            if isinstance(value, (int, float)):
                found[(cell, family)][key].append(float(value))
    return {key: {name: st.fmean(series) for name, series in axes.items() if series}
            for key, axes in found.items()}


def methods(root: Path, stage: str) -> Dict[str, str]:
    """
    ``{cell id: method}`` for one stage's catalogue, or empty for the cross.

    THE METHOD IS THE RULE THAT SKIPPED AN ARM. Both screens take one cell per
    method, so a shortlist whose score ranks read 1, 2 and 5 is the ordering
    with two already-represented methods stepped over - and the only way a
    reader can see that from the view is if the view says which method each arm
    belongs to. The catalogue is the stage's own, not a string split: a penalty
    id and its method share no reliable prefix.

    The cross ranked pairs, not methods; it crowned one arm out of eighteen and
    applied no per-method filter, so it has no catalogue and returns nothing.
    """
    if stage == "combination":
        return {}
    from federated_outlier_adaptation.training import reg_cells  # noqa: E402
    from federated_outlier_adaptation.training import study_lines as SL  # noqa: E402
    if stage == "aggregation":
        from federated_outlier_adaptation.training import agg_cells  # noqa: E402
        return {cell["id"]: SL.agg_method_of(cell)
                for cell in agg_cells.screen_cells()}
    import study_emit  # noqa: E402
    catalogue = reg_cells.screen_cells() + study_emit.hybrid_cells(Path(root))
    return {cell["id"]: cell["method"] for cell in catalogue}


def _block(stage, schedule, entries, test_entries, runs, baselines,
           shortlist, crowned, method_of) -> List[dict]:
    """One record's ranked population for one schedule, in both orderings."""
    a0, p0 = baselines
    val_place = {e["id"]: rank for rank, e in enumerate(entries, start=1)}
    test_place = {e["id"]: rank for rank, e in enumerate(test_entries, start=1)}
    rows = []
    for entry in entries:
        cell = entry["id"]
        if cell not in test_place:
            raise SystemExit(
                f"FATAL: {stage}/{schedule} ranks {cell!r} on validation and "
                "not on test. The two orderings are two readings of one set of "
                "runs; an arm in only one of them means they are not."
            )
        axes = runs.get((cell, schedule))
        if not axes or len(axes) != 4:
            raise SystemExit(
                f"FATAL: {stage}/{schedule} ranks {cell!r} and the runs under "
                "that stage's prefix carry no complete pair of axes for it. "
                "The preservation this view prints is read from those runs, "
                "and there is nothing to read."
            )
        for column, stored in (("val_adaptation", entry["adaptation"]),
                               ("test_adaptation", entry["adaptation_test"])):
            if abs(axes[column] - float(stored)) > AGREEMENT:
                raise SystemExit(
                    f"FATAL: {stage}/{schedule}/{cell}: the record stores "
                    f"{column} {float(stored)!r} and the runs read here give "
                    f"{axes[column]!r}. The selection scored a different set "
                    "of runs than this view is scoring."
                )
        rows.append({
            "stage": stage, "schedule": schedule, "cell": cell,
            "method": method_of.get(cell, ""),
            "role": ("crowned" if cell == crowned
                     else "shortlist" if cell in shortlist else ""),
            "ranked": len(entries),
            "val_adaptation": axes["val_adaptation"],
            "val_preservation": axes["val_preservation"],
            "val_score": ((axes["val_adaptation"] - a0)
                          - (p0 - axes["val_preservation"])),
            "val_rank": val_place[cell],
            "test_adaptation": axes["test_adaptation"],
            "test_preservation": axes["test_preservation"],
            "test_score": ((axes["test_adaptation"] - a0)
                           - (p0 - axes["test_preservation"])),
            "test_rank": test_place[cell],
            "crowned": "yes" if cell == crowned else "no",
        })

    # THE ORDERING THE SELECTING USED. `study_emit.ranked_by_trade` sorts on
    # the score descending with ties broken by cell id, so that an ordering is
    # a property of the numbers and not of the order the folders were read in.
    # Reproduced here rather than imported so that this view can be rebuilt
    # from a clone of the records without the emitter's own dependencies.
    for column, rank_column in (("val_score", "val_score_rank"),
                                ("test_score", "test_score_rank")):
        order = sorted(rows, key=lambda r: (-r[column], r["cell"]))
        for place, row in enumerate(order, start=1):
            row[rank_column] = place
    rows.sort(key=lambda r: r["val_score_rank"])
    return rows


def screen_rows(root, stage, name, prefix, baselines) -> List[dict]:
    """One shortlist record's ranked population, per schedule."""
    record = _record(root, name)
    basis = _basis(record, name)
    runs = read_axes(Path(root), prefix)
    method_of = methods(root, stage)
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
        out += _block(stage, schedule, val, test, runs, baselines,
                      set(shortlist), "", method_of)
    return out


def crowning_rows(root, stage, name, prefix, baselines) -> List[dict]:
    """
    The crowning's own ranked population, split back into the two schedules.

    The cross ranked both schedules in one list - it had to, because it crowned
    one pair out of the eighteen - so the per-schedule order this view prints is
    that one list filtered, never re-ranked. Filtering preserves order, and
    re-ranking here would be a third selection nobody ran.

    THE CROSS CUT NO SHORTLIST. It crowned, once, and the arm it crowned is the
    one row of these eighteen that carries a role. An earlier version of this
    file called the first three of each schedule a shortlist and printed them
    as one; there was no such shortlist, and naming three arms after a
    selection that picked one put two arms on the page under a description the
    record does not support.
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
    runs = read_axes(Path(root), prefix)
    rankings = record.get("rankings") or {}
    crowned = record.get("winner") or ""
    out: List[dict] = []
    for family, schedule in SCHEDULE.items():
        def mine(key):
            return [e for e in (rankings.get(key) or [])
                    if families.get(e["id"]) == family]
        out += _block(stage, schedule, mine(basis), mine("test"), runs,
                      baselines, set(), crowned, {})
    return out


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatted = []
    for row in rows:
        out = dict(row)
        for column in ("val_adaptation", "val_preservation", "val_score",
                       "test_adaptation", "test_preservation", "test_score"):
            out[column] = "%.6f" % row[column]
        formatted.append(out)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), restval="")
        writer.writeheader()
        writer.writerows(formatted)
    print(f"  -> {path}  ({len(rows)} rows)")


def rows_of(root: Path, tag: str = "d01") -> List[dict]:
    """Every ranked arm of every selecting stage, both axes, both orderings."""
    baselines = report_tables.baselines(Path(root))
    rows: List[dict] = []
    for stage, name, stem in RECORDS:
        prefix = f"{tag}_{stem}_"
        rows += (crowning_rows(root, stage, name, prefix, baselines)
                 if stage == "combination"
                 else screen_rows(root, stage, name, prefix, baselines))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="The study root ($FOA_STUDY_DIR).")
    ap.add_argument("--csv", required=True, type=Path,
                    help="Directory the CSV view is written into.")
    ap.add_argument("--study-tag", default="d01")
    args = ap.parse_args()

    rows = rows_of(args.root, args.study_tag)
    disagree = sum(1 for r in rows
                   if int(r["val_score_rank"]) != int(r["test_score_rank"]))
    skipped = sum(1 for r in rows
                  if int(r["val_rank"]) != int(r["val_score_rank"]))
    print("basis: the val_* columns are VALIDATION, the test_* columns are TEST;")
    print("val_score_rank is the ordering the selecting used (Eq. 2, w=1) and")
    print("val_rank is the adaptation ordering the record stores beside it.")
    print(f"{disagree} of {len(rows)} arms sit at a different place in the two")
    print(f"score orderings; {skipped} sit at a different place in the two")
    print("validation orderings.")
    write_csv(args.csv / "selection_axis.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
