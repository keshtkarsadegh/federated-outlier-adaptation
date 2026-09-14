#!/usr/bin/env python3
"""
The schedule, given a table of its own.

Every stage of the programme selected and reported the two schedules
SEPARATELY - which is right, because a rule that only exists on one of them
cannot be ranked against a rule that only exists on the other - and the cost of
that is that the comparison a reader makes first has no view. Parallel and
cyclic rows sit interleaved in `agg-winners.csv`, `reg-winners.csv` and
`combos.csv`, sorted by score and not by schedule, so "does it matter which way
round you run it?" is answered by scrolling three files and pairing rows by eye.

    python tools/export_schedule_views.py --root "$FOA_STUDY_DIR" \\
        --csv "$FOA_STUDY_DIR/tables/paper/"

THIS IS A CORE VIEW, not an extension. The schedule is an axis of the programme
rather than something screened beside it: every arm below is an arm a shipped
table already reports, read at the reporting horizon on the basis those tables
are measured on, and no `ctune_` folder is read anywhere here.

WHAT THE FIRST VIEW ANSWERS. Seven pairs, each the same thing run both ways -
the control, the best rule alone, the best penalty alone, each schedule's own
selected logit and NTD cell, the best combination of the cross, and the crowned
pair against the cyclic arm that was carried beside it. A pair is only worth
stating if both halves are the arm that schedule actually selected, which is why
five of the seven are two DIFFERENT cells: the cyclic grid crowned
`ntd_b0p01_t2` where the parallel one crowned `ntd_b0p01_t0p5`, and forcing one
cell across both schedules would report a setting one of them never chose.

PAIRED BY FOLD, and the mean is not the finding. The five folds are the same
five partitions for every arm, so a fold that is hard for one arm is hard for
all of them; differencing within a fold removes that shared difficulty, and
comparing two means does not. So every row carries its five per-fold
differences and how many of the five the cyclic half won, beside the mean -
`compare_arms.py` documents at length what reading a mean of this size without
its spread did to the combination stage, and this table is the same size of
effect.

    schedule_pairs.csv   seven pairs, cyclic minus parallel, TEST, full horizon
    schedule_top.csv     the study's eight strongest arms, with their schedule

WHICH ARMS ARE READ, NOT TYPED. The rule and the blend are the two halves of
`make_digits_p23.THE_PAIR`, which is where this repository records the best rule
and the best penalty of each schedule by TEST score - the very two rows this
table wants - so they are read from there rather than re-derived from two views
that may not have been regenerated. The logit and NTD cells come from
`tables/p13_reg_method_winners.json`, which crowns per method AND per schedule.
The best combination is ranked out of `tables/p15_combination_grid.json` under
this file's own score, and the crowned pair and the arm carried beside it out of
`tables/p15_stage_winner.json`. The control is the one unflagged cell of
`agg_cells.control_cells()`. Naming any of them here would let this view outlive
the selection that put the arm in it.

ONE BASIS. Both views are TEST, because both report; nothing here ranks a
selection. A0 and P0 are the shipped model's own two accuracies, read through
`report_tables` at call time and never written down.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))

import report_tables  # noqa: E402
from compare_arms import FAMILIES, score  # noqa: E402

from federated_outlier_adaptation.training import agg_cells  # noqa: E402
from federated_outlier_adaptation.training import study_lines as SL  # noqa: E402

#: How a selection record's family name maps onto the schedule a table is read
#: by, exactly as `export_extension_views.py` maps it: the records were written
#: with the runner's own "concurrent"/"sequential" and `report_tables` answers
#: "parallel"/"cyclic".
SCHEDULE = {"concurrent": "parallel", "sequential": "cyclic"}

#: The three stages whose arms ran at the reporting horizon in both schedules,
#: by the stem their run folders carry under the study tag. The screens are not
#: here: a screen is a selection and this view reports.
STAGE_STEMS = ("aggfull", "regfull", "combo")

#: The fold a run folder names, wherever the stem put it.
FOLD_IN_PATH = re.compile(r"_fold(\d+)")

#: How many arms the second view carries. Eight, because the interesting part of
#: that ranking is where the parallel side first appears in it and eight is the
#: first length that shows it.
TOP_N = 8

PAIR_COLUMNS = (("pair", "parallel_cell", "cyclic_cell",
                 "par_adapt", "par_presv", "par_score",
                 "cyc_adapt", "cyc_presv", "cyc_score")
                + tuple(f"diff_f{fold}" for fold in SL.FOLDS)
                + ("diff_mean", "cyclic_wins"))

TOP_COLUMNS = ("rank", "cell", "schedule", "stage", "folds",
               "adaptation", "preservation", "score")

#: The selection records each row's two arms are read out of.
REG_METHOD_WINNERS = "p13_reg_method_winners.json"
COMBINATION_GRID = "p15_combination_grid.json"
STAGE_WINNER = "p15_stage_winner.json"


# ------------------------------------------------------------- the numbers
def per_fold(root: Path, prefix: str) -> Dict[Tuple[str, str], Dict[int, Tuple[float, float]]]:
    """
    ``{(cell, schedule): {fold: (adaptation, preservation)}}`` under one prefix.

    `report_tables.read_runs` reads the same files and the same two fields with
    the same key rule, and then means the folds away. This table is paired by
    fold, so the fold has to survive the read - and nothing else about the read
    is restated: `cell_and_family` decides which schedule a stored summary
    belongs to and returns ``None`` for the half a penalty run was not selected
    for, and `in_study` drops the arrangement this study retired. Deciding
    either of those a second way here is how two views of one number begin to
    disagree without either of them failing.
    """
    found: Dict[Tuple[str, str], Dict[int, Tuple[float, float]]] = {}
    for name in sorted(glob.glob(f"{root}/{prefix}*/**/summary_0.json", recursive=True)):
        path = Path(name)
        cell, schedule = report_tables.cell_and_family(path, prefix)
        if cell is None or not report_tables.in_study(name):
            continue
        fold = FOLD_IN_PATH.search(name)
        if fold is None:
            continue
        body = next(iter(json.loads(path.read_text()).values()))

        # REPORT ON TEST. Selection reads the validation halves - that is what
        # a selection is allowed to see - and reporting reads the test ones.
        # Preservation is the SOURCE population and not the cohort being
        # adapted to; `report_tables` says at length what reading the wrong
        # series here costs, and these are the two fields it reads.
        final = body.get("final_evaluation") or {}
        clients = final.get("clients") or {}
        old = final.get("old") or {}
        adaptation = (clients["accuracy"] if isinstance(clients.get("accuracy"), (int, float))
                      else _last(body.get("heldout_client_accuracies")))
        preservation = (old["mean"] if isinstance(old.get("mean"), (int, float))
                        else _last(body.get("source_val_accuracies")))
        if adaptation is None or preservation is None:
            continue
        found.setdefault((cell, schedule), {})[int(fold.group(1))] = (adaptation, preservation)
    return found


def _last(series) -> Optional[float]:
    values = [value for value in (series or []) if value is not None]
    return float(values[-1]) if values else None


def read_stages(root: Path, tag: str) -> Tuple[dict, dict]:
    """
    Every arm of the three reporting stages, and which stage each came from.

    An arm is refused rather than merged if two stages hold runs of it: a rule,
    a penalty and a pair of the two have ids of three different shapes, so a
    collision means a stem has started matching a stage it does not name, and
    silently taking one of the two would decide a row's numbers by dictionary
    order.
    """
    found: Dict[Tuple[str, str], Dict[int, Tuple[float, float]]] = {}
    stages: Dict[Tuple[str, str], str] = {}
    for stem in STAGE_STEMS:
        for key, folds in per_fold(root, f"{tag}_{stem}_").items():
            if key in found:
                raise SystemExit(
                    f"FATAL: {key[0]!r} ({key[1]}) has runs under both "
                    f"{stages[key]!r} and {stem!r}. One of the two stems is "
                    "matching a stage it does not name, and which of them this "
                    "table reports would be decided by the order of this loop."
                )
            found[key], stages[key] = folds, stem
    if not found:
        raise SystemExit(
            f"FATAL: no run of any reporting stage under {root}/{tag}_*. This "
            "view is computed from the runs and cannot be computed without them."
        )
    return found, stages


def means(folds: Dict[int, Tuple[float, float]], a0: float, p0: float) -> Tuple[float, float, List[float]]:
    """``(adaptation, preservation, per-fold scores)`` over the study's folds."""
    return (st.mean(folds[fold][0] for fold in SL.FOLDS),
            st.mean(folds[fold][1] for fold in SL.FOLDS),
            [score(*folds[fold], a0, p0) for fold in SL.FOLDS])


# ------------------------------------------------------------- which arms
def _record(root: Path, name: str) -> dict:
    """One selection record, or a refusal that names the file."""
    path = Path(root) / "tables" / name
    if not path.is_file():
        raise SystemExit(
            f"FATAL: no {path}. Which arm each schedule contributes to this "
            "table is a selection result, and naming one here would report an "
            "arm the study never chose."
        )
    return json.loads(path.read_text())


def control_arm() -> str:
    """
    The plain control, from the cell table rather than named here.

    Two cells control for this stage and only one of them is plain: the other
    arms the oracle stop rule, which is a second control and a different row.
    Telling them apart by "runs no flag" rather than by position is what a
    reordered list cannot break.
    """
    plain = [cell["id"] for cell in agg_cells.control_cells() if not cell["flags"]]
    if len(plain) != 1:
        raise SystemExit(
            "FATAL: agg_cells.control_cells() no longer holds exactly one "
            f"unflagged control ({plain}). Which of them the control row of "
            "this table reports has stopped being decidable."
        )
    return plain[0]


def method_winner(root: Path, method: str, family: str) -> str:
    """The cell one penalty's grid crowned for ONE schedule, from its record."""
    entry = _record(root, REG_METHOD_WINNERS).get(f"{method}/{family}") or {}
    if not entry.get("measured") or not entry.get("winner"):
        raise SystemExit(
            f"FATAL: {REG_METHOD_WINNERS} crowns no measured {method!r} cell "
            f"for {family}. Each schedule's own winner is the point of this "
            "row; the other schedule's is not a substitute for it."
        )
    return entry["winner"]


def combination_families(root: Path) -> Dict[str, str]:
    """``{combination id: family}`` - a pair id is two ids joined by an
    underscore, so which schedule ran it is read off the grid it was emitted
    from rather than split out of the string."""
    found = {}
    for family, entries in (_record(root, COMBINATION_GRID).get("pairs") or {}).items():
        for agg_id, reg_id in entries:
            found[f"{agg_id}_{reg_id}"] = family
    if not found:
        raise SystemExit(
            f"FATAL: {COMBINATION_GRID} names no pair; the cross it records "
            "ran nothing for this table to report."
        )
    return found


def best_combination(root: Path, family: str, scored: dict, a0: float, p0: float) -> str:
    """
    The highest-scoring pair of the cross, under this file's own score.

    Ranked out of the grid record the cross was emitted from rather than off
    `tables/paper/combos.csv`, so this view does not depend on another view
    having been regenerated first. Ties go to the id, because a table whose top
    row moves with dictionary order is a table nobody can diff.
    """
    schedule = SCHEDULE[family]
    pairs = [cell for cell, ran in combination_families(root).items()
             if ran == family and len(scored.get((cell, schedule), {})) >= len(SL.FOLDS)]
    if not pairs:
        raise SystemExit(
            f"FATAL: the cross ran no complete {family} pair under {root}. "
            "The best combination is a row of this table and is not left out "
            "silently."
        )
    return max(pairs, key=lambda cell: (st.mean(means(scored[(cell, schedule)], a0, p0)[2]), cell))


def crowned_pair(root: Path) -> Dict[str, str]:
    """
    The pair the cross crowned, and the arm carried beside it on the other
    schedule.

    Both come out of one record. ``winner`` is the crowned cell, and ``ranked``
    is the crowning's own order - so the first cell of the OTHER schedule in it
    is the arm the size stage carried, which is what
    `paper_figures/make_numbers.CYCLIC_ARM` names and what this deliberately
    does not name a second time.
    """
    record = _record(root, STAGE_WINNER)
    families = combination_families(root)
    winner, crowned = record.get("winner"), record.get("family")
    if not winner or crowned not in FAMILIES:
        raise SystemExit(
            f"FATAL: {STAGE_WINNER} crowns no combination. The crowned pair is "
            "the last row of this table and is not guessed from a ranking."
        )
    found = {crowned: winner}
    for entry in record.get("ranked") or []:
        family = families.get(entry.get("id"))
        if family is not None and family not in found:
            found[family] = entry["id"]
    missing = [family for family in FAMILIES if family not in found]
    if missing:
        raise SystemExit(
            f"FATAL: {STAGE_WINNER} ranks no {missing} combination beside the "
            "one it crowned, so the crowned pair has nothing to be paired with."
        )
    return found


def pair_arms(root: Path, scored: dict, a0: float, p0: float) -> List[Tuple[str, Dict[str, str]]]:
    """``(pair name, {family: arm id})`` for every row of the first view."""
    # make_digits_p23 is where this repository records the best rule and the
    # best penalty of each schedule by TEST score. Imported inside the call
    # because it reads a study record of its own at import time.
    from make_digits_p23 import THE_PAIR

    control = control_arm()
    return [
        ("control", {family: control for family in FAMILIES}),
        ("rule_alone", {family: THE_PAIR[family][0] for family in FAMILIES}),
        ("blend", {family: THE_PAIR[family][1] for family in FAMILIES}),
        ("logit", {family: method_winner(root, "logit_l2", family) for family in FAMILIES}),
        ("ntd", {family: method_winner(root, "ntd", family) for family in FAMILIES}),
        ("best_combo", {family: best_combination(root, family, scored, a0, p0)
                        for family in FAMILIES}),
        ("crowned", crowned_pair(root)),
    ]


# --------------------------------------------------------------- the rows
def pair_rows(root: Path, scored: dict, a0: float, p0: float) -> List[dict]:
    """Each pair's two arms, and the cyclic half's per-fold lead over the other."""
    out: List[dict] = []
    for name, arms in pair_arms(root, scored, a0, p0):
        row = {"pair": name,
               "parallel_cell": arms["concurrent"], "cyclic_cell": arms["sequential"]}
        scores: Dict[str, List[float]] = {}
        for family, stem in (("concurrent", "par"), ("sequential", "cyc")):
            schedule = SCHEDULE[family]
            folds = scored.get((arms[family], schedule)) or {}
            absent = [fold for fold in SL.FOLDS if fold not in folds]
            if absent:
                raise SystemExit(
                    f"FATAL: {arms[family]!r} has no {schedule} run of fold(s) "
                    f"{absent} under {root}. Every arm here is named by a "
                    "selection record, so a missing one means the records and "
                    "the runs have come apart."
                )
            adaptation, preservation, scores[stem] = means(folds, a0, p0)
            row[f"{stem}_adapt"] = f"{adaptation:.6f}"
            row[f"{stem}_presv"] = f"{preservation:.6f}"
            row[f"{stem}_score"] = f"{st.mean(scores[stem]):.4f}"
        diffs = [cyc - par for cyc, par in zip(scores["cyc"], scores["par"])]
        for fold, diff in zip(SL.FOLDS, diffs):
            row[f"diff_f{fold}"] = f"{diff:.4f}"
        row["diff_mean"] = f"{st.mean(diffs):.4f}"
        row["cyclic_wins"] = f"{sum(1 for diff in diffs if diff > 0)}/{len(diffs)}"
        out.append(row)
    return out


def top_rows(scored: dict, stages: dict, a0: float, p0: float) -> List[dict]:
    """
    The study's strongest arms at the reporting horizon, schedule named.

    Ranked across the three stages together, which is the one place in this
    repository that happens - every other view ranks within a stage, within a
    schedule, or both. It is a fair ranking only because all three stages ran
    the same folds at the same horizon and are scored here by one rule; it is
    printed with the stage beside each row so that it cannot be read as a
    ranking of stages.
    """
    ranked = []
    for (cell, schedule), folds in scored.items():
        if len(folds) < len(SL.FOLDS) or any(fold not in folds for fold in SL.FOLDS):
            continue
        adaptation, preservation, scores = means(folds, a0, p0)
        ranked.append((st.mean(scores), cell, schedule, adaptation, preservation))
    ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
    return [{"rank": rank, "cell": cell, "schedule": schedule,
             "stage": stages[(cell, schedule)], "folds": len(SL.FOLDS),
             "adaptation": f"{adaptation:.6f}", "preservation": f"{preservation:.6f}",
             "score": f"{value:.4f}"}
            for rank, (value, cell, schedule, adaptation, preservation)
            in enumerate(ranked[:TOP_N], start=1)]


# ------------------------------------------------------------------ output
def write_csv(path: Path, columns, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}  ({len(rows)} rows)")


VIEWS = ("schedule_pairs", "schedule_top")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="The study root ($FOA_STUDY_DIR).")
    ap.add_argument("--csv", required=True, type=Path,
                    help="Directory the CSV views are written into.")
    ap.add_argument("--study-tag", default="d01")
    args = ap.parse_args()

    a0, p0 = report_tables.baselines(args.root)
    print(f"shipped on cohort A0={a0:.4f}   shipped on source P0={p0:.4f}")
    print("basis: TEST (final_evaluation) for both views. Selection ran on "
          "validation, and neither of these ranks a selection.")
    print("differences are cyclic minus parallel, in points, paired by fold.")
    scored, stages = read_stages(args.root, args.study_tag)
    write_csv(args.csv / "schedule_pairs.csv", PAIR_COLUMNS,
              pair_rows(args.root, scored, a0, p0))
    write_csv(args.csv / "schedule_top.csv", TOP_COLUMNS,
              top_rows(scored, stages, a0, p0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
