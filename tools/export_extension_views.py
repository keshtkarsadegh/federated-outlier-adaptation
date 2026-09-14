#!/usr/bin/env python3
"""
The four views the two joint-tuning extensions are read through.

The extensions of `REPRODUCE.md` §3 screen a server rule and a client penalty
**together** - the one thing the combination cross never did - and re-run the
cell each crowns at the reporting horizon. There are two of them because the
first tuned the pair that leads each schedule by TEST score and the study's own
aggregation shortlist was cut on VALIDATION, so `s23_combo_screen.txt` answered
the joint-grid question for a rule the programme did not choose and
`s25_combo_screen_selected.txt` answers it for the rule it did. Neither
supersedes the other and nothing the manuscript reports reads either: these are
tables to be read *beside* `tables/paper/combos.csv`, on the basis that table is
measured on, and they are kept in files of their own so that a reader can tell
at a glance which rows are the programme and which are an extension.

    python tools/export_extension_views.py --root "$FOA_STUDY_DIR" \\
        --csv "$FOA_STUDY_DIR/tables/paper/"

WHAT THE HORIZON VIEWS ANSWER. A tuned pair is only worth reporting against the
things it is supposed to beat, so each schedule contributes five arms - the
rule alone, the penalty alone, the untuned pair of those same two halves, the
tuned pair, and the pair the combination stage crowned - and every arm carries
its paired difference against the other four. THE CROWNED PAIR IS THE BASELINE,
not whichever of the eighteen combinations leads the column it would be read
in: the study shipped the crowned one, and an arm picked at report time for
topping a column is not an arm anybody was ever offered. A mean is not enough:
the combination stage's whole finding is that gains of this size sit inside the
fold spread, so the difference columns say how many of the five folds the row
actually won as well as by how much on average.

    extension_combo_tune.csv              five arms per schedule, TEST
    extension_combo_screen.csv            its screen's top five, VALIDATION
    extension_combo_tune_selected.csv     the same five for the SELECTED pair
    extension_combo_screen_selected.csv   its screen's top five, VALIDATION

The four are two pairs and they are cut by one reader over a `Grid`, because
the two extensions are read against each other: a second copy of this reading
could drift from the first in a way that looks like a result.

WHICH ARMS ARE READ, NOT TYPED. The pair is the grid's generator's `THE_PAIR`,
the untuned line its `THE_SHIPPED_LINES`, the tuned cell is read out of that
grid's own selection record, and the crowned pair comes through
`export_schedule_views.crowned_pair`, this repository's one reader of
`tables/p15_stage_winner.json` - the cell that record crowned, and for the other
schedule the first of its cells in the crowning's own order. Naming any of them
here would let a view outlive the selection that put the arm in it - the failure
`compare_arms.py` documents at length for the blends.

TWO BASES, AND THEY ARE NOT MIXED. The horizon views are TEST, because they
report; the screen views are VALIDATION, because a screen is a selection. Each
screen view is ranked by the rule that crowned its winner - `study_emit`'s own
`trade_score` over `select_reg_screen`'s summary - rather than by a rule
restated here, so the row at rank 1 is the row that grid's record names or this
file is wrong.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import statistics as st
import sys
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))

import report_tables  # noqa: E402
from compare_arms import (  # noqa: E402
    FAMILIES,
    baselines,
    paired,
    penalties,
    per_fold,
)
from export_schedule_views import crowned_pair  # noqa: E402

from federated_outlier_adaptation.training import reg_cells  # noqa: E402
from federated_outlier_adaptation.training import study_lines as SL  # noqa: E402

#: How a selection record's family name maps onto the schedule a run folder is
#: told apart by, exactly as `export_traces.py` maps it: `cell_and_family`
#: answers "parallel"/"cyclic" and the records were written with the runner's
#: own "concurrent"/"sequential".
SCHEDULE = {"concurrent": "parallel", "sequential": "cyclic"}

#: The five arms of one schedule, in the order the table argues them: what you
#: had, what you had instead, what the cross ran, what the joint screen chose,
#: and the pair the programme itself crowned.
ROLES = ("rule alone", "penalty alone", "untuned pair", "tuned pair",
         "crowned pair")

#: The four rows every row is differenced against, and the column stem each
#: difference is written under.
AGAINST = (("rule alone", "rule"), ("penalty alone", "penalty"),
           ("untuned pair", "untuned"), ("crowned pair", "crowned"))

COLUMNS = (("family", "role", "arm", "folds",
            "adaptation", "adaptation_sd", "preservation", "preservation_sd",
            "score", "score_sd")
           + tuple(f"{part}_vs_{stem}" for _, stem in AGAINST
                   for part in ("mean", "folds_won")))


def _screen_columns(knob: str) -> tuple:
    """
    The screen view's header, with the grid's own rule dial in it.

    The two grids move different knobs - a server step and an anchor half-life -
    and a column named for one of them in the other's file would be a header
    that says what was swept and is wrong. There is no shared name for the two
    that is not vaguer than either.
    """
    return ("family", "rank", "cell", "c_ewc", "c_kd", "T", "m", knob,
            "lam", "mix", "folds",
            "adaptation", "preservation", "score", "dials_at_edge")


SCREEN_COLUMNS = _screen_columns("server_eta")
SCREEN_COLUMNS_SELECTED = _screen_columns("anchor_halflife_r")

#: How many of the screen's cells each schedule contributes to the summary. A
#: screen ranks, so the interesting part of it is the top of the ranking; the
#: whole 216-cell field is in the run records for anyone who wants it.
SCREEN_TOP = 5


class Grid(NamedTuple):
    """
    One joint-tuning grid, and everything the two views need in order to read
    it without naming any of its arms.

    There are two of them and they are read the same way, so the reading is
    written once: what differs is which record the tuned row comes from, which
    generator named the pair, which catalogue the screen ranked and which stems
    the runs sit under. Naming an arm here instead - in either grid - is how a
    view starts outliving the selection that put the arm in it, which is the
    failure `compare_arms.py` documents at length for the blends.
    """

    record: str
    key: str
    module: str
    cells: Callable[[], list]
    by_family: Callable[[], dict]
    screen_tag: str
    full_tag: str
    knob: str
    columns: tuple
    horizon_view: str
    screen_view: str


#: The TEST-leading pair's grid (P23/P24) and the SELECTED pair's (P25/P26).
#: Two grids, two records, two pairs of views, and neither supersedes the other:
#: the second exists because the first tuned a pair the study did not select,
#: and they are written to be read beside each other.
GRIDS = {
    "combo_tune": Grid(
        record="p23_combo_tune_winners.json",
        key="combo-tune",
        module="make_digits_p23",
        cells=reg_cells.combo_tune_cells,
        by_family=reg_cells.combo_tune_by_family,
        screen_tag=SL.CTUNE_SCREEN_TAG,
        full_tag=SL.CTUNE_FULL_TAG,
        knob="server_eta",
        columns=SCREEN_COLUMNS,
        horizon_view="extension_combo_tune.csv",
        screen_view="extension_combo_screen.csv",
    ),
    "combo_tune_selected": Grid(
        record="p25_combo_tune_selected_winners.json",
        key="combo-tune-selected",
        module="make_digits_p25",
        cells=reg_cells.combo_tune_selected_cells,
        by_family=reg_cells.combo_tune_selected_by_family,
        screen_tag=SL.CTUNE_SEL_SCREEN_TAG,
        full_tag=SL.CTUNE_SEL_FULL_TAG,
        knob="anchor_halflife_r",
        columns=SCREEN_COLUMNS_SELECTED,
        horizon_view="extension_combo_tune_selected.csv",
        screen_view="extension_combo_screen_selected.csv",
    ),
}


# ------------------------------------------------------------- which arms
def tuned_winners(root: Path, grid: Grid) -> Dict[str, str]:
    """The cell each schedule's joint screen crowned, from its own record."""
    path = root / "tables" / grid.record
    if not path.is_file():
        raise SystemExit(
            f"FATAL: {path} is missing. Which cell the joint screen crowned is "
            "written there and nowhere else; inferring it from folder names "
            "would report an arm the selection never chose."
        )
    record = json.loads(path.read_text())
    return {family: record[f"{grid.key}/{family}"]["winner"]
            for family in FAMILIES
            if f"{grid.key}/{family}" in record}


def arm_sources(root: Path, family: str, grid: Grid) -> List[Tuple[str, str, str]]:
    """``(role, arm id, run-folder prefix)`` for one schedule's five arms."""
    # The generator is what named the pair and the line it copies; naming them a
    # second time here is how a view starts reporting a pair the screen never
    # tuned. Imported inside the call, as compare_arms imports study_emit, so
    # the module is only needed where it is used.
    generator = importlib.import_module(grid.module)

    rule, penalty = generator.THE_PAIR[family]
    winner = tuned_winners(root, grid).get(family)
    if winner is None:
        raise SystemExit(
            f"FATAL: the joint screen crowned no {family} cell. The tuned row "
            "is the point of this view and is not left out silently."
        )
    # The crowned pair is a combination cell whatever schedule crowned it, so
    # both halves ran under the cross's own stem; `crowned_pair` is what decides
    # which cell belongs to which schedule, and deciding that a second way here
    # is how two views of one selection begin to disagree without failing.
    return [("rule alone", rule, "d01_aggfull_"),
            ("penalty alone", penalty, "d01_regfull_"),
            ("untuned pair", generator.THE_SHIPPED_LINES[family], "d01_combo_"),
            ("tuned pair", winner, f"d01_{grid.full_tag}_{family}_"),
            ("crowned pair", crowned_pair(root)[family], "d01_combo_")]


# --------------------------------------------------------------- the rows
def summary_of(root: Path, prefix: str, arm: str, family: str,
               a0: float, p0: float) -> Optional[dict]:
    """
    One arm's TEST means and spreads, through `report_tables` rather than a
    second reader of the same folders.

    A penalty run stores BOTH schedules in one folder and is keyed by the one
    it was selected for, which is what `report_tables.cell_and_family` decides;
    reading the folders here would answer that question a second way.
    """
    rows = report_tables.summarise(report_tables.read_runs(root, prefix), a0, p0)
    for row in rows:
        if row["cell"] == arm and row["family"] == SCHEDULE[family]:
            return row
    return None


def rows_of(root: Path, a0: float, p0: float, grid: Grid) -> List[dict]:
    """The five arms of each schedule, each against the other four."""
    out: List[dict] = []
    for family in FAMILIES:
        sources = arm_sources(root, family, grid)
        # THE FOLD SCORES ARE READ ONCE PER PREFIX. The difference columns are
        # paired within a fold, so every arm of a schedule has to be scored by
        # one reader on one basis or the pairing is between two conventions.
        folds: Dict[str, Dict[int, float]] = {}
        for role, arm, prefix in sources:
            scored = (penalties(root, family, a0, p0) if prefix == "d01_regfull_"
                      else per_fold(root, prefix, family, a0, p0))
            if arm not in scored:
                raise SystemExit(
                    f"FATAL: no {family} run of {arm!r} under {root}/{prefix}*. "
                    "Every arm here is named by a selection record, so a missing "
                    "one means the records and the runs have come apart."
                )
            folds[role] = scored[arm]
        for role, arm, prefix in sources:
            summary = summary_of(root, prefix, arm, family, a0, p0)
            if summary is None:
                raise SystemExit(
                    f"FATAL: {arm!r} has no five-fold {family} summary under "
                    f"{root}/{prefix}*."
                )
            values = folds[role]
            scores = [values[fold] for fold in sorted(values)]
            row = {
                "family": family, "role": role, "arm": arm,
                "folds": len(scores),
                "adaptation": f"{summary['adaptation']:.6f}",
                "adaptation_sd": f"{summary['adaptation_sd']:.6f}",
                "preservation": f"{summary['preservation']:.6f}",
                "preservation_sd": f"{summary['preservation_sd']:.6f}",
                "score": f"{st.mean(scores):.4f}",
                "score_sd": f"{st.pstdev(scores):.4f}",
            }
            for other, stem in AGAINST:
                # A row is not differenced against itself: an empty cell says
                # "this is the baseline" where a zero would read as a measured
                # tie.
                result = None if other == role else paired(values, folds[other])
                row[f"mean_vs_{stem}"] = "" if result is None else f"{result['mean']:.4f}"
                row[f"folds_won_vs_{stem}"] = (
                    "" if result is None
                    else sum(1 for d in result["diffs"] if d > 0))
            out.append(row)
    return out


# ------------------------------------------------------------- the screen
def screen_rows(root: Path, tag: str, grid: Grid) -> List[dict]:
    """
    The joint screen's top cells per schedule, ranked the way it was crowned.

    Everything here - the summary, the score and the boundary report - is
    `study_emit`'s, so rank 1 is the cell the grid's own record names. Ranking
    it again with a rule of this file's own would let the two disagree without
    either failing.
    """
    import study_emit

    cells = grid.cells()
    by_family = grid.by_family()
    by_id = {cell["id"]: cell for cell in cells}
    a0, p0 = study_emit.shipped_baselines(root)
    summarised = study_emit.reg_selector.summarise(
        study_emit.reg_selector.collect(
            root, cells, f"{tag}_{grid.screen_tag}_"))

    out: List[dict] = []
    for family in FAMILIES:
        siblings = by_family[family]
        ids = {cell["id"] for cell in siblings}
        scored = [row for row in summarised
                  if row["id"] in ids and row["family"] == family
                  and row["adaptation"]["mean"] is not None]
        ranked = sorted(scored,
                        key=lambda row: (-study_emit.trade_score(row, a0, p0),
                                         row["id"]))
        for rank, row in enumerate(ranked[:SCREEN_TOP], start=1):
            cell = by_id[row["id"]]
            hits = study_emit.numeric_boundary(cell, siblings, "dials")
            entry = {
                "family": family, "rank": rank, "cell": cell["id"],
                grid.knob: "",
                "lam": f"{cell['hypers']['lam']:.6g}",
                "mix": f"{cell['hypers']['mix']:.6g}",
                "folds": len(row["folds"]),
                "adaptation": f"{row['adaptation']['mean']:.6f}",
                "preservation": f"{row['preservation']['mean']:.6f}",
                "score": f"{study_emit.trade_score(row, a0, p0):.4f}",
                "dials_at_edge": " ".join(_edge(hit) for hit in hits),
            }
            entry.update({name: f"{value:g}"
                          for name, value in cell["dials"].items()})
            out.append(entry)
    return out


def _edge(hit: str) -> str:
    """
    One boundary sentence reduced to the token a table column can carry.

    `study_emit.numeric_boundary` writes a sentence per hit - the cell id, the
    dial and its value, and the whole row it sits at the end of. The id is the
    column beside this one and the row is in `REG_GRID_RANGES.md`, so what is
    kept here is the dial at its value and which end of the row it is: the
    sentence is reused rather than the rule behind it restated, and a hit this
    file cannot parse is written through whole rather than dropped.
    """
    dial, _, rest = hit.partition(": ")[2].partition(" is the ")
    return f"{dial}:{rest.split(' ')[0]}" if rest else hit


# ------------------------------------------------------------------ output
def write_csv(path: Path, columns, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}  ({len(rows)} rows)")


#: Every file this tool writes, named rather than globbed, so that the
#: membership check that keeps extension arms out of the core bundle has a list
#: to subtract and a third view cannot quietly excuse itself from it.
VIEWS = tuple(name for grid in GRIDS.values()
              for name in (grid.horizon_view, grid.screen_view))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="The study root ($FOA_STUDY_DIR).")
    ap.add_argument("--csv", required=True, type=Path,
                    help="Directory the CSV views are written into.")
    ap.add_argument("--study-tag", default="d01")
    ap.add_argument("--grid", default="all", choices=("all",) + tuple(GRIDS),
                    help="Which joint grid to write the two views of.")
    args = ap.parse_args()

    a0, p0 = baselines(args.root)
    print(f"shipped on cohort A0={a0:.4f}   shipped on source P0={p0:.4f}")
    print("EXTENSION. Nothing the manuscript reports reads these views.")
    print("basis: TEST (final_evaluation) for the horizon tables; VALIDATION "
          "for the screens, which are selections.")
    chosen = tuple(GRIDS) if args.grid == "all" else (args.grid,)
    for name in chosen:
        grid = GRIDS[name]
        print(f"{name}: {grid.record}")
        write_csv(args.csv / grid.horizon_view, COLUMNS,
                  rows_of(args.root, a0, p0, grid))
        write_csv(args.csv / grid.screen_view, grid.columns,
                  screen_rows(args.root, args.study_tag, grid))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
