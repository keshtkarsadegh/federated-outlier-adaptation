#!/usr/bin/env python3
"""
The two views the joint-tuning extension is read through.

The extension of `REPRODUCE.md` §3 screens the leading rule and the leading
penalty of each schedule **together** - the one thing the combination cross
never did - and re-runs the cell it crowns at the reporting horizon. Nothing
the manuscript reports reads it, and nothing here writes into a core view: it
is a table to be read *beside* `tables/paper/combos.csv`, on the basis that
table is measured on, and it is kept in files of its own so that a reader can
tell at a glance which rows are the programme and which are the extension.

    python tools/export_extension_views.py --root "$FOA_STUDY_DIR" \\
        --csv "$FOA_STUDY_DIR/tables/paper/"

WHAT THE FIRST VIEW ANSWERS. A tuned pair is only worth reporting against the
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

    extension_combo_tune.csv     five arms per schedule, TEST, full horizon
    extension_combo_screen.csv   the screen's top five per schedule, VALIDATION

WHICH ARMS ARE READ, NOT TYPED. The pair is `make_digits_p23.THE_PAIR`, the
untuned line is `make_digits_p23.THE_SHIPPED_LINES`, the tuned cell is read out
of `tables/p23_combo_tune_winners.json`, and the crowned pair comes through
`export_schedule_views.crowned_pair`, this repository's one reader of
`tables/p15_stage_winner.json` - the cell that record crowned, and for the other
schedule the first of its cells in the crowning's own order. Naming any of them here
would let this view outlive the selection that put the arm in it - the failure
`compare_arms.py` documents at length for the blends.

TWO BASES, AND THEY ARE NOT MIXED. The first view is TEST, because it reports;
the second is VALIDATION, because it is a screen and a screen is a selection.
The second is ranked by the rule that crowned the winner - `study_emit`'s own
`trade_score` over `select_reg_screen`'s summary - rather than by a rule
restated here, so the row at rank 1 is the row `p23_combo_tune_winners.json`
names or this file is wrong.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

SCREEN_COLUMNS = ("family", "rank", "cell", "c_ewc", "c_kd", "T", "m",
                  "server_eta", "lam", "mix", "folds",
                  "adaptation", "preservation", "score", "dials_at_edge")

#: How many of the screen's cells each schedule contributes to the summary. A
#: screen ranks, so the interesting part of it is the top of the ranking; the
#: whole 216-cell field is in the run records for anyone who wants it.
SCREEN_TOP = 5


# ------------------------------------------------------------- which arms
def tuned_winners(root: Path) -> Dict[str, str]:
    """The cell each schedule's joint screen crowned, from its own record."""
    path = root / "tables" / "p23_combo_tune_winners.json"
    if not path.is_file():
        raise SystemExit(
            f"FATAL: {path} is missing. Which cell the joint screen crowned is "
            "written there and nowhere else; inferring it from folder names "
            "would report an arm the selection never chose."
        )
    record = json.loads(path.read_text())
    return {family: record[f"combo-tune/{family}"]["winner"]
            for family in FAMILIES
            if f"combo-tune/{family}" in record}


def arm_sources(root: Path, family: str) -> List[Tuple[str, str, str]]:
    """``(role, arm id, run-folder prefix)`` for one schedule's five arms."""
    # make_digits_p23 is what named the pair and the line it copies; naming
    # them a second time here is how a view starts reporting a pair the screen
    # never tuned. Imported inside the call, as compare_arms imports
    # study_emit, so the module is only needed where it is used.
    from make_digits_p23 import THE_PAIR, THE_SHIPPED_LINES

    rule, penalty = THE_PAIR[family]
    winner = tuned_winners(root).get(family)
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
            ("untuned pair", THE_SHIPPED_LINES[family], "d01_combo_"),
            ("tuned pair", winner, f"d01_ctunefull_{family}_"),
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


def rows_of(root: Path, a0: float, p0: float) -> List[dict]:
    """The five arms of each schedule, each against the other four."""
    out: List[dict] = []
    for family in FAMILIES:
        sources = arm_sources(root, family)
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
def screen_rows(root: Path, tag: str) -> List[dict]:
    """
    The joint screen's top cells per schedule, ranked the way it was crowned.

    Everything here - the summary, the score and the boundary report - is
    `study_emit`'s, so rank 1 is the cell `p23_combo_tune_winners.json` names.
    Ranking it again with a rule of this file's own would let the two disagree
    without either failing.
    """
    import study_emit

    cells = reg_cells.combo_tune_cells()
    by_family = reg_cells.combo_tune_by_family()
    by_id = {cell["id"]: cell for cell in cells}
    a0, p0 = study_emit.shipped_baselines(root)
    summarised = study_emit.reg_selector.summarise(
        study_emit.reg_selector.collect(
            root, cells, f"{tag}_{SL.CTUNE_SCREEN_TAG}_"))

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
                "server_eta": "",
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


VIEWS = ("extension_combo_tune", "extension_combo_screen")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="The study root ($FOA_STUDY_DIR).")
    ap.add_argument("--csv", required=True, type=Path,
                    help="Directory the CSV views are written into.")
    ap.add_argument("--study-tag", default="d01")
    args = ap.parse_args()

    a0, p0 = baselines(args.root)
    print(f"shipped on cohort A0={a0:.4f}   shipped on source P0={p0:.4f}")
    print("EXTENSION. Nothing the manuscript reports reads these two views.")
    print("basis: TEST (final_evaluation) for the horizon table; VALIDATION for "
          "the screen, which is a selection.")
    write_csv(args.csv / "extension_combo_tune.csv", COLUMNS, rows_of(args.root, a0, p0))
    write_csv(args.csv / "extension_combo_screen.csv", SCREEN_COLUMNS,
              screen_rows(args.root, args.study_tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
