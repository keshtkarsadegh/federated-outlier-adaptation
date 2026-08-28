"""
Rank the stage-6 screening cells and emit the full-horizon run.

    python tools/select_agg_screen.py --root $FOA_RESULTS_DIR/main_v6 \
        --epsilon 0.005 [--out stage6_full.txt] [--report stage6_selection.json]

Reads every ``coh6_agg_<cell>_fold<k>`` result under ``--root``, averages each
cell over its folds, applies a budget, ranks what is left per path, and writes
the winners out as a task file at the full horizon.  It reads JSON and writes
text: a login-node job.

What it ranks on, and why the budget is not optional
----------------------------------------------------
The study has two axes and they pull against each other.  A rule that throws the
old model away adapts beautifully; a rule that never moves preserves perfectly.
Ranking on adaptation alone selects the first, ranking on preservation alone
selects the second, and a single blended score hides which trade was made.

So the rule is a **constrained selection**: among the cells whose preservation
is within ``epsilon`` of the best preservation seen on that path, take the one
that adapts best.  ``epsilon`` is the budget - how much of the old population's
accuracy the owner is willing to spend - and it is a parameter because it is a
judgement, not a measurement.

Every figure is a **fold mean**.  Five folds of one cell are five runs over five
splits, so they are averaged before anything is compared and a cell wins only if
it wins on average.  The spread across folds is carried into the report, so a
cell that wins by less than its own fold-to-fold noise is visible as such.

Validation, not test
--------------------
Selection reads the pooled **validation** accuracy and the old data's
**validation** figures where the runs recorded them, falling back to the pooled
test figure only when a run has no validation series at all.  The test rows are
what the paper reports on; selecting on them would be choosing the winner by
the number that is supposed to judge it.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

from federated_outlier_adaptation.training.agg_cells import (
    FULL_ROUNDS,
    PATHS,
    screen_cells,
)

#: Folder prefix of a screening run.  A parameter rather than a constant so
#: that a second study reuses this selector instead of copying it - the
#: selection rule is the science, the prefix is only an address.
PARENT_PREFIX = "coh6_agg_"


def run_pattern(prefix: str = PARENT_PREFIX):
    """The compiled folder matcher for one prefix."""
    return re.compile(
        rf"^{re.escape(prefix)}(?P<cell>.+)_fold(?P<fold>\d+)(?:_.*)?$"
    )

#: A run folder is ``coh6_agg_<cell>_fold<k>``, optionally followed by whatever
#: the runner appends - ``_BaseTrainer_grid_search`` in practice, since the
#: driver builds its output path from ``<parent>_<trainer>_grid_search``.  The
#: fold number is therefore not the tail of the name, and reading it as one
#: silently found nothing at all.  Cell ids never contain ``_fold``, so the
#: split is unambiguous.
RUN_DIR = re.compile(rf"^{re.escape(PARENT_PREFIX)}(?P<cell>.+)_fold(?P<fold>\d+)(?:_.*)?$")


def parse_run_dir(name: str, prefix: str = PARENT_PREFIX):
    """``(cell id, fold)`` of a run folder, or ``None`` when it is not one."""
    match = run_pattern(prefix).match(name)
    if match is None:
        return None
    return match.group("cell"), int(match.group("fold"))


def _load(path: Path) -> Optional[dict]:
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _last(series) -> Optional[float]:
    """The last measured entry of a per-round series, skipping unmeasured ones."""
    if not series:
        return None
    for value in reversed(list(series)):
        if value is not None:
            return float(value)
    return None


def read_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    The two axes of one (cell, fold) run.

    ``adaptation`` is the pooled validation accuracy over the cohort - the
    selection signal.  ``preservation`` is the old data's five-fold mean from
    the final-model evaluation.  ``adaptation_test`` is carried alongside so the
    report can show what was selected *and* what it scored, without the second
    number ever entering the ranking.
    """
    payloads = sorted(run_dir.rglob("accuracies_*.json"))
    if not payloads:
        return None
    data = _load(payloads[0])
    if not data:
        return None

    final = data.get("final_evaluation") or {}
    clients = final.get("clients") or {}
    old = final.get("old") or {}

    adaptation = _last(data.get("pool_val_accuracies"))
    if adaptation is None:
        # A run with no validation series at all: fall back to the pooled test
        # figure and mark the record, so the fallback is visible in the report.
        adaptation = clients.get("accuracy")
        fallback = adaptation is not None
    else:
        fallback = False

    return {
        "adaptation": adaptation,
        "adaptation_is_test_fallback": fallback,
        "adaptation_test": clients.get("accuracy"),
        "preservation": old.get("mean"),
        "preservation_sd": old.get("sd"),
        "source_val_last": _last(data.get("source_val_accuracies")),
        "rounds": len(data.get("accuracies") or []),
        "path": str(payloads[0]),
    }


def collect(root: Path, cells, prefix: str = PARENT_PREFIX) -> Dict[str, Dict[str, Any]]:
    """Every cell's per-fold runs, keyed by cell id."""
    found: Dict[str, Dict[str, Any]] = {}
    for cell in cells:
        folds: Dict[int, Dict[str, Any]] = {}
        for run_dir in sorted(root.glob(f"{prefix}{cell['id']}_fold*")):
            if not run_dir.is_dir():
                continue
            parsed = parse_run_dir(run_dir.name, prefix)
            # The glob is by cell id, but ``fedadam_lr0p1`` also matches
            # ``fedadam_lr0p1_tau...``; the parsed id has to be this cell's.
            if parsed is None or parsed[0] != cell["id"]:
                continue
            record = read_run(run_dir)
            if record is not None:
                folds[parsed[1]] = record
        if folds:
            found[cell["id"]] = {"cell": cell, "folds": folds}
    return found


def _agg(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    usable = [v for v in values if v is not None]
    if not usable:
        return {"mean": None, "sd": None, "n": 0}
    return {
        "mean": mean(usable),
        "sd": pstdev(usable) if len(usable) > 1 else 0.0,
        "n": len(usable),
    }


def summarise(found: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per cell: the fold means of both axes, and the fold count."""
    rows = []
    for cell_id, entry in found.items():
        folds = entry["folds"]
        rows.append({
            "id": cell_id,
            "path": entry["cell"]["path"],
            "rule": entry["cell"]["rule"],
            "flags": entry["cell"]["flags"],
            "note": entry["cell"]["note"],
            "folds": sorted(folds),
            "adaptation": _agg([r["adaptation"] for r in folds.values()]),
            "adaptation_test": _agg([r["adaptation_test"] for r in folds.values()]),
            "preservation": _agg([r["preservation"] for r in folds.values()]),
            "test_fallback": any(r["adaptation_is_test_fallback"] for r in folds.values()),
        })
    return rows


def select(rows: List[Dict[str, Any]], epsilon: float) -> Dict[str, Any]:
    """
    The constrained winner of each path, and the ranking it came from.

    Among the cells whose fold-mean preservation is within ``epsilon`` of the
    best preservation on that path, the winner is the one with the best
    fold-mean adaptation.  A path where no cell measured both axes has no
    winner rather than an arbitrary one.
    """
    result: Dict[str, Any] = {"epsilon": epsilon, "paths": {}}
    for path in PATHS:
        usable = [
            row for row in rows
            if row["path"] == path
            and row["adaptation"]["mean"] is not None
            and row["preservation"]["mean"] is not None
        ]
        if not usable:
            result["paths"][path] = {"winner": None, "eligible": 0, "ranked": []}
            continue
        best_preservation = max(row["preservation"]["mean"] for row in usable)
        floor = best_preservation - epsilon
        eligible = [row for row in usable if row["preservation"]["mean"] >= floor]
        ranked = sorted(eligible, key=lambda r: r["adaptation"]["mean"], reverse=True)
        result["paths"][path] = {
            "best_preservation": best_preservation,
            "preservation_floor": floor,
            "eligible": len(eligible),
            "considered": len(usable),
            "winner": ranked[0]["id"] if ranked else None,
            "ranked": [
                {
                    "id": r["id"],
                    "adaptation": r["adaptation"]["mean"],
                    "adaptation_sd": r["adaptation"]["sd"],
                    "preservation": r["preservation"]["mean"],
                    "preservation_sd": r["preservation"]["sd"],
                    "folds": len(r["folds"]),
                }
                for r in ranked
            ],
        }
    return result


def emit_full(selection, rows, folds, out: Path, rounds: int) -> int:
    """Write the winners and the controls out at the full horizon."""
    import sys

    # Sibling module, imported by path so the tool works from any cwd.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_stage6_screen import task_line

    by_id = {row["id"]: row for row in rows}
    cells = {cell["id"]: cell for cell in screen_cells()}

    chosen: List[str] = []
    for path in PATHS:
        winner = selection["paths"].get(path, {}).get("winner")
        if winner:
            chosen.append(winner)
    # The controls always come along: a winner is only meaningful next to the
    # plain rule it is supposed to beat, measured at the same horizon.
    for cell_id, cell in cells.items():
        if cell["path"] == "control" and cell_id not in chosen:
            chosen.append(cell_id)

    lines = [
        "# stage6_full.txt - the stage-6 winners at the full horizon.",
        "#",
        f"# Emitted by tools/select_agg_screen.py at epsilon={selection['epsilon']:g}",
        "# from the 25-round screen. The screen ranked; this measures. Every line",
        f"# runs {rounds} rounds under stage 5's protocol, so these results and the",
        "# stage-5 baseline are directly comparable.",
        "#",
        "# THE RULE: among the cells whose fold-mean preservation is within epsilon",
        "# of the best on their path, the one that adapts best. Selection read the",
        "# pooled VALIDATION accuracy; the test rows are what the paper reports.",
        "#",
    ]
    for path in PATHS:
        info = selection["paths"].get(path, {})
        winner = info.get("winner")
        if winner:
            row = by_id[winner]
            lines.append(
                f"# {path}: {winner} ({row['note']}) - adaptation "
                f"{row['adaptation']['mean']:.4f}, preservation "
                f"{row['preservation']['mean']:.4f}, "
                f"{info.get('eligible')} of {info.get('considered')} cells eligible"
            )
        else:
            lines.append(f"# {path}: no cell measured both axes; nothing selected")
    lines.append("#")
    lines.append(f"# {len(chosen)} cells x {len(folds)} folds = {len(chosen) * len(folds)} tasks")
    lines.append("#")

    for cell_id in chosen:
        cell = cells[cell_id]
        lines.append(f"# --- {cell_id}: {cell['note']}")
        for fold in folds:
            index = list(cells).index(cell_id)
            lines.append(task_line(cell, fold, rounds, 63000 + index * 10 + fold))

    out.write_text("\n".join(lines) + "\n")
    return len(chosen) * len(folds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Results root holding coh6_agg_* runs.")
    parser.add_argument(
        "--epsilon", type=float, required=True,
        help="Preservation budget: how much old-data accuracy a winner may spend.",
    )
    parser.add_argument("--out", default="stage6_full.txt")
    parser.add_argument("--report", default="stage6_selection.json")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--rounds", type=int, default=FULL_ROUNDS)
    args = parser.parse_args()

    cells = screen_cells()
    found = collect(Path(args.root), cells)
    rows = summarise(found)
    if not rows:
        raise SystemExit(
            f"No stage-6 screening results under {args.root}; expected folders "
            f"named {PARENT_PREFIX}<cell>_fold<k>."
        )
    selection = select(rows, args.epsilon)
    tasks = emit_full(selection, rows, args.folds, Path(args.out), args.rounds)

    Path(args.report).write_text(json.dumps(
        {"epsilon": args.epsilon, "cells_found": len(rows),
         "cells_expected": len(cells), "selection": selection, "cells": rows},
        indent=2,
    ))
    print(f"read {len(rows)} of {len(cells)} cells under {args.root}")
    for path in PATHS:
        info = selection["paths"].get(path, {})
        print(f"  {path}: winner={info.get('winner')} "
              f"({info.get('eligible', 0)} of {info.get('considered', 0)} eligible)")
    print(f"wrote {args.out}: {tasks} task lines")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
